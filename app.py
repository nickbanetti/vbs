import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import time
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gemini Vision Scanner Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👁️ AI Vision Board Scanner Pro")
st.markdown("""
**Batch process workshop boards into a single dataset.**
Supports: **Gemini 3.0**, **1.5 Pro**, and **Flash**.
""")

# --- 2. HELPER FUNCTIONS ---

def get_valid_models(api_key):
    """
    Dynamically asks Google: 'What models can I actually use?'
    This prevents the 404 error by never letting you select a bad model.
    """
    try:
        genai.configure(api_key=api_key)
        all_models = list(genai.list_models())
        
        # Filter for models that support content generation
        valid_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        # Sort specifically to put Gemini 3 / Pro at the top
        def model_sort_key(name):
            if "gemini-3" in name: return 0
            if "gemini-1.5-pro" in name: return 1
            if "gemini-2.0" in name: return 2
            if "flash" in name: return 3
            return 4
            
        valid_models.sort(key=model_sort_key)
        return valid_models
    except Exception as e:
        st.error(f"API Key Validation Error: {e}")
        return []

def analyze_single_image(image_bytes, model_name, filename):
    """
    Runs the 3-Stage Logic on a single image.
    """
    model = genai.GenerativeModel(model_name)
    
    # --- STAGE 1: STRUCTURE ---
    structure_prompt = """
    Analyze this image layout.
    1. Is it a "Dot Voting" matrix, "Sticky Notes", or "Hybrid"?
    2. If Matrix/Voting, identify Row Headers (Categories) and Column Headers (Sentiment/Options).
    
    Return JSON: 
    {
        "board_type": "hybrid/voting/notes", 
        "row_headers": ["list"], 
        "column_headers": ["list"]
    }
    """
    
    try:
        r1 = model.generate_content(
            [{'mime_type': 'image/jpeg', 'data': image_bytes}, structure_prompt],
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        structure = json.loads(r1.text)
    except Exception as e:
        return None, f"Stage 1 Failed: {e}"

    # --- STAGE 2: CONTEXT ---
    rows = structure.get('row_headers', [])
    cols = structure.get('column_headers', [])
    
    context = ""
    if rows and cols:
        context = f"""
        I have identified this is a Matrix.
        ROWS found: {rows}
        COLUMNS found: {cols}
        CRITICAL TASK: Look at EVERY intersection (Row x Column) and COUNT the dots/pins.
        """
    else:
        context = "Focus strictly on reading handwritten sticky notes and grouping them."

    # --- STAGE 3: EXECUTION ---
    final_schema = {
        "type": "OBJECT",
        "properties": {
            "voting_data": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "row_label": {"type": "STRING"},
                        "column_label": {"type": "STRING"},
                        "dot_count": {"type": "INTEGER"},
                        "color_breakdown": {"type": "STRING"}
                    }
                }
            },
            "sticky_notes": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "text": {"type": "STRING"},
                        "category_context": {"type": "STRING"},
                        "confidence": {"type": "INTEGER"}
                    }
                }
            }
        },
        "required": ["voting_data", "sticky_notes"]
    }
    
    final_prompt = f"""
    {context}
    OUTPUT RULES:
    1. voting_data: Return one entry for every cell in the matrix. If empty, set dot_count to 0.
    2. sticky_notes: Extract all legible handwritten text.
    """
    
    try:
        r3 = model.generate_content(
            [{'mime_type': 'image/jpeg', 'data': image_bytes}, final_prompt],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=final_schema,
                temperature=0.0
            )
        )
        return json.loads(r3.text), None
    except Exception as e:
        return None, f"Stage 3 Failed: {e}"

# --- 3. SIDEBAR & CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key Strategy
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ API Key Loaded")
    else:
        api_key = st.text_input("Enter Google API Key", type="password")

    # 2. Dynamic Model Loading (The Fix for 404s)
    if api_key:
        valid_models = get_valid_models(api_key)
        if valid_models:
            selected_model = st.selectbox("Select Model", valid_models, index=0)
            if "gemini-3" in selected_model:
                st.caption("✨ Gemini 3 Active")
        else:
            st.error("Invalid Key or No Models Available")
            st.stop()
    else:
        st.warning("Enter API Key to continue")
        st.stop()

# --- 4. MAIN UI ---
uploaded_files = st.file_uploader(
    "Upload Board Images (Select Multiple)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files and st.button(f"🚀 Process {len(uploaded_files)} Images"):
    
    # Master Containers
    all_votes = []
    all_notes = []
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})...")
        
        image_bytes = file.getvalue()
        
        # RUN ANALYSIS
        data, error = analyze_single_image(image_bytes, selected_model, file.name)
        
        if error:
            st.error(f"Error processing {file.name}: {error}")
            if "429" in error: 
                st.warning("Rate limit hit. Pausing for 60s...")
                time.sleep(60)
        elif data:
            # Append Votes
            if data.get("voting_data"):
                for v in data["voting_data"]:
                    v["source_file"] = file.name
                    all_votes.append(v)
            
            # Append Notes
            if data.get("sticky_notes"):
                for n in data["sticky_notes"]:
                    n["source_file"] = file.name
                    all_notes.append(n)
        
        # Update Progress
        progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("✅ Batch Processing Complete!")
    
    # --- 5. EXPORT SECTION ---
    st.divider()
    st.subheader("📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    # PREPARE DATAFRAMES
    df_votes = pd.DataFrame(all_votes)
    df_notes = pd.DataFrame(all_notes)
    
    # OPTION 1: EXCEL (Tabs)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        if not df_votes.empty:
            df_votes.to_excel(writer, sheet_name='Voting Data', index=False)
        if not df_notes.empty:
            df_notes.to_excel(writer, sheet_name='Sticky Notes', index=False)
            
    with col1:
        st.download_button(
            label="📄 Download Excel (Tabs)",
            data=buffer.getvalue(),
            file_name="batch_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # OPTION 2: MASTER CSV
    # We merge everything into one massive list if possible, or just give the main one
    with col2:
        if not df_notes.empty:
            csv_data = df_notes.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download Master CSV (Notes)",
                data=csv_data,
                file_name="master_notes.csv",
                mime="text/csv"
            )
            
    # PREVIEW
    st.write("### Data Preview")
    if not df_votes.empty:
        st.write("**Voting Data**")
        st.dataframe(df_votes.head())
        
    if not df_notes.empty:
        st.write("**Sticky Notes**")
        st.dataframe(df_notes.head())
