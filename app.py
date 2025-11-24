import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import time
import io
import re

# --- 1. BANETTI BRANDING & CONFIG ---
st.set_page_config(
    page_title="Banetti Asset Intelligence",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Banetti Corporate CSS (Red/Navy Theme)
st.markdown("""
    <style>
        /* Primary Red Buttons */
        .stButton > button {
            background-color: #CF2E2E !important; 
            color: white !important;
            border-radius: 5px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #A52424 !important;
        }
        /* Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #F8F9FA;
        }
        /* Headers */
        h1, h2, h3 {
            color: #1C245D;
        }
        /* Progress Bar Color */
        .stProgress > div > div > div > div {
            background-color: #CF2E2E;
        }
    </style>
""", unsafe_allow_html=True)

# Header with Correct Banetti Logo
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://banetti.com/wp-content/uploads/2013/04/Baneti_Logo_Web21-300x133.png", width=150)
with col_title:
    st.title("Asset Intelligence Scanner")

st.markdown("""
**Enterprise Vision Processing for Workshop Boards & Asset Data.**
Unified extraction for **Sticky Notes**, **Dot-Voting Matrices**, and **Hybrid Layouts**.
""")

# --- 2. HELPER FUNCTIONS ---

def clean_json_string(json_str):
    """
    Cleans markdown formatting from JSON strings (e.g. ```json ... ```)
    """
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*', '', json_str)
    return json_str.strip()

def get_valid_models(api_key):
    """
    Prevents 404 Errors by asking Google exactly what this Key is allowed to touch.
    """
    try:
        genai.configure(api_key=api_key)
        all_models = list(genai.list_models())
        valid_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        def model_sort_key(name):
            # Priority Rank: Gemini 3 -> 1.5 Pro -> 2.0 -> Flash
            if "gemini-3" in name: return 0
            if "gemini-1.5-pro" in name: return 1
            if "gemini-2.0" in name: return 2
            if "flash" in name: return 3
            return 4
            
        valid_models.sort(key=model_sort_key)
        return valid_models
    except Exception as e:
        st.error(f"Authentication Error: {e}")
        return []

# --- 3. ANALYSIS ENGINE ---
def analyze_single_image(image_bytes, model_name, filename, status_container=None):
    model = genai.GenerativeModel(model_name)
    
    # STAGE 1: Structural Recognition
    if status_container:
        status_container.write("🔹 Stage 1: Identifying Board Architecture...")
        
    structure_prompt = """
    Analyze this workshop board.
    1. CLASSIFY: Is it "Dot Voting" (Matrix), "Sticky Notes" (Text), or "Hybrid"?
    2. MAPPING: If Matrix, identify Row Headers (Categories) and Column Headers (Sentiment/Options).
    
    Return JSON: {"board_type": "...", "row_headers": [], "column_headers": []}
    """
    
    try:
        r1 = model.generate_content(
            [{'mime_type': 'image/jpeg', 'data': image_bytes}, structure_prompt],
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        # Clean JSON before loading
        structure = json.loads(clean_json_string(r1.text))
        
        if status_container:
            status_container.write(f"✅ Detected: {structure.get('board_type', 'Unknown')}")
    except Exception as e:
        return None, f"Structure Analysis Failed: {e}"

    # STAGE 2: Context Injection
    if status_container:
        status_container.write("🔹 Stage 2: Formulating Counting Strategy...")
        
    rows = structure.get('row_headers', [])
    cols = structure.get('column_headers', [])
    
    context = ""
    if rows and cols:
        context = f"""
        MATRIX DETECTED.
        ROWS: {rows}
        COLUMNS: {cols}
        TASK: Count dots/pins at every intersection.
        """
    else:
        context = "TEXT DETECTED. Extract all handwritten notes and categorize them by spatial clusters."

    # STAGE 3: Final Extraction
    if status_container:
        status_container.write(f"🔹 Stage 3: Running Extraction on {model_name}...")
        
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
    REQUIREMENTS:
    1. voting_data: One entry per matrix cell. If empty, count=0.
    2. sticky_notes: Complete transcription of all legible text.
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
        return json.loads(clean_json_string(r3.text)), None
    except Exception as e:
        return None, f"Extraction Failed: {e}"

# --- 4. APPLICATION INTERFACE ---
with st.sidebar:
    st.header("System Configuration")
    
    # Secure Key Handling
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("Locked & Loaded (Secrets)")
    else:
        api_key = st.text_input("API Key", type="password")

    # Dynamic Model Selector
    if api_key:
        valid_models = get_valid_models(api_key)
        if valid_models:
            selected_model = st.selectbox("Processing Engine", valid_models, index=0)
            if "gemini-3" in selected_model:
                st.caption("🚀 Power Mode: Gemini 3 Enabled")
        else:
            st.error("No valid models found for this key.")
            st.stop()
    else:
        st.warning("Awaiting Credentials...")
        st.stop()

# Main Upload Area
uploaded_files = st.file_uploader(
    "Batch Upload Board Images", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files and st.button(f"Process {len(uploaded_files)} Files"):
    
    # Master Data Containers
    all_votes = []
    all_notes = []
    
    # GLOBAL PROGRESS BAR
    progress_bar = st.progress(0)
    
    # Batch Processing Loop
    for i, file in enumerate(uploaded_files):
        
        # LIVE STATUS CONTAINER
        with st.status(f"Processing **{file.name}**...", expanded=True) as status:
            image_bytes = file.getvalue()
            
            # Pass the status container to the function
            data, error = analyze_single_image(image_bytes, selected_model, file.name, status_container=status)
            
            if error:
                if "429" in str(error):
                    status.update(label=f"⚠️ Rate Limit Hit on {file.name}", state="error")
                    st.warning("API Rate Limit reached. Cooling down for 60s...")
                    # Visual Countdown
                    with st.empty():
                        for seconds in range(60, 0, -1):
                            st.write(f"⏳ Resuming in {seconds}s...")
                            time.sleep(1)
                        st.write("🔄 Resuming...")
                else:
                    status.update(label=f"❌ Failed: {file.name}", state="error")
                    st.error(f"Skipped {file.name}: {error}")
            
            elif data:
                status.update(label=f"✅ Completed: {file.name}", state="complete")
                
                # Aggregate Data
                if data.get("voting_data"):
                    for v in data["voting_data"]:
                        v["source_file"] = file.name
                        all_votes.append(v)
                
                if data.get("sticky_notes"):
                    for n in data["sticky_notes"]:
                        n["source_file"] = file.name
                        all_notes.append(n)
        
        # Update Global Progress
        progress_bar.progress((i + 1) / len(uploaded_files))

    st.success("Batch Processing Complete.")
    
    # --- 5. CONSOLIDATED EXPORT ---
    st.divider()
    st.subheader("Global Data Export")
    
    col1, col2 = st.columns(2)
    
    df_votes = pd.DataFrame(all_votes)
    df_notes = pd.DataFrame(all_notes)
    
    # Excel Export (Multi-Tab)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        if not df_votes.empty:
            df_votes.to_excel(writer, sheet_name='Dot Voting Data', index=False)
        if not df_notes.empty:
            df_notes.to_excel(writer, sheet_name='Sticky Notes Text', index=False)
            
    with col1:
        st.download_button(
            label="📥 Download Consolidated Excel",
            data=buffer.getvalue(),
            file_name="Banetti_Workshop_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Master CSV Export (Notes Only)
    with col2:
        if not df_notes.empty:
            csv_data = df_notes.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Master CSV (Text)",
                data=csv_data,
                file_name="Banetti_Master_Notes.csv",
                mime="text/csv"
            )
            
    # Data Previews
    if not df_votes.empty:
        st.write("### Global Voting Matrix")
        st.dataframe(df_votes, use_container_width=True)
        
    if not df_notes.empty:
        st.write("### Global Sticky Notes")
        st.dataframe(df_notes, use_container_width=True)
