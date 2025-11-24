import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gemini Vision Scanner",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👁️ AI Vision Board Scanner")
st.markdown("""
**Upload a workshop board to extract data.**
Supports: 
* 🟦 **Dot-Voting Matrices** (Counts pins by row/column)
* 📝 **Sticky Note Walls** (Extracts text & categories)
* 🌗 **Hybrid Boards** (Does both simultaneously)
""")

# --- 2. SIDEBAR & SECURE CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # SECURE API KEY HANDLING
    # 1. Try to load from Streamlit Secrets (Best for Cloud)
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ API Key loaded from Secrets")
    else:
        # 2. Fallback to Manual Entry (Best for Local Testing)
        api_key = st.text_input("Enter Google API Key", type="password", help="Get one at aistudio.google.com")
        if not api_key:
            st.warning("⚠️ API Key required to run.")
            st.stop()

    # MODEL SELECTION
    # We prioritize models capable of visual reasoning
    model_options = ["gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-1.5-flash"]
    selected_model = st.selectbox("Select Model", model_options, index=0)
    
    st.divider()
    st.info("Tip: 'Pro' models are better at counting small dots. 'Flash' is faster for text.")

# --- 3. CORE ANALYTICS ENGINE ---
def analyze_image_pipeline(image_bytes, model_name, api_key):
    """
    Executes the 3-Stage 'Chain of Thought' Analysis
    """
    genai.configure(api_key=api_key)
    
    # Initialize Model
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None

    status = st.empty()
    progress = st.progress(0)

    # --- STAGE 1: STRUCTURAL ANALYSIS ---
    status.text("🔹 Stage 1: Analyzing Board Layout & Legends...")
    progress.progress(25)
    
    structure_prompt = """
    Analyze this image layout.
    1. Is it a "Dot Voting" matrix, "Sticky Notes", or "Hybrid"?
    2. If Matrix/Voting, identify Row Headers (Categories) and Column Headers (Sentiment/Options).
    3. Identify the Legend (e.g., Blue=Dev, Red=Biz).
    
    Return JSON: 
    {
        "board_type": "hybrid/voting/notes", 
        "row_headers": ["list"], 
        "column_headers": ["list"],
        "legend": ["list"]
    }
    """
    
    try:
        r1 = model.generate_content(
            [{'mime_type': 'image/jpeg', 'data': image_bytes}, structure_prompt],
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        structure = json.loads(r1.text)
    except Exception as e:
        st.error(f"Stage 1 Failed (Structure Detection): {e}")
        return None

    # --- STAGE 2: STRATEGY INJECTION ---
    status.text(f"🔹 Stage 2: Detected {structure.get('board_type', 'Unknown')}. Building Counting Strategy...")
    progress.progress(50)
    
    # Dynamic Context Injection based on findings
    rows = structure.get('row_headers', [])
    cols = structure.get('column_headers', [])
    
    context_instruction = ""
    if rows and cols:
        context_instruction = f"""
        I have identified this is a Matrix.
        ROWS found: {rows}
        COLUMNS found: {cols}
        CRITICAL TASK: Look at EVERY intersection (Row x Column) and COUNT the dots/pins.
        """
    else:
        context_instruction = "Focus strictly on reading handwritten sticky notes and grouping them."

    # --- STAGE 3: EXECUTION ---
    status.text("🔹 Stage 3: Extracting Data & Counting Dots...")
    progress.progress(75)
    
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
                        "dot_count": {"type": "INTEGER", "description": "Exact count of pins/dots"},
                        "color_breakdown": {"type": "STRING", "description": "e.g. '3 blue, 2 red'"}
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
    {context_instruction}
    
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
        progress.progress(100)
        status.text("✅ Analysis Complete!")
        return json.loads(r3.text)
        
    except Exception as e:
        st.error(f"Stage 3 Failed (Extraction): {e}")
        if "429" in str(e):
            st.warning("⚠️ Quota Limit Reached. Please wait 60 seconds.")
        return None

# --- 4. MAIN UI LOGIC ---
uploaded_file = st.file_uploader("Upload Board Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display Image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Source Board", use_container_width=True)
    
    with col2:
        st.write("### Ready to Scan")
        if st.button("🚀 Run Analysis", type="primary"):
            
            # RUN THE PIPELINE
            image_bytes = uploaded_file.getvalue()
            result_data = analyze_image_pipeline(image_bytes, selected_model, api_key)
            
            if result_data:
                # --- RESULT TABS ---
                tab_votes, tab_notes, tab_json = st.tabs(["📊 Voting Grid", "📝 Sticky Notes", "🔍 Raw JSON"])
                
                # TAB 1: VOTING MATRIX
                with tab_votes:
                    if result_data.get("voting_data"):
                        df_votes = pd.DataFrame(result_data["voting_data"])
                        
                        # Attempt to create a Pivot Table (Grid View)
                        try:
                            grid_view = df_votes.pivot(index='row_label', columns='column_label', values='dot_count')
                            st.subheader("Matrix View")
                            st.dataframe(grid_view, use_container_width=True)
                            
                            # CSV Download (Grid)
                            csv_grid = grid_view.to_csv().encode('utf-8')
                            st.download_button("📥 Download Grid CSV", csv_grid, "voting_matrix.csv", "text/csv")
                        except:
                            st.dataframe(df_votes) # Fallback to list view
                            
                    else:
                        st.info("No voting/matrix data detected in this image.")

                # TAB 2: STICKY NOTES
                with tab_notes:
                    if result_data.get("sticky_notes"):
                        df_notes = pd.DataFrame(result_data["sticky_notes"])
                        st.dataframe(df_notes, use_container_width=True)
                        
                        # CSV Download (Notes)
                        csv_notes = df_notes.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Notes CSV", csv_notes, "sticky_notes.csv", "text/csv")
                    else:
                        st.info("No sticky notes detected.")

                # TAB 3: DEBUG
                with tab_json:
                    st.json(result_data)
