import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gemini Vision Scanner", layout="wide")

st.title("👁️ AI Vision Board Scanner")
st.markdown("Upload a **Sticky Note Wall**, **Dot Voting Board**, or **Hybrid** to extract data to CSV.")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google API Key", type="password")
    st.markdown("[Get an API Key](https://aistudio.google.com/app/apikey)")
    
    model_choice = st.selectbox(
        "Select Model", 
        ["gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-1.5-flash"],
        index=0
    )

# --- FUNCTIONS ---
def analyze_image(image_bytes, model_name, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # STAGE 1: STRUCTURE
    status_text.text("🔹 Stage 1: Analyzing Layout & Structure...")
    progress_bar.progress(33)
    
    structure_prompt = """
    Analyze this image layout.
    1. Is it a "Dot Voting" matrix, "Sticky Notes", or "Hybrid"?
    2. If Matrix, identify Row Headers (Categories) and Column Headers (Options).
    Return JSON: {"board_type": "...", "row_headers": [], "column_headers": []}
    """
    
    try:
        r1 = model.generate_content(
            [{'mime_type': 'image/jpeg', 'data': image_bytes}, structure_prompt],
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        structure = json.loads(r1.text)
    except Exception as e:
        st.error(f"Stage 1 Failed: {e}")
        return None

    # STAGE 2 & 3: EXTRACTION
    status_text.text(f"🔹 Stage 2: Detected {structure.get('board_type')}. Extracting Data...")
    progress_bar.progress(66)
    
    final_prompt = f"""
    Context: This is a {structure.get('board_type')} board.
    Rows Detected: {structure.get('row_headers')}
    Columns Detected: {structure.get('column_headers')}
    
    TASK:
    1. If Matrix: Count dots/pins at every intersection. Return 0 if empty.
    2. If Notes: Extract all handwritten text.
    
    Return strict JSON:
    {{
        "voting_data": [
            {{"row": "...", "col": "...", "count": 0}}
        ],
        "notes": [
            {{"text": "...", "section": "..."}}
        ]
    }}
    """
    
    try:
        r3 = model.generate_content(
            [{'mime_type': 'image/jpeg', 'data': image_bytes}, final_prompt],
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        return json.loads(r3.text)
    except Exception as e:
        st.error(f"Stage 3 Failed: {e}")
        return None

# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file and api_key:
    # Show Image
    st.image(uploaded_file, caption="Uploaded Board", use_container_width=True)
    
    if st.button("🚀 Run Analysis"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run Analysis
        image_bytes = uploaded_file.getvalue()
        data = analyze_image(image_bytes, model_choice, api_key)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis Complete!")
        
        if data:
            # --- TAB 1: VOTING DATA ---
            tab1, tab2 = st.tabs(["📊 Voting Matrix", "📝 Sticky Notes"])
            
            with tab1:
                if data.get("voting_data"):
                    df_votes = pd.DataFrame(data["voting_data"])
                    
                    # Pivot for Grid View
                    try:
                        grid = df_votes.pivot(index='row', columns='col', values='count')
                        st.dataframe(grid, use_container_width=True)
                        
                        # Download Button
                        csv = grid.to_csv().encode('utf-8')
                        st.download_button(
                            "📥 Download Grid CSV",
                            csv,
                            "voting_grid.csv",
                            "text/csv"
                        )
                    except:
                        st.dataframe(df_votes)
                else:
                    st.info("No voting matrix data detected.")

            # --- TAB 2: NOTES ---
            with tab2:
                if data.get("notes"):
                    df_notes = pd.DataFrame(data["notes"])
                    st.dataframe(df_notes, use_container_width=True)
                    
                    # Download Button
                    csv_notes = df_notes.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Notes CSV",
                        csv_notes,
                        "notes.csv",
                        "text/csv"
                    )
                else:
                    st.info("No sticky notes detected.")

elif not api_key:
    st.warning("⚠️ Please enter your API Key in the sidebar to start.")