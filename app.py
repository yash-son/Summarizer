import streamlit as st
import json
import os
from rag_engine import MedicalRAGEngine
from summarizer import MedicalSummarizer
from dotenv import load_dotenv

# Page config
st.set_page_config(page_title="MedSum: Radiology Simplifier", page_icon="🏥", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .report-container {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .summary-container {
        padding: 20px;
        border-radius: 10px;
        background-color: #e3f2fd;
        border-left: 5px solid #007bff;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize engines
@st.cache_resource
def init_engines():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Please set your GOOGLE_API_KEY in the environment or .env file.")
        st.stop()
    
    rag = MedicalRAGEngine()
    # Check if index exists, else create
    if not os.path.exists("./chroma_db"):
        with st.spinner("Initializing medical index..."):
            rag.load_and_index()
    
    summarizer = MedicalSummarizer()
    return rag, summarizer

try:
    rag_engine, summarizer = init_engines()
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()

# Sidebar: Load Data
st.sidebar.title("🏥 MedSum AI")
st.sidebar.markdown("---")

# Sample selection
with open("data/sample_reports.json", "r") as f:
    samples = json.load(f)

sample_options = {f"Patient {s['id']} - {s['type']}": s for s in samples}
selected_label = st.sidebar.selectbox("Select a Sample Report", list(sample_options.keys()))
selected_report = sample_options[selected_label]

st.sidebar.markdown("---")
st.sidebar.info("This tool uses RAG to simplify medical reports. Always consult with a doctor for medical advice.")

# Main UI
st.title("Radiology Report Simplifier")
st.markdown("Translating complex medical findings into patient-friendly summaries.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Original Radiology Report")
    st.markdown(f"""
    <div class="report-container">
        <strong>Report ID:</strong> {selected_report['id']}<br>
        <strong>Examination:</strong> {selected_report['type']}<br>
        <hr>
        <pre style="white-space: pre-wrap; font-family: sans-serif;">{selected_report['report']}</pre>
        <hr>
        <strong>Patient History:</strong> {selected_report['patient_history']}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("Patient-Friendly Summary")
    
    if st.button("Generate Summary"):
        with st.spinner("Simplifying..."):
            # Retrieve relevant context
            context_docs = rag_engine.search_reports(selected_report['report'])
            # Generate summary
            summary = summarizer.summarize(selected_report['report'], context_docs)
            st.markdown(f"""
            <div class="summary-container">
                {summary}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Click the button above to generate a simplified summary.")

st.markdown("---")
st.subheader("Ask a Question about this Report")
user_query = st.text_input("e.g., 'What is pneumonia?' or 'Are my lungs okay?'")

if user_query:
    with st.spinner("Searching and answering..."):
        context_docs = rag_engine.search_reports(user_query)
        answer = summarizer.ask_question(user_query, selected_report['report'], context_docs)
        st.chat_message("assistant").write(answer)
