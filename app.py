import streamlit as st
import os
import yaml
import time
from src.ingestion import extract_text_from_pdf, clean_text, process_and_save, sanitize_filename
from src.chunking import process_and_save_chunks
from src.vector_store import VectorDBManager
from src.rag_pipeline import RAGPipeline

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Sidebar and configuration setup
st.set_page_config(page_title="NCERT RAG Tutor", page_icon="📚", layout="wide")
config = load_config()

st.title("📚 NCERT RAG Tutor")
st.markdown("Upload your NCERT chapter PDF, ingest it, and ask questions! Ensure 'ollama run llama3' is running locally.")

# Session State for tracking progress
if 'ingested' not in st.session_state:
    st.session_state.ingested = False
if 'index_name' not in st.session_state:
    st.session_state.index_name = ""
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload and processing
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload NCERT PDF Chapter", type=['pdf'])
    
    if uploaded_file is not None and not st.session_state.ingested:
        if st.button("Process Document"):
            with st.spinner("Processing PDF and building Vector Index... This may take a moment."):
                # 1. Save uploaded file
                pdf_path = os.path.join(config["paths"]["raw_pdf_dir"], uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Ingestion
                st.text("1. Extracting Text...")
                extracted_yaml = process_and_save(pdf_path, config)
                
                # 3. Chunking
                st.text("2. Semantic Chunking...")
                chunks_yaml = process_and_save_chunks(extracted_yaml, config)
                
                # 4. Embeddings & Vector Store
                st.text("3. Building FAISS Index...")
                vdb = VectorDBManager(config)
                saved_index_path = vdb.build_from_chunks(chunks_yaml)
                
                st.session_state.index_name = os.path.basename(saved_index_path)
                st.session_state.ingested = True
                st.success(f"Processing Complete! Index: {st.session_state.index_name}")

if st.session_state.ingested:
    st.sidebar.success("Document loaded. Ready for questions.")
    
    # Init RAG Pipeline
    # Cache initialization so it doesn't reload embeddings every run
    @st.cache_resource
    def get_rag_pipeline(index):
        return RAGPipeline(index_name=index, config=config)
    
    try:
        rag = get_rag_pipeline(st.session_state.index_name)
    except Exception as e:
        st.error(f"Error loading RAG: {e}")
        st.stop()

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the textbook..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Fetch Answer
            response = rag.answer(prompt)
            
            answer_text = response["answer"]
            sources = ", ".join(response["sources"])
            
            if sources:
                answer_text += f"\n\n*Sources: {sources}*"
                
            message_placeholder.markdown(answer_text)
            
            with st.expander("View Retrieved Context"):
                st.text(response["context"])

        st.session_state.messages.append({"role": "assistant", "content": answer_text})
else:
    st.info("Please upload an NCERT PDF from the sidebar to get started.")

