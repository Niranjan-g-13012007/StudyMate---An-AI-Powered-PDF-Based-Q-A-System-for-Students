import os
import streamlit as st
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
from pdf_utils import extract_text_from_pdf, split_text_into_chunks
from vectorstore import VectorStore
from llm_granite import GraniteLLM

# Set page config
st.set_page_config(
    page_title="StudyMate - AI-Powered PDF Q&A",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
    .file-uploader {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize LLM
def init_llm():
    if st.session_state.llm is None:
        with st.spinner("Loading AI model..."):
            st.session_state.llm = GraniteLLM(
                model_name="ibm-granite/granite-3.3-2b-instruct",
                max_new_tokens=512,
                temperature=0.7
            )

def process_uploaded_files(uploaded_files) -> List[str]:
    """Process uploaded PDF files and return extracted text chunks."""
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(tmp_file_path)
            
            # Split text into chunks
            chunks = split_text_into_chunks(text)
            all_chunks.extend(chunks)
            
            # Add to uploaded files list
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    return all_chunks

def main():
    st.title("📚 StudyMate - AI-Powered PDF Q&A")
    st.markdown("Upload your study materials and ask questions!")  

    # Initialize LLM
    init_llm()

    # File Upload Section
    st.header("1. Upload Study Materials")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Process uploaded files
    if uploaded_files and len(uploaded_files) > 0:
        with st.spinner("Processing your documents..."):
            chunks = process_uploaded_files(uploaded_files)
            
            if chunks:
                # Initialize or update vector store
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = VectorStore()
                
                # Add chunks to vector store
                st.session_state.vector_store.add_documents(chunks)
                st.success(f"Processed {len(chunks)} text chunks from {len(uploaded_files)} file(s)")

    # Display uploaded files
    if st.session_state.uploaded_files:
        st.subheader("Uploaded Files")
        for file_name in st.session_state.uploaded_files:
            st.write(f"📄 {file_name}")

    # Q&A Section
    st.header("2. Ask a Question")
    question = st.text_area(
        "Type your question here...",
        height=100,
        key="question_input"
    )

    if st.button("Get Answer", key="get_answer"):
        if not question:
            st.warning("Please enter a question.")
        elif st.session_state.vector_store is None or len(st.session_state.vector_store.documents) == 0:
            st.warning("Please upload at least one PDF file first.")
        else:
            with st.spinner("Searching for relevant information..."):
                # Get relevant chunks
                relevant_chunks = st.session_state.vector_store.search(question, k=3)
                
                # Format context
                context = "\n\n".join([chunk for chunk, _ in relevant_chunks])
                
                # Generate prompt
                prompt = f"""You are StudyMate, an AI assistant helping students with their studies.
                
                Context from study materials:
                {context}
                
                Question: {question}
                
                Please provide a clear and concise answer based on the provided context. If the answer cannot be found in the context, say "I couldn't find the answer in the provided materials."
                """
                
                # Get answer from LLM
                answer = st.session_state.llm.generate_response(prompt)
                
                # Display answer
                st.subheader("Answer")
                st.markdown(answer)
                
                # Show sources
                with st.expander("View source materials"):
                    for i, (chunk, score) in enumerate(relevant_chunks, 1):
                        st.markdown(f"**Source {i} (Relevance: {score:.2f}):**")
                        st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))

if __name__ == "__main__":
    main()
