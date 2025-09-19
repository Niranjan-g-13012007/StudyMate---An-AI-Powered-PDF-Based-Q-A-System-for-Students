import os
import streamlit as st
from typing import List, Dict, Any, Optional
import tempfile
from pathlib import Path

# Import local modules
from pdf_utils import PDFProcessor
from vectorstore import VectorStore
from llm_granite import GraniteLLM
from config import MODEL_CONFIG, PDF_CONFIG, VECTOR_CONFIG, UI_CONFIG, PERFORMANCE_CONFIG

# Set page configuration
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"]
)

# Custom CSS for better UI
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
        border-radius: 5px;
        padding: 0.5rem 1rem;
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
    .success-msg {
        color: #4CAF50;
        font-weight: bold;
    }
    .warning-msg {
        color: #ff9800;
        font-weight: bold;
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
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

# Initialize PDF Processor
pdf_processor = PDFProcessor(
    chunk_size=PDF_CONFIG["chunk_size"], 
    chunk_overlap=PDF_CONFIG["chunk_overlap"]
)

@st.cache_resource
def get_llm_instance():
    """Get a cached instance of the Granite LLM model."""
    return GraniteLLM(
        model_name=MODEL_CONFIG["model_name"],
        max_new_tokens=MODEL_CONFIG["max_new_tokens"],
        temperature=MODEL_CONFIG["temperature"]
    )

def init_llm():
    """Initialize the Granite LLM model."""
    if st.session_state.llm is None:
        with st.spinner("🔧 Loading AI model (this may take a minute on first run)..."):
            # Show progress information
            progress_placeholder = st.empty()
            progress_placeholder.info("📥 Downloading model files (only on first run)...")
            
            st.session_state.llm = get_llm_instance()
            
            progress_placeholder.success("✅ Model loaded successfully!")
            progress_placeholder.empty()

def process_uploaded_files(uploaded_files) -> bool:
    """Process uploaded PDF files and update the vector store."""
    if not uploaded_files:
        return False
    
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name in [f['name'] for f in st.session_state.uploaded_files]:
            continue  # Skip already processed files
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract and clean text from PDF
            text = pdf_processor.extract_text_from_pdf(tmp_file_path)
            cleaned_text = pdf_processor.clean_text(text)
            
            # Split text into chunks using the PDFProcessor instance method
            chunks = pdf_processor.split_text(cleaned_text, 
                                           chunk_size=500, 
                                           chunk_overlap=100)
            
            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'source': uploaded_file.name,
                    'chunk_id': f"{uploaded_file.name}_chunk_{i}"
                })
            
            # Add to uploaded files list
            st.session_state.uploaded_files.append({
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'chunks': len(chunks)
            })
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            return False
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    # Update vector store if we have chunks
    if all_chunks:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()
        
        # Add chunks to vector store
        st.session_state.vector_store.add_texts(all_chunks)
        st.session_state.processing_done = True
        return True
    
    return False

def generate_response(question: str) -> str:
    """Generate a response to the user's question using the vector store and LLM."""
    if not question or not st.session_state.vector_store:
        return None
    
    try:
        # Get relevant chunks from vector store
        relevant_chunks = st.session_state.vector_store.search(question, top_k=VECTOR_CONFIG["top_k"])
        
        if not relevant_chunks:
            return "I couldn't find any relevant information in the uploaded documents."
        
        # Format context
        context_chunks = []
        for result in relevant_chunks:
            if isinstance(result, tuple) and len(result) == 2:
                chunk, _ = result
            else:
                chunk = result
            source = chunk.get('source', 'Unknown')
            text = chunk.get('text', '')
            context_chunks.append(f"Source: {source}\n{text}")
        
        context = "\n\n".join(context_chunks)
        
        # Generate prompt for LLM with more specific instructions
        prompt = f"""You are an expert research assistant. Your task is to provide accurate and concise information based on the provided context.
        
        Context from source materials:
        {context}
        
        Question: {question}
        
        Please provide a response that:
        1. Directly answers the question using only information from the provided context
        2. Is factually accurate and free from speculation
        3. If the answer isn't in the context, state: "The information needed to answer this question was not found in the provided materials."
        4. For numerical data, include exact figures and units when available
        5. Keep the response focused and to the point
        
        Answer:
        """
        
        # Get answer from LLM using the generate method
        answer = st.session_state.llm.generate(
            query=question,
            context=context,
            system_prompt="""You are an expert research assistant. Your task is to provide accurate and concise information based on the provided context.
            
            Please provide a response that:
            1. Directly answers the question using only information from the provided context
            2. Is factually accurate and free from speculation
            3. If the answer isn't in the context, state: "The information needed to answer this question was not found in the provided materials."
            4. For numerical data, include exact figures and units when available
            5. Keep the response focused and to the point
            """
        )
        return answer.strip()
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    # Header
    st.title("📚 StudyMate - AI-Powered PDF Q&A")
    st.markdown("Upload your study materials and get answers to your questions instantly!")
    
    # Show model status
    if st.session_state.llm is None:
        st.info("🤖 AI model will be loaded when you ask your first question. This may take a few minutes on the first run.")
    
    # File Upload Section
    st.header("1. Upload Study Materials")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Process uploaded files
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("🔍 Processing your documents..."):
            success = process_uploaded_files(uploaded_files)
            if success:
                st.success("✅ Documents processed successfully!")
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.subheader("📂 Uploaded Files")
        for file_info in st.session_state.uploaded_files:
            st.write(f"📄 {file_info['name']} ({file_info['chunks']} chunks)")
    
    # Q&A Section
    st.header("2. Ask a Question")
    question = st.text_area(
        "Type your question here...",
        height=100,
        key="question_input"
    )
    
    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question.")
        elif not st.session_state.vector_store or not st.session_state.uploaded_files:
            st.warning("Please upload and process at least one PDF file first.")
        else:
            # Initialize LLM if not already loaded
            if st.session_state.llm is None:
                init_llm()
            
            with st.spinner("🤔 Thinking..."):
                answer = generate_response(question)
                
                if answer:
                    st.subheader("💡 Answer")
                    st.markdown(answer)
                    
                    # Get relevant chunks
                    relevant_chunks = st.session_state.vector_store.search(question, top_k=VECTOR_CONFIG["top_k"])
                    with st.expander("🔍 View Source Materials"):
                        for i, result in enumerate(relevant_chunks, 1):
                            if isinstance(result, tuple) and len(result) == 2:
                                chunk, score = result
                                source = chunk.get('source', 'Unknown')
                                text = chunk.get('text', '')
                            else:
                                # Handle case where result is just the chunk dict
                                chunk = result
                                score = result.get('score', 1.0)
                                source = result.get('source', 'Unknown')
                                text = result.get('text', '')
                            
                            st.markdown(f"**Source {i} (Relevance: {score:.2f}):** {source}")
                            st.text(text[:500] + ("..." if len(text) > 500 else ""))
                            st.markdown("---")

if __name__ == "__main__":
    main()
