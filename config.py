# Configuration file for StudyMate AI

# Model Configuration
MODEL_CONFIG = {
    "model_name": "ibm/granite-2b-instruct-v1",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "device": "cpu",  # Force CPU for compatibility
}

# PDF Processing Configuration
PDF_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 100,
}

# Vector Store Configuration
VECTOR_CONFIG = {
    "top_k": 3,  # Number of relevant chunks to retrieve
}

# UI Configuration
UI_CONFIG = {
    "page_title": "StudyMate - AI-Powered PDF Q&A",
    "page_icon": "📚",
    "layout": "wide",
}

# Performance Optimizations (No longer needed with fast model)
PERFORMANCE_CONFIG = {
    "enable_model_caching": True,
    "lazy_model_loading": False, # Load model at start
    "show_progress_indicators": True,
}
