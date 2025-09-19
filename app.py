import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from flask import (
    Flask, 
    request, 
    render_template, 
    redirect, 
    url_for, 
    jsonify, 
    session, 
    send_from_directory,
    Response
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import local modules
from pdf_utils import PDFProcessor
from vectorstore import VectorStore
from llm_granite import GraniteLLM

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_FOLDER = 'uploads'
VECTOR_STORE_DIR = 'vector_store'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Test endpoint
@app.route('/api/test')
def test_endpoint():
    return jsonify({
        'status': 'success',
        'message': 'Flask is working!',
        'time': datetime.now().isoformat()
    })
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize components with error handling
try:
    print("Initializing components...")
    pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    print("PDF processor initialized successfully")
    
    vector_store = VectorStore(model_name='all-MiniLM-L6-v2')
    print("Vector store initialized successfully")
    
    # Initialize LLM with CPU
    print("Initializing LLM with CPU...")
    try:
        llm = GraniteLLM(
            model_name="ibm-granite/granite-3.3-2b-instruct",
            device='cpu',  # Force CPU usage
            load_in_4bit=False,  # Disable 4-bit quantization for CPU
            max_new_tokens=512,  # Reduce max tokens for CPU
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1
        )
        print("LLM initialized successfully")
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        print("Trying with smaller model...")
        # Try with a smaller model as fallback
        llm = GraniteLLM(
            model_name="gpt2",  # Much smaller model for testing
            device='cpu',
            load_in_4bit=False,
            max_new_tokens=100
        )
        print("Fallback model loaded successfully")
        
except Exception as e:
    print(f"Fatal error during initialization: {str(e)}")
    print("Please check the error message above and ensure all dependencies are properly installed.")
    print("You may need to run: pip install -r requirements.txt")
    import sys
    sys.exit(1)

# Load or initialize chat history
if not os.path.exists('chat_history.json'):
    with open('chat_history.json', 'w') as f:
        json.dump([], f)

def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_chat_history(history: List[Dict[str, Any]]) -> None:
    """Save chat history to a JSON file."""
    with open('chat_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_chat_history() -> List[Dict[str, Any]]:
    """Load chat history from a JSON file."""
    try:
        with open('chat_history.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

@app.route('/')
def index():
    """Render the main page with the chat interface."""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle PDF file uploads."""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No selected files'}), 400
    
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Save the file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process the PDF
                chunks = pdf_processor.process_pdf(filepath)
                
                # Add to vector store
                vector_store.add_texts(chunks)
                
                results.append({
                    'filename': filename,
                    'status': 'success',
                    'chunks_processed': len(chunks)
                })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': str(e)
                })
    
    return jsonify({'results': results})

@app.route('/api/chat', methods=['GET', 'POST'])
def chat():
    """Handle chat messages and generate responses."""
    if request.method == 'GET':
        question = request.args.get('question', '').strip()
    else:  # POST
        data = request.get_json()
        question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Search for relevant context
        search_results = vector_store.search(
            query=question,
            top_k=5,
            min_score=0.5
        )
        
        # Extract context from search results
        context = "\n\n".join([
            f"Document: {result.get('source', 'Unknown')}\n{result['text']}" 
            for result in search_results
        ])
        
        # Generate response
        system_prompt = (
            "You are a helpful AI assistant that answers questions based on the provided context. "
            "If the context doesn't contain the answer, say that you don't know. "
            "Be concise and accurate in your responses."
        )
        
        # Stream the response
        def generate():
            full_response = ""
            for chunk in llm.generate_stream(
                query=question,
                context=context,
                system_prompt=system_prompt
            ):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Save to chat history
            history = load_chat_history()
            history.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'response': full_response,
                'sources': list({result.get('source', 'Unknown') for result in search_results})
            })
            save_chat_history(history[-100:])  # Keep last 100 messages
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        return app.response_class(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

@app.route('/api/history', methods=['GET'])
def get_chat_history():
    """Retrieve chat history."""
    history = load_chat_history()
    return jsonify(history)

@app.route('/api/files', methods=['GET'])
def list_uploaded_files():
    """List all uploaded PDF files."""
    files = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            files.append({
                'name': filename,
                'size': os.path.getsize(filepath),
                'uploaded': os.path.getmtime(filepath)
            })
    return jsonify(files)

@app.route('/uploads/<filename>')
def uploaded_file(filename: str):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear all uploaded data and chat history."""
    try:
        # Clear uploads directory
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                app.logger.error(f"Error deleting {file_path}: {e}")
        
        # Clear vector store
        global vector_store
        vector_store = VectorStore()
        
        # Clear chat history
        save_chat_history([])
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create a test admin user if it doesn't exist
    if not os.path.exists('admin_created.flag'):
        from werkzeug.security import generate_password_hash
        admin_username = os.getenv('ADMIN_USERNAME', 'admin')
        admin_password = os.getenv('ADMIN_PASSWORD', 'admin123')
        # In a real app, you would save this to a database
        print('\n' + '='*50)
        print(f'Admin credentials:')
        print(f'Username: {admin_username}')
        print(f'Password: {admin_password}')
        print('='*50 + '\n')
        # Create a flag file to avoid showing this again
        with open('admin_created.flag', 'w') as f:
            f.write('1')
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
