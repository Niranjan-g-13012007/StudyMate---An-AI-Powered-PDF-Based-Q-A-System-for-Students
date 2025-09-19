# 📚 StudyMate: AI-Powered PDF Q&A System

## 🎯 Overview
StudyMate is an intelligent Q&A system that allows students to upload PDF study materials and ask questions about their content. Powered by IBM's Granite LLM and Streamlit, it provides accurate, context-aware answers by understanding the semantic meaning of both the documents and the questions.

## ✨ Features

- 📄 Upload and process multiple PDF study materials
- ❓ Ask natural language questions about your study content
- 🧠 Powered by IBM Granite 3.3B LLM for accurate answers
- 🔍 Semantic search to find relevant information in your documents
- 🎨 Clean, intuitive Streamlit web interface
- ⚡ Fast and responsive design
- 🔒 Local processing for maximum data privacy

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Basic knowledge of command line

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/studymate.git
   cd studymate
   ```

2. **Create and activate a virtual environment**
   - On Windows:
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install streamlit pypdf
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```
   This will start the Streamlit server and open the application in your default web browser.

## 🖥️ Usage

1. **Upload PDFs**
   - Click "Browse files" to upload one or more PDF documents
   - Click "Process Documents" to analyze the content

2. **Ask Questions**
   - Type your question in the text area
   - Click "Get Answer" to receive a response based on your documents

3. **View Sources**
   - Click "View Source Materials" to see which parts of your documents were used to generate the answer

## 🛠️ Technical Details

- **Backend**: Python with Streamlit
- **LLM**: IBM Granite 3.3B Instruct Model
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for efficient similarity search
- **PDF Processing**: PyPDF and pdfplumber

## 📂 Project Structure

```
studymate/
├── streamlit_app.py     # Main Streamlit application
├── llm_granite.py      # IBM Granite LLM wrapper
├── pdf_utils.py        # PDF processing utilities
├── vectorstore.py      # Vector store and similarity search
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
   FLASK_SECRET_KEY=your-secret-key-here
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=changeme
   MODEL_NAME=ibm-granite/granite-3.3-2b-instruct
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in your browser**
   ```bash
   http://localhost:5000
   ```

## 🖥️ Usage
- `llm_granite.py` — IBM Granite LLM wrapper
- `templates/index.html` — Web UI
- `static/style.css` — Basic styling
- `.env` — Configs
- `requirements.txt` — Dependencies

## Notes
- For IBM Granite, ensure you have enough GPU memory (recommended: >=16GB VRAM).
- All processing is local; no API keys required for IBM Granite.
