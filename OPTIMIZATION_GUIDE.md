# StudyMate Performance Optimization Guide

## 🚀 Performance Improvements Made

### 1. Model Caching
- **Problem**: Model was being reloaded on every Streamlit app restart
- **Solution**: Implemented `@st.cache_resource` decorator to cache the model in memory
- **Benefit**: Model loads only once per session, dramatically reducing startup time

### 2. Lazy Loading
- **Problem**: Model was loaded immediately when app started
- **Solution**: Model now loads only when user asks their first question
- **Benefit**: Faster initial app startup, better user experience

### 3. Configuration Management
- **Added**: `config.py` file for easy customization
- **Benefit**: Easy to modify model settings, chunk sizes, and other parameters

### 4. Better Progress Indicators
- **Added**: Clear loading messages and progress indicators
- **Benefit**: Users know what's happening during long operations

## 📊 Performance Comparison

### Before Optimization:
- ❌ Model loaded on every app restart (2-5 minutes)
- ❌ No progress indicators
- ❌ Hard-coded configuration
- ❌ Poor user experience during loading

### After Optimization:
- ✅ Model cached and loaded only once
- ✅ Lazy loading - model loads only when needed
- ✅ Clear progress indicators
- ✅ Configurable settings
- ✅ Much faster subsequent runs

## 🛠️ Usage Instructions

### First Run:
1. Start the app: `streamlit run streamlit_app.py`
2. Upload your PDF files and process them
3. Ask your first question (model will load here - takes 2-5 minutes)
4. Subsequent questions will be much faster!

### Subsequent Runs:
1. Start the app: `streamlit run streamlit_app.py`
2. App starts immediately (no model loading)
3. Questions are answered quickly using cached model

## ⚙️ Configuration Options

Edit `config.py` to customize:

```python
# Model settings
MODEL_CONFIG = {
    "model_name": "ibm-granite/granite-3.3-2b-instruct",
    "max_new_tokens": 512,
    "temperature": 0.7,
}

# PDF processing
PDF_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 100,
}
```

## 🧹 Cache Management

### Clear Cache (if needed):
```bash
python clear_cache.py
```

### Manual Cache Clearing:
- Streamlit cache: Delete `~/.streamlit/cache/`
- HuggingFace cache: Delete `~/.cache/huggingface/`

## 💡 Additional Tips

1. **Keep the app running**: Don't restart unless necessary
2. **Use smaller models**: For faster loading, consider smaller models
3. **Monitor memory**: Large models use significant RAM
4. **GPU acceleration**: If available, modify config to use CUDA

## 🐛 Troubleshooting

### If the app is still slow:
1. Run `python clear_cache.py` to clear all caches
2. Restart the Streamlit app
3. Check available RAM (model needs ~8GB)
4. Consider using a smaller model

### If you get memory errors:
1. Close other applications
2. Restart your computer
3. Consider using a cloud instance with more RAM

## 📈 Expected Performance

- **First question**: 2-5 minutes (model loading + inference)
- **Subsequent questions**: 10-30 seconds (inference only)
- **App restart**: Instant (no model loading)
- **Memory usage**: ~8GB RAM for the model
