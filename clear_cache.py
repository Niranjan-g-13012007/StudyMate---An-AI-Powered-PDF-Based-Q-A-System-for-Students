#!/usr/bin/env python3
"""
Script to clear Streamlit cache and temporary files.
Run this if you encounter issues with cached models or want to force a fresh start.
"""

import os
import shutil
import tempfile
from pathlib import Path

def clear_streamlit_cache():
    """Clear Streamlit cache directory."""
    try:
        # Get Streamlit cache directory
        cache_dir = Path.home() / ".streamlit" / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"✅ Cleared Streamlit cache: {cache_dir}")
        else:
            print("ℹ️  No Streamlit cache directory found")
    except Exception as e:
        print(f"❌ Error clearing Streamlit cache: {e}")

def clear_huggingface_cache():
    """Clear HuggingFace model cache."""
    try:
        # Get HuggingFace cache directory
        hf_cache_dir = Path.home() / ".cache" / "huggingface"
        if hf_cache_dir.exists():
            print(f"📁 HuggingFace cache directory: {hf_cache_dir}")
            print("⚠️  Warning: This will delete all cached HuggingFace models!")
            response = input("Do you want to clear HuggingFace cache? (y/N): ")
            if response.lower() == 'y':
                shutil.rmtree(hf_cache_dir)
                print("✅ Cleared HuggingFace cache")
            else:
                print("ℹ️  Skipped HuggingFace cache clearing")
        else:
            print("ℹ️  No HuggingFace cache directory found")
    except Exception as e:
        print(f"❌ Error clearing HuggingFace cache: {e}")

def clear_temp_files():
    """Clear temporary files."""
    try:
        temp_dir = Path(tempfile.gettempdir())
        pdf_temp_files = list(temp_dir.glob("tmp*.pdf"))
        if pdf_temp_files:
            for temp_file in pdf_temp_files:
                try:
                    temp_file.unlink()
                    print(f"🗑️  Removed temp file: {temp_file}")
                except:
                    pass
        else:
            print("ℹ️  No temporary PDF files found")
    except Exception as e:
        print(f"❌ Error clearing temp files: {e}")

def main():
    print("🧹 StudyMate Cache Cleaner")
    print("=" * 30)
    
    clear_streamlit_cache()
    clear_temp_files()
    clear_huggingface_cache()
    
    print("\n✨ Cache cleaning completed!")
    print("💡 Tip: Restart your Streamlit app for changes to take effect.")

if __name__ == "__main__":
    main()
