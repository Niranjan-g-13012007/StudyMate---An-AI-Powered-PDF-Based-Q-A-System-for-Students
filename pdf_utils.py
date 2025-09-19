import os
import re
from typing import List, Dict, Any, Optional
import pdfplumber
from pypdf import PdfReader
from tqdm import tqdm

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize PDF processor with chunking parameters.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file using pdfplumber for better formatting.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
        """
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in tqdm(pdf.pages, desc="Extracting pages"):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {e}")
            # Fallback to pypdf if pdfplumber fails
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()

    def split_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
        """Split text into chunks of specified size with overlap.
        
        Args:
            text: Input text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # If we're not at the end, try to find a good breaking point
            if end < text_length:
                # Look for a sentence end or whitespace
                break_pos = text.rfind('.', start, end)
                if break_pos == -1 or break_pos < start + chunk_size // 2:
                    break_pos = text.rfind(' ', start, end)
                if break_pos > start + chunk_size // 2:
                    end = break_pos + 1  # Include the period/space
            
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
                
            # Move start position, accounting for overlap
            start = max(start + 1, end - chunk_overlap)
        
        return chunks
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize the extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text:
            return []

        # Clean the text first
        text = self.clean_text(text)
        words = text.split()
        chunks = []
        
        if not words:
            return []
            
        # Calculate number of chunks needed
        num_chunks = max(1, (len(words) - self.chunk_size) // (self.chunk_size - self.chunk_overlap) + 1)
        
        for i in range(num_chunks):
            start = i * (self.chunk_size - self.chunk_overlap)
            end = start + self.chunk_size
            chunk_text = ' '.join(words[start:end])
            
            chunks.append({
                'text': chunk_text,
                'chunk_index': i,
                'start_word': start,
                'end_word': min(end, len(words)),
                'total_chunks': num_chunks
            })
            
        # Handle the last chunk if there are remaining words
        if len(words) > 0 and (not chunks or chunks[-1]['end_word'] < len(words)):
            start = max(0, len(words) - self.chunk_size)
            chunk_text = ' '.join(words[start:])
            chunks.append({
                'text': chunk_text,
                'chunk_index': len(chunks),
                'start_word': start,
                'end_word': len(words),
                'total_chunks': len(chunks) + 1
            })
            
        return chunks

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a PDF file and return chunked text with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing chunked text and metadata
        """
        print(f"Processing PDF: {os.path.basename(file_path)}")
        text = self.extract_text_from_pdf(file_path)
        chunks = self.chunk_text(text)
        
        # Add file metadata to each chunk
        for chunk in chunks:
            chunk.update({
                'source': os.path.basename(file_path),
                'source_type': 'pdf',
                'total_words': len(text.split())
            })
            
        return chunks
