import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Union
import torch
from tqdm import tqdm

# Handle InterpolationMode import for different torchvision versions
try:
    from torchvision.transforms import InterpolationMode
except ImportError:
    # Fallback for older torchvision versions
    class InterpolationMode:
        BILINEAR = 'bilinear'
        BICUBIC = 'bicubic'
        NEAREST = 'nearest'

from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """Initialize the VectorStore with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cuda', 'mps', 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.text_chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.dim: int = self.model.get_sentence_embedding_dimension()
        
    def add_texts(self, texts: List[Dict[str, Any]], batch_size: int = 32) -> None:
        """Add texts to the vector store with metadata.
        
        Args:
            texts: List of dictionaries containing 'text' and metadata
            batch_size: Batch size for encoding
        """
        if not texts:
            return
            
        # Extract just the text for encoding
        text_list = [item['text'] for item in texts]
        
        # Encode in batches to handle large numbers of texts
        embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc="Encoding texts"):
            batch = text_list[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)
        
        # Convert list of arrays to a single numpy array
        new_embeddings = np.vstack(embeddings)
        
        # Update stored texts and embeddings
        self.text_chunks.extend(texts)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self._build_index()
    
    def _build_index(self) -> None:
        """Build or rebuild the FAISS index."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index = faiss.IndexFlatIP(self.dim)  # Inner product for cosine similarity
            self.index.add(self.embeddings)
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        min_score: float = 0.2,  
        filter_condition: Optional[callable] = None,
        score_threshold: float = 0.5  
    ) -> List[Dict[str, Any]]:
        """Search for similar texts to the query with enhanced filtering and scoring.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            min_score: Minimum similarity score (0-1) for results
            filter_condition: Optional function to filter results based on metadata
            score_threshold: Minimum score threshold for including results
            
        Returns:
            List of dictionaries with 'text', 'score', and metadata, sorted by relevance
        """
        if not query.strip() or self.index is None or not self.text_chunks:
            return []
        
        try:
            # Encode the query with better error handling
            query_embedding = self.model.encode(
                [query], 
                convert_to_numpy=True, 
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=32,
                convert_to_tensor=False
            )
            
            # Search more candidates than needed to account for filtering
            n_candidates = min(max(top_k * 3, 10), len(self.text_chunks))
            scores, indices = self.index.search(query_embedding, n_candidates)
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                # Skip invalid indices
                if not (0 <= idx < len(self.text_chunks)):
                    continue
                    
                # Skip low-scoring results
                if score < min_score:
                    continue
                    
                # Get result with metadata
                result = {
                    **self.text_chunks[idx].copy(),
                    'score': float(score),
                    'relevance': min(1.0, max(0.0, (float(score) - min_score) / (1.0 - min_score)))  # Normalized score
                }
                
                # Apply custom filters if provided
                if filter_condition and not filter_condition(result):
                    continue
                    
                results.append(result)
                
                # Stop if we have enough high-quality results
                if len(results) >= top_k and score < score_threshold:
                    break
            
            # Sort by score in descending order
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Return only top_k results
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            return []
    
    def save_index(self, directory: str) -> None:
        """Save the index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, 'index.faiss'))
        
        # Save metadata
        with open(os.path.join(directory, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'text_chunks': self.text_chunks,
                'dim': self.dim,
                'model_name': self.model_name if hasattr(self, 'model_name') else 'all-MiniLM-L6-v2'
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_index(cls, directory: str, device: str = None) -> 'VectorStore':
        """Load a saved index from disk."""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist")
        
        # Load metadata
        with open(os.path.join(directory, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Create new VectorStore instance
        store = cls(model_name=metadata.get('model_name', 'all-MiniLM-L6-v2'), device=device)
        store.text_chunks = metadata['text_chunks']
        store.dim = metadata['dim']
        
        # Load FAISS index if it exists
        index_path = os.path.join(directory, 'index.faiss')
        if os.path.exists(index_path):
            store.index = faiss.read_index(index_path)
            
            # Rebuild embeddings from index (approximate, as FAISS doesn't store original vectors)
            if store.index.ntotal > 0:
                store.embeddings = store.index.reconstruct_n(0, store.index.ntotal)
        
        return store
