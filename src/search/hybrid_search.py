import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import math
import logging

logger = logging.getLogger(__name__)

class SimpleBM25:
    """A lightweight, self-contained implementation of the BM25 retrieval algorithm."""
    def __init__(self, corpus: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.documents = [doc["content"].lower().split() for doc in corpus]
        self.doc_len = [len(doc) for doc in self.documents]
        self.avg_doc_len = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.doc_count = len(self.documents)
        
        # Calculate Term Frequency (TF) and Document Frequency (DF)
        self.df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []
        
        for doc in self.documents:
            doc_tf = {}
            for word in doc:
                doc_tf[word] = doc_tf.get(word, 0) + 1
            self.tf.append(doc_tf)
            for word in doc_tf.keys():
                self.df[word] = self.df.get(word, 0) + 1
                
        # Precompute IDF (Inverse Document Frequency)
        self.idf: Dict[str, float] = {}
        for word, count in self.df.items():
            # Standard BM25 IDF formulation
            self.idf[word] = math.log((self.doc_count - count + 0.5) / (count + 0.5) + 1.0)

    def score(self, query: str) -> List[float]:
        query_words = query.lower().split()
        scores = []
        for idx in range(self.doc_count):
            score = 0.0
            doc_tf = self.tf[idx]
            d_len = self.doc_len[idx]
            for word in query_words:
                if word in doc_tf:
                    tf_val = doc_tf[word]
                    idf_val = self.idf.get(word, 0.0)
                    # BM25 tf scaling formula
                    numerator = tf_val * (self.k1 + 1)
                    denominator = tf_val + self.k1 * (1 - self.b + self.b * (d_len / self.avg_doc_len))
                    score += idf_val * (numerator / denominator)
            scores.append(score)
        return scores

class HybridSearchEngine:
    """
    Hybrid Search Engine combining Sparse (BM25) and Dense (SentenceTransformers) retrieval.
    Includes query expansion and Cross-Encoder reranking.
    """
    def __init__(
        self,
        corpus: List[Dict[str, str]],
        encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        hybrid_alpha: float = 0.5,
        query_expansion: bool = True
    ):
        self.corpus = corpus
        self.hybrid_alpha = hybrid_alpha
        self.query_expansion = query_expansion
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load custom BM25
        self.bm25 = SimpleBM25(corpus)
        
        # Load Sentence Transformer Encoder
        self.encoder = None
        self.encoder_model_name = encoder_model_name
        
        # Load Cross-Encoder
        self.cross_encoder = None
        self.cross_encoder_model_name = cross_encoder_model_name
        
    def _initialize_models(self):
        """Lazy loader for HuggingFace models to speed up initial launch."""
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.encoder_model_name).to(self.device)
                # Compute dense embeddings for corpus
                texts = [doc["content"] for doc in self.corpus]
                self.corpus_embeddings = self.encoder.encode(
                    texts, convert_to_tensor=True, show_progress_bar=False
                )
            except Exception as e:
                logger.error(f"Failed to initialize Dense Encoder: {e}")
                self.encoder = "fallback"
                
        if self.cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(self.cross_encoder_model_name).to(self.device)
            except Exception as e:
                logger.error(f"Failed to initialize Cross-Encoder: {e}")
                self.cross_encoder = "fallback"

    def expand_query(self, query: str) -> str:
        """Expands query with basic synonyms to increase recall."""
        synonyms = {
            "cv": "computer vision image cnn segment",
            "vision": "image picture computer vision segmentation cnn",
            "rag": "retrieval augmented generation vector document database search",
            "generative": "llm text generator rag knowledge",
            "forecasting": "time series prediction forecast patchtst tft lstm",
            "search": "retrieval hybrid dense query bm25 ranking"
        }
        words = query.lower().split()
        expanded = list(words)
        for word in words:
            if word in synonyms:
                expanded.extend(synonyms[word].split())
        return " ".join(dict.fromkeys(expanded)) # Remove duplicates

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self._initialize_models()
        
        original_query = query
        if self.query_expansion:
            query = self.expand_query(query)
            
        # 1. Sparse Scores (BM25)
        bm25_scores = np.array(self.bm25.score(query))
        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_norm = bm25_scores
            
        # 2. Dense Scores (Sentence Transformers)
        if self.encoder is not None and self.encoder != "fallback":
            try:
                from sentence_transformers import util
                query_emb = self.encoder.encode(original_query, convert_to_tensor=True)
                # Compute cosine similarities
                dense_scores = util.cos_sim(query_emb, self.corpus_embeddings)[0].cpu().numpy()
                dense_norm = (dense_scores - dense_scores.min()) / max(dense_scores.max() - dense_scores.min(), 1e-6)
            except Exception as e:
                logger.warning(f"Dense similarity computation failed: {e}")
                dense_norm = np.zeros_like(bm25_norm)
        else:
            dense_norm = np.zeros_like(bm25_norm)
            
        # 3. Hybrid Blending
        hybrid_scores = (1 - self.hybrid_alpha) * bm25_norm + self.hybrid_alpha * dense_norm
        
        # Get top candidates
        candidate_indices = np.argsort(hybrid_scores)[::-1][:top_k * 2]
        candidates = []
        for idx in candidate_indices:
            candidates.append({
                "corpus_idx": int(idx),
                "document": self.corpus[idx],
                "bm25_score": float(bm25_scores[idx]),
                "dense_score": float(dense_norm[idx] if 'dense_scores' in locals() else 0.0),
                "hybrid_score": float(hybrid_scores[idx])
            })
            
        # 4. Cross-Encoder Reranking
        if self.cross_encoder is not None and self.cross_encoder != "fallback" and len(candidates) > 0:
            try:
                pairs = [[original_query, c["document"]["content"]] for c in candidates]
                rerank_scores = self.cross_encoder.predict(pairs)
                
                # Assign rerank scores
                for idx, score in enumerate(rerank_scores):
                    candidates[idx]["rerank_score"] = float(score)
                    
                # Re-sort candidates based on Cross-Encoder relevance
                candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            except Exception as e:
                logger.warning(f"Cross-Encoder reranking failed: {e}")
                
        # Limit to top_k
        return candidates[:top_k]
