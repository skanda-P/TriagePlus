"""
Hybrid RAG Engine with BM25+Dense retrieval.
- Dense-heavy weighting (0.3 BM25 + 0.7 Dense) for best medical Q&A retrieval
- Metadata filtering by question_type and focus_area
- Support for parent-child document retrieval
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
import threading

logger = logging.getLogger(__name__)


class HybridRAGEngine:
    """Hybrid BM25 + Dense vector retrieval for medical Q&A."""
    
    def __init__(self, embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased"):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.medquad_index = None
        self.medquad_chunks = None
        self.bm25_index = None
        self.conversations_index = None
        self._rag_health = {"medquad": False, "conversations": False, "issues": {}}
        self._load_indices()
    
    def _load_indices(self):
        """Load FAISS indices and prepare BM25."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            
            logger.info(f"Initializing embeddings with {self.embedding_model}...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            
            # Verify embedding dimension
            test_embedding = self.embeddings.embed_query("test")
            embedding_dim = len(test_embedding)
            logger.info(f"Embedding dimension: {embedding_dim}")
            
            # Load MedQuAD index
            faiss_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/faiss"))
            medquad_path = os.path.join(faiss_dir, "medquad")
            
            if os.path.exists(medquad_path):
                logger.info(f"Loading MedQuAD index from {medquad_path}...")
                self.medquad_index = FAISS.load_local(
                    medquad_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Load metadata for BM25 and filtering
                metadata_path = os.path.join(medquad_path, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.index_metadata = json.load(f)
                        logger.info(f"Index metadata: {self.index_metadata}")
                
                # Build BM25 index from documents
                self._build_bm25_from_faiss(medquad_path)
                
                self._rag_health["medquad"] = True
                logger.info(f"MedQuAD index loaded: {self.medquad_index.index.ntotal} vectors")
            else:
                self._rag_health["issues"]["medquad"] = f"Index not found at {medquad_path}"
                logger.warning(f"MedQuAD index not found at {medquad_path}")
            
        except Exception as e:
            self._rag_health["issues"]["global"] = f"Failed to load RAG indices: {str(e)}"
            logger.error(f"RAG initialization error: {e}", exc_info=True)
    
    def _build_bm25_from_faiss(self, faiss_path: str):
        """Build BM25 index from FAISS documents."""
        try:
            # Try to load pickled FAISS index to extract documents
            index_pkl = os.path.join(faiss_path, "index.pkl")
            if os.path.exists(index_pkl):
                import pickle
                with open(index_pkl, 'rb') as f:
                    data = pickle.load(f)
                    if hasattr(data, 'docstore'):
                        docs = []
                        for i in range(len(data.docstore._dict)):
                            doc = data.docstore._dict[i]
                            docs.append(doc.page_content)
                        
                        # Tokenize for BM25
                        tokenized_docs = [doc.split() for doc in docs]
                        self.bm25_index = BM25Okapi(tokenized_docs)
                        self.medquad_chunks = docs
                        logger.info(f"BM25 index built with {len(docs)} documents")
                        return
        except Exception as e:
            logger.warning(f"Could not build BM25 from FAISS pickle: {e}")
        
        # Fallback: use generic BM25 with empty index
        logger.warning("BM25 index will use lightweight fallback")
        self.bm25_index = None
    
    def query_medquad_hybrid(
        self,
        query: str,
        k: int = 5,
        question_type_filter: Optional[str] = None,
        focus_area_filter: Optional[str] = None,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Hybrid retrieval: BM25 + Dense similarity.
        
        Args:
            query: User query
            k: Number of results
            question_type_filter: Filter by 'symptoms', 'treatment', 'prognosis', etc.
            focus_area_filter: Filter by disease/condition
            bm25_weight: Weight for BM25 score (default 0.3)
            dense_weight: Weight for dense similarity (default 0.7)
            threshold: Minimum combined score
        
        Returns:
            List of {"text": str, "metadata": dict, "score": float}
        """
        
        if not self.medquad_index:
            logger.warning("MedQuAD index not available")
            return []
        
        try:
            # Dense retrieval
            dense_results = self.medquad_index.similarity_search_with_score(query, k=k*2)
            
            # Normalize dense scores (FAISS L2 distance, lower is better)
            dense_dict = {}
            for doc, score in dense_results:
                # Convert L2 distance to similarity (1 / (1 + distance))
                similarity = 1.0 / (1.0 + score)
                
                # Apply metadata filters
                if question_type_filter and doc.metadata.get('question_type') != question_type_filter:
                    continue
                if focus_area_filter and doc.metadata.get('focus_area') != focus_area_filter:
                    continue
                
                doc_id = doc.metadata.get('question', '')[:100]  # Use question as key
                dense_dict[doc_id] = {
                    'document': doc,
                    'dense_score': similarity
                }
            
            # BM25 retrieval (if available)
            bm25_dict = {}
            if self.bm25_index and self.medquad_chunks:
                query_tokens = query.lower().split()
                bm25_scores = self.bm25_index.get_scores(query_tokens)
                
                for i, score in enumerate(bm25_scores):
                    if score > 0 and i < len(self.medquad_chunks):
                        chunk_text = self.medquad_chunks[i]
                        # Normalize BM25 score (0-1 range)
                        normalized_score = min(score / 50.0, 1.0)  # Heuristic normalization
                        bm25_dict[chunk_text[:100]] = normalized_score
            
            # Combine scores
            combined_results = []
            for doc_id, dense_info in dense_dict.items():
                doc = dense_info['document']
                dense_score = dense_info['dense_score']
                bm25_score = bm25_dict.get(doc_id, 0.0)
                
                # Hybrid score
                combined_score = (bm25_weight * bm25_score) + (dense_weight * dense_score)
                
                if combined_score >= threshold:
                    combined_results.append({
                        'text': doc.page_content,
                        'metadata': doc.metadata,
                        'score': combined_score,
                        'dense_score': dense_score,
                        'bm25_score': bm25_score
                    })
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            
            return combined_results[:k]
        
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}", exc_info=True)
            return []
    
    def query_medquad_by_condition(self, condition: str, k: int = 3) -> List[Dict]:
        """Query MedQuAD by disease/condition name (for verification node)."""
        return self.query_medquad_hybrid(
            query=condition,
            k=k,
            question_type_filter=None,  # All question types
            focus_area_filter=condition,
            bm25_weight=0.7,  # BM25-heavy for exact condition matching
            dense_weight=0.3
        )
    
    def query_medquad_by_symptoms(self, symptoms: str, k: int = 5) -> List[Dict]:
        """Query MedQuAD for symptom-related information (symptom description → question)."""
        return self.query_medquad_hybrid(
            query=symptoms,
            k=k,
            question_type_filter='symptoms',
            bm25_weight=0.3,
            dense_weight=0.7
        )
    
    def get_health_status(self) -> Dict:
        """Return RAG health status."""
        return self._rag_health


# Singleton instance
_rag_engine = None
_rag_lock = threading.Lock()


def get_rag_engine() -> HybridRAGEngine:
    """Get or create RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        with _rag_lock:
            if _rag_engine is None:
                _rag_engine = HybridRAGEngine()
    return _rag_engine


def load_rag_models():
    """Warmup function to run in background thread on startup."""
    get_rag_engine()
