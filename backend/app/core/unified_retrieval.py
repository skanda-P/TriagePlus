#!/usr/bin/env python3
"""
Unified hybrid retrieval system combining:
- MedQuAD (medical corpus) with 0.3 BM25 + 0.7 Dense
- Conversations (few-shot examples) with 0.4 BM25 + 0.6 Dense  
- MedDialog (Q&A index) with 0.5 BM25 + 0.5 Dense

Parallel search across all three indices with result merging.
"""

import os
import pickle
import logging
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path(__file__).parent.parent.parent / "data"
FAISS_DIR = DATA_DIR / "faiss"

# Model
EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


class UnifiedRetriever:
    """Unified retrieval across MedQuAD, Conversations, and MedDialog."""
    
    def __init__(self):
        self.model = None
        self.medquad_index = None
        self.medquad_bm25 = None
        self.medquad_chunks = None
        
        self.conversations_index = None
        self.conversations_bm25 = None
        self.conversations_chunks = None
        
        self.meddialog_index = None
        self.meddialog_bm25 = None
        self.meddialog_chunks = None
        
        self._load_indices()
    
    def _load_indices(self):
        """Load all FAISS indices, BM25 indexes, and metadata."""
        logger.info("Loading unified retrieval indices...")
        
        # Load model
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
        
        # Load MedQuAD
        self._load_medquad_index()
        
        # Load Conversations
        self._load_conversations_index()
        
        # Load MedDialog
        self._load_meddialog_index()
    
    def _load_medquad_index(self):
        """Load MedQuAD index with BM25."""
        try:
            medquad_dir = FAISS_DIR / "medquad"
            if medquad_dir.exists():
                index_path = medquad_dir / "medquad.index"
                metadata_path = medquad_dir / "medquad_metadata.pkl"
                bm25_path = medquad_dir / "medquad_bm25.pkl"
                
                if index_path.exists() and metadata_path.exists():
                    self.medquad_index = faiss.read_index(str(index_path))
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    self.medquad_chunks = metadata.get('chunks', [])
                    logger.info(f"Loaded MedQuAD index: {self.medquad_index.ntotal} chunks")
                    
                    # Load or build BM25
                    if bm25_path.exists():
                        with open(bm25_path, 'rb') as f:
                            self.medquad_bm25 = pickle.load(f)
                        logger.info("Loaded MedQuAD BM25 index")
                    else:
                        self._build_bm25('medquad')
        except Exception as e:
            logger.warning(f"Could not load MedQuAD index: {e}")
    
    def _load_conversations_index(self):
        """Load Conversations index with BM25."""
        try:
            conversations_dir = FAISS_DIR / "conversations"
            if conversations_dir.exists():
                index_path = conversations_dir / "conversations.index"
                metadata_path = conversations_dir / "conversations_metadata.pkl"
                bm25_path = conversations_dir / "conversations_bm25.pkl"
                
                if index_path.exists() and metadata_path.exists():
                    self.conversations_index = faiss.read_index(str(index_path))
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    self.conversations_chunks = metadata.get('chunks', [])
                    logger.info(f"Loaded Conversations index: {self.conversations_index.ntotal} chunks")
                    
                    if bm25_path.exists():
                        with open(bm25_path, 'rb') as f:
                            self.conversations_bm25 = pickle.load(f)
                        logger.info("Loaded Conversations BM25 index")
                    else:
                        self._build_bm25('conversations')
        except Exception as e:
            logger.warning(f"Could not load Conversations index: {e}")
    
    def _load_meddialog_index(self):
        """Load MedDialog index with BM25."""
        try:
            meddialog_dir = FAISS_DIR / "meddialog"
            if meddialog_dir.exists():
                index_path = meddialog_dir / "meddialog.index"
                metadata_path = meddialog_dir / "meddialog_metadata.pkl"
                bm25_path = meddialog_dir / "meddialog_bm25.pkl"
                
                if index_path.exists() and metadata_path.exists():
                    self.meddialog_index = faiss.read_index(str(index_path))
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    self.meddialog_chunks = metadata.get('chunks', [])
                    logger.info(f"Loaded MedDialog index: {self.meddialog_index.ntotal} Q&A pairs")
                    
                    if bm25_path.exists():
                        with open(bm25_path, 'rb') as f:
                            self.meddialog_bm25 = pickle.load(f)
                        logger.info("Loaded MedDialog BM25 index")
                    else:
                        self._build_bm25('meddialog')
        except Exception as e:
            logger.warning(f"Could not load MedDialog index: {e}")
    
    def _build_bm25(self, source: str):
        """Build BM25 index for a source if not present."""
        try:
            if source == 'medquad' and self.medquad_chunks:
                texts = [c.get('answer_chunk', c.get('answer', '')) for c in self.medquad_chunks]
            elif source == 'conversations' and self.conversations_chunks:
                texts = [c.get('doctor_few_shot', c.get('full_text', '')) for c in self.conversations_chunks]
            elif source == 'meddialog' and self.meddialog_chunks:
                texts = [c.get('full_text', '') for c in self.meddialog_chunks]
            else:
                return
            
            tokenized = [t.lower().split() for t in texts]
            bm25 = BM25Okapi(tokenized)
            
            # Save BM25
            if source == 'medquad':
                self.medquad_bm25 = bm25
                bm25_path = FAISS_DIR / "medquad" / "medquad_bm25.pkl"
            elif source == 'conversations':
                self.conversations_bm25 = bm25
                bm25_path = FAISS_DIR / "conversations" / "conversations_bm25.pkl"
            elif source == 'meddialog':
                self.meddialog_bm25 = bm25
                bm25_path = FAISS_DIR / "meddialog" / "meddialog_bm25.pkl"
            
            bm25_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bm25_path, 'wb') as f:
                pickle.dump(bm25, f)
            logger.info(f"Built and saved BM25 for {source}: {len(texts)} docs")
        except Exception as e:
            logger.warning(f"Could not build BM25 for {source}: {e}")
    
    def _hybrid_search(self, query: str, index, bm25, chunks: List[Dict], 
                       top_k: int, bm25_weight: float, dense_weight: float, source: str) -> List[Dict]:
        """Perform hybrid BM25 + Dense search and merge results."""
        if not index or not self.model or not chunks:
            return []
        
        try:
            # Dense search
            query_emb = self.model.encode([query], normalize_embeddings=True).astype('float32')
            dense_distances, dense_indices = index.search(query_emb, top_k * 2)
            
            # BM25 search
            query_tokens = query.lower().split()
            bm25_scores = np.zeros(len(chunks))
            if bm25:
                bm25_scores = bm25.get_scores(query_tokens)
            
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
            
            # Collect unique results from both
            all_indices = set(dense_indices[0].tolist() + bm25_top_indices.tolist())
            all_indices = [i for i in all_indices if 0 <= i < len(chunks)]
            
            # Compute hybrid scores
            results = []
            for idx in all_indices:
                # Normalize dense score (L2 distance -> similarity)
                dense_score = 0.0
                if idx in dense_indices[0]:
                    pos = list(dense_indices[0]).index(idx)
                    dist = dense_distances[0][pos]
                    dense_score = 1.0 / (1.0 + dist)
                
                # Normalize BM25 score (0-1 range)
                bm25_score = min(bm25_scores[idx] / 50.0, 1.0) if bm25 else 0.0
                
                # Hybrid score
                hybrid_score = (bm25_weight * bm25_score) + (dense_weight * dense_score)
                
                if hybrid_score > 0:
                    chunk = chunks[idx]
                    results.append({
                        'source': source,
                        'chunk': chunk,
                        'hybrid_score': hybrid_score,
                        'dense_score': dense_score,
                        'bm25_score': bm25_score,
                        'index': idx
                    })
            
            # Sort by hybrid score and return top_k
            results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search for {source}: {e}")
            return []
    
    def retrieve_medquad(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve from MedQuAD with 0.3 BM25 + 0.7 Dense."""
        return self._hybrid_search(
            query, self.medquad_index, self.medquad_bm25, 
            self.medquad_chunks, top_k, 0.3, 0.7, 'medquad'
        )
    
    def retrieve_conversations(self, query: str, symptom: str = None, top_k: int = 3) -> List[Dict]:
        """Retrieve from Conversations with 0.4 BM25 + 0.6 Dense."""
        search_text = f"{query} {symptom}" if symptom else query
        return self._hybrid_search(
            search_text, self.conversations_index, self.conversations_bm25,
            self.conversations_chunks, top_k, 0.4, 0.6, 'conversations'
        )
    
    def retrieve_meddialog(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve from MedDialog Q&A with 0.5 BM25 + 0.5 Dense."""
        return self._hybrid_search(
            query, self.meddialog_index, self.meddialog_bm25,
            self.meddialog_chunks, top_k, 0.5, 0.5, 'meddialog'
        )
    
    def retrieve_parallel(self, query: str, symptom: str = None, top_k_per_source: int = 5) -> Dict[str, List[Dict]]:
        """Retrieve from all three sources in parallel."""
        logger.info(f"Parallel retrieval: query='{query}', symptom='{symptom}'")
        
        results = {
            'medquad': self.retrieve_medquad(query, top_k_per_source),
            'conversations': self.retrieve_conversations(query, symptom, top_k_per_source),
            'meddialog': self.retrieve_meddialog(query, top_k_per_source)
        }
        
        logger.info(f"Retrieved: {len(results['medquad'])} from MedQuAD, "
                    f"{len(results['conversations'])} from Conversations, "
                    f"{len(results['meddialog'])} from MedDialog")
        return results
    
    def get_fewshot_examples(self, query: str, symptom: str = None, num_examples: int = 3) -> List[str]:
        """
        Get top few-shot examples (doctor turns) from Conversations.
        Pre-extracted, ready for system prompt.
        """
        conv_results = self.retrieve_conversations(query, symptom, num_examples)
        
        few_shot = []
        for result in conv_results:
            chunk = result.get('chunk', {})
            doctor_turn = chunk.get('doctor_few_shot', '')
            if doctor_turn:
                few_shot.append(doctor_turn)
        
        return few_shot[:num_examples]


# Singleton instance
_retriever_instance: Optional[UnifiedRetriever] = None
_retriever_lock = threading.Lock()


def get_unified_retriever() -> UnifiedRetriever:
    """Get singleton retriever instance (thread-safe)."""
    global _retriever_instance
    if _retriever_instance is None:
        with _retriever_lock:
            if _retriever_instance is None:
                _retriever_instance = UnifiedRetriever()
    return _retriever_instance