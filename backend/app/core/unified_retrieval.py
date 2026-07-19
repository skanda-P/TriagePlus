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
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)
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
        self.medquad_metadata = None
        self.medquad_bm25 = None
        
        self.conversations_index = None
        self.conversations_metadata = None
        self.conversations_bm25 = None
        
        self.meddialog_index = None
        self.meddialog_metadata = None
        self.meddialog_bm25 = None
        
        self._load_indices()
    
    def _load_indices(self):
        """Load all FAISS indices and metadata."""
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
        """Load MedQuAD index."""
        try:
            medquad_dir = FAISS_DIR / "medquad"
            if medquad_dir.exists():
                index_path = medquad_dir / "medquad.index"
                metadata_path = medquad_dir / "medquad_metadata.pkl"
                
                if index_path.exists() and metadata_path.exists():
                    self.medquad_index = faiss.read_index(str(index_path))
                    with open(metadata_path, 'rb') as f:
                        self.medquad_metadata = pickle.load(f)
                    logger.info(f"Loaded MedQuAD index: {self.medquad_index.ntotal} chunks")
        except Exception as e:
            logger.warning(f"Could not load MedQuAD index: {e}")
    
    def _load_conversations_index(self):
        """Load Conversations index with pre-extracted doctor turns."""
        try:
            conversations_dir = FAISS_DIR / "conversations"
            if conversations_dir.exists():
                index_path = conversations_dir / "conversations.index"
                metadata_path = conversations_dir / "conversations_metadata.pkl"
                
                if index_path.exists() and metadata_path.exists():
                    self.conversations_index = faiss.read_index(str(index_path))
                    with open(metadata_path, 'rb') as f:
                        self.conversations_metadata = pickle.load(f)
                    logger.info(f"Loaded Conversations index: {self.conversations_index.ntotal} chunks")
                    logger.info(f"  Pre-extracted doctor turns available: True")
        except Exception as e:
            logger.warning(f"Could not load Conversations index: {e}")
    
    def _load_meddialog_index(self):
        """Load MedDialog index."""
        try:
            meddialog_dir = FAISS_DIR / "meddialog"
            if meddialog_dir.exists():
                index_path = meddialog_dir / "meddialog.index"
                metadata_path = meddialog_dir / "meddialog_metadata.pkl"
                bm25_path = meddialog_dir / "meddialog_bm25.pkl"
                
                if index_path.exists() and metadata_path.exists():
                    self.meddialog_index = faiss.read_index(str(index_path))
                    with open(metadata_path, 'rb') as f:
                        self.meddialog_metadata = pickle.load(f)
                    if bm25_path.exists():
                        with open(bm25_path, 'rb') as f:
                            self.meddialog_bm25 = pickle.load(f)
                    logger.info(f"Loaded MedDialog index: {self.meddialog_index.ntotal} Q&A pairs")
        except Exception as e:
            logger.warning(f"Could not load MedDialog index: {e}")
    
    def retrieve_medquad(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve from MedQuAD with 0.3 BM25 + 0.7 Dense."""
        if not self.medquad_index or not self.model:
            return []
        
        try:
            # Dense search
            query_embedding = self.model.encode(query, convert_to_tensor=False).astype('float32').reshape(1, -1)
            distances, indices = self.medquad_index.search(query_embedding, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                chunk = self.medquad_metadata['chunks'][idx]
                results.append({
                    'source': 'medquad',
                    'chunk_id': chunk.get('chunk_id'),
                    'text': chunk.get('answer', ''),
                    'question_type': chunk.get('question_type', ''),
                    'focus_area': chunk.get('focus_area', ''),
                    'score': float(distances[0][i]),
                    'distance_type': 'l2'
                })
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving from MedQuAD: {e}")
            return []
    
    def retrieve_conversations(self, query: str, symptom: str = None, top_k: int = 3) -> List[Dict]:
        """
        Retrieve from Conversations with 0.4 BM25 + 0.6 Dense.
        Returns pre-extracted doctor turns for few-shot.
        """
        if not self.conversations_index or not self.model:
            return []
        
        try:
            # Combine query and symptom for better matching
            search_text = f"{query} {symptom}" if symptom else query
            
            # Dense search
            query_embedding = self.model.encode(search_text, convert_to_tensor=False).astype('float32').reshape(1, -1)
            distances, indices = self.conversations_index.search(query_embedding, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                chunk = self.conversations_metadata['chunks'][idx]
                results.append({
                    'source': 'conversations',
                    'chunk_id': chunk.get('chunk_id'),
                    'specialty': chunk.get('specialty', ''),
                    'file': chunk.get('file', ''),
                    'doctor_few_shot': chunk.get('doctor_few_shot', ''),  # Pre-extracted!
                    'full_text': chunk.get('full_text', ''),
                    'num_turns': chunk.get('num_turns', 0),
                    'score': float(distances[0][i]),
                    'distance_type': 'l2'
                })
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving from Conversations: {e}")
            return []
    
    def retrieve_meddialog(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve from MedDialog Q&A index with 0.5 BM25 + 0.5 Dense."""
        if not self.meddialog_index or not self.model:
            return []
        
        try:
            # Dense search
            query_embedding = self.model.encode(query, convert_to_tensor=False).astype('float32').reshape(1, -1)
            distances, indices = self.meddialog_index.search(query_embedding, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                chunk = self.meddialog_metadata['chunks'][idx]
                results.append({
                    'source': 'meddialog',
                    'qa_id': chunk.get('qa_id'),
                    'patient_question': chunk.get('patient_question', ''),
                    'doctor_answer': chunk.get('doctor_answer', ''),
                    'patient_followup': chunk.get('patient_followup', ''),
                    'score': float(distances[0][i]),
                    'distance_type': 'l2'
                })
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving from MedDialog: {e}")
            return []
    
    def retrieve_parallel(self, query: str, symptom: str = None, top_k_per_source: int = 5) -> Dict[str, List[Dict]]:
        """
        Retrieve from all three sources in parallel.
        Returns results organized by source.
        """
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
            doctor_turn = result.get('doctor_few_shot', '')
            if doctor_turn:
                few_shot.append(doctor_turn)
        
        return few_shot[:num_examples]


# Singleton instance
_retriever_instance: Optional[UnifiedRetriever] = None


def get_unified_retriever() -> UnifiedRetriever:
    """Get singleton retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = UnifiedRetriever()
    return _retriever_instance
