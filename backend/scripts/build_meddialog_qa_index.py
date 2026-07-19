#!/usr/bin/env python3
"""
Build FAISS index for MedDialog Q&A dataset.
- Each Q&A pair (Description + Doctor answer) = 1 atomic chunk
- Uses PubMedBERT embeddings (768-dim)
- Hybrid search: 0.5 BM25 + 0.5 Dense (balanced for direct Q&A)
"""

import json
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path(__file__).parent.parent / "data"
MEDDIALOG_FILE = DATA_DIR / "en_medical_dialog.json"
FAISS_DIR = DATA_DIR / "faiss" / "meddialog"
METADATA_FILE = FAISS_DIR / "meddialog_metadata.pkl"

# Model
EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


def load_meddialog() -> List[Dict]:
    """Load MedDialog dataset."""
    logger.info(f"Loading MedDialog from {MEDDIALOG_FILE}")
    
    if not MEDDIALOG_FILE.exists():
        logger.error(f"MedDialog file not found: {MEDDIALOG_FILE}")
        return []
    
    with open(MEDDIALOG_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to chunks
    chunks = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            chunk = {
                'chunk_id': i,
                'qa_id': item.get('id', i),
                'patient_question': item.get('Description', ''),
                'doctor_answer': item.get('Doctor', ''),
                'patient_followup': item.get('Patient', ''),
                'full_text': f"{item.get('Description', '')} {item.get('Doctor', '')}"
            }
            chunks.append(chunk)
    elif isinstance(data, dict):
        # Handle dict format
        chunk_id = 0
        for key, item in data.items():
            chunk = {
                'chunk_id': chunk_id,
                'qa_id': item.get('id', key),
                'patient_question': item.get('Description', ''),
                'doctor_answer': item.get('Doctor', ''),
                'patient_followup': item.get('Patient', ''),
                'full_text': f"{item.get('Description', '')} {item.get('Doctor', '')}"
            }
            chunks.append(chunk)
            chunk_id += 1
    
    logger.info(f"Loaded {len(chunks)} Q&A pairs from MedDialog")
    return chunks


def build_bm25_index(chunks: List[Dict]) -> BM25Okapi:
    """Build BM25 index for keyword search."""
    logger.info("Building BM25 index...")
    
    # Tokenize documents
    tokenized_docs = [
        (chunk['full_text'].lower().split() + chunk['patient_question'].lower().split())
        for chunk in chunks
    ]
    
    bm25 = BM25Okapi(tokenized_docs)
    logger.info(f"Built BM25 index with {len(chunks)} documents")
    return bm25


def build_faiss_index(chunks: List[Dict], model: SentenceTransformer) -> Tuple[faiss.IndexIDMap, np.ndarray]:
    """Build FAISS index with PubMedBERT embeddings."""
    logger.info(f"Building FAISS index for {len(chunks)} chunks...")
    
    # Extract text for embedding
    texts = [chunk['full_text'] for chunk in chunks]
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    ids = np.arange(len(chunks), dtype=np.int64)
    index.add_with_ids(embeddings, ids)
    
    logger.info(f"Created FAISS index: {index.ntotal} vectors, {dimension}D")
    return index, embeddings


def save_index(index: faiss.Index, bm25: BM25Okapi, chunks: List[Dict], embeddings: np.ndarray):
    """Save FAISS index and metadata."""
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss_path = FAISS_DIR / "meddialog.index"
    faiss.write_index(index, str(faiss_path))
    logger.info(f"Saved FAISS index to {faiss_path}")
    
    # Save BM25 index
    bm25_path = FAISS_DIR / "meddialog_bm25.pkl"
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 index to {bm25_path}")
    
    # Save metadata
    metadata = {
        'chunks': chunks,
        'embeddings': embeddings,
        'total_chunks': len(chunks),
        'model_name': EMBEDDING_MODEL,
        'dimension': embeddings.shape[1] if embeddings.size > 0 else 0,
        'hybrid_weights': {'bm25': 0.5, 'dense': 0.5}
    }
    
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved metadata to {METADATA_FILE}")
    
    # Save summary stats
    summary = {
        'total_qa_pairs': len(chunks),
        'embedding_dimension': metadata['dimension'],
        'hybrid_weights': metadata['hybrid_weights'],
        'chunking_strategy': 'atomic_qa_pairs',
        'use_case': 'direct_qa_answering'
    }
    
    summary_path = FAISS_DIR / "meddialog_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")


def main():
    """Main execution."""
    logger.info("Starting MedDialog Q&A index build...")
    
    # Load model
    logger.info(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Load MedDialog
    chunks = load_meddialog()
    
    if not chunks:
        logger.error("No MedDialog Q&A pairs loaded!")
        return
    
    # Build BM25 index
    bm25 = build_bm25_index(chunks)
    
    # Build FAISS index
    index, embeddings = build_faiss_index(chunks, model)
    
    # Save indices and metadata
    save_index(index, bm25, chunks, embeddings)
    
    logger.info("✓ MedDialog Q&A index built successfully!")
    logger.info(f"  - {len(chunks)} Q&A pairs indexed")
    logger.info(f"  - Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  - Hybrid search: 0.5 BM25 + 0.5 Dense")
    logger.info(f"  - Use case: Direct Q&A answering")


if __name__ == '__main__':
    main()
