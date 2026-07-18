#!/usr/bin/env python3
"""
Build FAISS index for medical conversations with 3-turn sliding window.
- Turn: Doctor message + Patient response = 1 turn
- Chunk: 3 consecutive turns with overlap of 2 turns
- Pre-extracts doctor turns for few-shot prompting
- Uses PubMedBERT embeddings (768-dim)
"""

import json
import os
import pickle
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path(__file__).parent.parent.parent / "backend" / "data"
CONVERSATIONS_DIR = DATA_DIR / "prompts"
FAISS_DIR = DATA_DIR / "faiss" / "conversations"
METADATA_FILE = FAISS_DIR / "conversations_metadata.pkl"

# Model
EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


def parse_conversation(file_path: str) -> List[Dict]:
    """Parse conversation file into doctor-patient turns."""
    turns = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by "Doctor:" and "Patient:" markers
        lines = content.split('\n')
        current_speaker = None
        current_text = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('Doctor:') or line_stripped.startswith('D:'):
                if current_speaker == 'Patient' and current_text:
                    # We have a complete patient response, now we have a doctor message
                    patient_text = '\n'.join(current_text).strip()
                    turns.append({
                        'speaker': 'Patient',
                        'text': patient_text
                    })
                    current_text = []
                
                current_speaker = 'Doctor'
                doctor_text = line_stripped.replace('Doctor:', '').replace('D:', '').strip()
                if doctor_text:
                    current_text.append(doctor_text)
            
            elif line_stripped.startswith('Patient:') or line_stripped.startswith('P:'):
                if current_speaker == 'Doctor' and current_text:
                    doctor_text = '\n'.join(current_text).strip()
                    turns.append({
                        'speaker': 'Doctor',
                        'text': doctor_text
                    })
                    current_text = []
                
                current_speaker = 'Patient'
                patient_text = line_stripped.replace('Patient:', '').replace('P:', '').strip()
                if patient_text:
                    current_text.append(patient_text)
            
            else:
                if current_speaker and line.strip():
                    current_text.append(line.strip())
        
        # Add final turn
        if current_text:
            turns.append({
                'speaker': current_speaker,
                'text': '\n'.join(current_text).strip()
            })
    
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
    
    return turns


def create_turn_pairs(turns: List[Dict]) -> List[Dict]:
    """Convert individual speaker turns into doctor+patient pairs (1 turn)."""
    pairs = []
    i = 0
    
    while i < len(turns):
        if i + 1 < len(turns) and turns[i]['speaker'] == 'Patient' and turns[i+1]['speaker'] == 'Doctor':
            pair = {
                'patient': turns[i]['text'],
                'doctor': turns[i+1]['text'],
                'index': len(pairs)
            }
            pairs.append(pair)
            i += 2
        else:
            i += 1
    
    return pairs


def create_sliding_window_chunks(turn_pairs: List[Dict], window_size: int = 3, overlap: int = 2) -> List[Dict]:
    """Create 3-turn chunks with sliding window and overlap."""
    chunks = []
    stride = window_size - overlap  # stride = 3 - 2 = 1 (slide by 1)
    
    for i in range(0, len(turn_pairs) - window_size + 1, stride):
        chunk_turns = turn_pairs[i:i+window_size]
        
        # Create chunk text (all doctor+patient exchanges)
        chunk_text = " ".join([
            f"Patient: {turn['patient']} Doctor: {turn['doctor']}"
            for turn in chunk_turns
        ])
        
        # Extract ONLY doctor turns for few-shot (pre-extracted)
        doctor_turns = [turn['doctor'] for turn in chunk_turns]
        doctor_few_shot = " ".join(doctor_turns)
        
        chunk = {
            'chunk_id': len(chunks),
            'start_turn_idx': i,
            'end_turn_idx': i + window_size - 1,
            'num_turns': window_size,
            'full_text': chunk_text,
            'doctor_few_shot': doctor_few_shot,  # Pre-extracted for system prompt
            'doctor_turns': doctor_turns,
            'turns': chunk_turns
        }
        chunks.append(chunk)
    
    logger.info(f"Created {len(chunks)} chunks from {len(turn_pairs)} turns")
    return chunks


def load_all_conversations() -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Load all conversations from prompts directory."""
    all_chunks = []
    chunks_by_specialty = defaultdict(list)
    
    if not CONVERSATIONS_DIR.exists():
        logger.warning(f"Conversations directory not found: {CONVERSATIONS_DIR}")
        return [], {}
    
    # Iterate through specialties
    for specialty_dir in CONVERSATIONS_DIR.iterdir():
        if not specialty_dir.is_dir():
            continue
        
        specialty_name = specialty_dir.name
        specialty_chunks = []
        
        # Process each conversation file
        for conv_file in specialty_dir.glob("*.txt"):
            logger.info(f"Processing {specialty_name}/{conv_file.name}")
            
            turns = parse_conversation(str(conv_file))
            turn_pairs = create_turn_pairs(turns)
            chunks = create_sliding_window_chunks(turn_pairs)
            
            # Add metadata
            for chunk in chunks:
                chunk['specialty'] = specialty_name
                chunk['file'] = conv_file.name
                chunk['file_path'] = str(conv_file.relative_to(CONVERSATIONS_DIR))
                specialty_chunks.append(chunk)
                all_chunks.append(chunk)
        
        chunks_by_specialty[specialty_name] = specialty_chunks
        logger.info(f"  {specialty_name}: {len(specialty_chunks)} chunks")
    
    return all_chunks, chunks_by_specialty


def build_faiss_index(chunks: List[Dict], model: SentenceTransformer) -> Tuple[faiss.IndexIDMap, np.ndarray]:
    """Build FAISS index with PubMedBERT embeddings."""
    logger.info(f"Building FAISS index for {len(chunks)} chunks...")
    
    # Extract text for embedding (full conversation text)
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


def save_index(index: faiss.Index, chunks: List[Dict], embeddings: np.ndarray):
    """Save FAISS index and metadata."""
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss_path = FAISS_DIR / "conversations.index"
    faiss.write_index(index, str(faiss_path))
    logger.info(f"Saved FAISS index to {faiss_path}")
    
    # Save metadata with pre-extracted doctor turns
    metadata = {
        'chunks': chunks,
        'embeddings': embeddings,
        'total_chunks': len(chunks),
        'model_name': EMBEDDING_MODEL,
        'dimension': embeddings.shape[1] if embeddings.size > 0 else 0,
        'hybrid_weights': {'bm25': 0.4, 'dense': 0.6}
    }
    
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved metadata to {METADATA_FILE}")
    
    # Save summary stats
    summary = {
        'total_chunks': len(chunks),
        'embedding_dimension': metadata['dimension'],
        'specialties': list(set(chunk['specialty'] for chunk in chunks)),
        'hybrid_weights': metadata['hybrid_weights'],
        'few_shot_included': True,
        'pre_extracted_doctor_turns': True
    }
    
    summary_path = FAISS_DIR / "conversations_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")


def main():
    """Main execution."""
    logger.info("Starting conversations index build...")
    
    # Load model
    logger.info(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Load conversations
    logger.info(f"Loading conversations from {CONVERSATIONS_DIR}")
    all_chunks, chunks_by_specialty = load_all_conversations()
    
    if not all_chunks:
        logger.error("No conversations loaded!")
        return
    
    logger.info(f"Loaded {len(all_chunks)} total chunks across {len(chunks_by_specialty)} specialties")
    
    # Build FAISS index
    index, embeddings = build_faiss_index(all_chunks, model)
    
    # Save index and metadata
    save_index(index, all_chunks, embeddings)
    
    logger.info("✓ Conversations index built successfully!")
    logger.info(f"  - {len(all_chunks)} chunks indexed")
    logger.info(f"  - Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  - Hybrid search: 0.4 BM25 + 0.6 Dense")
    logger.info(f"  - Doctor turns pre-extracted for few-shot")
    logger.info(f"  - Specialties: {', '.join(chunks_by_specialty.keys())}")


if __name__ == '__main__':
    main()
