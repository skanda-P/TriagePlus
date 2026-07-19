#!/usr/bin/env python3
"""
Build FAISS indices for MedQuAD with proper chunking strategy.
- Atomic QA pairs as chunks
- Long answers split by paragraph with parent-child linking
- Metadata preservation (focus_area, question_type, source)
"""

import os
import csv
import json
from typing import List, Dict, Tuple
from pathlib import Path
import sys

# Setup paths
BACKEND_DIR = Path(__file__).parent.parent
DATA_DIR = BACKEND_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss"
MODEL_DIR = BACKEND_DIR / "models"

os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def chunk_answer_by_paragraph(answer: str, max_tokens: int = 500) -> List[Tuple[str, int]]:
    """
    Split long answers by paragraph while keeping track of chunk indices.
    Returns list of (paragraph_text, paragraph_index) tuples.
    
    Args:
        answer: Full answer text
        max_tokens: Approx token threshold (char count / 4 as proxy)
    
    Returns:
        List of (paragraph_text, paragraph_index) if answer is long,
        or single-item list with full answer if short
    """
    char_threshold = max_tokens * 4  # Rough token-to-char conversion
    
    if len(answer) <= char_threshold:
        return [(answer, 0)]
    
    # Split by double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
    
    if len(paragraphs) <= 1:
        # No clear paragraph structure, split into sentences
        sentences = [s.strip() for s in answer.split('. ') if s.strip()]
        result = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence if sentence.endswith('.') else sentence + '.'
            if current_length + len(sentence) <= char_threshold:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                if current_chunk:
                    result.append((' '.join(current_chunk), len(result)))
                current_chunk = [sentence]
                current_length = len(sentence)
        
        if current_chunk:
            result.append((' '.join(current_chunk), len(result)))
        
        return result if result else [(answer, 0)]
    else:
        # Use existing paragraphs
        return [(para, i) for i, para in enumerate(paragraphs)]


def load_medquad_csv(csv_path: str) -> List[Dict]:
    """
    Load MedQuAD from CSV and create chunks with metadata.
    
    CSV columns: question, answer, source, focus_area
    """
    chunks = []
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found at {csv_path}")
        return chunks
    
    print(f"Loading MedQuAD from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        if reader.fieldnames != ['question', 'answer', 'source', 'focus_area']:
            print(f"WARNING: CSV columns are {reader.fieldnames}, expected ['question', 'answer', 'source', 'focus_area']")
        
        for row_idx, row in enumerate(reader):
            question = row.get('question', '').strip()
            answer = row.get('answer', '').strip()
            source = row.get('source', '').strip()
            focus_area = row.get('focus_area', '').strip()
            
            if not question or not answer:
                print(f"WARNING: Row {row_idx} missing question or answer, skipping")
                continue
            
            # Infer question_type from question text
            question_lower = question.lower()
            if any(word in question_lower for word in ['what is', 'what are', 'how does', 'why']):
                question_type = 'symptoms'
            elif any(word in question_lower for word in ['treat', 'cure', 'medication', 'medicine', 'therapy']):
                question_type = 'treatment'
            elif any(word in question_lower for word in ['prognosis', 'outlook', 'survival', 'life expectancy']):
                question_type = 'prognosis'
            elif any(word in question_lower for word in ['cause', 'risk', 'susceptibility', 'who']):
                question_type = 'susceptibility'
            else:
                question_type = 'general'
            
            # Split long answers by paragraph
            answer_chunks = chunk_answer_by_paragraph(answer)
            
            for chunk_text, chunk_idx in answer_chunks:
                chunk = {
                    'question': question,
                    'answer_chunk': chunk_text,
                    'source': source,
                    'focus_area': focus_area,
                    'question_type': question_type,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(answer_chunks),
                    'full_answer': answer if chunk_idx == 0 else None  # Store full answer in first chunk
                }
                chunks.append(chunk)
    
    print(f"Loaded {len(chunks)} chunks from {row_idx + 1} QA pairs")
    return chunks


def build_faiss_index(chunks: List[Dict], model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
    """Build FAISS index with embeddings and metadata."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    
    if not chunks:
        print("ERROR: No chunks to index")
        return None
    
    print(f"Building embeddings with {model_name}...")
    
    # Initialize embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"ERROR initializing embeddings: {e}")
        return None
    
    # Create documents with metadata
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk['answer_chunk'],
            metadata={
                'question': chunk['question'],
                'source': chunk['source'],
                'focus_area': chunk['focus_area'],
                'question_type': chunk['question_type'],
                'chunk_index': chunk['chunk_index'],
                'total_chunks': chunk['total_chunks'],
                'full_answer': chunk.get('full_answer', '')
            }
        )
        documents.append(doc)
    
    print(f"Creating FAISS index with {len(documents)} documents...")
    
    try:
        index = FAISS.from_documents(documents, embeddings)
        print(f"FAISS index created successfully with {index.index.ntotal} vectors")
        return index
    except Exception as e:
        print(f"ERROR creating FAISS index: {e}")
        return None


def save_index_metadata(chunks: List[Dict], output_dir: str):
    """Save metadata about the index for validation."""
    metadata = {
        'total_chunks': len(chunks),
        'embedding_model': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'embedding_dim': 768,
        'chunking_strategy': 'atomic_qa_pairs_with_paragraph_splitting',
        'question_types': list(set(c['question_type'] for c in chunks)),
        'sources': list(set(c['source'] for c in chunks)),
        'focus_areas': list(set(c['focus_area'] for c in chunks))
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved index metadata to {metadata_path}")


def main():
    print("=" * 80)
    print("MedQuAD FAISS Index Builder")
    print("=" * 80)
    
    # Paths
    csv_path = DATA_DIR / "medquad.csv"
    medquad_index_dir = FAISS_DIR / "medquad"
    
    os.makedirs(medquad_index_dir, exist_ok=True)
    
    # Load MedQuAD
    chunks = load_medquad_csv(str(csv_path))
    
    if not chunks:
        print("ERROR: Failed to load MedQuAD data")
        sys.exit(1)
    
    # Build index
    index = build_faiss_index(chunks)
    
    if not index:
        print("ERROR: Failed to build FAISS index")
        sys.exit(1)
    
    # Verify index
    if index.index.ntotal < 1000:
        print(f"WARNING: Index has only {index.index.ntotal} vectors (expected 1000+)")
    else:
        print(f"SUCCESS: Index has {index.index.ntotal} vectors (target: 1000+)")
    
    # Save index
    print(f"Saving index to {medquad_index_dir}...")
    index.save_local(str(medquad_index_dir))
    
    # Save metadata
    save_index_metadata(chunks, str(medquad_index_dir))
    
    print("\n" + "=" * 80)
    print("MedQuAD index built successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
