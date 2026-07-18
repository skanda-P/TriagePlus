#!/usr/bin/env python3
"""
MedDialog Few-Shot Retrieval Index Builder
Converts MedDialog conversations into retrievable examples for question templates
"""

import json
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedDialogIndexBuilder:
    """Build retrieval index from MedDialog conversations"""
    
    def __init__(self, data_dir: str = "backend/data", model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        self.data_dir = data_dir
        self.meddialog_file = os.path.join(data_dir, "en_medical_dialog.json")
        self.index_dir = os.path.join(data_dir, "meddialog_index")
        
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 768
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.conversations = []
        self.doctor_questions = []  # Store extracted doctor questions for retrieval
        self.embeddings = []
        
    def load_conversations(self):
        """Load MedDialog JSON conversations"""
        if not os.path.exists(self.meddialog_file):
            logger.warning(f"MedDialog file not found: {self.meddialog_file}")
            return False
        
        logger.info(f"Loading MedDialog from {self.meddialog_file}")
        
        try:
            with open(self.meddialog_file, 'r') as f:
                data = json.load(f)
                
                # Handle both dict and list formats
                if isinstance(data, dict):
                    self.conversations = list(data.values()) if 'conversations' not in data else data['conversations']
                else:
                    self.conversations = data
            
            logger.info(f"Loaded {len(self.conversations)} conversations")
            return True
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
            return False
    
    def extract_doctor_questions(self):
        """
        Extract doctor questions and patient descriptions for few-shot retrieval
        Format: {patient_description, doctor_question, context}
        """
        logger.info("Extracting doctor questions and patient context...")
        
        for conv_idx, conversation in enumerate(self.conversations):
            if not isinstance(conversation, dict):
                continue
            
            turns = conversation.get('turns', [])
            if not turns:
                continue
            
            patient_description = ""
            
            for turn_idx, turn in enumerate(turns):
                speaker = turn.get('speaker', '').lower()
                content = turn.get('content', '').strip()
                
                if not content:
                    continue
                
                if speaker == 'patient':
                    patient_description = content
                elif speaker == 'doctor':
                    # Extract doctor's question/response
                    if patient_description:  # Only if we have patient context
                        self.doctor_questions.append({
                            'patient_symptom': patient_description,
                            'doctor_response': content,
                            'turn_index': turn_idx,
                            'conversation_id': conv_idx,
                            'source': 'meddialog'
                        })
        
        logger.info(f"Extracted {len(self.doctor_questions)} doctor question templates")
    
    def build_embeddings(self):
        """Generate embeddings for all questions"""
        logger.info("Generating embeddings...")
        
        if not self.doctor_questions:
            logger.warning("No questions to embed")
            return
        
        # Combine patient symptom + doctor response for context
        texts_to_embed = [
            f"Patient: {q['patient_symptom']} Doctor: {q['doctor_response']}"
            for q in self.doctor_questions
        ]
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, normalize_embeddings=True)
            all_embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + batch_size}/{len(texts_to_embed)} texts")
        
        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Generated {len(self.embeddings)} embeddings (dim: {self.embedding_dim})")
    
    def build_index(self):
        """Build FAISS index"""
        logger.info("Building FAISS index...")
        
        if len(self.embeddings) == 0:
            logger.error("No embeddings to index")
            return False
        
        # Add embeddings to index
        self.index.add(self.embeddings)
        logger.info(f"Index built with {self.index.ntotal} vectors")
        return True
    
    def search_similar_questions(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for similar doctor question templates
        Used for few-shot prompting
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.doctor_questions):
                question_data = self.doctor_questions[int(idx)]
                question_data['similarity_score'] = 1 - (distance / 2)  # Convert L2 distance to similarity
                results.append(question_data)
        
        return results
    
    def save_index(self):
        """Save index and metadata"""
        logger.info(f"Saving index to {self.index_dir}")
        
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.index_dir, "meddialog.index"))
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'model': 'microsoft/BiomedNLP-PubMedBERT-base',
            'num_questions': len(self.doctor_questions),
            'questions': self.doctor_questions
        }
        
        with open(os.path.join(self.index_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Index saved successfully")
    
    def build(self):
        """Execute full pipeline"""
        if not self.load_conversations():
            logger.error("Failed to load conversations")
            return False
        
        self.extract_doctor_questions()
        self.build_embeddings()
        
        if not self.build_index():
            logger.error("Failed to build index")
            return False
        
        self.save_index()
        logger.info("MedDialog index built successfully!")
        return True

if __name__ == "__main__":
    builder = MedDialogIndexBuilder()
    builder.build()
