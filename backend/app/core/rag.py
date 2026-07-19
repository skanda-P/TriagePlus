import os
import json
import threading

class RAGQueryEngine:
    def __init__(self):
        self._rag_health = {"medquad": True, "conversations": True, "meddialog": True, "issues": {}}
        self.medquad_index = None
        self.conversations_index = None
        self.meddialog_index = None
        self.embeddings = None
        self._load_indices()
        
    def _load_indices(self):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            
            # Use GPU if available, fallback to CPU
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
            
            model_kwargs = {'device': device}
            self.embeddings = HuggingFaceEmbeddings(
                model_name="NeuML/pubmedbert-base-embeddings",
                model_kwargs=model_kwargs
            )
            
            # FAISS indices in backend/data/faiss/
            faiss_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/faiss"))
            medquad_path = os.path.join(faiss_dir, "medquad")
            conversations_path = os.path.join(faiss_dir, "conversations")
            meddialog_path = os.path.join(faiss_dir, "meddialog")
            
            if os.path.exists(medquad_path):
                self.medquad_index = FAISS.load_local(
                    medquad_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            else:
                self._rag_health["medquad"] = False
                self._rag_health["issues"]["medquad"] = "Index not found. Running in degraded mode."
                
            if os.path.exists(conversations_path):
                self.conversations_index = FAISS.load_local(
                    conversations_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            else:
                self._rag_health["conversations"] = False
                self._rag_health["issues"]["conversations"] = "Index not found. Running in degraded mode."
                
            if os.path.exists(meddialog_path):
                self.meddialog_index = FAISS.load_local(
                    meddialog_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            else:
                self._rag_health["meddialog"] = False
                self._rag_health["issues"]["meddialog"] = "Index not found. Running in degraded mode."
                
        except Exception as e:
            self._rag_health["medquad"] = False
            self._rag_health["conversations"] = False
            self._rag_health["meddialog"] = False
            self._rag_health["issues"]["global"] = f"Failed to load RAG indices: {str(e)}"
            print(f"RAG initialization failed: {e}")

    def query_medquad(self, query: str, k: int = 3, threshold: float = 1.2):
        if not self.medquad_index:
            return []
            
        try:
            results = self.medquad_index.similarity_search_with_score(query, k=k)
            filtered_results = [res[0].page_content for res in results if res[1] <= threshold]
            return filtered_results
        except Exception as e:
            print(f"Error querying MedQuAD FAISS: {e}")
            return []

    def query_conversations(self, query: str, k: int = 2, threshold: float = 2.0):
        if not self.conversations_index:
            return []
            
        try:
            results = self.conversations_index.similarity_search_with_score(query, k=k)
            filtered_results = []
            for doc, score in results:
                if score <= threshold and "doctor_followup" in doc.metadata:
                    filtered_results.append(doc.metadata["doctor_followup"])
            return filtered_results
        except Exception as e:
            print(f"Error querying Conversations FAISS: {e}")
            return []

    def query_meddialog(self, query: str, k: int = 3, threshold: float = 2.0):
        if not self.meddialog_index:
            return []
            
        try:
            results = self.meddialog_index.similarity_search_with_score(query, k=k)
            filtered_results = []
            for doc, score in results:
                if score <= threshold and "doctor_answer" in doc.metadata:
                    filtered_results.append(doc.metadata["doctor_answer"])
            return filtered_results
        except Exception as e:
            print(f"Error querying MedDialog FAISS: {e}")
            return []

# Singleton instance
_rag_engine = None
_rag_lock = threading.Lock()

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        with _rag_lock:
            if _rag_engine is None:
                _rag_engine = RAGQueryEngine()
    return _rag_engine

def load_rag_models():
    """Warmup function to run in background thread on startup"""
    get_rag_engine()
