import os
import json
import logging
from pathlib import Path
import numpy as np
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recover_faiss")

def recover():
    base_dir = Path(r"D:\BTech\hackathons\triageplus\triagePlus\RAG")
    data_dir = base_dir / "ml_training" / "data"
    faiss_dir = base_dir / "faiss"
    faiss_dir.mkdir(exist_ok=True)
    
    conv_root = data_dir / "conversations" / "prompts"
    conv_texts = []
    conv_meta = []
    
    if conv_root.is_dir():
        for specialty_dir in conv_root.iterdir():
            if not specialty_dir.is_dir():
                continue
            for txt_file in specialty_dir.rglob("*.txt"):
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [l.strip() for l in f if l.strip()]
                dialogue_lines = [l for l in lines if l.startswith("D:") or l.startswith("P:")]
                window_size = 6  # 3 turns
                step = 4         # overlap of 1 turn (2 lines)
                for i in range(0, len(dialogue_lines), step):
                    chunk_lines = dialogue_lines[i:i+window_size]
                    if len(chunk_lines) >= 2:
                        text = "\n".join(chunk_lines)
                        conv_texts.append(text)
                        conv_meta.append({"specialty": specialty_dir.name, "source": str(txt_file), "text": text})
    
    logger.info(f"Reconstructed {len(conv_texts)} texts/metadata for Index A.")
    
    emb_path = faiss_dir / "embeddings_a.npy"
    logger.info(f"Loading {emb_path}...")
    vectors = np.load(str(emb_path))
    logger.info(f"Loaded vectors shape: {vectors.shape}")
    
    if len(conv_texts) != vectors.shape[0]:
        logger.error(f"Mismatch: {len(conv_texts)} texts vs {vectors.shape[0]} vectors!")
        return
        
    d = vectors.shape[1]
    index_a = faiss.IndexFlatIP(d)
    index_a.add(vectors)
    
    faiss.write_index(index_a, str(faiss_dir / "index_a.faiss"))
    logger.info(f"Saved index_a.faiss")
    
    with open(faiss_dir / "index_a_meta.json", "w", encoding="utf-8") as f:
        json.dump(conv_meta, f, indent=2)
    logger.info(f"Saved index_a_meta.json")

if __name__ == "__main__":
    recover()
