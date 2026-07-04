import os, sys, json, logging, zipfile, subprocess, shutil, random
from pathlib import Path
import datetime
import re

log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "debug.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("setup_and_train")

def run_cmd(cmd: list[str], cwd: Path = None):
    logger.debug(f"Running command: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        raise RuntimeError("Command failed")
    return result.stdout.strip()

def extract_conversations():
    zip_path = Path(__file__).parent.parent / "conversations.zip"
    dest_dir = Path(__file__).parent / "data" / "conversations"
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {zip_path} to {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Extraction complete.")

def download_hf_datasets():
    from datasets import load_dataset
    data_dir = Path(__file__).parent / "data"
    
    logger.info("Downloading MedDialog (processed.en) via load_dataset...")
    meddialog_path = data_dir / "en_medical_dialog.json"
    out_path = data_dir / "meddialog_en_train.jsonl"
    try:
        meddialog = load_dataset("UCSD26/medical_dialog", name="processed.en", split="train", download_mode="force_redownload", trust_remote_code=True)
        meddialog.to_json(str(out_path))
        logger.info(f"MedDialog saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to load MedDialog via load_dataset: {e}")
        if meddialog_path.is_file():
            logger.info("Converting local en_medical_dialog.json to JSONL...")
            with meddialog_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
                data = json.load(f_in)
                for obj in data:
                    desc = obj.get("Description", "")
                    utts = []
                    if "Patient" in obj:
                        utts.append("Patient: " + str(obj["Patient"]))
                    if "Doctor" in obj:
                        utts.append("Doctor: " + str(obj["Doctor"]))
                    json.dump({"description": desc, "utterances": utts}, f_out)
                    f_out.write("\n")
            logger.info(f"MedDialog saved to {out_path}")
        else:
            raise FileNotFoundError(f"MedDialog fallback file not found: {meddialog_path}")

    logger.info("Preparing MedQuAD dataset...")
    medquad_csv = data_dir / "medquad.csv"
    out_path_q = data_dir / "medquad.jsonl"
    if medquad_csv.is_file():
        import pandas as pd
        df = pd.read_csv(medquad_csv)
        df = df.rename(columns={c: c.title() for c in df.columns})
        df = df.fillna("")
        with out_path_q.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                json.dump({"Question": row["Question"], "Answer": row["Answer"], "Focus_Area": row.get("Focus_Area", "")}, f)
                f.write("\n")
    else:
        medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        medquad["train"].to_json(str(out_path_q))

def semantic_chunks(text, sentence_embedder, threshold=0.62, max_tokens=380):
    import numpy as np
    sentences = re.split(r'(?<=[.!?]) +', text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences
        
    embs = sentence_embedder.encode(sentences, normalize_embeddings=True)
    chunks = []
    current = [sentences[0]]
    
    for i in range(1, len(sentences)):
        sim = float(np.dot(embs[i], embs[i-1]))
        current_len = sum(len(s.split()) for s in current)
        if sim < threshold or current_len > max_tokens:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks

def build_faiss_index_a():
    import numpy as np, faiss
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    data_dir = Path(__file__).parent / "data"
    faiss_dir = Path(__file__).parent.parent / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading MedCPT Article Encoder on {device} for Index A...")
    article_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    article_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
    
    def encode_batch(pairs, batch_size=64):
        embs = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            inputs = article_tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = article_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(emb)
        return np.vstack(embs)

    logger.info("Building Index A (conversations + MedDialog)...")
    
    index_a_pairs = []
    index_a_meta = []
    
    # 1. Synthetic Conversations
    conv_root = data_dir / "conversations" / "prompts"
    if conv_root.is_dir():
        for specialty_dir in conv_root.iterdir():
            if not specialty_dir.is_dir():
                continue
            chunks_for_specialty = []
            for txt_file in specialty_dir.rglob("*.txt"):
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [l.strip() for l in f if l.strip()]
                dialogue_lines = [l for l in lines if l.startswith("D:") or l.startswith("P:")]
                window_size = 6
                step = 4
                for i in range(0, len(dialogue_lines), step):
                    chunk_lines = dialogue_lines[i:i+window_size]
                    if len(chunk_lines) >= 2:
                        text = "\n".join(chunk_lines)
                        chunks_for_specialty.append({
                            "title": specialty_dir.name,
                            "body": text,
                            "meta": {
                                "text": text,
                                "source": "synthetic_conv",
                                "specialty": specialty_dir.name
                            }
                        })
            
            # Per-specialty cap
            if len(chunks_for_specialty) > 150:
                chunks_for_specialty = random.sample(chunks_for_specialty, 150)
                
            for c in chunks_for_specialty:
                index_a_pairs.append([c["title"], c["body"]])
                index_a_meta.append(c["meta"])

    # 2. MedDialog
    meddialog_path = data_dir / "meddialog_en_train.jsonl"
    if meddialog_path.is_file():
        seen_desc = set()
        meddialog_entries = []
        
        with open(meddialog_path, "r", encoding="utf-8") as f:
            for ln in f:
                obj = json.loads(ln)
                desc = obj.get("description", "").strip()
                if not desc:
                    continue
                    
                norm_desc = re.sub(r'[^\w\s]', '', desc.lower())
                if norm_desc in seen_desc:
                    continue
                seen_desc.add(norm_desc)
                
                specialty = None
                if any(kw in norm_desc for kw in ["fever", "headache", "cold", "sore throat", "fatigue", "body ache"]):
                    specialty = "General Medicine"
                
                meddialog_entries.append({
                    "description": desc,
                    "utterances": obj.get("utterances", []),
                    "specialty": specialty,
                    "len": len(desc)
                })
        
        # Subsample to ~3500 entries (favoring General Medicine)
        gen_entries = [e for e in meddialog_entries if e["specialty"] == "General Medicine"]
        other_entries = [e for e in meddialog_entries if e["specialty"] is None]
        
        gen_target = min(len(gen_entries), 3500)
        sampled_gen = random.sample(gen_entries, gen_target)
        other_target = min(len(other_entries), max(0, 3500 - gen_target))
        sampled_other = random.sample(other_entries, other_target)
        
        final_meddialog = sampled_gen + sampled_other
        
        for e in final_meddialog:
            desc = e["description"]
            utts = e["utterances"]
            
            if utts:
                # Opening complaint
                head = "\n".join(utts[:8])
                index_a_pairs.append([desc, head])
                index_a_meta.append({
                    "text": head,
                    "source": "meddialog_desc",
                    "specialty": e["specialty"]
                })
                
                # Sliding window
                for i in range(0, len(utts), 4):
                    win = utts[i:i+6]
                    if len(win) >= 2:
                        win_text = "\n".join(win)
                        index_a_pairs.append(["MedDialog Follow-up", win_text])
                        index_a_meta.append({
                            "text": win_text,
                            "source": "meddialog_turnwin",
                            "specialty": e["specialty"]
                        })

    logger.info(f"Embedding {len(index_a_pairs)} chunks for Index A...")
    if index_a_pairs:
        vectors_a = encode_batch(index_a_pairs, batch_size=64)
        d = vectors_a.shape[1]
        index_a = faiss.IndexFlatIP(d)
        faiss.normalize_L2(vectors_a)
        index_a.add(np.asarray(vectors_a, dtype=np.float32))
        faiss.write_index(index_a, str(faiss_dir / "index_a.faiss"))
        with open(faiss_dir / "index_a_meta.json", "w", encoding="utf-8") as f:
            json.dump(index_a_meta, f)
        logger.info(f"Index A saved.")

def build_faiss_index_b():
    import numpy as np, faiss
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    
    data_dir = Path(__file__).parent / "data"
    faiss_dir = Path(__file__).parent.parent / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading MedCPT Article Encoder on {device} for Index B...")
    article_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    article_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
    
    def encode_batch(pairs, batch_size=64):
        embs = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            inputs = article_tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = article_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(emb)
        return np.vstack(embs)

    logger.info(f"Loading all-MiniLM-L6-v2 on {device} for fast semantic chunking...")
    chunking_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))

    logger.info("Building Index B (MedQuAD + Symptom2Disease)...")
    index_b_pairs = []
    index_b_meta = []
    
    medquad_path = data_dir / "medquad.jsonl"
    if medquad_path.is_file():
        with open(medquad_path, "r", encoding="utf-8") as f:
            for ln in f:
                obj = json.loads(ln)
                q = str(obj.get("Question", ""))
                a = str(obj.get("Answer", ""))
                fa = str(obj.get("Focus_Area", "Medical Knowledge"))
                
                if q:
                    index_b_pairs.append([fa, q])
                    index_b_meta.append({"text": q, "source": "medquad_question", "extra": {"focus_area": fa}})
                    
                if a:
                    chunks = semantic_chunks(a, chunking_embedder, threshold=0.62, max_tokens=380)
                    for chunk in chunks:
                        index_b_pairs.append([fa, chunk])
                        index_b_meta.append({"text": chunk, "source": "medquad_answer_chunk", "extra": {"focus_area": fa}})

    symptom_file = data_dir / "Symptom2Disease.csv"
    if symptom_file.is_file():
        import csv
        with open(symptom_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "")
                text = row.get("text", "")
                if text:
                    index_b_pairs.append([label, text])
                    index_b_meta.append({"text": text, "source": "symptom2disease", "extra": {"disease_label": label}})

    logger.info(f"Embedding {len(index_b_pairs)} chunks for Index B...")
    if index_b_pairs:
        vectors_b = encode_batch(index_b_pairs, batch_size=64)
        d = vectors_b.shape[1]
        index_b = faiss.IndexFlatIP(d)
        faiss.normalize_L2(vectors_b)
        index_b.add(np.asarray(vectors_b, dtype=np.float32))
        faiss.write_index(index_b, str(faiss_dir / "index_b.faiss"))
        with open(faiss_dir / "index_b_meta.json", "w", encoding="utf-8") as f:
            json.dump(index_b_meta, f)
        logger.info(f"Index B saved.")

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / "backend" / ".env"
    load_dotenv(env_path)

    parser = argparse.ArgumentParser(description="TriagePlus Embeddings Setup")
    parser.add_argument("--index", choices=["a", "b", "both"], default="both", help="Which index to build (a, b, or both)")
    args = parser.parse_args()

    extract_conversations()
    download_hf_datasets()
    
    if args.index in ["a", "both"]:
        build_faiss_index_a()
    if args.index in ["b", "both"]:
        build_faiss_index_b()
        
    logger.info("Setup completed successfully.")
