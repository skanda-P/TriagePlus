# setup_and_train.py
"""Setup script for TriagePlus project.
- Extracts conversations.zip
- Downloads required datasets (Hugging Face & Kaggle)
- Builds FAISS Index A (conversation + MedDialog)
- Builds FAISS Index B (MedQuAD + MedlinePlus XML)
- Performs department inference using Gemini 2.5-flash
- Writes predictions to models/gemini_predictions.json
"""

import os, sys, json, logging, zipfile, subprocess, shutil
from pathlib import Path
import datetime

# ---------------------------------------------------------------------------
# Logging configuration (DEBUG to console and rotating file)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def run_cmd(cmd: list[str], cwd: Path = None):
    """Run a shell command, log output, raise on failure."""
    logger.debug(f"Running command: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            f"Command failed: {' '.join(cmd)}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        raise RuntimeError("Command failed")
    logger.debug(f"Command succeeded. STDOUT: {result.stdout.strip()}")
    return result.stdout.strip()

# ---------------------------------------------------------------------------
# 1. Extract conversations.zip
# ---------------------------------------------------------------------------
def extract_conversations():
    zip_path = Path(__file__).parent.parent / "conversations.zip"
    dest_dir = Path(__file__).parent / "data" / "conversations"
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {zip_path} to {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Extraction complete.")

# ---------------------------------------------------------------------------
# 2. Download Hugging Face datasets (MedDialog & MedQuAD)
# ---------------------------------------------------------------------------
def download_hf_datasets():
    from datasets import load_dataset
    data_dir = Path(__file__).parent / "data"
    # MedDialog (processed.en) – try HF first, fallback to local JSON
    logger.info("Downloading MedDialog (processed.en) via load_dataset...")
    meddialog_path = data_dir / "en_medical_dialog.json"
    out_path = data_dir / "meddialog_en_train.jsonl"
    try:
        meddialog = load_dataset(
            "UCSD26/medical_dialog",
            name="processed.en",
            split="train",
            download_mode="force_redownload",
            trust_remote_code=True,
        )
        meddialog.to_json(str(out_path))
        logger.info(f"MedDialog saved to {out_path}")
    except Exception as e:
        logger.error(f"Failed to load MedDialog via load_dataset: {e}")
        if meddialog_path.is_file():
            import json
            logger.info("Converting local en_medical_dialog.json to JSONL...")
            with meddialog_path.open("r", encoding="utf-8") as f_in, out_path.open(
                "w", encoding="utf-8"
            ) as f_out:
                data = json.load(f_in)
                for obj in data:
                    json.dump(obj, f_out)
                    f_out.write("\n")
            logger.info(f"MedDialog saved to {out_path}")
        else:
            raise FileNotFoundError(f"MedDialog fallback file not found: {meddialog_path}")

    # MedQuAD QA pairs – try CSV first, else HF
    logger.info("Preparing MedQuAD dataset...")
    medquad_csv = data_dir / "medquad.csv"
    out_path_q = data_dir / "medquad.jsonl"
    if medquad_csv.is_file():
        import pandas as pd, json
        df = pd.read_csv(medquad_csv)
        df = df.rename(columns={c: c.title() for c in df.columns})
        with out_path_q.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                json.dump({"Question": row["Question"], "Answer": row["Answer"]}, f)
                f.write("\n")
        logger.info(f"MedQuAD CSV converted to {out_path_q}")
    else:
        medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        medquad["train"].to_json(str(out_path_q))
        logger.info(f"MedQuAD saved to {out_path_q}")

# ---------------------------------------------------------------------------
# 3. Download Kaggle datasets (mtsamples & Symptom2Disease)
# ---------------------------------------------------------------------------
def download_kaggle_datasets():
    logger.info("Kaggle dataset download is disabled per user request.")
    # No operation – placeholder to keep function signature

# ---------------------------------------------------------------------------
# 4. Download MedlinePlus XML (public URL, no auth)
# ---------------------------------------------------------------------------
def download_medlineplus():
    logger.info("MedlinePlus XML download is disabled per user request.")
    # No operation – placeholder

# ---------------------------------------------------------------------------
# 5. Build FAISS indexes
# ---------------------------------------------------------------------------
def build_faiss_indexes():
    import numpy as np, faiss
    from sentence_transformers import SentenceTransformer
    data_dir = Path(__file__).parent / "data"
    faiss_dir = Path(__file__).parent.parent / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # ---- Index A: conversation chunks + MedDialog ----
    logger.info("Building Index A (conversation + MedDialog)...")
    conv_texts = []
    conv_meta = []
    conv_root = data_dir / "conversations"
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
    logger.info(f"Embedding {len(conv_texts)} texts for Index A")
    batch_size = 64
    vectors_list = []
    total_texts = len(conv_texts)
    for start_idx in range(0, total_texts, batch_size):
        batch_texts = conv_texts[start_idx:start_idx + batch_size]
        batch_vectors = embedder.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        vectors_list.append(batch_vectors)
        batch_num = start_idx // batch_size + 1
        if batch_num % 5 == 0 or start_idx + batch_size >= total_texts:
            logger.info(f"Embedded {min(start_idx + batch_size, total_texts)}/{total_texts} texts for Index A (batch {batch_num})")
    vectors = np.vstack(vectors_list)
    np.save(faiss_dir / "embeddings_a.npy", vectors)
    logger.info(f"Embeddings for Index A saved to {faiss_dir / 'embeddings_a.npy'}")
    d = vectors.shape[1]
    index_a = faiss.IndexFlatIP(d)
    index_a.add(np.asarray(vectors, dtype=np.float32))
    faiss.write_index(index_a, str(faiss_dir / "index_a.faiss"))
    with open(faiss_dir / "index_a_meta.json", "w", encoding="utf-8") as f:
        json.dump(conv_meta, f, indent=2)
    logger.info(f"Index A saved to {faiss_dir / 'index_a.faiss'}")
    # ---- Index B: MedQuAD + MedlinePlus ----
    logger.info("Building Index B (MedQuAD + MedlinePlus)...")
    texts_b = []
    medquad_path = data_dir / "medquad.jsonl"
    if medquad_path.is_file():
        import json
        with open(medquad_path, "r", encoding="utf-8") as f:
            for ln in f:
                obj = json.loads(ln)
                q = obj.get("Question", "")
                a = obj.get("Answer", "")
                if q or a:
                    texts_b.append(f"{q}\n{a}")
    medline_path = data_dir / "medlineplus.xml"
    if medline_path.is_file():
        import xml.etree.ElementTree as ET
        tree = ET.parse(medline_path)
        root = tree.getroot()
        for topic in root.findall('.//topic'):
            parts = []
            for tag in ["summary", "also-called", "see-reference"]:
                el = topic.find(tag)
                if el is not None and el.text:
                    parts.append(el.text.strip())
            if parts:
                texts_b.append(" ".join(parts))
    logger.info(f"Embedding {len(texts_b)} texts for Index B")
    batch_size = 64
    vectors_list_b = []
    total_texts_b = len(texts_b)
    for start_idx in range(0, total_texts_b, batch_size):
        batch_texts = texts_b[start_idx:start_idx + batch_size]
        batch_vectors = embedder.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        vectors_list_b.append(batch_vectors)
        batch_num = start_idx // batch_size + 1
        if batch_num % 5 == 0 or start_idx + batch_size >= total_texts_b:
            logger.info(f"Embedded {min(start_idx + batch_size, total_texts_b)}/{total_texts_b} texts for Index B (batch {batch_num})")
    vectors_b = np.vstack(vectors_list_b)
    index_b = faiss.IndexFlatIP(d)
    index_b.add(np.asarray(vectors_b, dtype=np.float32))
    faiss.write_index(index_b, str(faiss_dir / "index_b.faiss"))
    with open(faiss_dir / "index_b_meta.json", "w", encoding="utf-8") as f:
        json.dump([{"text": t} for t in texts_b], f, indent=2)
    logger.info(f"Index B saved to {faiss_dir / 'index_b.faiss'}")

# ---------------------------------------------------------------------------
# 6. Collect real samples (conversations, MedDialog, optional Kaggle CSVs)
# ---------------------------------------------------------------------------
def collect_real_samples():
    data_dir = Path(__file__).parent / "data"
    real_samples = []
    # Conversation samples
    conv_root = data_dir / "conversations"
    for specialty_dir in conv_root.iterdir():
        if not specialty_dir.is_dir():
            continue
        for txt_file in specialty_dir.rglob("*.txt"):
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]
            patient_turns = [ln[2:].strip() for ln in lines if ln.startswith("P:")]
            if not patient_turns:
                continue
            first = patient_turns[0]
            real_samples.append({"input": first, "label": specialty_dir.name})
            if len(patient_turns) >= 3:
                accum = " ".join(patient_turns[:3])
                real_samples.append({"input": accum, "label": specialty_dir.name})
    # MedDialog samples
    meddialog_path = data_dir / "meddialog_en_train.jsonl"
    if meddialog_path.is_file():
        import json
        with open(meddialog_path, "r", encoding="utf-8") as f:
            for ln in f:
                obj = json.loads(ln)
                if "description" in obj:
                    real_samples.append({"input": obj["description"], "label": "unknown"})
    # Kaggle CSV samples (if present)
    mtsamples_dir = data_dir / "mtsamples"
    if mtsamples_dir.is_dir():
        import csv
        for csv_file in mtsamples_dir.rglob("*.csv"):
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    complaint = row.get("chief complaint", "")
                    transcription = row.get("transcription", "")
                    if complaint and transcription:
                        real_samples.append({"input": f"{complaint} {transcription}", "label": "unknown"})
    symptom_dir = data_dir / "symptom2disease"
    if symptom_dir.is_dir():
        import csv
        for csv_file in symptom_dir.rglob("*.csv"):
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "")
                    if text:
                        real_samples.append({"input": text, "label": "unknown"})
    logger.info(f"Collected {len(real_samples)} real samples for inference.")
    return real_samples

# ---------------------------------------------------------------------------
# 7. Gemini department inference via Gemini 2.5-Flash
# ---------------------------------------------------------------------------
def run_department_inference(samples):
    # Add ml-training directory to sys.path so gemini_inference can be imported directly
    ml_dir = str(Path(__file__).parent)
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    from gemini_inference import infer_department
    predictions = []
    total = len(samples)
    for i, sample in enumerate(samples, 1):
        dept, confidence, _diag = infer_department(sample["input"])
        predictions.append({
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "patient_summary": sample["input"],
            "department": dept,
            "confidence": confidence,
            "doctor": "Placeholder Doctor",
        })
        if i % 50 == 0 or i == total:
            logger.info(f"Inference progress: {i}/{total}")
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "gemini_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Gemini predictions written to {out_path}")

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / "backend" / ".env"
    load_dotenv(env_path)

    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError("GEMINI_API_KEY is not set. Export it before running.")
    extract_conversations()
    download_hf_datasets()
    # Skipping Kaggle and MedlinePlus downloads as per user request
    build_faiss_indexes()
    # samples = collect_real_samples()
    # run_department_inference(samples)
    logger.info("Setup and inference completed successfully.")
