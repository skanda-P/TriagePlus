#!/usr/bin/env python3
"""
Build FAISS index for MedQuAD.

- Atomic QA pairs as chunks; long answers split by paragraph with parent-child
  linking.
- Metadata preserved (focus_area, question_type, source).
- Embeddings: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract,
  L2-normalized at encode time so the runtime's normalize-on-query path lands
  in a consistent cosine-like similarity space.
- Hybrid search weights: 0.3 BM25 + 0.7 Dense.

Output files (written into backend/data/faiss/medquad/, matches the runtime
reader in app/core/unified_retrieval._load_medquad_index):
  - medquad.index           (raw faiss index, IndexIDMap(IndexFlatL2))
  - medquad_metadata.pkl    ({chunks, embeddings, total_chunks, model_name,
                              dimension, hybrid_weights})
  - medquad_bm25.pkl        (BM25Okapi over chunk answer text)
  - medquad_summary.json    (human-readable summary, not read at runtime)
"""

import csv
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import re

if TYPE_CHECKING:
    import faiss  # noqa: F401  (type-only; real import is local in functions)


def _tokenize(text):
    """Shared word-boundary tokenizer; matches runtime query tokenization."""
    return re.findall(r"\b\w+\b", (text or "").lower())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories - resolved relative to backend/ so the same script works whether
# invoked from backend/ or repo root.
BACKEND_DIR = Path(__file__).parent.parent
DATA_DIR = BACKEND_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss"
MEDQUAD_INDEX_DIR = FAISS_DIR / "medquad"

# Source data
MEDQUAD_CSV = DATA_DIR / "medquad.csv"

# Model
EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# Hybrid search weights - documented in README and consumed by the runtime's
# retrieve_medquad() call (which hard-codes 0.3/0.7); stored in metadata for
# human readability only.
HYBRID_WEIGHTS = {"bm25": 0.3, "dense": 0.7}


def chunk_answer_by_paragraph(answer: str, max_tokens: int = 500) -> List[Tuple[str, int]]:
    """Split long answers by paragraph, keeping track of chunk indices.

    Returns list of (paragraph_text, paragraph_index) tuples. Short answers
    return a single (full_answer, 0) tuple.
    """
    char_threshold = max_tokens * 4  # rough token -> char proxy

    if len(answer) <= char_threshold:
        return [(answer, 0)]

    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]

    if len(paragraphs) <= 1:
        # No clear paragraph structure - fall back to sentence splitting.
        sentences = [s.strip() for s in answer.split(". ") if s.strip()]
        result: List[Tuple[str, int]] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence if sentence.endswith(".") else sentence + "."
            if current_length + len(sentence) <= char_threshold:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                if current_chunk:
                    result.append((" ".join(current_chunk), len(result)))
                current_chunk = [sentence]
                current_length = len(sentence)

        if current_chunk:
            result.append((" ".join(current_chunk), len(result)))

        return result if result else [(answer, 0)]

    return [(para, i) for i, para in enumerate(paragraphs)]


def load_medquad_csv(csv_path: Path) -> List[Dict]:
    """Load MedQuAD CSV into chunk dicts with metadata.

    CSV columns: question, answer, source, focus_area.
    """
    chunks: List[Dict] = []

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return chunks

    logger.info(f"Loading MedQuAD from {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != ["question", "answer", "source", "focus_area"]:
            logger.warning(
                f"CSV columns are {reader.fieldnames}, "
                "expected ['question', 'answer', 'source', 'focus_area']"
            )

        for row_idx, row in enumerate(reader):
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()
            source = row.get("source", "").strip()
            focus_area = row.get("focus_area", "").strip()

            if not question or not answer:
                logger.warning(f"Row {row_idx} missing question or answer, skipping")
                continue

            question_lower = question.lower()
            if any(w in question_lower for w in ["what is", "what are", "how does", "why"]):
                question_type = "symptoms"
            elif any(w in question_lower for w in
                      ["treat", "cure", "medication", "medicine", "therapy"]):
                question_type = "treatment"
            elif any(w in question_lower for w in
                      ["prognosis", "outlook", "survival", "life expectancy"]):
                question_type = "prognosis"
            elif any(w in question_lower for w in
                      ["cause", "risk", "susceptibility", "who"]):
                question_type = "susceptibility"
            else:
                question_type = "general"

            answer_chunks = chunk_answer_by_paragraph(answer)
            for chunk_text, chunk_idx in answer_chunks:
                chunks.append({
                    "question": question,
                    "answer_chunk": chunk_text,
                    "source": source,
                    "focus_area": focus_area,
                    "question_type": question_type,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(answer_chunks),
                    # Store the full answer only on the first chunk so the
                    # runtime can use it as additional context if desired.
                    "full_answer": answer if chunk_idx == 0 else None,
                })

    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def build_faiss_index(
    chunks: List[Dict],
    model: SentenceTransformer,
) -> Tuple["faiss.Index", np.ndarray]:
    """Embed chunks with PubMedBERT (L2-normalized) and build an IDMap+FlatL2 index.

    Mirrors the strategy used by build_conversations_index.py so all three
    indices share the same on-disk format and ID semantics.
    """
    import faiss  # local import so module import doesn't hard-require faiss

    if not chunks:
        logger.error("No chunks to index")
        return None, None

    logger.info(f"Encoding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    texts = [c["answer_chunk"] for c in chunks]
    embeddings = np.asarray(
        model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalized -> cosine ~ 1/(1+L2)
        ),
        dtype="float32",
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    ids = np.arange(len(chunks), dtype=np.int64)
    index.add_with_ids(embeddings, ids)

    logger.info(f"Built FAISS index: {index.ntotal} vectors, {dimension}D")
    return index, embeddings


def build_bm25_index(chunks: List[Dict]) -> BM25Okapi:
    """BM25 over the chunk answer text. Uses the shared word-boundary tokenizer
    so build-time tokens match the runtime query tokenizer."""
    texts = [c.get("answer_chunk", c.get("answer", "")) for c in chunks]
    tokenized = [_tokenize(t) for t in texts]
    return BM25Okapi(tokenized)


def save_index(
    index: "faiss.Index",
    bm25: BM25Okapi,
    chunks: List[Dict],
    embeddings: np.ndarray,
):
    """Write all artifact files the runtime expects, plus a human summary."""
    import faiss

    MEDQUAD_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss_path = MEDQUAD_INDEX_DIR / "medquad.index"
    faiss.write_index(index, str(faiss_path))
    logger.info(f"Saved FAISS index to {faiss_path}")

    bm25_path = MEDQUAD_INDEX_DIR / "medquad_bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 index to {bm25_path}")

    metadata = {
        "chunks": chunks,
        "embeddings": embeddings,
        "total_chunks": len(chunks),
        "model_name": EMBEDDING_MODEL,
        "dimension": embeddings.shape[1] if embeddings.size > 0 else 0,
        "hybrid_weights": HYBRID_WEIGHTS,
    }
    metadata_path = MEDQUAD_INDEX_DIR / "medquad_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved metadata to {metadata_path}")

    summary = {
        "total_chunks": len(chunks),
        "embedding_dimension": metadata["dimension"],
        "embedding_model": EMBEDDING_MODEL,
        "chunking_strategy": "atomic_qa_pairs_with_paragraph_splitting",
        "hybrid_weights": HYBRID_WEIGHTS,
        "question_types": list(set(c["question_type"] for c in chunks)),
        "sources": list(set(c["source"] for c in chunks)),
        "focus_areas": list(set(c["focus_area"] for c in chunks)),
    }
    summary_path = MEDQUAD_INDEX_DIR / "medquad_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")


def main():
    logger.info("=" * 80)
    logger.info("MedQuAD FAISS Index Builder")
    logger.info("=" * 80)

    if not MEDQUAD_CSV.exists():
        logger.error(f"MedQuAD CSV not found: {MEDQUAD_CSV}")
        sys.exit(1)

    chunks = load_medquad_csv(MEDQUAD_CSV)
    if not chunks:
        logger.error("Failed to load MedQuAD data")
        sys.exit(1)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    index, embeddings = build_faiss_index(chunks, model)
    if index is None:
        logger.error("Failed to build FAISS index")
        sys.exit(1)

    if index.ntotal < 1000:
        logger.warning(f"Index has only {index.ntotal} vectors (expected 1000+)")
    else:
        logger.info(f"SUCCESS: Index has {index.ntotal} vectors")

    bm25 = build_bm25_index(chunks)
    save_index(index, bm25, chunks, embeddings)

    logger.info("=" * 80)
    logger.info("MedQuAD index built successfully!")
    logger.info(f"  - {len(chunks)} chunks indexed")
    logger.info(f"  - Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  - Hybrid search: {HYBRID_WEIGHTS['bm25']} BM25 + {HYBRID_WEIGHTS['dense']} Dense")
    logger.info(f"  - Output dir: {MEDQUAD_INDEX_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
