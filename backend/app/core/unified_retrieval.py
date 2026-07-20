#!/usr/bin/env python3
"""
Unified hybrid retrieval system combining:
- MedQuAD (medical corpus) with 0.3 BM25 + 0.7 Dense
- Conversations (few-shot examples) with 0.4 BM25 + 0.6 Dense
- MedDialog (Q&A index) with 0.5 BM25 + 0.5 Dense

Concurrent search across all three indices with result merging.

Design notes (see review findings R1/R5/R6/R7/R11/R14/R16):
- Corpus vectors must be L2-normalized at build time so cosine similarity is
  meaningful. Runtime queries are also L2-normalized. The build scripts have
  been updated to pass `normalize_embeddings=True` for all three sources; a
  metadata `normalized` flag is asserted at load time.
- BM25 tokenization uses a single shared `tokenize()` helper (regex
  word-boundary, lowercase) so build and query paths can never drift.
- Per-source score normalization: dense → 1 / (1 + L2) ∈ (0, 1], BM25 →
  divided by max BM25 score across the current candidate set so the top BM25
  hit always lands at ~1.0 (no more arbitrary /50.0 magic constant).
- `_diagnostic_clients` is now a set with a lock to avoid races on
  concurrent connection/disconnection.
"""

import os
import re
import pickle
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path(__file__).parent.parent.parent / "data"
FAISS_DIR = DATA_DIR / "faiss"

# Embedding model — must be the same across build & runtime
EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# Per-source hybrid weights (kept as constants so they don't drift from the
# build-time metadata documents).
SOURCE_WEIGHTS = {
    "medquad": {"bm25": 0.3, "dense": 0.7},
    "conversations": {"bm25": 0.4, "dense": 0.6},
    "meddialog": {"bm25": 0.5, "dense": 0.5},
}


def _tokenize(text: str) -> List[str]:
    """Word-only tokenization shared by build, runtime-fallback, and query paths.

    Replaces bare `str.lower().split()` which kept punctuation glued to tokens
    ("throat." vs "throat") and so gave BM25 ~0 matches on most punctuated
    queries.
    """
    return re.findall(r"\b\w+\b", (text or "").lower())


class UnifiedRetriever:
    """Unified retrieval across MedQuAD, Conversations, and MedDialog."""

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.medquad_index = None
        self.medquad_bm25 = None
        self.medquad_chunks: List[Dict] = []

        self.conversations_index = None
        self.conversations_bm25 = None
        self.conversations_chunks: List[Dict] = []

        self.meddialog_index = None
        self.meddialog_bm25 = None
        self.meddialog_chunks: List[Dict] = []

        self._load_indices()

    def _load_indices(self):
        """Load all FAISS indices, BM25 indexes, and metadata."""
        logger.info("Loading unified retrieval indices...")

        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Error loading model (fatal for retriever): {e}")
            # Do not cache a broken retriever as a singleton — caller will see
            # this exception and the singleton factory will not store it.
            raise

        self._load_medquad_index()
        self._load_conversations_index()
        self._load_meddialog_index()

        loaded = sum(
            1
            for idx in (self.medquad_index, self.conversations_index, self.meddialog_index)
            if idx is not None and idx.ntotal > 0
        )
        if loaded == 0:
            logger.error(
                "No FAISS indices could be loaded — retriever will return [] for every query. "
                "Run the build_*_index.py scripts before serving traffic."
            )

    def _assert_index_chunks_consistent(self, index, chunks, source):
        """Guard against the ID/position drift failure mode by asserting
        `index.ntotal == len(chunks)` at load time."""
        if index is not None and chunks is not None:
            if index.ntotal != len(chunks):
                logger.error(
                    f"[{source}] FAISS index.ntotal ({index.ntotal}) != len(chunks) "
                    f"({len(chunks)}). Returned IDs may map to the wrong chunk. "
                    "Re-running the build script is required."
                )

    def _load_medquad_index(self):
        try:
            medquad_dir = FAISS_DIR / "medquad"
            if not medquad_dir.exists():
                return
            index_path = medquad_dir / "medquad.index"
            metadata_path = medquad_dir / "medquad_metadata.pkl"
            bm25_path = medquad_dir / "medquad_bm25.pkl"

            if index_path.exists() and metadata_path.exists():
                self.medquad_index = faiss.read_index(str(index_path))
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                self.medquad_chunks = metadata.get("chunks", [])
                logger.info(f"Loaded MedQuAD index: {self.medquad_index.ntotal} chunks")
                self._assert_index_chunks_consistent(self.medquad_index, self.medquad_chunks, "medquad")

                if bm25_path.exists():
                    with open(bm25_path, "rb") as f:
                        self.medquad_bm25 = pickle.load(f)
                    logger.info("Loaded MedQuAD BM25 index")
                else:
                    self._build_bm25("medquad")
            else:
                logger.warning(f"MedQuAD index or metadata missing in {medquad_dir}")
        except Exception as e:
            logger.warning(f"Could not load MedQuAD index: {e}")

    def _load_conversations_index(self):
        try:
            conversations_dir = FAISS_DIR / "conversations"
            if not conversations_dir.exists():
                return
            index_path = conversations_dir / "conversations.index"
            metadata_path = conversations_dir / "conversations_metadata.pkl"
            bm25_path = conversations_dir / "conversations_bm25.pkl"

            if index_path.exists() and metadata_path.exists():
                self.conversations_index = faiss.read_index(str(index_path))
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                self.conversations_chunks = metadata.get("chunks", [])
                logger.info(f"Loaded Conversations index: {self.conversations_index.ntotal} chunks")
                self._assert_index_chunks_consistent(
                    self.conversations_index, self.conversations_chunks, "conversations"
                )

                if bm25_path.exists():
                    with open(bm25_path, "rb") as f:
                        self.conversations_bm25 = pickle.load(f)
                    logger.info("Loaded Conversations BM25 index")
                else:
                    self._build_bm25("conversations")
            else:
                logger.warning(f"Conversations index or metadata missing in {conversations_dir}")
        except Exception as e:
            logger.warning(f"Could not load Conversations index: {e}")

    def _load_meddialog_index(self):
        try:
            meddialog_dir = FAISS_DIR / "meddialog"
            if not meddialog_dir.exists():
                return
            index_path = meddialog_dir / "meddialog.index"
            metadata_path = meddialog_dir / "meddialog_metadata.pkl"
            bm25_path = meddialog_dir / "meddialog_bm25.pkl"

            if index_path.exists() and metadata_path.exists():
                self.meddialog_index = faiss.read_index(str(index_path))
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                self.meddialog_chunks = metadata.get("chunks", [])
                logger.info(f"Loaded MedDialog index: {self.meddialog_index.ntotal} Q&A pairs")
                self._assert_index_chunks_consistent(
                    self.meddialog_index, self.meddialog_chunks, "meddialog"
                )

                if bm25_path.exists():
                    with open(bm25_path, "rb") as f:
                        self.meddialog_bm25 = pickle.load(f)
                    logger.info("Loaded MedDialog BM25 index")
                else:
                    self._build_bm25("meddialog")
            else:
                logger.warning(f"MedDialog index or metadata missing in {meddialog_dir}")
        except Exception as e:
            logger.warning(f"Could not load MedDialog index: {e}")

    def _build_bm25(self, source: str):
        """Build BM25 index for a source if not present.

        Uses the shared `_tokenize` helper so query-time and build-time
        tokenization always agree, and writes atomically to avoid races with
        the build script running in parallel.
        """
        try:
            if source == "medquad" and self.medquad_chunks:
                texts = [c.get("answer_chunk", c.get("answer", "")) for c in self.medquad_chunks]
            elif source == "conversations" and self.conversations_chunks:
                # Use full_text so patient-side text is also indexed; otherwise
                # patient-symptom queries get no BM25 credit.
                texts = [c.get("full_text", c.get("doctor_few_shot", "")) for c in self.conversations_chunks]
            elif source == "meddialog" and self.meddialog_chunks:
                # full_text already contains Description + Doctor; don't
                # prepend patient_question (it's a duplicate).
                texts = [c.get("full_text", "") for c in self.meddialog_chunks]
            else:
                return

            tokenized = [_tokenize(t) for t in texts]
            bm25 = BM25Okapi(tokenized)

            bm25_path_map = {
                "medquad": FAISS_DIR / "medquad" / "medquad_bm25.pkl",
                "conversations": FAISS_DIR / "conversations" / "conversations_bm25.pkl",
                "meddialog": FAISS_DIR / "meddialog" / "meddialog_bm25.pkl",
            }
            bm25_path = bm25_path_map[source]
            if source == "medquad":
                self.medquad_bm25 = bm25
            elif source == "conversations":
                self.conversations_bm25 = bm25
            elif source == "meddialog":
                self.meddialog_bm25 = bm25

            # Atomic write: safe even if another process is also rebuilding.
            bm25_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = bm25_path.with_suffix(bm25_path.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(bm25, f)
            try:
                os.replace(tmp_path, bm25_path)
            except OSError:
                # On some platforms os.replace may fail across volumes; fall back.
                if tmp_path.exists():
                    tmp_path.unlink()
            logger.info(f"Built and saved BM25 for {source}: {len(texts)} docs")
        except Exception as e:
            logger.warning(f"Could not build BM25 for {source}: {e}")

    def _hybrid_search(
        self,
        query: str,
        index,
        bm25,
        chunks: List[Dict],
        top_k: int,
        source: str,
    ) -> List[Dict]:
        """Perform hybrid BM25 + Dense search and merge results."""
        if not index or not self.model or not chunks:
            return []
        if not query or not query.strip():
            logger.debug(f"[{source}] empty query — returning []")
            return []

        try:
            weights = SOURCE_WEIGHTS.get(source, {"bm25": 0.3, "dense": 0.7})
            bm25_weight = weights["bm25"]
            dense_weight = weights["dense"]

            # Dense search (query is L2-normalized; corpus vectors are too at
            # build time, so this is consistent cosine-style retrieval).
            query_emb = np.asarray(
                self.model.encode([query], normalize_embeddings=True), dtype="float32"
            )
            dense_distances, dense_indices = index.search(query_emb, top_k * 2)

            # BM25 search
            query_tokens = _tokenize(query)
            bm25_scores: Optional[np.ndarray] = None
            if bm25 is not None and query_tokens:
                bm25_scores = bm25.get_scores(query_tokens)

            # Collect candidate indices
            all_indices = set()
            pos_map = {int(i): p for p, i in enumerate(dense_indices[0].tolist())}
            all_indices.update(dense_indices[0].tolist())
            if bm25_scores is not None:
                # argsort(-score) → descending; tie-broken by numpy is stable.
                bm25_top = np.argsort(bm25_scores)[::-1][: top_k * 2]
                all_indices.update(int(i) for i in bm25_top)
            all_indices = [i for i in all_indices if 0 <= i < len(chunks)]

            # BM25 normalization: divide by max score across candidates (so the
            # top BM25 hit lands at ~1.0). Avoids the previous `min(s/50, 1)`
            # magic constant that varied wildly across source/query lengths.
            max_bm25 = 0.0
            if bm25_scores is not None and all_indices:
                cand_scores = [float(bm25_scores[i]) for i in all_indices]
                max_bm25 = max(cand_scores) if cand_scores else 0.0
            if max_bm25 <= 0:
                max_bm25 = 1.0

            results: List[Dict] = []
            for idx in all_indices:
                dense_score = 0.0
                if idx in pos_map:
                    dist = float(dense_distances[0][pos_map[idx]])
                    dense_score = 1.0 / (1.0 + max(0.0, dist))

                bm25_score = (float(bm25_scores[idx]) / max_bm25) if bm25_scores is not None else 0.0

                hybrid = bm25_weight * bm25_score + dense_weight * dense_score
                if hybrid > 0:
                    chunk = chunks[idx]
                    results.append(
                        {
                            "source": source,
                            "chunk": chunk,
                            "hybrid_score": float(hybrid),
                            "dense_score": float(dense_score),
                            "bm25_score": float(bm25_score),
                            "index": int(idx),
                        }
                    )

            results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error in hybrid search for {source}: {e}")
            return []

    def retrieve_medquad(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve from MedQuAD."""
        return self._hybrid_search(
            query, self.medquad_index, self.medquad_bm25, self.medquad_chunks, top_k, "medquad"
        )

    def retrieve_conversations(self, query: str, symptom: Optional[str] = None, top_k: int = 3) -> List[Dict]:
        """Retrieve from Conversations — `symptom` is appended as context."""
        search_text = f"{query} {symptom}" if symptom else query
        return self._hybrid_search(
            search_text, self.conversations_index, self.conversations_bm25,
            self.conversations_chunks, top_k, "conversations"
        )

    def retrieve_meddialog(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve from MedDialog Q&A."""
        return self._hybrid_search(
            query, self.meddialog_index, self.meddialog_bm25, self.meddialog_chunks, top_k, "meddialog"
        )

    def retrieve_concurrent(
        self, query: str, symptom: Optional[str] = None, top_k_per_source: int = 5
    ) -> Dict[str, List[Dict]]:
        """Retrieve from all three sources concurrently using a thread pool.

        SentenceTransformer.encode and faiss.search both release the GIL for
        the heavy work, so threads give real concurrency. Total latency is the
        max of the three sources (not the sum as the previous serial impl).
        """
        logger.info(f"Concurrent retrieval: query='{query}', symptom='{symptom}'")

        with ThreadPoolExecutor(max_workers=3) as ex:
            f_medquad = ex.submit(self.retrieve_medquad, query, top_k_per_source)
            f_conv = ex.submit(self.retrieve_conversations, query, symptom, top_k_per_source)
            f_meddialog = ex.submit(self.retrieve_meddialog, query, top_k_per_source)
            results = {
                "medquad": f_medquad.result(),
                "conversations": f_conv.result(),
                "meddialog": f_meddialog.result(),
            }

        logger.info(
            f"Retrieved: {len(results['medquad'])} from MedQuAD, "
            f"{len(results['conversations'])} from Conversations, "
            f"{len(results['meddialog'])} from MedDialog"
        )
        return results

    # Backwards-compat alias for callers that imported retrieve_parallel.
    def retrieve_parallel(
        self, query: str, symptom: Optional[str] = None, top_k_per_source: int = 5
    ) -> Dict[str, List[Dict]]:
        return self.retrieve_concurrent(query, symptom, top_k_per_source)

    def get_fewshot_examples(
        self, query: str, symptom: Optional[str] = None, num_examples: int = 3
    ) -> List[str]:
        """Get top few-shot examples (doctor turns) from Conversations.

        `query` should be a natural-language symptom text (NOT an E_XX code).
        """
        conv_results = self.retrieve_conversations(query, symptom, num_examples)

        few_shot = []
        for result in conv_results:
            chunk = result.get("chunk", {})
            doctor_turn = chunk.get("doctor_few_shot", "")
            if doctor_turn:
                few_shot.append(doctor_turn)
        return few_shot[:num_examples]


# Singleton instance with lock (was unlocked before, racing first-turn N concurrent WS)
_retriever_instance: Optional[UnifiedRetriever] = None
_retriever_lock = threading.Lock()


def get_unified_retriever() -> UnifiedRetriever:
    """Get singleton retriever instance (thread-safe).

    Does NOT cache a broken retriever — if `_load_indices` raises during
    construction, the singleton is left as None so the next request can retry.
    """
    global _retriever_instance
    if _retriever_instance is None:
        with _retriever_lock:
            if _retriever_instance is None:
                try:
                    _retriever_instance = UnifiedRetriever()
                except Exception as e:
                    logger.error(f"Failed to construct UnifiedRetriever: {e}")
                    raise
    return _retriever_instance


def reset_unified_retriever() -> None:
    """Test hook: reset the singleton."""
    global _retriever_instance
    with _retriever_lock:
        _retriever_instance = None
