# TriagePlus — Embedding & Chunking Technical Specification

Scope: this document covers only embedding model selection and chunking methodology for Index A (conversational/case retrieval) and Index B (medical knowledge retrieval). It does not create or recommend creating any new synthetic conversation data — all rebalancing below uses only real data already available (existing scripted conversation files, real MedDialog consultations, real MedQuAD/Symptom2Disease rows).

---

## 1. Embedding model selection

### 1.1 Why the current model is a mismatch for this data

`all-MiniLM-L6-v2` (currently used) is a general-purpose sentence-similarity model:
- 384-dimensional output, 256 word-piece token limit.
- Trained on general web/NLI sentence pairs, not biomedical text — general-domain tokenizers routinely misinterpret medical vocabulary (multi-syllable Latin/Greek clinical terms fragment into more subword pieces than the tokenizer was optimized for).
- Symmetric encoder — the same model and vector space is used for both short queries and long documents, which is a bad structural fit for the question→long-answer and short-complaint→long-conversation shapes present in MedQuAD and MedDialog.

### 1.2 Candidates evaluated

| Model | Type | Dim | Token limit | Domain | Fit for this project |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` (current) | Symmetric, general | 384 | 256 | General | Baseline; causes the truncation and generic-vocabulary issues already identified |
| `pritamdeka/S-PubMedBert-MS-MARCO` / `NeuML/pubmedbert-base-embeddings` | Symmetric, `sentence-transformers`-native | 768 | 512 | Biomedical (PubMed-pretrained, MS MARCO retrieval fine-tune) | Strong, drop-in replacement — same `SentenceTransformer(...)` API as today, minimal code change |
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | Symmetric, entity-level | 768 | 25 (designed for short entity names/phrases, not passages) | Biomedical concept normalization | Not suitable as the primary passage embedder — built for linking short terms to a controlled vocabulary (e.g., "high blood pressure" ↔ "hypertension"), not for indexing paragraphs. Useful only as a future add-on for symptom-term synonym expansion, not for this chunking pipeline. |
| `ncbi/MedCPT-Query-Encoder` + `ncbi/MedCPT-Article-Encoder` | **Asymmetric dual-encoder** | 768 | Query encoder: short text; Article encoder: up to 512 tokens | Biomedical — pretrained on 255M real PubMed query-article pairs (search-log supervision, not synthetic labels) | **Recommended primary choice** — see 1.3 |

Supporting evidence from current literature: PubMedBERT-based sentence embeddings have been shown to outperform general-domain models (including MiniLM and even larger general models like `gte-base`) on biomedical semantic search benchmarks, and domain-specific pretraining has repeatedly been shown to matter more than raw parameter count for retrieval quality in this space — a fine-tuned biomedical model with fewer parameters can beat a larger general-domain one on medical retrieval tasks. MedCPT specifically has demonstrated state-of-the-art zero-shot biomedical retrieval performance, outperforming general dense retrievers many times its size, precisely because it was trained on real query→article click/search behavior rather than generic sentence-similarity objectives — which is structurally the same problem this project has (short patient query → longer knowledge/case document).

### 1.3 Recommendation: MedCPT (dual-encoder), with PubMedBERT-embeddings as the lower-effort fallback

**Primary recommendation — `ncbi/MedCPT-Query-Encoder` + `ncbi/MedCPT-Article-Encoder`:**

This is the best structural fit for this project specifically because the retrieval pattern here is inherently asymmetric — a short patient utterance or symptom summary (query-shaped) is used to search longer stored content (MedQuAD answers, MedDialog conversations, case exemplars — article-shaped). MedCPT is a **dual-encoder**, meaning it is two separate models sharing a vector space:
- **Query Encoder** — used only at query time, on short text (the live patient message or running symptom summary).
- **Article Encoder** — used only when building the index, on the longer content being stored (conversation windows, MedQuAD chunks, Symptom2Disease rows).

This maps directly onto the small-to-big / parent-child design already planned for this project — rather than being a chunking-side workaround, it is the model's native design, trained end-to-end on that exact shape of problem using 255 million real PubMed search-log query→article pairs (not synthetic or LLM-generated training data).

Trade-off: this requires loading two models instead of one, and it is not a native `sentence-transformers` model — it's loaded via `transformers.AutoModel` directly, so there's a small integration cost. Both encoders are BERT-base-sized (~110M parameters each), which is entirely practical for CPU inference: the Article Encoder only runs at index-build time (an offline batch job, so speed doesn't matter), and the Query Encoder runs once per user turn at request time (typically well under the latency already added by the existing three sequential LLM calls per turn, so it will not be the bottleneck).

**Fallback recommendation — `pritamdeka/S-PubMedBert-MS-MARCO` (or `NeuML/pubmedbert-base-embeddings`):**

If minimizing engineering effort matters more than the last few points of retrieval quality, this is a single symmetric model that's a genuine drop-in replacement for the current `SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')` call — same API, same `_search_faiss` code, just a different model string and a 384→768 dimension change in the FAISS index. It's still meaningfully better than MiniLM (biomedical pretraining, double the token limit), just without the asymmetric query/document specialization MedCPT provides.

### 1.4 Loading each model

**MedCPT:**
```python
import torch
from transformers import AutoTokenizer, AutoModel

query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")

article_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
article_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")

def encode_query(text: str) -> torch.Tensor:
    inputs = query_tokenizer([text], truncation=True, padding=True,
                              return_tensors="pt", max_length=64)
    with torch.no_grad():
        return query_model(**inputs).last_hidden_state[:, 0, :]  # [CLS] pooling

def encode_article(title: str, body: str) -> torch.Tensor:
    # Article encoder expects a [title, body]-style pair per item
    inputs = article_tokenizer([[title, body]], truncation=True, padding=True,
                                return_tensors="pt", max_length=512)
    with torch.no_grad():
        return article_model(**inputs).last_hidden_state[:, 0, :]
```

**PubMedBERT-embeddings fallback (native `sentence-transformers`):**
```python
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
vec = embedder.encode(["patient query text"], normalize_embeddings=True)
```

---

## 2. Chunking methodology — Index A (conversational / case-pattern retrieval)

**Global rule for this section:** all content that goes *into* the index (from any source) is embedded with the **Article Encoder**. Only the live patient text, embedded at query time inside `_search_faiss`, uses the **Query Encoder**.

### 2.1 Existing scripted conversation files (`conversations/*.txt`)
No new files are created. The existing sliding-window scheme is kept (3-turn window, 1-turn overlap over `D:`/`P:` lines), but:
- Chunk text is passed through the Article Encoder as `(title, body)` = `(specialty_label, windowed_dialogue_text)` — using the folder's specialty name as the "title" half of the pair gives the Article Encoder useful topical context it's specifically trained to use.
- **Per-specialty cap applied at index-build time only** (not by writing new files): reservoir-sample each specialty folder down to a fixed maximum (e.g. 150 chunks) before embedding, so Respiratory's current 5,091 chunks can't numerically dominate retrieval the way they do today. This only *subsets* existing real data — nothing is fabricated.

### 2.2 MedDialog (`meddialog_en_train.jsonl`)
Schema per consultation: `{"description": "<short patient blurb>", "utterances": ["patient: ...", "doctor: ...", ...]}`.

**(a) Opening-complaint exemplars (parent-child pair, both from real data):**
```python
{
  "embed_input": ("meddialog case", "\n".join(utterances[:8])),  # Article Encoder input: (title, body)
  "text": "\n".join(utterances[:8]),          # payload returned to the LLM
  "source": "meddialog_desc",
  "specialty": None,                           # MedDialog carries no specialty label — leave explicit
}
```
Note: `description` itself is not the thing embedded here — the *conversation excerpt* is embedded via the Article Encoder (since it's index-side content), with `description` optionally concatenated into the "title" slot of the `(title, body)` pair to give the encoder a short topical anchor, e.g. `(description, "\n".join(utterances[:8]))`.

**(b) Mid-conversation follow-up style exemplars:** same sliding-window technique as 2.1, applied to `utterances`, tagged `"source": "meddialog_turnwin"`.

**(c) Deduplication and subsampling (real-data-only rebalancing — this replaces the previously suggested "create a General Medicine folder"):**
1. Exact-dedup on normalized `description` text (lowercase, strip punctuation) to remove verbatim forum repeats.
2. **Keyword-based filtering to surface real General Medicine content:** since MedDialog has no specialty labels, apply a simple keyword match against each real `description` (fever, headache, cold, sore throat, fatigue, body ache, and similar non-specific presenting complaints) and tag matching entries `"specialty": "General"`. This directly and legitimately closes today's 19-chunk General Medicine gap in Index A using real patient-authored text, without writing any new conversation content.
3. Stratified random subsampling of the remainder (stratified by description length bucket, so short/uninformative entries aren't over-represented in what survives) to cap total MedDialog contribution at roughly 3,000–5,000 entries, keeping Index A's total size in the same order of magnitude as the existing corpus rather than 40x larger.

---

## 3. Chunking methodology — Index B (medical knowledge retrieval)

Same global rule: everything indexed uses the Article Encoder; the live query (patient's symptom summary) uses the Query Encoder at request time.

### 3.1 MedQuAD (`medquad.csv` / `medquad.jsonl`)
Schema: `question, answer, source, focus_area`.

Replace the current single `question+answer` blob with two record types, both Article-Encoder-embedded:

**Question record** — embedded as `(focus_area, question)`:
```python
{"embed_input": (focus_area, question), "text": question,
 "source": "medquad_question", "extra": {"focus_area": focus_area, "answer_ref": answer_id}}
```

**Answer chunk records** — the answer is split before embedding, since MedCPT's Article Encoder still has a 512-token limit and MedQuAD answers run up to ~29,000 characters (well beyond that):
- **Baseline method: sentence-aware recursive chunking.** Split on sentence boundaries only (never mid-sentence). Target **~350–400 tokens per chunk** (leaving headroom under the 512-token limit for the `focus_area` title portion of the pair), with **~15–20% overlap** between consecutive chunks so a fact split across a chunk boundary isn't fully lost to either side.
- **Better fit for this dataset: semantic/topic-boundary chunking.** MedQuAD answers are lightly-edited NIH/NLM web pages with implicit sub-headings (visible directly in the source text — e.g. "How Glaucoma Develops" runs straight into "Open-angle Glaucoma" with no boundary marker). This method embeds each sentence individually, then starts a new chunk wherever the cosine similarity between consecutive sentence embeddings drops below a threshold — i.e., it finds where the topic actually shifts, instead of cutting at an arbitrary token count:
  ```python
  def semantic_chunks(sentences, sentence_embedder, threshold=0.62, max_tokens=380):
      embs = sentence_embedder(sentences)  # one embedding per sentence
      chunks, current, current_len = [], [sentences[0]], 0
      for i in range(1, len(sentences)):
          sim = float(embs[i] @ embs[i-1])
          current_len = sum(len(s.split()) for s in current)
          if sim < threshold or current_len > max_tokens:
              chunks.append(" ".join(current))
              current = [sentences[i]]
          else:
              current.append(sentences[i])
      if current:
          chunks.append(" ".join(current))
      return chunks
  ```
  Each resulting chunk is embedded as `(focus_area, chunk_text)` and inherits the parent `question`/`focus_area`/`source` metadata.
- **Optional stretch method, for reference:** proposition-based chunking (per Chen et al., *"Dense X Retrieval: What Retrieval Granularity Should We Use?"*, 2023) — decomposing each answer into atomic factual statements via an offline LLM pass, one embedding per proposition. Shown in that work to outperform sentence/passage chunking on QA retrieval benchmarks, at the cost of one extra LLM batch pass over the ~16K MedQuAD rows at index-build time (not in the live request path). Recommended only after the two simpler methods above are implemented and validated.

**Retrieval-time cap (companion to chunking, not a substitute for it):** at query time, cap what gets concatenated into the LLM's knowledge context to the top 3 chunks with a hard total character cap (e.g. ~800 characters), so long-tail retrieved chunks can't bloat the prompt even after proper chunking.

### 3.2 Symptom2Disease.csv
Schema: `label` (disease), `text` (short first-person symptom paragraph — already comfortably under the token limit).

No chunking required. Embed directly as `(label, text)`:
```python
{"embed_input": (label, text), "text": text, "source": "symptom2disease", "extra": {"disease_label": label}}
```

### 3.3 MedlinePlus XML (only if enabled later — currently absent)
Structured (`<summary>`, `<also-called>`, `<see-reference>`). Chunk `summary` using the same sentence-aware/semantic method as MedQuAD answers. Treat `also-called` as a synonym table for future query expansion (e.g. patient says "high blood pressure," content is indexed under "hypertension") rather than as content to embed itself.

---

## 4. Summary of chunking parameters

| Source | Unit embedded | Chunking method | Target size | Overlap | Encoder used |
|---|---|---|---|---|---|
| `conversations/*.txt` | 3-turn dialogue window | Fixed sliding window (existing) | ~6 lines | 1-turn | Article |
| MedDialog — opening exemplar | First 8 `utterances` | None (fixed prefix) | ≤8 turns | n/a | Article |
| MedDialog — turn window | Sliding `utterances` window | Fixed sliding window | 3 turns | 1-turn | Article |
| MedQuAD — question | `question` field | None | as-is | n/a | Article |
| MedQuAD — answer | `answer` field | Sentence-aware or semantic chunking | ~350–400 tokens | ~15–20% | Article |
| Symptom2Disease | `text` field | None | as-is | n/a | Article |
| Live patient query (either index) | Patient message / summary | None | as-is | n/a | **Query** |

---

## 5. FAISS index structural notes

- Switching from MiniLM (384-dim) to MedCPT/PubMedBERT (768-dim) requires rebuilding both `index_a.faiss` and `index_b.faiss` from scratch with the new dimension — existing `.faiss` files are not compatible and must be regenerated, not migrated in place.
- `IndexFlatIP` (exact brute-force cosine similarity via inner product on normalized vectors) remains appropriate at the resulting scale (Index A in the low-to-mid thousands after capping/subsampling; Index B in the tens of thousands after MedQuAD chunking). If either index later grows past roughly the 100K–500K vector range, revisit with `IndexHNSWFlat` for approximate, sub-linear-time search — not needed at the sizes produced by this plan.

## 6. Validation after rebuilding
- **Token-length audit:** run the Article Encoder's tokenizer over every chunk and assert the resulting token count is under 512 before accepting the build — catches any chunking miscalculation immediately rather than silently truncating.
- **Distribution audit:** re-run a per-specialty/per-source count over the rebuilt `index_a_meta.json` (as done previously) and confirm no specialty exceeds its configured cap, and that the keyword-filtered General Medicine count from real MedDialog data is non-trivial (not zero).
- **Retrieval spot-check:** for a handful of known queries (including "fever," "headache," "mild fatigue"), print top-5 hits from both indices using the Query Encoder and manually confirm topical relevance before wiring the new indices into the live inference path.
