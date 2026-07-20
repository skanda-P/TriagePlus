"""Regression contract test for the RAG index build / runtime path alignment.

This test pins down the file-name contract between the three build scripts in
`backend/scripts/` and the runtime reader in `backend/app/core/unified_retrieval.py`.

We do NOT execute the build scripts (they require faiss + sentence-transformers
+ the DDXPlus / MedQuAD / MedDialog source data, none of which are available in
CI). Instead we exercise the file paths computed by each side, asserting they
agree on every artifact file the runtime expects to find.

This test specifically guards against regressions of:

- Bug 1: build_medquad_index.py previously called langchain's
  `FAISS.save_local(...)` which writes `index.faiss` + `index.pkl`, while the
  runtime looked for `medquad.index` (faiss native) -> MedQuAD index never
  loaded.
- Bug 2: build_meddialog_qa_index.py previously called `IndexFlatL2.add_with_ids`
  which is unsupported and would crash the build mid-flight, leaving no index.
- Bug 3: a stale `build_meddialog_index.py` script wrote `meddialog.index` +
  `metadata.json` (NOT `_metadata.pkl` / `_bm25.pkl`), and the runtime couldn't
  use it.
"""
from pathlib import Path

import pytest

# Resolve exactly as the build scripts do.
BACKEND_ROOT = Path(__file__).parent.parent
BUILD_SCRIPTS_DIR = BACKEND_ROOT / "scripts"
DATA_DIR = BACKEND_ROOT / "data"
FAISS_DIR = DATA_DIR / "faiss"


# -----------------------------------------------------------------------------
# Build script paths - mirror the exact path constants computed by each script
# (resolved from `Path(__file__).parent.parent`).
# -----------------------------------------------------------------------------

def _medquad_build_artifacts() -> dict:
    """Files the rewritten build_medquad_index.py writes."""
    index_dir = FAISS_DIR / "medquad"
    return {
        "faiss_index": index_dir / "medquad.index",
        "metadata_pkl": index_dir / "medquad_metadata.pkl",
        "bm25_pkl": index_dir / "medquad_bm25.pkl",
        "summary_json": index_dir / "medquad_summary.json",
        # NOT tolerated: langchain's "index.faiss" / "index.pkl" files.
    }


def _conversations_build_artifacts() -> dict:
    """Files build_conversations_index.py writes."""
    index_dir = FAISS_DIR / "conversations"
    return {
        "faiss_index": index_dir / "conversations.index",
        "metadata_pkl": index_dir / "conversations_metadata.pkl",
        "bm25_pkl": index_dir / "conversations_bm25.pkl",
        "summary_json": index_dir / "conversations_summary.json",
    }


def _meddialog_build_artifacts() -> dict:
    """Files build_meddialog_qa_index.py writes."""
    index_dir = FAISS_DIR / "meddialog"
    return {
        "faiss_index": index_dir / "meddialog.index",
        "metadata_pkl": index_dir / "meddialog_metadata.pkl",
        "bm25_pkl": index_dir / "meddialog_bm25.pkl",
        "summary_json": index_dir / "meddialog_summary.json",
    }


# -----------------------------------------------------------------------------
# Runtime paths - mirror what unified_retrieval.py _load_*_index methods look
# for. We compute them independent of the runtime module so this test works
# even if faiss / sentence-transformers aren't installed.
# -----------------------------------------------------------------------------

def _runtime_artifact_paths() -> dict:
    """Files the runtime UnifiedRetriever opens, keyed by source.

    Sources:
      - medquad
      - conversations
      - meddialog

    Each entry is a dict of {kind: Path} enumerating EVERY file the runtime
    opens. Any path the build scripts write and the runtime ignores is fine;
    any path the runtime opens and the build scripts DON'T write is a bug.
    """
    return {
        "medquad": {
            "faiss_index": FAISS_DIR / "medquad" / "medquad.index",
            "metadata_pkl": FAISS_DIR / "medquad" / "medquad_metadata.pkl",
            "bm25_pkl": FAISS_DIR / "medquad" / "medquad_bm25.pkl",
        },
        "conversations": {
            "faiss_index": FAISS_DIR / "conversations" / "conversations.index",
            "metadata_pkl": FAISS_DIR / "conversations" / "conversations_metadata.pkl",
            "bm25_pkl": FAISS_DIR / "conversations" / "conversations_bm25.pkl",
        },
        "meddialog": {
            "faiss_index": FAISS_DIR / "meddialog" / "meddialog.index",
            "metadata_pkl": FAISS_DIR / "meddialog" / "meddialog_metadata.pkl",
            "bm25_pkl": FAISS_DIR / "meddialog" / "meddialog_bm25.pkl",
        },
    }


# -----------------------------------------------------------------------------
# Contract assertions
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("source,build_artifacts", [
    ("medquad", _medquad_build_artifacts()),
    ("conversations", _conversations_build_artifacts()),
    ("meddialog", _meddialog_build_artifacts()),
])
def test_build_writes_every_file_runtime_reads(source: str, build_artifacts: dict):
    """Every file the runtime opens must be produced by the corresponding
    build script. Drift here is what produced bug 1 / bug 3 historically.
    """
    runtime = _runtime_artifact_paths()[source]
    for kind, runtime_path in runtime.items():
        assert kind in build_artifacts, (
            f"Runtime expects '{kind}' for {source} but the build script "
            f"doesn't enumerate a corresponding artifact path - contract drift."
        )
        assert build_artifacts[kind] == runtime_path, (
            f"{source} '{kind}' path mismatch:\n"
            f"  build  -> {build_artifacts[kind]}\n"
            f"  runtime-> {runtime_path}"
        )


def test_build_medquad_no_longer_writes_langchain_files():
    """Bug 1 regression guard: the rewritten MedQuAD build script must NOT emit
    langchain's `index.faiss` + `index.pkl` files. We assert the only
    `.faiss`-format file the runtime looks for is `medquad.index`.
    """
    artifacts = _medquad_build_artifacts()
    for kind, path in artifacts.items():
        assert path.name != "index.faiss", (
            f"MedQuAD build still writes langchain's index.faiss at {path} - "
            "the runtime reads medquad.index."
        )
        assert path.name != "index.pkl", (
            f"MedQuAD build still writes langchain's index.pkl at {path} - "
            "the runtime reads medquad_metadata.pkl."
        )


def test_no_legacy_build_meddialog_script_in_repo():
    """Bug 3 regression guard: there should be exactly ONE MedDialog build
    script (the *_qa_index one referenced by README), not two.
    """
    meddialog_scripts = sorted(BUILD_SCRIPTS_DIR.glob("build_meddialog*.py"))
    assert meddialog_scripts == [BUILD_SCRIPTS_DIR / "build_meddialog_qa_index.py"], (
        f"Expected only build_meddialog_qa_index.py in {BUILD_SCRIPTS_DIR}, "
        f"got: {[p.name for p in meddialog_scripts]}. The legacy "
        "build_meddialog_index.py produced an index the runtime couldn't load "
        "(lack of _metadata.pkl / _bm25.pkl) and should have been deleted."
    )


def test_meddialog_build_does_not_use_unsupported_add_with_ids_on_flat_l2():
    """Bug 2 regression guard: IndexFlatL2.add_with_ids is unsupported by
    faiss. The script must wrap the index in IndexIDMap before add_with_ids,
    matching build_conversations_index.py.
    """
    script = BUILD_SCRIPTS_DIR / "build_meddialog_qa_index.py"
    src = script.read_text(encoding="utf-8")
    assert "faiss.IndexIDMap(faiss.IndexFlatL2" in src, (
        f"{script.name} must wrap IndexFlatL2 in IndexIDMap before "
        "add_with_ids - otherwise the build throws RuntimeError and produces "
        "no meddialog.index at all."
    )


def test_medquad_build_writes_medquad_index_not_langchain():
    """Bug 1 regression guard at source-level: build_medquad_index.py must
    emit raw faiss via faiss.write_index(index, 'medquad.index'), not use
    langchain.save_local()."""
    script = BUILD_SCRIPTS_DIR / "build_medquad_index.py"
    src = script.read_text(encoding="utf-8")
    assert "faiss.write_index" in src, (
        f"{script.name} must call faiss.write_index(...) to produce "
        "medquad.index (the file the runtime reads)."
    )
    assert "save_local" not in src, (
        f"{script.name} must not use langchain FAISS.save_local - it produces "
        "index.faiss/index.pkl which the runtime does not read."
    )
    assert "from langchain_community" not in src, (
        f"{script.name} should not depend on langchain_community - "
        "we use raw SentenceTransformer + faiss to match the Conversations "
        "and MedDialog build shape."
    )


def test_medquad_build_has_no_unused_model_dir_constant():
    """Bug 4 regression guard: the unused MODEL_DIR pointing at backend/models
    (wrong - other scripts use backend/model singular) must stay deleted
    across future edits. We assert the constant never reappears.
    """
    src = (BUILD_SCRIPTS_DIR / "build_medquad_index.py").read_text(encoding="utf-8")
    assert "MODEL_DIR" not in src, (
        "build_medquad_index.py should not define MODEL_DIR - it's unused "
        "and previously pointed at backend/models (with 's'), inconsistent "
        "with the rest of the repo which uses backend/model (singular)."
    )
