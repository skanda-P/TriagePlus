import os
import json
import re
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR.parent / "faiss"

# Ensure FAISS dir exists
FAISS_DIR.mkdir(parents=True, exist_ok=True)

def chunk_medquad_entry(question: str, answer: str, focus_area: str) -> list[Document]:
    """
    Structure-aware chunking for MedQuAD entries as per architecture docs.
    """
    docs = []
    
    # Chunk type 1: The question itself
    docs.append(Document(
        page_content=question.strip(),
        metadata={
            "source": "medquad",
            "chunk_type": "question",
            "focus_area": focus_area,
            "parent_question": question.strip(),
        }
    ))
    
    if not answer or not answer.strip():
        return docs
    
    # Chunk type 2: Answer sections (split on double newlines/headers)
    sections = re.split(r'\n\s*\n|\n(?=[A-Z][^.]{5,50}\s*\n)', answer.strip())
    sections = [s.strip() for s in sections if s.strip() and len(s.strip()) > 30]
    
    if not sections:
        sections = [answer.strip()]
    
    for i, section in enumerate(sections):
        word_count = len(section.split())
        
        if word_count <= 400:
            docs.append(Document(
                page_content=section,
                metadata={
                    "source": "medquad",
                    "chunk_type": "answer_section",
                    "focus_area": focus_area,
                    "parent_question": question.strip(),
                    "section_index": i,
                    "parent_answer_preview": answer[:300],
                }
            ))
        else:
            # Sub-split oversized sections at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', section)
            current_chunk = []
            current_words = 0
            
            for sentence in sentences:
                s_words = len(sentence.split())
                if current_words + s_words > 350 and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    docs.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "source": "medquad",
                            "chunk_type": "answer_section",
                            "focus_area": focus_area,
                            "parent_question": question.strip(),
                            "section_index": i,
                            "parent_answer_preview": answer[:300],
                        }
                    ))
                    current_chunk = [current_chunk[-1], sentence] if current_chunk else [sentence]
                    current_words = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_words += s_words
            
            if current_chunk:
                docs.append(Document(
                    page_content=' '.join(current_chunk),
                    metadata={
                        "source": "medquad",
                        "chunk_type": "answer_section",
                        "focus_area": focus_area,
                        "parent_question": question.strip(),
                        "section_index": i,
                        "parent_answer_preview": answer[:300],
                    }
                ))
    return docs

def build_medquad_index():
    print("Building MedQuAD (Clinical Facts) Knowledge Base...")
    docs = []
    
    with open(DATA_DIR / "medquad.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Use structure-aware chunking
            chunks = chunk_medquad_entry(
                question=data.get("question", ""),
                answer=data.get("answer", ""),
                focus_area=data.get("focus_area", "")
            )
            docs.extend(chunks)
            
    print(f"Loaded {len(docs)} MedQuAD chunks. Embedding (this may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(FAISS_DIR / "medquad"))
    print(f"Saved MedQuAD index to {FAISS_DIR / 'medquad'}")

def build_conversation_index():
    print("Building Conversation Aid Knowledge Base from MedDialog...")
    docs = []
    skipped = 0
    total = 0

    with open(DATA_DIR / "meddialog_en_train.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            utterances = data.get("utterances", [])
            if len(utterances) < 2:
                skipped += 1
                continue

            # utterances are plain strings already prefixed with
            # "Patient:" / "Doctor:" -- just join them, don't treat as dicts.
            content = "\n".join(u.strip() for u in utterances if u and u.strip())

            if not content:
                skipped += 1
                continue

            description = data.get("description", "").strip()

            docs.append(Document(
                page_content=content,
                metadata={
                    "source": "meddialog",
                    "conv_id": str(i),
                    "description": description,
                    "num_turns": len(utterances),
                }
            ))

    print(f"Processed {total} lines: {len(docs)} loaded, {skipped} skipped.")

    if not docs:
        print("ERROR: No documents were loaded. Check the file structure.")
        return

    print(f"Loaded {len(docs)} MedDialog conversations. Embedding...")
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(FAISS_DIR / "conversations"))
    print(f"Saved Conversation index to {FAISS_DIR / 'conversations'}")

if __name__ == "__main__":
    build_conversation_index()
    build_medquad_index()
    print("All Knowledge Bases built successfully!")
