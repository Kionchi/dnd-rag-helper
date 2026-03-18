from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from .summarizer import SessionNote


def append_note_to_markdown(note: SessionNote, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"- [{note.start_sec:.0f}s–{note.end_sec:.0f}s] {note.text}\n")


def get_session_notes_vectorstore(
    db_path: Path,
    collection_name: str = "session_notes",
    embedding_model: str = "nomic-embed-text",
) -> Chroma:
    embeddings = OllamaEmbeddings(model=embedding_model)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(db_path),
        embedding_function=embeddings,
    )


def upsert_note_into_chroma(
    note: SessionNote,
    system_name: str,
    session_id: str,
    db_path: Path,
    collection_name: str = "session_notes",
    embedding_model: str = "nomic-embed-text",
) -> None:
    vectordb = get_session_notes_vectorstore(
        db_path=db_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    doc = Document(
        page_content=note.text,
        metadata={
            "system_name": system_name,
            "session_id": session_id,
            "start_sec": note.start_sec,
            "end_sec": note.end_sec,
            "source_file": "session_current.md",
        },
    )
    vectordb.add_documents([doc])