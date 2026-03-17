import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

HEADER_KEYS = ("h1", "h2", "h3")

def load_env() -> None:
    """Load environment variables from .env (optional, for future use)."""
    load_dotenv()


def load_markdown_documents(data_dir: str) -> List:
    """Load all .md files from data_dir as LangChain documents."""
    print(f"Loading Markdown files from directory: {data_dir} ...")

    loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} Markdown documents.")
    return docs


def split_markdown_to_chunks(docs: List) -> List:
    """
    Split Markdown documents:
    1. Logically by headers (MarkdownHeaderTextSplitter),
    2. Then into smaller chunks (RecursiveCharacterTextSplitter).
    """
    print("Splitting documents by Markdown headers...")

    # Header hierarchy we want to track and keep in metadata
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    header_docs = []
    for idx, doc in enumerate(docs, start=1):
        print(f"  Processing document {idx}/{len(docs)}: {doc.metadata.get('source', '')}")
        split_docs = header_splitter.split_text(doc.page_content)

        # Add metadata: source_file and keep header hierarchy in metadata
        source_path = doc.metadata.get("source", "")
        source_file = os.path.basename(source_path)
        for d in split_docs:
            md = d.metadata or {}
            md["source_file"] = source_file
            # MarkdownHeaderTextSplitter adds h1 / h2 / h3 keys as metadata
            d.metadata = md

        header_docs.extend(split_docs)

    print(f"After header-based splitting: {len(header_docs)} logical sections.")

    print("Splitting sections into embedding-friendly chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    final_chunks = text_splitter.split_documents(header_docs)
    print(f"After final splitting: {len(final_chunks)} chunks.")
    final_chunks = _inject_header_context_into_chunks(final_chunks)
    return final_chunks


def _inject_header_context_into_chunks(chunks: List[Document]) -> List[Document]:
    """Prepend header hierarchy (h1 > h2 > h3) to each chunk's page_content so every chunk
    is self-descriptive and retrievable by entity name (e.g. 'Fireball' + 'damage')."""
    result = []
    for doc in chunks:
        parts = [doc.metadata.get(k) for k in HEADER_KEYS if doc.metadata.get(k)]
        parts = [str(p).strip() for p in parts if p]
        if parts:
            prefix = " > ".join(parts) + ": "
            result.append(
                Document(page_content=prefix + doc.page_content, metadata=dict(doc.metadata))
            )
        else:
            result.append(Document(page_content=doc.page_content, metadata=dict(doc.metadata)))
    return result

def build_vector_store(chunks: List, db_dir: str, embedding_model: str = "nomic-embed-text") -> Chroma:
    """Build a local Chroma vector store in db_dir using Ollama embeddings."""
    print(f"Creating Chroma vector store in directory: {db_dir} ...")
    print(f"Using Ollama embedding model: {embedding_model}")

    persist_directory = Path(db_dir)
    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = OllamaEmbeddings(model=embedding_model)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )

    print("Persisting vector store to disk...")
    vectordb.persist()
    print("Chroma vector store saved.")
    return vectordb


def run_ingestion(
    data_dir: str = "data/DND5E.SRD.Wiki-master",
    db_dir: str = "db",
    embedding_model: str = "nomic-embed-text",
) -> None:
    """Main entry point for the Markdown -> Chroma ingestion pipeline (Ollama local)."""
    print("=== Starting ingestion process (Ollama) ===")
    load_env()

    docs = load_markdown_documents(data_dir)
    chunks = split_markdown_to_chunks(docs)
    build_vector_store(chunks, db_dir, embedding_model=embedding_model)


if __name__ == "__main__":
    run_ingestion()