"""
D&D SRD RAG bot: answers questions using the ingested Chroma vector store and Ollama.
Run from project root: py src/bot.py
"""
import re
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "phi4:14b"

STOPWORDS = frozenset(
    {"a", "an", "the", "how", "does", "do", "what", "when", "where", "tell", "me", "about", "more", "spell", "work", "is", "are", "can", "of", "in", "to", "and", "for"}
)


def _extract_entity(question: str) -> str:
    """Extract the main entity (e.g. spell/monster/rule name) from the question for re-ranking."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", question)
    candidates = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    if not candidates:
        return question.strip() or "unknown"
    entity = max(candidates, key=len)
    return entity.strip().title()


def _retrieve_and_rerank(vectorstore, question: str, entity: str, k_retrieve: int = 30, k_keep: int = 12):
    """Metadata-first retrieval + vector fallback.

    1) Try exact entity match via Markdown header metadata (h3).
    2) Fallback to vector similarity search.
    3) Optional: merge + re-rank to keep '{entity}: ' chunks first.
    """
    # 1) Deterministic lookup by header metadata (works because ingestion stores h3='Fireball')
    res = vectorstore.get(where={"h3": entity}, include=["documents", "metadatas"], limit=8)
    docs_meta = []
    for text, meta in zip(res.get("documents") or [], res.get("metadatas") or []):
        if text:
            docs_meta.append(Document(page_content=text, metadata=meta or {}))

    # 2) Vector fallback (bigger pool)
    docs_vec = vectorstore.similarity_search(question, k=k_retrieve)

    # 3) Merge (metadata first), dedupe, then re-rank so '{entity}: ' chunks come first
    combined = list(docs_meta)
    seen = {d.page_content[:200] for d in combined}
    for d in docs_vec:
        key = d.page_content[:200]
        if key not in seen:
            seen.add(key)
            combined.append(d)

    if not combined:
        return []

    prefix = f"{entity}: "

    def rank_key(doc):
        content = doc.page_content.strip()
        if content.startswith(prefix):
            return (0, content)
        if prefix.lower() in content[:80].lower():
            return (1, content)
        return (2, content)

    combined_sorted = sorted(combined, key=rank_key)

    # final dedupe + trim
    seen2 = set()
    final_docs = []
    for d in combined_sorted:
        key = d.page_content[:200]
        if key not in seen2:
            seen2.add(key)
            final_docs.append(d)

    return final_docs[:k_keep]

def load_vector_store():
    """Load Chroma vector store from db/ using the same embeddings as ingestion."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )


def main():
    print("Loading D&D SRD vector store...")
    vectorstore = load_vector_store()

    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about D&D 5e rules using the SRD (System Reference Document).
Use ONLY the following context from the SRD. If the answer is not in the context, say so and do not invent rules.

Context from SRD:
{context}"""),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def get_context(question: str):
        entity = _extract_entity(question)
        docs = _retrieve_and_rerank(vectorstore, question, entity)
        return format_docs(docs)

    chain = (
        {"context": get_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Ready. Ask about D&D 5e rules (e.g. 'How does Fireball work?'). Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        print("Bot:", end=" ", flush=True)
        try:
            for chunk in chain.stream(question):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"\nError: {e}")
        print()


if __name__ == "__main__":
    main()