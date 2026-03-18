"""
D&D SRD + Session Notes RAG bot.

- SRD rules are grounded only in the SRD Chroma collection (default: "langchain")
- Session notes are used for campaign/table-specific questions
"""
import re
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "phi4:14b"

STOPWORDS = frozenset(
    {
        "a", "an", "the", "how", "does", "do", "what", "when", "where", "tell", "me", "about",
        "more", "spell", "work", "is", "are", "can", "of", "in", "to", "and", "for",
        "please"
    }
)


def _extract_entity(question: str) -> str:
    """Extract a likely entity name (e.g., Fireball) from the question for SRD retrieval bias."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", question)
    candidates = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    if not candidates:
        return question.strip() or "unknown"
    return max(candidates, key=len).strip().title()


def _retrieve_and_rerank(vectorstore, question: str, entity: str, k_retrieve: int = 30, k_keep: int = 12):
    """Metadata-first retrieval for SRD (tries h3=entity, then similarity fallback)."""
    # 1) Deterministic lookup by Markdown header metadata (h3)
    res = vectorstore.get(where={"h3": entity}, include=["documents", "metadatas"], limit=8)
    docs_meta = []
    for text, meta in zip(res.get("documents") or [], res.get("metadatas") or []):
        if text:
            docs_meta.append(Document(page_content=text, metadata=meta or {}))

    # 2) Vector fallback
    docs_vec = vectorstore.similarity_search(question, k=k_retrieve)

    # 3) Merge + re-rank by prefix
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


def load_srd_vector_store():
    """
    Load the SRD rules vector store.
    IMPORTANT: during ingestion you used Chroma.from_documents without specifying collection_name,
    so the default collection name is typically "langchain".
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )


def load_session_notes_store():
    """Load the session notes vector store (collection_name=session_notes)."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name="session_notes",
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )


def main():
    print("Loading SRD vector store...")
    srd_store = load_srd_vector_store()

    print("Loading session notes vector store...")
    session_notes_store = load_session_notes_store()

    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a professional Dungeon Master assistant.

You will receive two context sections:
1) SESSION NOTES: what happened during the actual game (campaign/table facts).
2) SRD RULES CONTEXT: official D&D 5e SRD rule text.

Rules for answering:
- If the user asks about rules, mechanics, spells, actions, or DCs: use ONLY SRD RULES CONTEXT.
- If the user asks about what happened in the campaign (events, negotiations, what we did): use ONLY SESSION NOTES.
- If the requested rule is not present in SRD RULES CONTEXT: do NOT invent rules.
  Instead, suggest an appropriate Ability Check (and briefly state which ability + a reasonable DC guess).
- Be grounded and concise. Fantasy-scholar tone.

SESSION NOTES:
{session_context}

SRD RULES CONTEXT:
{srd_context}
""",
            ),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def get_context(question: str):
        entity = _extract_entity(question)

        # SRD context
        srd_docs = _retrieve_and_rerank(srd_store, question, entity)
        srd_context = format_docs(srd_docs)

        # Session notes context
        notes_docs = session_notes_store.similarity_search(question, k=5)
        session_context = format_docs(notes_docs)

        return {
            "session_context": session_context,
            "srd_context": srd_context,
        }

    chain = (
        {"session_context": lambda q: get_context(q)["session_context"],
         "srd_context": lambda q: get_context(q)["srd_context"],
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(
        "Ready. Ask about D&D rules or about what happened last session.\n"
        "Examples: 'How much damage does Fireball do?' or 'What did we do last session?'\n"
        "Type 'quit' or 'exit' to stop.\n"
    )

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