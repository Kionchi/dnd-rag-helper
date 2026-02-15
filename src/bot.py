"""
D&D SRD RAG bot: answers questions using the ingested Chroma vector store and Ollama.
Run from project root: py src/bot.py
"""
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "phi4:14b"


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
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 30},
)

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

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
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