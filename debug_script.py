from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

p = Path(".").resolve()
vs = Chroma(persist_directory=str(p / "db"), embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

# 1) Cheap lexical probe (no embedding): fetch a bunch and scan
# If this finds nothing, Fireball likely isn't in this collection.
all_docs = vs.get(include=["documents", "metadatas"], limit=20000)
docs = all_docs.get("documents", []) or []
metas = all_docs.get("metadatas", []) or []

hits = []
for i, text in enumerate(docs):
    if text and "fireball" in text.lower():
        hits.append((i, text[:120], metas[i] if i < len(metas) else None))
        if len(hits) >= 10:
            break

print("Lexical hits containing 'fireball':", len(hits))
for h in hits:
    print(" -", h[1], "| meta:", h[2])


res = vs.get(where={"source_file": "Fireball.md"}, include=["documents", "metadatas"], limit=3)
print("Fireball.md matches:", len(res.get("documents") or []))
if res.get("documents"):
    print(res["metadatas"][0])
    print(res["documents"][0][:200])    