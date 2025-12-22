"""
Simple retrieval test for the `pdf_docs` Chroma collection.
Run this after you've run the extractor (so the collection has content).
"""
from extractor import collection, embedding_model
import sys

def encode_query(text: str):
    # use the same encoder instance as extractor
    vec = embedding_model.encode([text])[0]
    return vec.tolist()

def main():
    query_text = "summary of the main chart" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print("Querying for:\n", query_text)
    vec = encode_query(query_text)

    try:
        results = collection.query(query_embeddings=[vec], n_results=5)
    except Exception as e:
        print("Chroma query failed:", e)
        return

    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    dists = results.get("distances", [])

    # results may be nested per-query
    if isinstance(docs, list) and len(docs) and isinstance(docs[0], list):
        docs = docs[0]
    if isinstance(metas, list) and len(metas) and isinstance(metas[0], list):
        metas = metas[0]
    if isinstance(dists, list) and len(dists) and isinstance(dists[0], list):
        dists = dists[0]

    if not docs:
        print("No documents retrieved. Make sure extractor ran and ingested data into Chroma.")
        return

    print(f"Retrieved {len(docs)} items:\n")
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None
        snippet = meta.get("snippet") or (doc[:300] if isinstance(doc, str) else "")
        print(f"Rank {i+1} — distance={dist} — page={meta.get('page')} — type={meta.get('type')}")
        print("Snippet:", snippet)
        print("---")

if __name__ == '__main__':
    main()
