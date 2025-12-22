"""Inspect ChromaDB collection contents for debugging.
Run this to see whether documents exist in the `pdf_docs` collection.
"""
from extractor import collection

def main():
    print("Inspecting Chroma collection 'pdf_docs'...")
    try:
        total = collection.count()
        print(f"collection.count() -> {total}")
        # Request explicit inclusion of documents/metadatas for newer Chroma builds
        data = collection.get(include=["documents", "metadatas"], limit=5)
        print('get() returned keys:', list(data.keys()))
        docs = data.get('documents')
        if docs:
            sample = docs if isinstance(docs, list) else [docs]
            print(f"Sample documents (up to {len(sample)} shown):")
            for idx, doc in enumerate(sample):
                if idx >= 5:
                    break
                print(f"  #{idx + 1}: {str(doc)[:120]}")
        else:
            print('No documents returned by get(). Use `limit` argument for more results.')
    except Exception as e:
        print('collection.get() failed:', e)
        try:
            # Try count() if available
            cnt = collection.count()
            print('collection.count() ->', cnt)
        except Exception as e2:
            print('collection.count() failed:', e2)

if __name__ == '__main__':
    main()
