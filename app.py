# app.py
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import requests
import pdfplumber
from PIL import Image
import pytesseract
from extractor import (
    extract_text,
    extract_images_from_pdf,
    generate_caption_openrouter,
    store_in_chromadb,
    extract_tables_from_pdf,
    store_tables_in_mongodb,
    image_to_data_uri,
)
from sentence_transformers import SentenceTransformer
import chromadb
from pymongo import MongoClient
import mimetypes
import base64

# ---------- SETUP ----------
st.set_page_config(page_title="PDF Chatbot (Qwen Multimodal)", layout="wide")
st.title("📄 PDF Chatbot (Qwen2.5-VL Multimodal)")

# ---------- Initialize ChromaDB ----------
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("pdf_docs")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda x: embedding_model.encode(x).tolist()

# ---------- MongoDB for tables ----------
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["mytestdb"]
tables_collection = mongo_db["pdf_tables"]

# ---------- OpenRouter / Qwen config ----------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("Set OPENROUTER_API_KEY environment variable")
    st.stop()

OPENROUTER_ENDPOINT = "https://api.openrouter.ai/v1/chat/completions"
QWEN_MODEL = "qwen/qwen2.5-vl-7b-instruct"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# ---------- Sidebar settings ----------
st.sidebar.header("Settings")
chunk_size = st.sidebar.slider("PDF Text Chunk Size (words)", min_value=100, max_value=1500, value=800, step=100)
fast_mode = st.sidebar.checkbox(
    "Fast mode (text only, skip images/tables)",
    value=True,
    help="Recommended for large PDFs (200+ pages). Uses text only and skips OCR on empty pages.",
)

# ---------- Tesseract OCR ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- Helper functions ----------
def openrouter_call_with_image(query, context_texts, image_path=None, max_tokens=512):
    """
    Send multimodal request to OpenRouter Qwen: includes context text and optional image (data URI).
    context_texts: string with retrieved text/table info
    image_path: filesystem path to image to include
    """
    user_content = []
    # add context as system/user text
    if context_texts:
        # split or keep as a single text block
        user_content.append({"type": "text", "text": f"Context:\n{context_texts}"})
    # add main question
    user_content.append({"type": "text", "text": f"Question: {query}"})
    # add image if present
    if image_path:
        data_uri = image_to_data_uri(image_path)
        user_content.append({"type": "input_image", "image_url": data_uri})

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in answering questions about documents, charts, tables and images. Use the provided context and images to answer precisely.",
            },
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(OPENROUTER_ENDPOINT, headers=HEADERS, json=payload, timeout=60)
    except Exception as e:
        # Network / DNS / connectivity issue – fall back to a simple context echo
        fallback = "⚠️ Unable to reach OpenRouter (network error).\n\n" "Here is the most relevant context I found:\n" + (context_texts[:2000] if context_texts else "No context available.")
        return fallback

    if resp.status_code == 200:
        try:
            j = resp.json()
            return j["choices"][0]["message"]["content"].strip()
        except Exception:
            return "⚠️ OpenRouter returned an invalid response."
    else:
        return f"⚠️ OpenRouter API Error: {resp.status_code} {resp.text}"

def fetch_table_context(query):
    table_text = ""
    docs = tables_collection.find()
    for doc in docs:
        for row in doc["data"]:
            row_text = ", ".join([f"{k}: {v}" for k, v in row.items()])
            table_text += f"Table {doc['table_number']} ({doc['pdf_name']}): {row_text}\n"
    return table_text

def fetch_text_context(query, n_results=5):
    query_embedding = embed_fn(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    context = "\n".join([f"Page {m['page']} ({m['type']}): {d}" for m, d in zip(metadatas, docs)])
    return context, metadatas, docs

# ---------- Upload PDF ----------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded successfully!")

    if st.button("Process PDF"):
        with st.spinner("⏳ Processing PDF..."):
            # Text (always extracted)
            # In fast_mode we also skip OCR on empty pages for speed.
            text_chunks = extract_text(
                pdf_path,
                chunk_size=chunk_size,
                use_ocr=not fast_mode,
            )

            image_infos = []
            tables_data = []

            if not fast_mode:
                # Images + captions (slower)
                images = extract_images_from_pdf(pdf_path)
                for img_path, page in images:
                    try:
                        caption = generate_caption_openrouter(img_path)
                    except Exception:
                        caption = ""
                        try:
                            caption = pytesseract.image_to_string(Image.open(img_path))
                        except Exception:
                            caption = ""
                    image_infos.append((caption, page, img_path))

                # Tables (slower, uses Camelot + MongoDB)
                tables_data = extract_tables_from_pdf(pdf_path)
                if tables_data:
                    store_tables_in_mongodb(tables_data)

            # Store in Chroma (text always, images only when fast_mode is off)
            store_in_chromadb(text_chunks, image_infos)

        if fast_mode:
            st.success("✅ PDF processed in fast text-only mode (no images/tables)")
        else:
            st.success("✅ PDF processed: text/images → ChromaDB, tables → MongoDB")

# ---------- Ask Questions ----------
st.header("Ask a question about your PDF")
query = st.text_input("Type your question here:")

if query:
    query_lower = query.lower()
    # Determine if this is a multimodal question
    is_image_related = any(k in query_lower for k in ["image", "figure", "chart", "graph", "figure", "plot", "diagram", "table"])
    # Get text context (Chroma)
    context_text, metadatas, docs = fetch_text_context(query)

    if is_image_related:
        # retrieve best matching image doc from Chroma
        # we search for top image hits specifically
        query_embedding = embed_fn(query)
        results = collection.query(query_embeddings=[query_embedding], where={"$or":[{"type":"image"},{"type":"text"}]}, n_results=5)
        # results structure may vary; defensively search metadatas for image
        image_candidate = None
        try:
            metads = results["metadatas"][0]
            docs_res = results["documents"][0]
            for m, d in zip(metads, docs_res):
                if m.get("type") == "image":
                    image_candidate = m.get("image_path")  # stored earlier
                    break
        except Exception:
            image_candidate = None

        # fallback: use first image in folder if none found
        if not image_candidate:
            # try any image doc in collection metadata
            all_docs = collection.get(include=["metadatas", "documents"])
            for idx, meta in enumerate(all_docs["metadatas"]):
                if meta.get("type") == "image":
                    image_candidate = meta.get("image_path")
                    break

        # fetch table context also if question mentions table
        table_ctx = fetch_table_context(query) if "table" in query_lower else ""
        combined_context = context_text + "\n" + table_ctx

        answer = openrouter_call_with_image(query, combined_context, image_path=image_candidate)
    else:
        # text-only: just send context + query (no image)
        combined_context = context_text
        answer = openrouter_call_with_image(query, combined_context, image_path=None)

    st.subheader("📌 Answer:")
    st.markdown(answer)
