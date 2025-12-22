# extractor.py
import os
import base64
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import camelot
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import mimetypes
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv
load_dotenv()

# ---------- CONFIG ----------
PDF_PATH = "sample.pdf"  # Replace with your PDF path
IMAGE_FOLDER = "pdf_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Configurable system paths (set via .env)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH = os.getenv("POPPLER_PATH")  # e.g. C:\path\to\poppler\bin

# Tesseract - make configurable via env
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------- ChromaDB + Embeddings ----------
# Create Chroma client in a compatibility-safe way.
# Newer chromadb versions accept `persist_directory` directly, older versions use `Settings`.
try:
    # chromadb>=1.0 exposes PersistentClient on the top-level module
    if hasattr(chromadb, "PersistentClient"):
        chroma_client = chromadb.PersistentClient(path="chroma_db")
    else:
        raise AttributeError("PersistentClient not available")
except Exception:
    try:
        # Try direct constructor with persist_directory (pre-1.0 API)
        chroma_client = chromadb.Client(persist_directory="chroma_db")
    except TypeError:
        try:
            from chromadb.config import Settings

            chroma_client = chromadb.Client(Settings(persist_directory="chroma_db", chroma_db_impl="duckdb+parquet"))
        except Exception:
            # Fallback to in-memory client; warn user that persistence may be disabled.
            print("Warning: chromadb Client persist configuration not available; using default in-memory client.")
            chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection("pdf_docs")
# Multilingual encoder so cross-language queries (English/Tamil/Tanglish) map to the same space
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def embed_fn(text: str):
    # ensure a plain Python list[float] is returned for Chroma
    vec = embedding_model.encode([text])[0]
    return vec.tolist()

# ---------- MongoDB ----------
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["mytestdb"]
tables_collection = mongo_db["pdf_tables"]

# ---------- OpenRouter / Qwen config ----------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
# Use a broadly available OpenRouter model for text QA
QWEN_MODEL = "meta-llama/llama-3.1-8b-instruct"

# build headers only when key exists; avoid raising at import time
HEADERS = {"Content-Type": "application/json"}
if OPENROUTER_API_KEY:
    HEADERS["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"

# ---------- STEP 1: Extract text ----------
def extract_text(
    pdf_path,
    chunk_size=500,
    use_ocr: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Extract text from PDF and split into chunks of chunk_size words.
    Returns a list of (text_chunk, page_number) tuples.
    """
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages) or 1
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                words = text.split()
                for j in range(0, len(words), chunk_size):
                    chunk = " ".join(words[j:j + chunk_size])
                    text_chunks.append((chunk, i))
            elif use_ocr:
                # fallback: OCR the page to text
                try:
                    if POPPLER_PATH:
                        images = convert_from_path(pdf_path, first_page=i, last_page=i, dpi=200, poppler_path=POPPLER_PATH)
                    else:
                        images = convert_from_path(pdf_path, first_page=i, last_page=i, dpi=200)
                except Exception:
                    images = []
                if images:
                    ocr_text = pytesseract.image_to_string(images[0], lang="eng")
                    words = ocr_text.split()
                    for j in range(0, len(words), chunk_size):
                        chunk = " ".join(words[j:j + chunk_size])
                        text_chunks.append((chunk, i))
            if progress_callback:
                try:
                    progress_callback(i, total_pages)
                except Exception:
                    pass
    return text_chunks

# ---------- STEP 2: Extract images ----------
def extract_images_from_pdf(
    pdf_path,
    output_folder=IMAGE_FOLDER,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Extract embedded images from the PDF and save them to disk.
    Return list of tuples: (image_path, page_number)
    """
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    total_pages = doc.page_count or 1
    for page_number, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            image_name = os.path.join(output_folder, f"page{page_number}_img{img_index}.{image_ext}")
            try:
                with open(image_name, "wb") as f:
                    f.write(image_bytes)
                image_paths.append((image_name, page_number))
            except Exception:
                continue

        # Also export full-page raster image (useful for charts that are not embedded images)
        try:
            pix = page.get_pixmap(dpi=200)
            page_image_path = os.path.join(output_folder, f"page{page_number}_full.png")
            pix.save(page_image_path)
            image_paths.append((page_image_path, page_number))
        except Exception:
            pass

        if progress_callback:
            try:
                progress_callback(page_number, total_pages)
            except Exception:
                pass

    return image_paths

# ---------- Utility: encode image to data URI ----------
def image_to_data_uri(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

# ---------- STEP 3: Generate captions using Qwen via OpenRouter ----------
def generate_caption_openrouter(image_path):
    """
    Ask Qwen to generate a short caption for the image.
    Returns caption string.
    """
    # If API key not configured, fallback to local OCR caption
    if not OPENROUTER_API_KEY:
        try:
            text = pytesseract.image_to_string(Image.open(image_path))
            return text.strip()[:500]
        except Exception:
            return ""

    # Avoid sending extremely large multipart payloads where possible — embed small images or provide URLs.
    data_uri = image_to_data_uri(image_path)
    system = "You are an image captioning assistant. Provide a short descriptive caption (1-2 sentences) of the image focusing on charts, tables, text, or important visual elements."
    # Use a single string message with the data URI to improve compatibility
    user_text = (
        "Caption the following image in 1-2 sentences. If it's a chart or table, mention axis labels or table headers if visible.\n"
        f"Image data (data URI):\n{data_uri}\n"
    )
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.0,
        "max_tokens": 200
    }

    try:
        resp = requests.post(OPENROUTER_ENDPOINT, headers=HEADERS, json=payload, timeout=60)
    except Exception as e:
        # network/timeout — fallback to OCR
        try:
            return pytesseract.image_to_string(Image.open(image_path)).strip()
        except Exception:
            return ""

    if resp.status_code != 200:
        # fallback: use OCR if API failed
        try:
            return pytesseract.image_to_string(Image.open(image_path)).strip()
        except Exception:
            raise RuntimeError(f"OpenRouter caption error: {resp.status_code} {resp.text}")

    try:
        j = resp.json()
    except Exception:
        try:
            return pytesseract.image_to_string(Image.open(image_path)).strip()
        except Exception:
            return ""

    # defensive parsing
    try:
        # many chat APIs return choices -> message -> content
        return j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return ""

# ---------- STEP 4: Store in ChromaDB (text chunks + image captions metadata) ----------
def store_in_chromadb(
    text_chunks,
    image_infos,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
):
    # Store text chunks
    text_count = 0
    image_count = 0
    total_items = len(text_chunks) + len(image_infos)
    processed = 0
    for idx, (text, page) in enumerate(text_chunks):
        try:
            snippet = text[:400]
            collection.add(
                ids=[f"text_{page}_{idx}"],
                documents=[text],
                metadatas=[{"page": page, "type": "text", "snippet": snippet}],
                embeddings=[embed_fn(text)]
            )
            text_count += 1
            processed += 1
            if progress_callback and total_items:
                try:
                    progress_callback(processed, total_items, "text")
                except Exception:
                    pass
        except Exception as e:
            print(f"Chroma add text error for page {page} idx {idx}:", e)

    # Store image captions along with image path in metadata
    for idx, (caption, page, image_path) in enumerate(image_infos):
        doc_id = f"image_{page}_{idx}"
        try:
            collection.add(
                ids=[doc_id],
                documents=[caption],
                metadatas=[{"page": page, "type": "image", "image_path": image_path}],
                embeddings=[embed_fn(caption if caption else "")]
            )
            image_count += 1
            processed += 1
            if progress_callback and total_items:
                try:
                    progress_callback(processed, total_items, "image")
                except Exception:
                    pass
        except Exception as e:
            print(f"Chroma add image error for {image_path}:", e)
    # Try to persist Chroma client if supported
    try:
        if hasattr(chroma_client, "persist"):
            chroma_client.persist()
    except Exception as e:
        print("Chroma persist() failed:", e)
    return {"text_count": text_count, "image_count": image_count}

# ---------- STEP 5: Extract tables ----------
def extract_tables_from_pdf(pdf_path):
    """
    Extract tables from a PDF using Camelot.
    Returns a list of dicts with table data and page numbers.
    """
    tables_data = []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")  # or flavor="lattice"
    except Exception as e:
        print("Camelot read_pdf failed:", e)
        tables = []
    for i, table in enumerate(tables, start=1):
        tables_data.append({
            "table_number": i,
            "page": table.page,
            "data": table.df.to_dict(orient="records"),
            "pdf_name": pdf_path
        })
    return tables_data

def store_tables_in_mongodb(tables_data):
    """
    Store extracted tables into MongoDB.
    """
    if tables_data:
        tables_collection.insert_many(tables_data)

# ---------- MAIN (for standalone testing) ----------
if __name__ == "__main__":
    print("Extracting text...")
    text_chunks = extract_text(PDF_PATH)
    print(f"Text chunks extracted: {len(text_chunks)}")

    print("Extracting images...")
    images = extract_images_from_pdf(PDF_PATH)
    print(f"Images extracted: {len(images)}")
    image_infos = []
    for img_path, page in images:
        try:
            caption = generate_caption_openrouter(img_path)
        except Exception as e:
            caption = ""  # fallback to OCR or empty
            try:
                caption = pytesseract.image_to_string(Image.open(img_path))
            except Exception:
                pass
        image_infos.append((caption, page, img_path))
        print(f"Page {page} caption: {caption}")

    print("Storing text and captions in ChromaDB...")
    res = store_in_chromadb(text_chunks, image_infos)
    print("Chroma insertion result:", res)

    print("Extracting tables...")
    tables_data = extract_tables_from_pdf(PDF_PATH)
    print(f"Tables extracted: {len(tables_data)}")
    try:
        store_tables_in_mongodb(tables_data)
        print("Tables stored in MongoDB (if Mongo is running).")
    except Exception as e:
        print("Failed to store tables in MongoDB:", e)
    print("Done.")
