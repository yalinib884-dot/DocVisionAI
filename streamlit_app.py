"""Streamlit UI for the PDF chatbot pipeline."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import fitz
import streamlit as st
from dotenv import load_dotenv

# Load .env early so downstream modules pick up credentials
load_dotenv()

from extractor import (  # noqa: E402
    extract_text,
    extract_images_from_pdf,
    generate_caption_openrouter,
    store_in_chromadb,
    extract_tables_from_pdf,
    store_tables_in_mongodb,
    pytesseract,
    Image,
    collection as chroma_collection,
)
from model import answer_query, call_llm_openrouter  # noqa: E402

st.set_page_config(page_title="PDF Chatbot", layout="wide")

HISTORY_FILE = Path("chat_history.json")
DEFAULT_SESSION_TITLE = "New chat"


def create_chat_session(title: str | None = None) -> dict:
    return {
        "id": str(uuid4()),
        "title": title or DEFAULT_SESSION_TITLE,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "messages": [],
    }


def derive_session_title(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            first_line = message["content"].strip().splitlines()[0]
            if not first_line:
                continue
            trimmed = first_line[:60]
            return trimmed + ("…" if len(first_line) > 60 else "")
    return DEFAULT_SESSION_TITLE


def _sanitize_session(session: dict) -> dict:
    if not isinstance(session, dict):
        return create_chat_session()
    session_id = str(session.get("id") or uuid4())
    messages = []
    for message in session.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    title = session.get("title")
    if not isinstance(title, str) or not title.strip():
        title = derive_session_title(messages)
    created_at = session.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        created_at = datetime.utcnow().isoformat(timespec="seconds")
    return {
        "id": session_id,
        "title": title,
        "created_at": created_at,
        "messages": messages,
    }


def _migrate_legacy_history(entries: list[dict]) -> dict:
    session = create_chat_session("Previous chat")
    messages = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if isinstance(question, str) and question.strip():
            messages.append({"role": "user", "content": question})
        if isinstance(answer, str) and answer.strip():
            messages.append({"role": "assistant", "content": answer})
    session["messages"] = messages
    session["title"] = derive_session_title(messages)
    return session


def load_chat_sessions() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(raw, dict) and isinstance(raw.get("sessions"), list):
        return [_sanitize_session(session) for session in raw["sessions"]]

    if isinstance(raw, list):
        return [_migrate_legacy_history(raw)]

    return []


def save_chat_sessions(sessions: list[dict]) -> None:
    try:
        payload = {"sessions": [_sanitize_session(session) for session in sessions]}
        HISTORY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        # Persisting history is best-effort; ignore disk errors quietly
        pass


def get_current_session() -> dict:
    sessions = st.session_state.get("chat_sessions", [])
    current_id = st.session_state.get("current_session_id")
    for session in sessions:
        if session.get("id") == current_id:
            return session

    new_session = create_chat_session()
    sessions.append(new_session)
    st.session_state["chat_sessions"] = sessions
    st.session_state["current_session_id"] = new_session["id"]
    save_chat_sessions(st.session_state["chat_sessions"])
    return new_session


def start_new_chat() -> None:
    new_session = create_chat_session()
    st.session_state.setdefault("chat_sessions", []).append(new_session)
    st.session_state["current_session_id"] = new_session["id"]
    save_chat_sessions(st.session_state["chat_sessions"])


def switch_chat(session_id: str) -> None:
    if session_id == st.session_state.get("current_session_id"):
        return
    for session in st.session_state.get("chat_sessions", []):
        if session.get("id") == session_id:
            st.session_state["current_session_id"] = session_id
            save_chat_sessions(st.session_state["chat_sessions"])
            break


st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; }
    .block-container { padding-top: 2rem; }
    [data-testid="stSidebar"] { background-color: #0b1220; color: #e2e8f0; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #c7d2fe; background-color: rgba(37, 99, 235, 0.15); border-radius: 8px; margin-right: 0.5rem; }
    .stTabs [aria-selected="true"] { background-color: #2563eb !important; color: #fff !important; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35); }
    .status-card { background: rgba(30, 64, 175, 0.35); border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.65rem; }
    .status-card span { display: block; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #bfdbfe; }
    .status-card strong { font-size: 1.35rem; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 PDF Chatbot — Retrieval + LLM")

if "chat_sessions" not in st.session_state:
    sessions = load_chat_sessions()
    if not sessions:
        sessions = [create_chat_session()]
        save_chat_sessions(sessions)
    st.session_state["chat_sessions"] = sessions
    save_chat_sessions(st.session_state["chat_sessions"])

if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = st.session_state["chat_sessions"][-1]["id"]

if "pdf_meta" not in st.session_state:
    st.session_state["pdf_meta"] = {}


def _status_card(label: str, value: str) -> str:
    return f'<div class="status-card"><span>{label}</span><strong>{value}</strong></div>'

st.sidebar.header("Ingestion Settings")
chunk_size = st.sidebar.slider(
    "Text chunk size (words)", min_value=100, max_value=1000, value=500, step=50
)
fast_mode = st.sidebar.checkbox(
    "Fast mode (text only, skip images/tables)",
    value=True,
    help="When enabled, only text is stored in Chroma (no images/tables) for faster processing.",
)
handwriting_mode = st.sidebar.checkbox(
    "Handwritten PDF (use handwriting OCR)",
    value=False,
    help="Enable this when your PDF pages are handwritten; uses a handwriting-focused OCR engine if available.",
)

st.sidebar.header("Retrieval Settings")
top_k = st.sidebar.slider(
    "Results to retrieve",
    min_value=1,
    max_value=10,
    value=st.session_state.get("top_k", 5),
    step=1,
)
st.session_state["top_k"] = top_k

with st.sidebar.expander("Help", expanded=False):
    st.markdown(
        "- Add your OpenRouter key to `.env` as `OPENROUTER_API_KEY`.\n"
        "- Ingestion writes embeddings to `chroma_db/`; rerun if you replace the PDF.\n"
        "- Questions use the shared Chroma collection (`pdf_docs`).\n"
        "- You can ask questions in English, Tamil, or Tanglish; answers follow the same style."
    )

st.sidebar.header("Chats")
sessions = st.session_state["chat_sessions"]
current_session_id = st.session_state["current_session_id"]

st.sidebar.button("New Chat", use_container_width=True, on_click=start_new_chat, type="primary")

if sessions:
    st.sidebar.subheader("Your chats")
    ordered_sessions = sorted(
        sessions,
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )

    for session in ordered_sessions:
        session_id = session.get("id")
        if not session_id:
            continue
        title = session.get("title") or DEFAULT_SESSION_TITLE
        message_count = len(session.get("messages", []))
        created_at = session.get("created_at", "")
        label = f"{title}\n{message_count} msgs | {created_at}"
        is_active = session_id == current_session_id
        if st.sidebar.button(
            label,
            key=f"chat_entry_{session_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            switch_chat(session_id)
else:
    st.sidebar.caption("No chats yet. Ask a question to begin.")


@st.cache_data(show_spinner=False)
def _save_upload(upload) -> Path:
    """Persist the uploaded PDF to a temporary location."""
    suffix = Path(upload.name).suffix or ".pdf"
    tmp_path = Path(tempfile.mkstemp(suffix=suffix)[1])
    tmp_path.write_bytes(upload.getvalue())
    return tmp_path


def run_ingestion(
    pdf_path: Path,
    status: dict | None = None,
    progress: dict | None = None,
    handwriting_mode: bool = False,
) -> dict[str, int]:
    """Replicate the CLI ingestion pipeline for a given PDF."""
    progress = progress or {}

    def _update_progress(key: str, current: int, total: int, phase: str | None = None) -> None:
        handle = progress.get(key)
        if not handle:
            return
        widget = handle.get("widget")
        label = handle.get("label", "")
        if not widget:
            return
        percent = int(min(100, max(0, (current / total) * 100))) if total else 100
        detail = f"{label} ({current}/{total})" if total else label
        if phase:
            detail = f"{detail} – {phase}"
        try:
            widget.progress(percent, text=detail)
        except Exception:
            pass

    def _finalize_progress(key: str, phase: str | None = None) -> None:
        handle = progress.get(key)
        if not handle or not handle.get("widget"):
            return
        label = handle.get("label", "")
        detail = label
        if phase:
            detail = f"{label} – {phase}"
        try:
            handle["widget"].progress(100, text=detail)
        except Exception:
            pass

    if status and status.get("text"):
        status["text"].markdown(_status_card("Text", "Processing…"), unsafe_allow_html=True)

    text_chunks = extract_text(
        str(pdf_path),
        chunk_size=chunk_size,
        ocr_engine="handwriting" if handwriting_mode else "tesseract",
        progress_callback=lambda current, total: _update_progress("text", current, total, "pages"),
    )
    st.toast(f"Text extraction complete — {len(text_chunks)} chunks", icon="📄")
    if status and status.get("text"):
        status["text"].markdown(_status_card("Text Chunks", f"{len(text_chunks)}"), unsafe_allow_html=True)
    _finalize_progress("text", "completed")

    image_infos = []
    if fast_mode:
        # Skip image extraction/captioning entirely in fast text-only mode
        if status and status.get("images"):
            status["images"].markdown(_status_card("Image Entries", "Skipped (fast mode)"), unsafe_allow_html=True)
        _finalize_progress("images", "skipped")
    else:
        if status and status.get("images"):
            status["images"].markdown(_status_card("Image Entries", "Processing…"), unsafe_allow_html=True)

        images = extract_images_from_pdf(
            str(pdf_path),
            progress_callback=lambda current, total: _update_progress("images", current, total, "pages"),
        )
        if images:
            st.toast(f"Found {len(images)} page/image assets", icon="🖼️")
        if status and status.get("images"):
            status["images"].markdown(_status_card("Image Entries", f"{len(images)}"), unsafe_allow_html=True)
        _finalize_progress("images", "captured")
        for idx, (img_path, page) in enumerate(images, start=1):
            try:
                caption = generate_caption_openrouter(img_path)
            except Exception:
                caption = ""
                try:
                    caption = pytesseract.image_to_string(Image.open(img_path))
                except Exception:
                    caption = ""
            image_infos.append((caption, page, img_path))
            if idx % 5 == 0:
                st.toast(f"Captioned {idx}/{len(images)} images…", icon="🖼️")

    if status and status.get("chroma"):
        status["chroma"].markdown(_status_card("Stored Embeddings", "Updating…"), unsafe_allow_html=True)

    result = store_in_chromadb(
        text_chunks,
        image_infos,
        progress_callback=lambda processed, total, phase: _update_progress("chroma", processed, total, phase),
    )
    if status and status.get("chroma"):
        total_entries = result.get("text_count", 0) + result.get("image_count", 0)
        status["chroma"].markdown(
            _status_card("Stored Embeddings", f"{total_entries}"), unsafe_allow_html=True
        )
    _finalize_progress("chroma", "stored")

    if fast_mode:
        # Skip table extraction in fast mode
        if status and status.get("tables"):
            status["tables"].markdown(_status_card("Tables", "Skipped (fast mode)"), unsafe_allow_html=True)
        _finalize_progress("tables", "skipped")
        result["table_count"] = 0
    else:
        if status and status.get("tables"):
            status["tables"].markdown(_status_card("Tables", "Processing…"), unsafe_allow_html=True)

        tables_data = extract_tables_from_pdf(str(pdf_path))
        if tables_data:
            st.toast(f"Extracted {len(tables_data)} tables", icon="📊")
        if status and status.get("tables"):
            status["tables"].markdown(_status_card("Tables", f"{len(tables_data)}"), unsafe_allow_html=True)
        _finalize_progress("tables", "completed")
        try:
            store_tables_in_mongodb(tables_data)
        except Exception:
            # MongoDB is optional during local experimentation
            pass

        result["table_count"] = len(tables_data)
    return result


def clear_collection() -> None:
    """Remove all embeddings from the shared Chroma collection."""
    try:
        # Requesting an empty include avoids pulling large embeddings while still returning ids
        data = chroma_collection.get(include=[])
        ids = data.get("ids", []) if data else []
        if not ids:
            st.toast("Collection already empty.", icon="ℹ️")
            return
        chroma_collection.delete(ids=ids)
        st.toast(f"Cleared {len(ids)} entries from Chroma.", icon="🧹")
    except Exception as exc:
        st.error(f"Failed to clear collection: {exc}")


ingest_tab, qa_tab = st.tabs(["Ingest PDF", "Ask Questions"])

with ingest_tab:
    status_col, main_col = st.columns([1, 3])

    with status_col:
        status_col.markdown("#### PDF Snapshot")
        page_placeholder = status_col.empty()
        file_placeholder = status_col.empty()
        status_col.markdown("#### Processing Details")
        text_placeholder = status_col.empty()
        image_placeholder = status_col.empty()
        table_placeholder = status_col.empty()
        chroma_placeholder = status_col.empty()

    meta = st.session_state.get("pdf_meta", {})
    page_placeholder.markdown(
        _status_card("Total Pages", str(meta.get("pages") or "—")),
        unsafe_allow_html=True,
    )
    file_placeholder.markdown(
        _status_card("Current File", meta.get("name", "No upload yet")),
        unsafe_allow_html=True,
    )

    last_stats = st.session_state.get("last_run_stats")
    if last_stats:
        text_placeholder.markdown(
            _status_card("Text Chunks", str(last_stats.get("text_count", "—"))),
            unsafe_allow_html=True,
        )
        image_placeholder.markdown(
            _status_card("Image Entries", str(last_stats.get("image_count", "—"))),
            unsafe_allow_html=True,
        )
        table_placeholder.markdown(
            _status_card("Tables", str(last_stats.get("table_count", "—"))),
            unsafe_allow_html=True,
        )
        total_embeddings = (last_stats.get("text_count", 0) or 0) + (last_stats.get("image_count", 0) or 0)
        chroma_placeholder.markdown(
            _status_card("Stored Embeddings", str(total_embeddings)),
            unsafe_allow_html=True,
        )
    else:
        text_placeholder.markdown(_status_card("Text Chunks", "—"), unsafe_allow_html=True)
        image_placeholder.markdown(_status_card("Image Entries", "—"), unsafe_allow_html=True)
        table_placeholder.markdown(_status_card("Tables", "—"), unsafe_allow_html=True)
        chroma_placeholder.markdown(_status_card("Stored Embeddings", "—"), unsafe_allow_html=True)

    status_map = {
        "text": text_placeholder,
        "images": image_placeholder,
        "tables": table_placeholder,
        "chroma": chroma_placeholder,
    }

    with main_col:
        st.markdown("#### Upload & Process")
        upload = st.file_uploader("Upload a PDF to index", type="pdf")
        progress_container = st.container()
        if upload:
            saved_path = _save_upload(upload)
            st.info(f"Saved upload to `{saved_path}`")

            page_count = None
            try:
                with fitz.open(str(saved_path)) as doc:
                    page_count = doc.page_count
            except Exception:
                page_count = None

            if meta.get("name") != upload.name:
                st.session_state["last_run_stats"] = None
                text_placeholder.markdown(_status_card("Text Chunks", "—"), unsafe_allow_html=True)
                image_placeholder.markdown(_status_card("Image Entries", "—"), unsafe_allow_html=True)
                table_placeholder.markdown(_status_card("Tables", "—"), unsafe_allow_html=True)
                chroma_placeholder.markdown(_status_card("Stored Embeddings", "—"), unsafe_allow_html=True)

            st.session_state["pdf_meta"] = {
                "path": str(saved_path),
                "pages": page_count or 0,
                "name": upload.name,
            }
            page_placeholder.markdown(
                _status_card("Total Pages", str(page_count) if page_count else "—"),
                unsafe_allow_html=True,
            )
            file_placeholder.markdown(
                _status_card("Current File", upload.name),
                unsafe_allow_html=True,
            )

            clear_before = st.checkbox(
                "Clear existing embeddings before processing", value=True
            )

            if st.button("Process PDF", type="primary"):
                progress_container.empty()
                with progress_container:
                    st.markdown("#### Processing Progress")
                    text_progress = st.progress(0, text="Text extraction")
                    image_progress = st.progress(0, text="Image processing")
                    chroma_progress = st.progress(0, text="Embedding storage")
                    tables_progress = st.progress(0, text="Table extraction")
                progress_map = {
                    "text": {"widget": text_progress, "label": "Text extraction"},
                    "images": {"widget": image_progress, "label": "Image processing"},
                    "chroma": {"widget": chroma_progress, "label": "Embedding storage"},
                    "tables": {"widget": tables_progress, "label": "Table extraction"},
                }

                with st.spinner("Extracting text/images and updating Chroma…"):
                    if clear_before:
                        clear_collection()
                    st.toast("Processing started…", icon="⌛")
                    res = run_ingestion(
                        saved_path,
                        status=status_map,
                        progress=progress_map,
                        handwriting_mode=handwriting_mode,
                    )
                st.session_state["last_run_stats"] = res
                total_embeddings = res.get("text_count", 0) + res.get("image_count", 0)
                chroma_placeholder.markdown(
                    _status_card("Stored Embeddings", str(total_embeddings)),
                    unsafe_allow_html=True,
                )
                text_placeholder.markdown(
                    _status_card("Text Chunks", str(res.get("text_count", 0))),
                    unsafe_allow_html=True,
                )
                image_placeholder.markdown(
                    _status_card("Image Entries", str(res.get("image_count", 0))),
                    unsafe_allow_html=True,
                )
                table_placeholder.markdown(
                    _status_card("Tables", str(res.get("table_count", 0))),
                    unsafe_allow_html=True,
                )
                st.success(
                    f"Embeddings updated — text chunks: {res.get('text_count', 0)}, "
                    f"image entries: {res.get('image_count', 0)}"
                )

with qa_tab:
    st.markdown("#### Conversation")

    sessions = st.session_state["chat_sessions"]
    current_session = get_current_session()

    chat_container = st.container()

    for message in current_session.get("messages", []):
        role = message.get("role")
        content = message.get("content", "").strip()
        if not content:
            continue
        display_role = "assistant" if role == "assistant" else "user"
        with chat_container.chat_message(display_role):
            st.markdown(content)

    prompt = st.chat_input("Ask a question about the indexed PDFs")

    if prompt:
        prompt = prompt.strip()
        if not prompt:
            st.warning("Please enter a non-empty question.")
        else:
            user_message = {"role": "user", "content": prompt}
            current_session.setdefault("messages", []).append(user_message)
            current_session["title"] = derive_session_title(current_session["messages"])
            save_chat_sessions(st.session_state["chat_sessions"])

            with chat_container.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Retrieving context…"):
                retrieval = answer_query(prompt, k=st.session_state.get("top_k", 5))

            answer_text = ""

            if retrieval.get("error"):
                error_message = retrieval["error"]
                st.error(error_message)
                answer_text = f"Error: {error_message}"
            else:
                context = retrieval.get("context", "")
                if context:
                    with st.spinner("Calling LLM via OpenRouter…"):
                        llm_resp = call_llm_openrouter(prompt, context)

                    if llm_resp.get("error"):
                        error_message = llm_resp["error"]
                        st.error(error_message)
                        answer_text = f"Error: {error_message}"
                    else:
                        answer_text = (llm_resp.get("answer") or "").strip()
                        if not answer_text:
                            answer_text = "No answer returned by the model."
                else:
                    answer_text = "No context retrieved. Ingest a PDF and try again."

            if answer_text:
                with chat_container.chat_message("assistant"):
                    st.markdown(answer_text)
                current_session["messages"].append(
                    {"role": "assistant", "content": answer_text}
                )
                current_session["title"] = derive_session_title(current_session["messages"])
                save_chat_sessions(st.session_state["chat_sessions"])
