"""Retrieval and answer helper functions for the PDF chatbot.

This module does two main things:

- `answer_query` → retrieves the most relevant chunks from the `pdf_docs`
    Chroma collection.
- `call_llm_openrouter` → despite the legacy name, this now calls **OpenAI**
    (ChatGPT / GPT‑4 family) to generate the final answer from the retrieved
    context, and falls back to a simple local answer if the API cannot be
    reached.
"""
from typing import List, Dict, Any
import os
import re

import requests
from langdetect import detect, DetectorFactory, LangDetectException

from extractor import collection, embedding_model

try:  # OpenAI client for ChatGPT / GPT-4 family
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

# OpenAI configuration (single, preferred backend)
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_openai_client = OpenAI(api_key=_OPENAI_KEY) if (_OPENAI_KEY and OpenAI is not None) else None


DetectorFactory.seed = 0

_TAMIL_CHAR_PATTERN = re.compile(r"[\u0B80-\u0BFF]")
_TANGLISH_KEYWORDS = {
    "enna",
    "epdi",
    "epdi",
    "seri",
    "seriya",
    "illai",
    "illa",
    "unga",
    "ungal",
    "enga",
    "intha",
    "anna",
    "akka",
    "machan",
    "dei",
    "poda",
    "po",
    "vaa",
    "sapadu",
    "velai",
    "thambi",
    "thala",
    "sollu",
    "yen",
    "ypa",
    "thalaiva",
    "thalaivar",
    "sapdu",
    "sapidalama",
    "lamma",
    "nanba",
    "nanri",
    "yaaru",
    "kadasi",
    "podhum",
}


def _contains_tamil_script(text: str) -> bool:
    return bool(text and _TAMIL_CHAR_PATTERN.search(text))


def _looks_like_tanglish(text: str) -> bool:
    if not text:
        return False
    if _contains_tamil_script(text):
        return False
    lowered = text.lower()
    for keyword in _TANGLISH_KEYWORDS:
        if keyword in lowered:
            return True
    return False


def _detect_language_code(text: str) -> str:
    if not text or not text.strip():
        return "en"
    if _contains_tamil_script(text):
        return "ta"
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def _response_style_for_question(question: str) -> str:
    if _contains_tamil_script(question):
        return "tamil"
    if _looks_like_tanglish(question):
        return "tanglish"
    lang_code = _detect_language_code(question)
    if lang_code == "ta":
        return "tamil"
    return "english"


def _encode_query(text: str) -> List[float]:
    # Use the same encoder as extractor to ensure compatibility
    vec = embedding_model.encode([text])[0]
    return vec.tolist()


def answer_query(query: str, k: int = 5) -> Dict[str, Any]:
    """Retrieve top-k relevant chunks for `query` and return context + sources.

    Returns a dict with:
    - 'context': concatenated snippets to feed an LLM
    - 'sources': list of dicts {id, page, type, snippet, distance}
    - 'raw': raw Chroma query response
    """
    qvec = _encode_query(query)

    try:
        results = collection.query(query_embeddings=[qvec], n_results=k)
    except Exception as e:
        return {"error": f"Chroma query failed: {e}"}

    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    dists = results.get("distances", [])

    # normalize nested batch responses: many chroma clients return lists per query
    if isinstance(docs, list) and len(docs) and isinstance(docs[0], list):
        docs = docs[0]
    if isinstance(metas, list) and len(metas) and isinstance(metas[0], list):
        metas = metas[0]
    if isinstance(dists, list) and len(dists) and isinstance(dists[0], list):
        dists = dists[0]

    sources = []
    context_parts: List[str] = []

    for i, doc in enumerate(docs or []):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None
        snippet = meta.get("snippet") or (doc[:400] if isinstance(doc, str) else "")
        src = {
            "id": (meta.get("id") or f"doc_{i}"),
            "page": meta.get("page"),
            "type": meta.get("type"),
            "snippet": snippet,
            "distance": dist,
        }
        sources.append(src)
        # build context parts; prefer snippet then full doc
        context_parts.append(snippet if snippet else (doc if isinstance(doc, str) else ""))

    # simple concatenation; downstream LLM code can truncate or format as needed
    context = "\n---\n".join([c for c in context_parts if c])

    return {"context": context, "sources": sources, "raw": results}


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Return a human-readable list of sources for display in UI/logging."""
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"{i}. page={s.get('page')} type={s.get('type')} distance={s.get('distance')}\n   {s.get('snippet')}")
    return "\n\n".join(lines)


def call_llm_openrouter(question: str, context: str, temperature: float = 0.0, max_tokens: int = 400) -> Dict[str, Any]:
    """Call OpenRouter / Qwen model with the combined context and question.

    Returns dict {answer: str, raw: dict} or {'error': msg} on failure.
    Falls back to returning the context if no API key is configured.
    """
    style = _response_style_for_question(question)
    if style == "tamil":
        style_instruction = (
            "The user is interacting in Tamil. Reply fully in Tamil script, keep the tone natural, "
            "and include page references in the format 'பக்கம் X'."
        )
        response_hint = "Reply entirely in Tamil script."
    elif style == "tanglish":
        style_instruction = (
            "The user writes Tamil words using the Latin alphabet (Tanglish). Reply in the same Tanglish style "
            "— Tamil vocabulary but English letters — and show page references as 'page X'."
        )
        response_hint = "Reply in Tanglish (Tamil words spelled with English letters)."
    else:
        style_instruction = "Respond in clear English while referencing source pages (e.g., 'page X')."
        response_hint = "Reply in English."

    prompt_system = (
        "You are an assistant that answers user questions using the provided context and sources. "
        "Keep answers concise, quote key facts when helpful, and always mention source pages when available. "
        f"{style_instruction}"
    )
    user_text = (
        f"Context (may be English even if the question is another language):\n{context[:15000]}\n\n"
        f"Question: {question}\n"
        f"Use only the context; if information is missing, say so. {response_hint}"
    )

    def _build_fallback_answer(reason: str) -> Dict[str, Any]:
        """Return a local, context-based fallback answer when remote LLM is unavailable."""

        base = (context[:2000] + "\n\n" + "Answer (fallback, no remote LLM): " + question)[:4000]
        if style == "tamil":
            prefix = (
                "LLM API கிடைக்கவில்லை (" + reason + "). "
                "அதனால், கீழே உள்ள பதில் context அடிப்படையில் ஒரு எளிய சுருக்கம் மட்டுமே.\n"
            )
            base = prefix + base
        elif style == "tanglish":
            prefix = (
                "LLM API not available (" + reason + "). "
                "So this is a simple fallback answer using only the retrieved context.\n"
            )
            base = prefix + base
        else:
            prefix = (
                "LLM backend is unavailable (" + reason + "). "
                "Showing a simple fallback answer based only on retrieved context.\n"
            )
            base = prefix + base
        return {"answer": base, "raw": None}

    # Single remote backend: OpenAI
    if _openai_client is not None:
        try:
            resp = _openai_client.chat.completions.create(
                model=_OPENAI_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": user_text},
                ],
            )
            content = resp.choices[0].message.content or ""
            return {
                "answer": content.strip(),
                "raw": resp.to_dict_recursive() if hasattr(resp, "to_dict_recursive") else None,
            }
        except Exception as e:
            return _build_fallback_answer(f"OpenAI request failed: {e}")

    # No usable OpenAI client – always return a safe local fallback.
    return _build_fallback_answer("no usable OpenAI client configured")
