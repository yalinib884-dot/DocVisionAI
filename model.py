"""Retrieval and source-attribution helper functions.

This module provides `answer_query` which retrieves top-k chunks from the
`pdf_docs` Chroma collection and returns merged context and source metadata.
"""
from typing import List, Dict, Any
import re

import requests
from langdetect import detect, DetectorFactory, LangDetectException

from extractor import collection, embedding_model, OPENROUTER_API_KEY, OPENROUTER_ENDPOINT, QWEN_MODEL, HEADERS


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

    if not OPENROUTER_API_KEY:
        # no remote key — return a safe fallback combining context and question
        base = (context[:2000] + "\n\n" + "Answer (fallback): " + question)[:4000]
        if style == "tamil":
            base = "API விசை கிடைக்காததால், உருவாக்கப்பட்ட பதில் ஆங்கிலத்தில் மட்டுமே உள்ளது.\n" + base
        elif style == "tanglish":
            base = "API key missing, so fallback answer is in English. Tanglish mode is unavailable in fallback.\n" + base
        return {"answer": base, "raw": None}

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": user_text}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(OPENROUTER_ENDPOINT, headers=HEADERS, json=payload, timeout=60)
    except Exception as e:
        return {"error": f"OpenRouter request failed: {e}"}

    if resp.status_code != 200:
        return {"error": f"OpenRouter API error {resp.status_code}: {resp.text}"}

    try:
        j = resp.json()
    except Exception as e:
        return {"error": f"Invalid JSON response: {e}"}

    # defensive parsing - many chat APIs return choices -> message -> content
    try:
        content = j.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            # try older style
            content = j.get("choices", [{}])[0].get("text", "")
    except Exception:
        content = ""

    return {"answer": content.strip(), "raw": j}
