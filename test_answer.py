"""Test script: retrieve context for a query and call the LLM to get an answer.

Usage:
    python test_answer.py "your query here"
"""
import sys
from model import answer_query, format_sources, call_llm_openrouter


def main():
    query = "How many drawings are mentioned in the patent form?" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print("Query:", query)
    res = answer_query(query, k=5)
    if res.get("error"):
        print("Retrieval error:", res["error"])
        return

    print("\nRetrieved context (short):\n")
    print(res["context"][:1000])

    print("\nSources:\n")
    print(format_sources(res["sources"]))

    print("\nCalling LLM (OpenRouter/Qwen)...\n")
    llm = call_llm_openrouter(query, res["context"], temperature=0.0, max_tokens=300)
    if llm.get("error"):
        print("LLM error:", llm["error"])
    else:
        print("Answer:\n", llm.get("answer"))


if __name__ == '__main__':
    main()
