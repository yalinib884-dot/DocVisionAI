"""List OpenAI models available for the current API key.

Usage (from project root, with .env loaded):

    python list_openai_models.py

This prints model IDs so you can choose one (e.g. gpt-4o, gpt-4o-mini)
for OPENAI_MODEL in your .env.
"""

import os

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("The openai package is not installed. Run 'pip install openai'.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY in your .env before running this script.")

client = OpenAI(api_key=api_key)

print("Available OpenAI models for this key:\n")
for model in client.models.list().data:
    model_id = getattr(model, "id", "<no-id>")
    print(f"- {model_id}")
