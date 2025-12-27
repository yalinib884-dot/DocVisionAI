"""Small helper script to list Gemini models available for the current API key.

Usage (from project root, with .env loaded via python-dotenv in your shell):

    python list_gemini_models.py

It will print model names and which methods they support (e.g., generateContent).
"""

import os

from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
except ImportError:
    raise SystemExit("google-generativeai is not installed. Run 'pip install google-generativeai'.")

api_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("google_api_key")
    or os.getenv("googlre-api-key")
)

if not api_key:
    raise SystemExit("No Gemini/Google API key found in .env. Set GEMINI_API_KEY or GOOGLE_API_KEY or google_api_key.")

genai.configure(api_key=api_key)

print("Available Gemini models for this API key:\n")
for model in genai.list_models():
    name = getattr(model, "name", "<no-name>")
    methods = getattr(model, "supported_generation_methods", [])
    print(f"- {name}  (methods: {', '.join(methods)})")
