PDF Chatbot — Windows 11 (16GB RAM) setup

This project extracts text, tables, and images from PDFs and stores embeddings + tables for a retrieval-based chatbot.

Quick setup notes for Windows 11 (16GB RAM):

1) System dependencies
- Install Poppler (used by `pdf2image`)
  - Download: https://github.com/oschwartz10612/poppler-windows/releases
  - Extract and set `POPPLER_PATH` in `.env` or add the `poppler` bin folder to PATH.

- Install Tesseract OCR
  - Installer: https://github.com/tesseract-ocr/tesseract/releases
  - Or via Chocolatey (Admin PowerShell): `choco install tesseract`
  - Set `TESSERACT_CMD` in `.env` if installed in a non-standard location.

- Install Ghostscript (may be required by some table-extraction tools)
  - https://www.ghostscript.com/download/gsdnld.html

2) Python env (PowerShell commands)
```powershell
python -m pip install -U pip
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3) Ingestion + verification workflow
```powershell
# activate the virtual environment each session
venv\Scripts\Activate.ps1

# populate embeddings + captions (writes to chroma_db/)
python extractor.py

# inspect persisted docs (count + sample snippets)
python inspect_chroma.py

# retrieval-only smoke test
python test_retrieval.py "summary of the main chart"

# end-to-end answer (requires OPENROUTER_API_KEY in .env)
python test_answer.py "summary of the main chart"
```

Remote LLM configuration
- Set `OPENROUTER_API_KEY` inside `.env` (loaded via `python-dotenv`).
- Calls go to `https://openrouter.ai/api/v1/chat/completions` with model `meta-llama/llama-3.1-8b-instruct`.
- If DNS errors occur, switch the adapter to public DNS (e.g., 8.8.8.8 / 1.1.1.1) and retry `curl https://openrouter.ai/api/v1`.

4) Optional Streamlit UI
```powershell
venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```
The UI lets you ingest PDFs and ask questions from a browser tab, reusing the same Chroma collection and OpenRouter model.

Notes for 16GB RAM
- Avoid running large local multimodal LLMs on-device. Use a managed cloud model (OpenRouter/Qwen or OpenAI) for multimodal reasoning.
- For image captioning, prefer the remote API if you don't have a GPU. Use pytesseract as a fallback for scanned text.
- The chatbot now understands English, Tamil, and Tanglish (Tamil written with Latin characters). Make sure to clear/rebuild the Chroma collection after updating to the multilingual embeddings.

If you'd like, I can add a Dockerfile for a Linux-based extraction environment (recommended for Camelot reliability).