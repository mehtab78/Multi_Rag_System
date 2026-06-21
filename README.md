# Marginalia — RAG Q&A over a PDF

A document-grounded Q&A chatbot that answers strictly from a source PDF and
footnotes every claim back to the passage it came from.

**Live demo:** <https://multi-rag-system.vercel.app/>

The live app is the Next.js project in [`web/`](web/) — see
[`web/README.md`](web/README.md) for its architecture, local dev, and deploy
instructions.

## What's in the repo root

This repo previously also contained a local Gradio + OpenSearch + Ollama
version of the same idea. That stack has been removed in favor of the
serverless `web/` app, which is simpler to run and deploy and is what's
actually live. The only Python left at the root is the script that builds the
static document corpus the web app retrieves from:

```text
scripts/build_embeddings.py   Extract + chunk + embed a PDF -> web/data/embeddings.json
files/tani.pdf                 Sample source document
requirements.txt               Deps for build_embeddings.py (google-genai, pdfplumber, python-dotenv)
```

To index a different PDF:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set GEMINI_API_KEY
python3 scripts/build_embeddings.py files/your.pdf
```

This writes `web/data/embeddings.json`; commit it and redeploy `web/` to
update the live corpus.

## License

MIT. See `LICENSE`.
