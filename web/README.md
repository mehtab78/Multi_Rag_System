# Marginalia — serverless RAG demo (Vercel)

A document-grounded Q&A chatbot that answers **strictly from one source document**
and footnotes every claim back to the passage it came from. This is the
Vercel-deployable companion to the full Python RAG system in the repo root.

## Why a separate app?

The root project uses **OpenSearch** (vector DB) + **Ollama** (embeddings) +
**Gradio**. None of those run on Vercel, which is serverless (no persistent
servers, no GB-scale model hosts). This app keeps the exact RAG pipeline —
ingest → chunk → embed → retrieve → cited generation — but in a serverless shape:

| Stage | Root project | This app (serverless) |
|---|---|---|
| Embeddings | Ollama `nomic-embed-text` | Gemini `gemini-embedding-001` (768-d) |
| Vector store | OpenSearch (kNN) | Precomputed JSON + in-memory cosine |
| Generation | Gemini | Gemini `gemini-2.5-flash` (thinking off) |
| UI | Gradio | Next.js (App Router) |

The document vectors are **precomputed once** and shipped in
`data/embeddings.json`, so the only runtime dependency is the Gemini API.

## Architecture

```
Browser ──POST /api/chat──▶ Next.js route (Node serverless fn)
                              1. embed query   → gemini-embedding-001
                              2. cosine search → data/embeddings.json (in-memory)
                              3. build prompt  → grounded, "cite with [n]"
                              4. stream answer → gemini-2.5-flash  (NDJSON)
```

The response is NDJSON: a `meta` line with the retrieved sources, then `token`
lines streaming the answer, then `done`.

## Local development

```bash
cd web
npm install
echo "GEMINI_API_KEY=your_key_here" > .env.local
npm run dev   # http://localhost:3000
```

## Deploy on Vercel (GitHub import)

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to <https://vercel.com/new> and **Import** the `Multi_Rag_System` repo.
3. **Set _Root Directory_ to `web`** (so Vercel builds this Next.js app, not the
   Python root). Framework preset auto-detects as **Next.js**.
4. Under **Environment Variables**, add:
   - `GEMINI_API_KEY` = your Google AI Studio key
5. Click **Deploy**. Vercel gives you a live `*.vercel.app` URL.

> The key is server-side only (used in the API route); it is never exposed to
> the browser and is not committed to the repo.

## Rebuilding the document set

To index a different PDF, regenerate the embeddings file from the repo root:

```bash
python scripts/build_embeddings.py files/your.pdf   # writes web/data/embeddings.json
```

Then redeploy (push to GitHub → Vercel auto-builds).
