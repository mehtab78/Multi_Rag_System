"""
One-time precompute: extract + chunk a PDF and embed each chunk with Gemini,
writing a static JSON the Vercel serverless app loads for in-memory retrieval.

Why this exists: Vercel is serverless (no OpenSearch / Ollama). Instead of a
live vector DB we ship the document vectors as a file and do cosine search in
the function. Run locally whenever the document set changes:

    python scripts/build_embeddings.py files/tani.pdf

Output: web/data/embeddings.json
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # Matryoshka truncation keeps the JSON small; plenty for ~150 chunks.
MAX_CHARS = 900
OVERLAP = 150
BATCH = 20          # each content counts toward the 100 req/min free-tier embed quota
BATCH_PAUSE = 15.0  # 20 items / 15s = 80/min, safely under the limit


def extract_pages(pdf_path: Path):
    """Return list of (page_number, text) for pages that contain text."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            text = re.sub(r"[ \t]+", " ", text).strip()
            if text:
                pages.append((i, text))
    return pages


def chunk_text(text: str):
    """Greedy ~MAX_CHARS chunks with OVERLAP, preferring paragraph/sentence breaks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blob = "\n\n".join(paragraphs) if paragraphs else text
    chunks = []
    start = 0
    while start < len(blob):
        end = min(start + MAX_CHARS, len(blob))
        if end < len(blob):
            # back off to the nearest sentence/space boundary for cleaner chunks
            window = blob[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"), window.rfind(" "))
            if cut > MAX_CHARS * 0.5:
                end = start + cut + 1
        chunk = blob[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(blob):
            break
        start = max(end - OVERLAP, start + 1)
    return chunks


def l2_normalize(vec):
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _retry_delay_seconds(err) -> float:
    m = re.search(r"retry in ([\d.]+)s", str(err))
    return float(m.group(1)) + 1.0 if m else 20.0


def embed_batch(client, texts, task_type, attempts=6):
    for attempt in range(attempts):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBED_DIM,
                ),
            )
            return [l2_normalize(e.values) for e in resp.embeddings]
        except Exception as err:  # noqa: BLE001 - retry only on rate limits
            if "429" not in str(err) or attempt == attempts - 1:
                raise
            delay = _retry_delay_seconds(err)
            print(f"  rate limited, waiting {delay:.0f}s (attempt {attempt + 1})")
            time.sleep(delay)


def main():
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "files" / "tani.pdf"
    if not pdf_path.is_absolute():
        pdf_path = ROOT / pdf_path
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set (.env).")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    print(f"Extracting {pdf_path.name} ...")
    pages = extract_pages(pdf_path)
    records = []
    for page_number, text in pages:
        for piece in chunk_text(text):
            records.append({
                "id": f"p{page_number}-{len(records):03d}",
                "page": page_number,
                "file_name": pdf_path.name,
                "content": piece,
            })
    print(f"  {len(pages)} pages -> {len(records)} chunks")

    print(f"Embedding with {EMBED_MODEL} (dim={EMBED_DIM}) ...")
    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        vectors = embed_batch(client, [r["content"] for r in batch], "RETRIEVAL_DOCUMENT")
        for r, v in zip(batch, vectors):
            r["embedding"] = v
        print(f"  embedded {min(i + BATCH, len(records))}/{len(records)}")
        if i + BATCH < len(records):
            time.sleep(BATCH_PAUSE)

    out_path = ROOT / "web" / "data" / "embeddings.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": EMBED_MODEL,
        "dim": EMBED_DIM,
        "source": pdf_path.name,
        "chunks": records,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path.relative_to(ROOT)} ({len(records)} chunks, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
