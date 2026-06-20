// Serverless RAG core: in-memory cosine retrieval over precomputed chunk
// embeddings (web/data/embeddings.json) + the grounded-answer prompt.
// No vector DB — the document set is small and ships with the deploy.

import data from "@/data/embeddings.json";

export type Chunk = {
  id: string;
  page: number;
  file_name: string;
  content: string;
  embedding: number[];
};

type EmbeddingsFile = {
  model: string;
  dim: number;
  source: string;
  chunks: Chunk[];
};

const db = data as unknown as EmbeddingsFile;

export const EMBED_MODEL = db.model;
export const EMBED_DIM = db.dim;
export const SOURCE_DOC = db.source;
export const CHUNK_COUNT = db.chunks.length;
export const PAGE_COUNT = new Set(db.chunks.map((c) => c.page)).size;

export type Retrieved = {
  rank: number;
  id: string;
  page: number;
  file_name: string;
  content: string;
  score: number;
};

function l2normalize(v: number[]): number[] {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) s += a[i] * b[i];
  return s;
}

// Doc vectors are L2-normalized at build time; normalizing the query here
// makes the dot product equal to cosine similarity.
// `extra` carries session-uploaded chunks (see /api/upload) that never touch
// disk server-side — the browser sends them back on each question and they're
// merged into the search pool for that request only.
export function search(
  queryEmbedding: number[],
  topK = 5,
  extra: Chunk[] = []
): Retrieved[] {
  const q = l2normalize(queryEmbedding);
  const pool = extra.length ? [...db.chunks, ...extra] : db.chunks;
  const scored = pool.map((c) => ({ c, score: dot(q, c.embedding) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map((s, i) => ({
    rank: i + 1,
    id: s.c.id,
    page: s.c.page,
    file_name: s.c.file_name,
    content: s.c.content,
    score: s.score,
  }));
}

const MAX_EXTRA_CHUNKS = 300;

// Defensive validation for chunks arriving from the client on /api/chat —
// never trust shape/length of embeddings sent back by the browser.
export function sanitizeExtraChunks(input: unknown): Chunk[] {
  if (!Array.isArray(input)) return [];
  const out: Chunk[] = [];
  for (const item of input) {
    if (out.length >= MAX_EXTRA_CHUNKS) break;
    if (!item || typeof item !== "object") continue;
    const c = item as Partial<Chunk>;
    if (
      typeof c.id === "string" &&
      typeof c.file_name === "string" &&
      typeof c.content === "string" &&
      typeof c.page === "number" &&
      Array.isArray(c.embedding) &&
      c.embedding.length === EMBED_DIM &&
      c.embedding.every((x) => typeof x === "number" && Number.isFinite(x))
    ) {
      out.push({
        id: c.id,
        file_name: c.file_name,
        content: c.content.slice(0, 4000),
        page: c.page,
        embedding: c.embedding,
      });
    }
  }
  return out;
}

export const SYSTEM_INSTRUCTION = `You are a precise research assistant answering questions about a single source document.
Rules:
- Answer ONLY using the numbered documents provided. Do not use outside knowledge.
- Cite every factual claim inline with bracketed numerals matching the document numbers, e.g. [1] or [2][3].
- If the documents do not contain the answer, say so directly instead of guessing.
- Be clear and concise. Prefer short paragraphs over long ones.`;

export function buildPrompt(question: string, hits: Retrieved[]): string {
  const context = hits
    .map(
      (h) =>
        `[Document ${h.rank}] (file: ${h.file_name}, page: ${h.page}, relevance: ${h.score.toFixed(
          3
        )})\n${h.content}`
    )
    .join("\n\n---\n\n");

  return `RETRIEVED DOCUMENTS:\n\n${context}\n\nQUESTION:\n${question}\n\nGrounded answer (with [n] citations):`;
}
