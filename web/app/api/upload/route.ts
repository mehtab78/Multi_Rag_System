import { GoogleGenAI } from "@google/genai";
import { chunkText } from "@/lib/chunk";
import { EMBED_MODEL, EMBED_DIM, type Chunk } from "@/lib/rag";

export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_BYTES = 4 * 1024 * 1024; // stay under Vercel's request body limit
const MAX_CHUNKS = 80; // bound embedding calls + maxDuration for one upload
const EMBED_BATCH = 25;

// Session-only document upload: extract -> chunk -> embed, return the
// embedded chunks to the browser. Nothing is written server-side — the
// browser holds these and resends them with each /api/chat question so they
// can be merged into retrieval for that session, with no new infra.
export async function POST(req: Request) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return Response.json(
      { error: "GEMINI_API_KEY is not configured on the server." },
      { status: 500 }
    );
  }

  let file: File | null = null;
  try {
    const form = await req.formData();
    const f = form.get("file");
    if (f instanceof File) file = f;
  } catch {
    return Response.json({ error: "Expected multipart/form-data with a 'file' field." }, { status: 400 });
  }

  if (!file) {
    return Response.json({ error: "No file provided." }, { status: 400 });
  }
  if (file.type && file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    return Response.json({ error: "Only PDF files are supported." }, { status: 400 });
  }
  if (file.size > MAX_BYTES) {
    return Response.json(
      { error: `File is ${(file.size / 1e6).toFixed(1)}MB; the limit is ${MAX_BYTES / 1e6}MB.` },
      { status: 413 }
    );
  }

  const buf = Buffer.from(await file.arrayBuffer());

  let pages: { page: number; text: string }[];
  try {
    pages = await extractPdfPages(buf);
  } catch (e: any) {
    return Response.json(
      { error: `Could not read PDF: ${e?.message ?? e}` },
      { status: 400 }
    );
  }
  if (!pages.length) {
    return Response.json({ error: "No extractable text found in that PDF." }, { status: 400 });
  }

  const records: { id: string; page: number; content: string }[] = [];
  for (const { page, text } of pages) {
    for (const piece of chunkText(text)) {
      records.push({ id: `p${page}-${records.length}`, page, content: piece });
      if (records.length >= MAX_CHUNKS) break;
    }
    if (records.length >= MAX_CHUNKS) break;
  }
  const truncated = records.length >= MAX_CHUNKS;

  const ai = new GoogleGenAI({ apiKey });
  const chunks: Chunk[] = [];
  try {
    for (let i = 0; i < records.length; i += EMBED_BATCH) {
      const batch = records.slice(i, i + EMBED_BATCH);
      const resp = await ai.models.embedContent({
        model: EMBED_MODEL,
        contents: batch.map((b) => b.content),
        config: { taskType: "RETRIEVAL_DOCUMENT", outputDimensionality: EMBED_DIM },
      });
      const embeddings = resp.embeddings ?? [];
      batch.forEach((b, j) => {
        const values = embeddings[j]?.values;
        if (!values) return;
        chunks.push({
          id: b.id,
          page: b.page,
          file_name: file!.name,
          content: b.content,
          embedding: l2normalize(values),
        });
      });
    }
  } catch (e: any) {
    return Response.json(
      { error: `Embedding failed: ${e?.message ?? e}` },
      { status: 502 }
    );
  }

  return Response.json({
    file_name: file.name,
    pages: pages.length,
    chunks,
    truncated,
  });
}

function l2normalize(v: number[]): number[] {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

async function extractPdfPages(buf: Buffer): Promise<{ page: number; text: string }[]> {
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const doc = await pdfjs.getDocument({
    data: new Uint8Array(buf),
    useWorkerFetch: false,
    isEvalSupported: false,
    disableFontFace: true,
  }).promise;

  const pages: { page: number; text: string }[] = [];
  try {
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      const content = await page.getTextContent();
      const text = content.items
        .map((it: any) => ("str" in it ? it.str : ""))
        .join(" ")
        .replace(/\s+/g, " ")
        .trim();
      if (text) pages.push({ page: i, text });
    }
  } finally {
    await doc.destroy();
  }
  return pages;
}
