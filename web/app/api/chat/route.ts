import { NextRequest } from "next/server";
import { GoogleGenAI } from "@google/genai";
import {
  search,
  buildPrompt,
  SYSTEM_INSTRUCTION,
  EMBED_MODEL,
  EMBED_DIM,
  sanitizeExtraChunks,
} from "@/lib/rag";

export const runtime = "nodejs";
export const maxDuration = 30;

const GEN_MODEL = "gemini-2.5-flash";

// NDJSON stream: one JSON object per line.
//   {type:"meta", sources:[...]}  -> retrieved chunks (sent before generation)
//   {type:"token", text:"..."}    -> streamed answer text
//   {type:"done"} | {type:"error", message}
export async function POST(req: NextRequest) {
  let question = "";
  let topK = 5;
  let extraChunks: ReturnType<typeof sanitizeExtraChunks> = [];
  try {
    const body = await req.json();
    question = (body.question ?? "").toString();
    if (Number.isFinite(body.topK)) topK = Math.max(1, Math.min(8, body.topK));
    extraChunks = sanitizeExtraChunks(body.extraChunks);
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!question.trim()) {
    return Response.json({ error: "Question is required" }, { status: 400 });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return Response.json(
      { error: "GEMINI_API_KEY is not configured on the server." },
      { status: 500 }
    );
  }

  const ai = new GoogleGenAI({ apiKey });

  // 1. Embed the query in the same space as the precomputed chunks.
  let queryEmbedding: number[];
  try {
    const emb = await ai.models.embedContent({
      model: EMBED_MODEL,
      contents: question,
      config: { taskType: "RETRIEVAL_QUERY", outputDimensionality: EMBED_DIM },
    });
    const values = emb.embeddings?.[0]?.values;
    if (!values || !values.length) throw new Error("empty embedding");
    queryEmbedding = values;
  } catch (e: any) {
    return Response.json(
      { error: `Embedding failed: ${e?.message ?? e}` },
      { status: 502 }
    );
  }

  // 2. Retrieve, then 3. stream a grounded, cited answer.
  const hits = search(queryEmbedding, topK, extraChunks);
  const prompt = buildPrompt(question, hits);

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj: unknown) =>
        controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"));

      send({
        type: "meta",
        sources: hits.map((h) => ({
          rank: h.rank,
          id: h.id,
          page: h.page,
          file_name: h.file_name,
          score: Number(h.score.toFixed(4)),
          preview: h.content.replace(/\s+/g, " ").slice(0, 260),
        })),
      });

      try {
        const result = await ai.models.generateContentStream({
          model: GEN_MODEL,
          contents: prompt,
          config: {
            temperature: 0.6,
            maxOutputTokens: 1024,
            systemInstruction: SYSTEM_INSTRUCTION,
            // gemini-2.5-flash is a thinking model; thinking tokens would eat the
            // output budget and truncate the answer. Disable it for this RAG demo.
            thinkingConfig: { thinkingBudget: 0 },
          },
        });
        for await (const chunk of result) {
          const text = chunk.text;
          if (text) send({ type: "token", text });
        }
        send({ type: "done" });
      } catch (e: any) {
        send({ type: "error", message: e?.message ?? String(e) });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}
