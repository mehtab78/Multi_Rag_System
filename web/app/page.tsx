import Chat, { type Stats } from "./chat";
import {
  SOURCE_DOC,
  PAGE_COUNT,
  CHUNK_COUNT,
  EMBED_MODEL,
} from "@/lib/rag";

export default function Home() {
  const stats: Stats = {
    source: SOURCE_DOC,
    pages: PAGE_COUNT,
    chunks: CHUNK_COUNT,
    embedModel: EMBED_MODEL,
    genModel: "gemini-2.5-flash",
  };

  return (
    <main className="shell">
      <header className="masthead">
        <div className="kicker">Retrieval-Augmented · Cited · Grounded</div>
        <h1 className="title">
          Margin<em>alia</em>
        </h1>
        <p className="lede">
          A chatbot that answers strictly from one source document — and footnotes
          every claim back to the passage it came from.
        </p>
      </header>

      <Chat stats={stats} />

      <footer className="foot">
        <span>Serverless RAG · embed → cosine retrieval → grounded generation</span>
        <span>Next.js on Vercel · Gemini</span>
      </footer>
    </main>
  );
}
