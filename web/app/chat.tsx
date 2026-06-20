"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Source = {
  rank: number;
  id: string;
  page: number;
  file_name: string;
  score: number;
  preview: string;
};

type Turn = {
  id: number;
  question: string;
  answer: string;
  sources: Source[];
  status: "retrieving" | "composing" | "done" | "error";
  error?: string;
};

export type Stats = {
  source: string;
  pages: number;
  chunks: number;
  embedModel: string;
  genModel: string;
};

const EXAMPLES = [
  "How does RAG work?",
  "What are the advantages of RAG over fine-tuning?",
  "How is retrieval quality evaluated?",
  "Summarize the main contributions.",
];

const STEPS = ["Ingest", "Chunk", "Embed", "Retrieve", "Cite"];

// Split answer text into prose + clickable [n] footnote markers.
function renderProse(text: string, turnId: number, onCite: (n: number) => void) {
  const paragraphs = text.split(/\n{2,}/);
  return paragraphs.map((para, pi) => {
    const parts = para.split(/(\[\d+\])/g);
    return (
      <p key={pi}>
        {parts.map((part, i) => {
          const m = part.match(/^\[(\d+)\]$/);
          if (m) {
            const n = Number(m[1]);
            return (
              <button
                key={i}
                className="fn"
                onClick={() => onCite(n)}
                aria-label={`Source ${n}`}
              >
                {n}
              </button>
            );
          }
          return <span key={i}>{part}</span>;
        })}
      </p>
    );
  });
}

export default function Chat({ stats }: { stats: Stats }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const fieldRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const scrollToCite = useCallback((turnId: number, n: number) => {
    const el = document.getElementById(`fn-${turnId}-${n}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 1200);
  }, []);

  const ask = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || busy) return;
      setBusy(true);
      setInput("");
      const id = Date.now();
      setTurns((t) => [
        ...t,
        { id, question: q, answer: "", sources: [], status: "retrieving" },
      ]);

      const update = (patch: Partial<Turn>) =>
        setTurns((t) => t.map((x) => (x.id === id ? { ...x, ...patch } : x)));

      try {
        setActiveStep(2); // Embed
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, topK: 5 }),
        });

        if (!res.ok || !res.body) {
          const msg = await res.json().catch(() => ({ error: res.statusText }));
          update({ status: "error", error: msg.error ?? "Request failed" });
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let answer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.trim()) continue;
            let evt: any;
            try {
              evt = JSON.parse(line);
            } catch {
              continue;
            }
            if (evt.type === "meta") {
              setActiveStep(3); // Retrieve
              update({ sources: evt.sources, status: "composing" });
            } else if (evt.type === "token") {
              setActiveStep(4); // Cite
              answer += evt.text;
              update({ answer });
            } else if (evt.type === "done") {
              update({ status: "done" });
            } else if (evt.type === "error") {
              update({ status: "error", error: evt.message });
            }
          }
        }
        update({ status: "done" });
      } catch (e: any) {
        update({ status: "error", error: e?.message ?? "Network error" });
      } finally {
        setBusy(false);
        setActiveStep(-1);
      }
    },
    [busy]
  );

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    ask(input);
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(input);
    }
  };

  const grow = () => {
    const el = fieldRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  };

  return (
    <div className="grid">
      <aside className="rail">
        <div className="card">
          <div className="card-label">The Source</div>
          <div className="source-name">
            <span className="dot" />
            {stats.source}
          </div>
          <div className="stats">
            <div className="stat">
              <span className="k">pages</span>
              <span className="v">{stats.pages}</span>
            </div>
            <div className="stat">
              <span className="k">chunks</span>
              <span className="v">{stats.chunks}</span>
            </div>
            <div className="stat">
              <span className="k">embed</span>
              <span className="v">{stats.embedModel}</span>
            </div>
            <div className="stat">
              <span className="k">answer</span>
              <span className="v">{stats.genModel}</span>
            </div>
          </div>

          <ul className="pipeline">
            <li className="pipe-label">Pipeline</li>
            {STEPS.map((s, i) => (
              <li
                key={s}
                className={`step${i === activeStep ? " active" : ""}`}
              >
                <span className="n">{i + 1}</span>
                {s}
              </li>
            ))}
          </ul>
        </div>
      </aside>

      <section className="main">
        <div className="thread">
          {turns.length === 0 && (
            <div className="empty">
              Ask a question and the system retrieves the most relevant passages
              from <b>{stats.source}</b>, then composes an answer that{" "}
              <b>cites each source</b> as a footnote. Nothing is invented — if
              the document doesn&apos;t cover it, the answer says so.
            </div>
          )}

          {turns.map((turn) => (
            <article key={turn.id}>
              <div className="msg-q">{turn.question}</div>

              <div className="answer" style={{ marginTop: 18 }}>
                <div className="answer-tag">Grounded answer</div>

                {turn.status === "retrieving" && (
                  <div className="status">
                    <span className="pulse" /> retrieving passages…
                  </div>
                )}

                {turn.answer && (
                  <div className="prose">
                    {renderProse(turn.answer, turn.id, (n) =>
                      scrollToCite(turn.id, n)
                    )}
                    {turn.status === "composing" && <span className="caret" />}
                  </div>
                )}

                {turn.status === "composing" && !turn.answer && (
                  <div className="status">
                    <span className="pulse" /> composing…
                  </div>
                )}

                {turn.error && <div className="err">⚠ {turn.error}</div>}

                {turn.sources.length > 0 && (
                  <div className="apparatus">
                    <div className="apparatus-label">
                      Sources · {turn.sources.length} passages
                    </div>
                    {turn.sources.map((s) => (
                      <div
                        key={s.rank}
                        id={`fn-${turn.id}-${s.rank}`}
                        className="footnote"
                      >
                        <span className="marker">{s.rank}</span>
                        <span>
                          <span className="meta">
                            {s.file_name} · p.{s.page} · score{" "}
                            {s.score.toFixed(3)}
                          </span>
                          <br />
                          <span className="quote">“{s.preview}…”</span>
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </article>
          ))}
          <div ref={endRef} />
        </div>

        <form className="composer" onSubmit={onSubmit}>
          <div className="composer-row">
            <textarea
              ref={fieldRef}
              className="field"
              rows={1}
              placeholder="Ask the document a question…"
              value={input}
              disabled={busy}
              onChange={(e) => {
                setInput(e.target.value);
                grow();
              }}
              onKeyDown={onKey}
            />
            <button className="send" type="submit" disabled={busy || !input.trim()}>
              {busy ? "…" : "Ask ↵"}
            </button>
          </div>
        </form>

        <div className="examples">
          <span className="ex-label">Suggested readings</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              className="chip"
              disabled={busy}
              onClick={() => ask(ex)}
            >
              {ex}
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
