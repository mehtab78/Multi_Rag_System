// Greedy ~MAX_CHARS chunks with OVERLAP, preferring paragraph/sentence breaks.
// Mirrors scripts/build_embeddings.py so session-uploaded docs chunk the same
// way as the precomputed corpus.
const MAX_CHARS = 900;
const OVERLAP = 150;

export function chunkText(text: string): string[] {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);
  const blob = paragraphs.length ? paragraphs.join("\n\n") : text;

  const chunks: string[] = [];
  let start = 0;
  while (start < blob.length) {
    let end = Math.min(start + MAX_CHARS, blob.length);
    if (end < blob.length) {
      const window = blob.slice(start, end);
      const cut = Math.max(
        window.lastIndexOf(". "),
        window.lastIndexOf("\n"),
        window.lastIndexOf(" ")
      );
      if (cut > MAX_CHARS * 0.5) end = start + cut + 1;
    }
    const chunk = blob.slice(start, end).trim();
    if (chunk) chunks.push(chunk);
    if (end >= blob.length) break;
    start = Math.max(end - OVERLAP, start + 1);
  }
  return chunks;
}
