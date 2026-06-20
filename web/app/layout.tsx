import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Marginalia — document-grounded Q&A",
  description:
    "A retrieval-augmented chatbot that answers strictly from a source document and cites every claim. Serverless RAG on Vercel.",
};

// Fonts are loaded at runtime from Google Fonts (no build-time fetch), with
// Georgia / system-mono fallbacks declared in globals.css.
const FONTS_HREF =
  "https://fonts.googleapis.com/css2?" +
  "family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400..700&" +
  "family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400..600&" +
  "family=JetBrains+Mono:wght@400..700&display=swap";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link rel="stylesheet" href={FONTS_HREF} />
      </head>
      <body>{children}</body>
    </html>
  );
}
