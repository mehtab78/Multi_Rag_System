import unittest

from generation import build_rag_prompt, format_retrieved_contexts, format_source_citations


class GenerationTests(unittest.TestCase):
    def test_format_retrieved_contexts_includes_source_metadata(self):
        hits = [
            {
                "_id": "fallback-id",
                "_score": 0.42,
                "_source": {
                    "content": "RAG retrieves relevant context before generation.",
                    "content_type": "text",
                    "file_name": "rag.pdf",
                    "page_number": 7,
                    "chunk_id": "chunk-123",
                },
            }
        ]

        context = format_retrieved_contexts(hits)

        self.assertIn("[Document 1]", context)
        self.assertIn("File: rag.pdf", context)
        self.assertIn("Page: 7", context)
        self.assertIn("Chunk ID: chunk-123", context)
        self.assertIn("RAG retrieves relevant context", context)

    def test_build_rag_prompt_includes_question_and_context(self):
        prompt = build_rag_prompt("Context block", "What is RAG?")

        self.assertIn("RETRIEVED DOCUMENTS:", prompt)
        self.assertIn("Context block", prompt)
        self.assertIn("USER QUESTION:", prompt)
        self.assertIn("What is RAG?", prompt)

    def test_format_source_citations_includes_page_and_chunk(self):
        citations = format_source_citations(
            [
                {
                    "_score": 0.12345,
                    "_source": {
                        "file_name": "rag.pdf",
                        "page_number": 3,
                        "chunk_id": "abcdef1234567890",
                    },
                }
            ]
        )

        self.assertIn("Sources:", citations)
        self.assertIn("rag.pdf, page 3", citations)
        self.assertIn("abcdef123456", citations)


if __name__ == "__main__":
    unittest.main()
