import unittest
from unittest.mock import patch

import retrieval


class RetrievalTests(unittest.TestCase):
    def test_hybrid_search_fuses_duplicate_keyword_and_semantic_hits(self):
        keyword_hits = [
            {
                "_id": "a",
                "_score": 10,
                "_source": {"file_name": "a.pdf", "content_type": "text", "content": "alpha"},
            },
            {
                "_id": "b",
                "_score": 9,
                "_source": {"file_name": "b.pdf", "content_type": "text", "content": "beta"},
            },
        ]
        semantic_hits = [
            {
                "_id": "b",
                "_score": 0.9,
                "_source": {"file_name": "b.pdf", "content_type": "text", "content": "beta"},
            },
            {
                "_id": "c",
                "_score": 0.8,
                "_source": {"file_name": "c.pdf", "content_type": "text", "content": "gamma"},
            },
        ]

        with (
            patch.object(retrieval, "RERANK_RESULTS", False),
            patch.object(retrieval, "keyword_search", return_value=keyword_hits),
            patch.object(retrieval, "semantic_search", return_value=semantic_hits),
        ):
            results = retrieval.hybrid_search("test", top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["_id"], "b")
        self.assertEqual(results[0]["retrieval_sources"], ["keyword", "semantic"])

    def test_rerank_results_prefers_query_overlap(self):
        hits = [
            {
                "_score": 0.01,
                "_source": {"content": "retrieval augmented generation improves grounded answers"},
            },
            {
                "_score": 0.01,
                "_source": {"content": "unrelated content about deployment"},
            },
        ]

        results = retrieval.rerank_results("grounded retrieval generation", hits, top_k=2)

        self.assertTrue(results[0]["_source"]["content"].startswith("retrieval augmented"))
        self.assertGreater(results[0]["rerank_score"], results[1]["rerank_score"])


if __name__ == "__main__":
    unittest.main()
