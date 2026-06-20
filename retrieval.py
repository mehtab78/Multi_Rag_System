"""
Retrieval module for the Multi-RAG System.

This module provides functions for:
- Keyword search (BM25-style)
- Semantic search (KNN with embeddings)
- Hybrid search (combined scoring)
"""

import os
import re
from typing import List, Dict, Any
from helper import EMBEDDING_DIMENSION, get_embedding, get_opensearch_client

DEFAULT_INDEX_NAME = os.getenv("INDEX_NAME", "pdf_content_index")
VALID_SEARCH_TYPES = {"keyword", "semantic", "hybrid"}
RERANK_RESULTS = os.getenv("RERANK_RESULTS", "true").lower() in {"1", "true", "yes"}


def _normalize_top_k(top_k: int) -> int:
    """Clamp retrieval count to a practical range for demos and prompts."""
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        value = 5
    return max(1, min(value, 25))


def _source_key(hit: Dict[str, Any]) -> str:
    source = hit.get("_source", {})
    return str(hit.get("_id") or source.get("chunk_id") or "|".join(
        [
            str(source.get("file_name", "")),
            str(source.get("page_number", "")),
            str(source.get("content_type", "")),
            str(source.get("content", ""))[:300],
        ]
    ))


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def rerank_results(query_text: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Lightweight lexical reranker for the final candidate set.

    It intentionally avoids another model call so the public demo remains cheap and deterministic.
    """
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return hits[:top_k]

    reranked = []
    for hit in hits:
        source = hit.get("_source", {})
        content_tokens = _tokenize(source.get("content", ""))
        overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
        base_score = float(hit.get("_score") or 0)
        rerank_score = (0.8 * base_score) + (0.2 * overlap)
        updated_hit = {**hit, "_score": rerank_score, "base_score": base_score, "rerank_score": rerank_score}
        reranked.append(updated_hit)

    return sorted(reranked, key=lambda hit: hit.get("rerank_score", 0), reverse=True)[:top_k]


def keyword_search(
    query_text: str, 
    index_name: str = DEFAULT_INDEX_NAME, 
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Perform keyword search using OpenSearch.
    
    Args:
        query_text: The query text to search for
        index_name: OpenSearch index name
        top_k: Number of results to return
    
    Returns:
        List of search results with metadata
    """
    if not query_text or not query_text.strip():
        print("Warning: Empty query provided to keyword_search")
        return []
    top_k = _normalize_top_k(top_k)

    try:
        client = get_opensearch_client()
    except Exception as e:
        print(f"✗ Cannot connect to OpenSearch: {e}")
        return []

    try:
        # Create a keyword search query
        search_query = {
            "size": top_k,
            "query": {
                "match": {
                    "content": {
                        "query": query_text,
                        "fuzziness": "AUTO"
                    }
                }
            },
            "_source": ["content", "content_type", "file_name", "page_number", "chunk_id"],
        }

        response = client.search(index=index_name, body=search_query)
        hits = response["hits"]["hits"]
        if RERANK_RESULTS:
            hits = rerank_results(query_text, hits, top_k)
        
        print(f"✓ Keyword search found {len(hits)} results")
        return hits
        
    except Exception as e:
        print(f"✗ Keyword search error: {str(e)}")
        return []


def semantic_search(
    query_text: str, 
    index_name: str = DEFAULT_INDEX_NAME, 
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using vector embeddings (KNN).
    
    Args:
        query_text: The query text to search for
        index_name: OpenSearch index name
        top_k: Number of results to return
    
    Returns:
        List of search results with metadata
    """
    if not query_text or not query_text.strip():
        print("Warning: Empty query provided to semantic_search")
        return []
    top_k = _normalize_top_k(top_k)

    try:
        client = get_opensearch_client()
    except Exception as e:
        print(f"✗ Cannot connect to OpenSearch: {e}")
        return []

    try:
        # Get embedding for the query using Ollama
        print(f"Generating embedding for query: '{query_text[:50]}...'")
        query_embedding = get_embedding(query_text)
        
        # Verify embedding
        if not query_embedding or len(query_embedding) != EMBEDDING_DIMENSION:
            print(
                f"✗ Invalid embedding: expected {EMBEDDING_DIMENSION} dimensions, "
                f"got {len(query_embedding) if query_embedding else 0}"
            )
            return []
        
        print(f"✓ Embedding generated successfully ({EMBEDDING_DIMENSION} dimensions)")

        # Create a semantic search query using KNN
        search_query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k
                    }
                }
            },
            "_source": ["content", "content_type", "file_name", "page_number", "chunk_id"],
        }

        response = client.search(index=index_name, body=search_query)
        hits = response["hits"]["hits"]
        if RERANK_RESULTS:
            hits = rerank_results(query_text, hits, top_k)
        
        print(f"✓ Semantic search found {len(hits)} results")
        return hits
        
    except Exception as e:
        print(f"✗ Semantic search error: {str(e)}")
        return []


def hybrid_search(
    query_text: str, 
    index_name: str = DEFAULT_INDEX_NAME, 
    top_k: int = 20,
    keyword_weight: float = 0.45,
    semantic_weight: float = 0.55,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining keyword and semantic search with weighted scoring.
    
    Args:
        query_text: The query text to search for
        index_name: OpenSearch index name
        top_k: Number of results to return
    
    Returns:
        List of search results with combined scores
    """
    if not query_text or not query_text.strip():
        print("Warning: Empty query provided to hybrid_search")
        return []
    top_k = _normalize_top_k(top_k)

    try:
        keyword_hits = keyword_search(query_text, index_name=index_name, top_k=top_k * 2)
        semantic_hits = semantic_search(query_text, index_name=index_name, top_k=top_k * 2)

        fused: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(keyword_hits, 1):
            key = _source_key(hit)
            fused.setdefault(key, {**hit, "_score": 0.0})
            fused[key]["_score"] += keyword_weight / (rank + 60)
            fused[key]["retrieval_sources"] = fused[key].get("retrieval_sources", []) + ["keyword"]

        for rank, hit in enumerate(semantic_hits, 1):
            key = _source_key(hit)
            fused.setdefault(key, {**hit, "_score": 0.0})
            fused[key]["_score"] += semantic_weight / (rank + 60)
            fused[key]["retrieval_sources"] = fused[key].get("retrieval_sources", []) + ["semantic"]

        hits = sorted(fused.values(), key=lambda hit: hit.get("_score", 0), reverse=True)
        if RERANK_RESULTS:
            hits = rerank_results(query_text, hits, top_k)
        else:
            hits = hits[:top_k]
        print(f"✓ Hybrid search found {len(hits)} fused results")
        return hits

    except Exception as e:
        print(f"✗ Hybrid search error: {str(e)}")
        print("Falling back to keyword search...")
        return keyword_search(query_text, index_name=index_name, top_k=top_k)


def format_search_results(
    results: List[Dict[str, Any]], 
    max_content_length: int = 300
) -> None:
    """
    Format and print search results for display.
    
    Args:
        results: Search results from OpenSearch
        max_content_length: Maximum content length to display
    
    Returns:
        None (prints formatted results)
    """
    if not results:
        print("No results found.")
        return

    print("="*80)
    print(f"SEARCH RESULTS ({len(results)} results)")
    print("="*80)

    for i, hit in enumerate(results, 1):
        score = hit.get("_score", 0)
        source = hit.get("_source", {})
        
        content = source.get("content", "")
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        content_type = source.get("content_type", "text")
        file_name = source.get("file_name", "unknown")
        page_number = source.get("page_number", "unknown")
        chunk_id = source.get("chunk_id") or hit.get("_id", "unknown")
        
        print(f"\n[Result {i}]")
        print(f"  Score: {score:.4f}")
        print(f"  Type: {content_type}")
        print(f"  File: {file_name}")
        print(f"  Page: {page_number}")
        print(f"  Chunk ID: {chunk_id}")
        print(f"  Content: {content}")

    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPENSEARCH RETRIEVAL TEST")
    print("="*80)

    query = "Compare RAG vs fine-tuning"
    index = "pdf_content_index"

    # Test keyword search
    print("\n1. KEYWORD SEARCH")
    print("-" * 40)
    keyword_results = keyword_search(query, index_name=index, top_k=5)
    format_search_results(keyword_results)

    # Test semantic search
    print("\n2. SEMANTIC SEARCH")
    print("-" * 40)
    semantic_results = semantic_search(query, index_name=index, top_k=5)
    format_search_results(semantic_results)

    # Test hybrid search
    print("\n3. HYBRID SEARCH")
    print("-" * 40)
    hybrid_results = hybrid_search(query, index_name=index, top_k=5)
    format_search_results(hybrid_results)
