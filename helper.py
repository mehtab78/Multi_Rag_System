"""
Helper utilities for the Multi-RAG System.

This module provides functions for:
- Generating embeddings via Ollama API
- Managing OpenSearch client connections
"""

import os
import requests
from typing import Optional, List
from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))


def get_embedding(prompt: str, model: str = EMBEDDING_MODEL) -> Optional[List[float]]:
    """
    Generate an embedding vector for the given prompt using Ollama API.
    
    Args:
        prompt: The text prompt to embed
        model: Ollama model to use (default: nomic-embed-text)
    
    Returns:
        List of embedding dimensions, or None on failure
    """
    if not prompt or not prompt.strip():
        print("Warning: Empty prompt provided for embedding.")
        return None

    ollama_host = os.getenv("OLLAMA_HOST", "localhost")
    ollama_port = os.getenv("OLLAMA_PORT", "11434")
    url = f"http://{ollama_host}:{ollama_port}/api/embeddings"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        embedding = result.get("embedding")
        
        if embedding is None:
            print(f"Warning: No embedding returned from Ollama for prompt: {prompt[:50]}...")
            return None
            
        if len(embedding) != EMBEDDING_DIMENSION:
            print(
                "Warning: Embedding dimension mismatch. "
                f"Expected {EMBEDDING_DIMENSION}, got {len(embedding)}"
            )
            return None
            
        return embedding
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Ollama at {ollama_host}:{ollama_port}")
        print("Ensure Ollama is running: docker ps | grep ollama")
        return None
    except requests.exceptions.Timeout:
        print(f"Error: Timeout connecting to Ollama at {ollama_host}:{ollama_port}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error from Ollama: {e}")
        return None
    except ValueError as e:
        print(f"Error: Invalid JSON response from Ollama: {e}")
        return None
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None


def get_opensearch_client(
    host: Optional[str] = None, 
    port: Optional[int] = None, 
    log_connection: bool = False
) -> OpenSearch:
    """
    Create and return an OpenSearch client connection.
    
    Args:
        host: OpenSearch host (default: from OPENSEARCH_HOST env var or localhost)
        port: OpenSearch port (default: from OPENSEARCH_PORT env var or 9200)
        log_connection: Whether to log connection status
    
    Returns:
        OpenSearch client instance
    
    Raises:
        ConnectionError: If cannot connect to OpenSearch
    """
    host = host or os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(port or os.getenv("OPENSEARCH_PORT", "9200"))

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        timeout=30,
        max_retries=4,
        retry_on_timeout=True,
        use_ssl=False,
        verify_certs=False
    )

    if log_connection:
        try:
            if client.ping():
                print(f"✓ Connected to OpenSearch at {host}:{port}")
            else:
                print(f"⚠ OpenSearch responded but ping failed at {host}:{port}")
        except Exception as e:
            print(f"✗ Failed to connect to OpenSearch at {host}:{port}: {e}")
            raise ConnectionError(f"Cannot connect to OpenSearch at {host}:{port}") from e

    return client


def check_ollama_health() -> bool:
    """
    Check if Ollama service is available and healthy.
    
    Returns:
        True if Ollama is healthy, False otherwise
    """
    ollama_host = os.getenv("OLLAMA_HOST", "localhost")
    ollama_port = os.getenv("OLLAMA_PORT", "11434")
    url = f"http://{ollama_host}:{ollama_port}/api/tags"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return True
    except Exception:
        return False


def check_opensearch_health() -> bool:
    """
    Check if OpenSearch service is available and healthy.
    
    Returns:
        True if OpenSearch is healthy, False otherwise
    """
    try:
        client = get_opensearch_client()
        return client.ping()
    except Exception:
        return False


if __name__ == "__main__":
    # Test connections
    print("="*50)
    print("Testing Service Connections")
    print("="*50)
    
    # Test OpenSearch
    print("\nTesting OpenSearch...")
    try:
        client = get_opensearch_client(log_connection=True)
    except ConnectionError as e:
        print(f"✗ OpenSearch connection failed: {e}")
    
    # Test Ollama
    print("\nTesting Ollama...")
    if check_ollama_health():
        print("✓ Ollama is healthy")
        # Test embedding
        test_embedding = get_embedding("test")
        if test_embedding:
            print(f"✓ Embedding generated: {len(test_embedding)} dimensions")
    else:
        print("✗ Ollama is not available")
