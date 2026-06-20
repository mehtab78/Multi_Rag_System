"""
Validate local configuration for the RAG demo.

This command avoids printing secret values. Use --services to also check
OpenSearch and Ollama connectivity.
"""

import argparse
import os
import sys

from dotenv import load_dotenv


REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]


def validate_env() -> list[str]:
    load_dotenv()
    issues = []

    for key in REQUIRED_ENV_VARS:
        value = os.getenv(key)
        if not value or value.strip() in {"", "your_gemini_api_key_here"}:
            issues.append(f"{key} is missing or still set to the placeholder value.")

    try:
        embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
        if embedding_dimension <= 0:
            issues.append("EMBEDDING_DIMENSION must be a positive integer.")
    except ValueError:
        issues.append("EMBEDDING_DIMENSION must be an integer.")

    return issues


def validate_services() -> list[str]:
    from helper import check_ollama_health, check_opensearch_health

    issues = []
    if not check_opensearch_health():
        issues.append("OpenSearch is not reachable.")
    if not check_ollama_health():
        issues.append("Ollama is not reachable.")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LocalRAG setup.")
    parser.add_argument("--services", action="store_true", help="also check OpenSearch and Ollama")
    args = parser.parse_args()

    issues = validate_env()
    if args.services:
        issues.extend(validate_services())

    if issues:
        print("Setup check failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Setup check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
