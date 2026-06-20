"""
Run lightweight evaluation examples for the RAG pipeline.

By default this checks whether retrieved chunks contain the expected source
hints from eval_examples.json. Pass --generate to also call Gemini and check
simple expected-answer trait keywords.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from generation import generate_rag_response
from retrieval import hybrid_search, keyword_search, semantic_search


def load_examples(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def retrieve(question, search_type, top_k):
    if search_type == "keyword":
        return keyword_search(question, top_k=top_k)
    if search_type == "semantic":
        return semantic_search(question, top_k=top_k)
    return hybrid_search(question, top_k=top_k)


def source_hint_score(hits, expected_hints):
    retrieved_text = "\n".join(
        hit.get("_source", {}).get("content", "")
        for hit in hits
    ).lower()
    matched = [
        hint for hint in expected_hints
        if hint.lower() in retrieved_text
    ]
    total = len(expected_hints)
    return {
        "matched": matched,
        "missing": [hint for hint in expected_hints if hint not in matched],
        "score": len(matched) / total if total else 1.0,
    }


def answer_trait_score(answer, expected_traits):
    answer_text = answer.lower()
    matched = []
    missing = []

    for trait in expected_traits:
        trait_terms = [term for term in trait.lower().replace("-", " ").split() if len(term) > 4]
        if any(term in answer_text for term in trait_terms):
            matched.append(trait)
        else:
            missing.append(trait)

    total = len(expected_traits)
    return {
        "matched": matched,
        "missing": missing,
        "score": len(matched) / total if total else 1.0,
    }


def evaluate_examples(examples, search_type="hybrid", top_k=5, generate=False, model_name="gemini-2.0-flash"):
    results = []

    for example in examples:
        question = example["question"]
        hits = retrieve(question, search_type=search_type, top_k=top_k)
        source_score = source_hint_score(hits, example.get("expected_source_hints", []))

        answer = None
        answer_score = None
        if generate:
            answer = generate_rag_response(
                question,
                search_type=search_type,
                top_k=top_k,
                stream=False,
                model_name=model_name,
            )
            answer_score = answer_trait_score(answer, example.get("expected_answer_traits", []))

        results.append(
            {
                "id": example["id"],
                "question": question,
                "retrieved_count": len(hits),
                "source_hints": source_score,
                "answer_traits": answer_score,
                "answer_preview": answer[:500] if answer else None,
            }
        )

    return results


def print_summary(results):
    source_scores = [item["source_hints"]["score"] for item in results]
    answer_scores = [
        item["answer_traits"]["score"]
        for item in results
        if item["answer_traits"] is not None
    ]

    print("Evaluation summary")
    print(f"- Examples: {len(results)}")
    print(f"- Avg source hint score: {sum(source_scores) / len(source_scores):.2f}" if source_scores else "- Avg source hint score: n/a")
    if answer_scores:
        print(f"- Avg answer trait score: {sum(answer_scores) / len(answer_scores):.2f}")

    for item in results:
        print(
            f"- {item['id']}: sources={item['source_hints']['score']:.2f}, "
            f"retrieved={item['retrieved_count']}"
        )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate RAG examples.")
    parser.add_argument("--examples", default="eval_examples.json", help="path to evaluation examples JSON")
    parser.add_argument("--output", default="eval_results.json", help="path to write JSON results")
    parser.add_argument("--search-type", choices=["keyword", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="also generate answers with Gemini")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    args = parser.parse_args()

    examples = load_examples(args.examples)
    results = evaluate_examples(
        examples,
        search_type=args.search_type,
        top_k=args.top_k,
        generate=args.generate,
        model_name=args.model,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
