"""
Universal Evaluation Runner for Medical RAG Framework.
Evaluates Retrieval Engine (Hit Rate@K, MRR@K, Precision@K, Recall@K)
across Hybrid, Dense, and BM25 search modes, and evaluates RAG Generation Quality.

Usage:
  python run_evaluation.py --dataset data/eval_dataset.json --top-k 4
"""

import sys
import json
import argparse
import pathlib
from typing import List, Dict, Any

from src import config
from src.retrieval.retriever import ClinicalRetriever
from src.generation.generator import ClinicalRAGGenerator
from src.evaluation.evaluator import ClinicalRAGEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Medical RAG Retrieval & Generation Quality")
    parser.add_argument("--dataset", default=config.EVAL_DATASET_PATH, help=f"Path to evaluation dataset JSON (default: {config.EVAL_DATASET_PATH})")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_EVAL_TOP_K, help=f"Top K retrieved chunks for evaluation (default: {config.DEFAULT_EVAL_TOP_K})")
    parser.add_argument("--output", default=config.EVAL_RESULTS_PATH, help=f"Path to output evaluation results JSON (default: {config.EVAL_RESULTS_PATH})")
    parser.add_argument("--skip-generation", action="store_true", help="Skip LLM generation step for faster retrieval-only evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("🏥 Medical RAG Quantitative Evaluation & Benchmarking Suite")
    print("=" * 80)

    evaluator = ClinicalRAGEvaluator(dataset_path=args.dataset)
    dataset = evaluator.dataset

    if not dataset:
        print(f"❌ Error: Evaluation dataset at '{args.dataset}' is empty or missing.")
        sys.exit(1)

    print(f"📋 Loaded {len(dataset)} benchmark queries from '{args.dataset}'")
    print(f"⚙️ Evaluation Settings: Top K = {args.top_k} | Search Modes: [hybrid, dense, bm25]\n")

    # Initialize Retrievers for both documents
    retrievers = {
        "hypertension": ClinicalRetriever(
            json_chunks_path=str(config.DATA_PROCESSED_DIR / "hypertension_sections_output.json"),
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name=config.DEFAULT_COLLECTION_NAME,
            embedding_model_name=config.EMBEDDING_MODEL_NAME
        ),
        "diabetes": ClinicalRetriever(
            json_chunks_path=str(config.DATA_PROCESSED_DIR / "paddle_sections_output.json"),
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name="med_guidelines_BAAI_bge_small_en_v1_5",
            embedding_model_name=config.EMBEDDING_MODEL_NAME
        )
    }

    # Initialize Generator for RAG generation quality evaluation
    generator = None
    if not args.skip_generation:
        generator = ClinicalRAGGenerator(
            retriever=retrievers["hypertension"],
            model_name=config.DEFAULT_LLM_MODEL
        )

    mode_results: Dict[str, List[Dict[str, Any]]] = {
        "hybrid": [],
        "dense": [],
        "bm25": []
    }
    generation_results: List[Dict[str, Any]] = []

    print("🔍 Running Retrieval & Generation Benchmarks...")
    print("-" * 80)

    for item in dataset:
        qid = item["query_id"]
        doc_key = item.get("document", "hypertension")
        question = item["question"]

        retriever = retrievers.get(doc_key, retrievers["hypertension"])

        print(f"▶ Query [{qid}] ({doc_key}): \"{question}\"")

        # Evaluate across search modes
        for mode in ["hybrid", "dense", "bm25"]:
            chunks = retriever.retrieve(query=question, top_k=args.top_k, mode=mode)
            ret_metrics = evaluator.evaluate_retrieval_query(item, chunks, top_k=args.top_k)
            ret_metrics["mode"] = mode
            mode_results[mode].append(ret_metrics)

        # Evaluate RAG Generation quality (on hybrid mode)
        if generator and not args.skip_generation:
            # Switch generator's retriever to target document
            generator.retriever = retriever
            gen_res = generator.generate(query=question, top_k=args.top_k, mode="hybrid")
            answer = gen_res.get("answer", "")
            retrieved_chunks = gen_res.get("citations", [])

            gen_metrics = evaluator.evaluate_generation_query(item, answer, retrieved_chunks)
            gen_metrics["generated_answer"] = answer
            generation_results.append(gen_metrics)

    # Compute Summary Statistics
    summary_by_mode = {}
    for mode, res in mode_results.items():
        summary_by_mode[mode] = evaluator.summarize_results(res)

    gen_summary = evaluator.summarize_results(generation_results) if generation_results else {}

    # Print Comparative Benchmark Table
    print("\n" + "=" * 80)
    print("📊 COMPARATIVE RETRIEVAL BENCHMARK RESULTS (Top-K = {})".format(args.top_k))
    print("=" * 80)
    print("{:<12} | {:<12} | {:<12} | {:<12} | {:<12}".format("Search Mode", "Hit Rate @K", "MRR @K", "Precision @K", "Recall @K"))
    print("-" * 72)
    for mode in ["hybrid", "dense", "bm25"]:
        s = summary_by_mode[mode]
        print("{:<12} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f}".format(
            mode.upper(), s["mean_hit_rate"], s["mean_mrr"], s["mean_precision"], s["mean_recall"]
        ))
    print("-" * 72)

    if generation_results:
        print("\n" + "=" * 80)
        print("🤖 RAG GENERATION QUALITY METRICS (LLM Response)")
        print("=" * 80)
        print(f"  • Faithfulness / Groundedness : {gen_summary.get('mean_faithfulness', 0.0):.4f}")
        print(f"  • Answer Relevance           : {gen_summary.get('mean_answer_relevance', 0.0):.4f}")
        citation_rate = sum(1 for g in generation_results if g.get("citation_present")) / max(len(generation_results), 1)
        print(f"  • Citation Grounding Rate    : {citation_rate:.4f}")
        print("=" * 80)

    # Save Output JSON
    final_report = {
        "dataset_size": len(dataset),
        "top_k": args.top_k,
        "summary_by_mode": summary_by_mode,
        "generation_summary": gen_summary,
        "mode_detailed_results": mode_results,
        "generation_detailed_results": generation_results
    }

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Full evaluation report saved to: '{out_path}'\n")


if __name__ == "__main__":
    main()
