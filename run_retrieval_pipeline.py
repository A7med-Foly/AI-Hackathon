"""
Clinical Retrieval Pipeline Runner - Day 2 Verification with BAAI/bge-small-en-v1.5.
Executes Hybrid BM25 + Dense Vector retrieval with Reciprocal Rank Fusion on NICE Medical Guidelines.
"""

import sys
import json
import argparse
from src.retrieval.retriever import ClinicalRetriever


def format_evidence_card(idx: int, item: dict):
    meta = item.get("metadata", item)
    content = item.get("content", meta.get("content", ""))
    sec_num = meta.get("section_number", "")
    sec_title = meta.get("section_title", "")
    page_num = meta.get("page_number", 1)
    rrf_score = item.get("rrf_score")
    score = item.get("score")
    layout_meta = meta.get("layout_metadata", {})
    bboxes = layout_meta.get("bounding_boxes", [])

    header_str = f"[{idx}] "
    if sec_num:
        header_str += f"Section {sec_num}: "
    header_str += f"{sec_title} (Page {page_num})"

    print(f"\n📌 {header_str}")
    if rrf_score is not None:
        dense_r = item.get("dense_rank")
        bm25_r = item.get("bm25_rank")
        print(f"   Score: {rrf_score:.4f} (RRF Fusion) | Dense Rank: #{dense_r or '-'} | BM25 Rank: #{bm25_r or '-'}")
    elif score is not None:
        print(f"   Score: {score:.4f}")

    print(f"   Hierarchy: {' > '.join(meta.get('hierarchy_path', []))}")
    print(f"   Evidence Snippet: {content[:180]}...")
    if bboxes:
        bbox_sample = bboxes[0].get("bbox", [])
        print(f"   Layout Ground-Truth: {len(bboxes)} bounding boxes matched (e.g. BBox {bbox_sample})")
    if layout_meta.get("page_image_url"):
        print(f"   Visual Evidence URL: {layout_meta['page_image_url'][:80]}...")
    print("-" * 75)


def run_retrieval_pipeline(query: str = None, top_k: int = 3, mode: str = "hybrid", model_name: str = "BAAI/bge-small-en-v1.5"):
    retriever = ClinicalRetriever(
        json_chunks_path="paddle_sections_output.json",
        embedding_model_name=model_name
    )
    retriever.initialize()

    test_queries = [
        "What is the first-line pharmacological treatment for adults with type 2 diabetes?",
        "When should SGLT-2 inhibitors be prescribed for diabetes?",
        "HbA1c target for adults managed with lifestyle and a single drug"
    ]

    if query:
        queries_to_run = [query]
    else:
        queries_to_run = test_queries

    print(f"\n==============================================================")
    print(f"🔍 [Medical RAG Clinical Retrieval Engine] Model: {model_name} | Mode: {mode.upper()}")
    print(f"==============================================================")

    for q in queries_to_run:
        print(f"\n❓ Query: \"{q}\"")
        results = retriever.retrieve(query=q, top_k=top_k, mode=mode)
        
        if not results:
            print("   ⚠️ No matching evidence chunks found.")
            continue

        for idx, item in enumerate(results, start=1):
            format_evidence_card(idx, item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Retrieval Engine CLI")
    parser.add_argument("--query", type=str, help="Custom clinical search query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of evidence chunks to return (default: 3)")
    parser.add_argument("--mode", type=str, choices=["hybrid", "dense", "bm25"], default="hybrid", help="Retrieval mode")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model name")

    args = parser.parse_args()
    run_retrieval_pipeline(query=args.query, top_k=args.top_k, mode=args.mode, model_name=args.model)
