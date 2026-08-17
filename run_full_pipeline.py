"""
End-to-End Clinical RAG Pipeline Runner.
Initializes PaddleOCR Ingestion, ChromaDB + BGE Vector Store, Hybrid BM25 RRF Retriever,
FastAPI API Backend, and Evidence-Grounded Generator.
"""

import os
import sys
import uvicorn
from src.generation.generator import ClinicalRAGGenerator


def main():
    print("=" * 70)
    print("🚀 Medical RAG End-to-End Clinical Decision Support Engine (Day 3)")
    print("=" * 70)

    # 1. Initialize Generator & Retriever
    generator = ClinicalRAGGenerator()
    generator.retriever.initialize()

    # 2. Run Sample Verification Queries
    test_queries = [
        "What is the first-line pharmacological treatment for adults with type 2 diabetes?",
        "HbA1c target for adults managed with lifestyle and a single drug"
    ]

    for q in test_queries:
        print(f"\n❓ Query: \"{q}\"")
        res = generator.generate(query=q, top_k=3)
        print(f"💡 Response:\n{res['answer']}\n")
        print(f"📌 Citations ({len(res['citations'])} chunks matched):")
        for cit in res['citations']:
            print(f"   - Section {cit['section_number']}: {cit['section_title']} (Page {cit['page_number']}) [RRF: {cit['rrf_score']:.4f}]")
        print("-" * 70)

    # 3. Launch FastAPI Server with Visual Evidence Grounding UI Panel
    print("\n🌐 Starting FastAPI Web Server at http://127.0.0.1:8000 ...")
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
