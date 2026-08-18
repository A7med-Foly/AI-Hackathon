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
    print("🏥 Medical RAG End-to-End Clinical Decision Support Engine")
    print("=" * 70)

    # Launch FastAPI Server with Visual Evidence Grounding UI Panel
    print("\n🌐 Starting FastAPI Web Server at http://127.0.0.1:8000 ...")
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
