"""
End-to-End Clinical RAG Pipeline Runner.
Initializes PaddleOCR Ingestion, ChromaDB + BGE Vector Store, Hybrid BM25 RRF Retriever,
FastAPI API Backend, and Evidence-Grounded Generator.
"""

import os
import uvicorn
from src import config
from src.generation.generator import ClinicalRAGGenerator


def main():
    print("=" * 70)
    print("🏥 Medical RAG End-to-End Clinical Decision Support Engine")
    print("=" * 70)

    # Launch FastAPI Server with Visual Evidence Grounding UI Panel
    print(f"\n🌐 Starting FastAPI Web Server at http://{config.SERVER_HOST}:{config.SERVER_PORT} ...")
    uvicorn.run("src.api.server:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=False)


if __name__ == "__main__":
    main()
