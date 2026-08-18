"""
Universal Document Ingestion & Indexing Pipeline CLI.
Ingests any target PaddleOCR (.json / .md) guideline into ChromaDB Vector Store.
Usage:
  python ingest_document.py \
    --json Guideline-for-the-pharmacological-treatment-of-hypertension-in-adults.json \
    --md Guideline-for-the-pharmacological-treatment-of-hypertension-in-adults.md \
    --doc-name "Guideline for the pharmacological treatment of hypertension in adults" \
    --doc-slug "hypertension" \
    --page-offset 12
"""

import os
import sys
import json
import argparse
import pathlib

from src.ingestion.paddle_section_parser import PaddleSectionDetector
from src.retrieval.vector_store import VectorStoreManager


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest any Medical Guideline OCR output into ChromaDB & BM25.")
    parser.add_argument("--json", required=True, help="Path to PaddleOCR .json layout file")
    parser.add_argument("--md", required=False, help="Path to PaddleOCR .md text file")
    parser.add_argument("--doc-name", default="Medical Guideline", help="Full title of the document")
    parser.add_argument("--doc-slug", default="hypertension", help="Short slug identifier (e.g. hypertension)")
    parser.add_argument("--page-offset", type=int, default=0, help="Number of front-matter cover pages before Page 1")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="SentenceTransformer model name")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="ChromaDB persistence directory")
    return parser.parse_args()


def main():
    args = parse_args()

    json_path = pathlib.Path(args.json)
    if not json_path.exists():
        print(f"❌ Error: JSON file '{args.json}' not found.")
        sys.exit(1)

    print("=" * 70)
    print(f"🏥 Universal Guideline Ingestion: {args.doc_name}")
    print("=" * 70)

    # 1. Parse JSON / MD into Chunks & Section Tree
    print(f"\n1️⃣ Parsing layout metadata from {args.json} (page_offset={args.page_offset})...")
    detector = PaddleSectionDetector(
        document_name=args.doc_name,
        source_url="https://www.who.int/publications/i/item/9789240033987",
        max_chunk_tokens=600,
        chunk_overlap_tokens=100,
        page_offset=args.page_offset
    )

    with open(json_path, "r", encoding="utf-8") as f:
        raw_pages = json.load(f)

    parsed_payload = detector.parse_from_pages(raw_pages)
    chunks = parsed_payload.get("flat_chunks", [])
    tree = parsed_payload.get("hierarchy_tree", [])

    output_filename = f"{args.doc_slug}_sections_output.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(parsed_payload, f, indent=2)

    print(f"   ✅ Section parsing complete: {len(tree)} root sections, {len(chunks)} chunks.")
    print(f"   💾 Saved section structure to '{output_filename}'")

    # 2. Build ChromaDB Dense Vector Index
    collection_name = f"med_guidelines_{args.doc_slug}_{args.embedding_model.replace('/', '_').replace('-', '_').replace('.', '_')}"
    print(f"\n2️⃣ Indexing dense vector embeddings into ChromaDB collection '{collection_name}'...")
    
    vec_manager = VectorStoreManager(
        persist_dir=args.chroma_dir,
        collection_name=collection_name,
        embedding_model_name=args.embedding_model
    )
    vec_manager.ingest_chunks(chunks)

    print("\n" + "=" * 70)
    print(f"🎉 Ingestion & Indexing Complete for '{args.doc_name}'!")
    print(f"   - ChromaDB Collection: {collection_name}")
    print(f"   - Section Output File: {output_filename}")
    print("=" * 70)


if __name__ == "__main__":
    main()
