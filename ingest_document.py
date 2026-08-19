"""
Universal Document Ingestion & Indexing Pipeline CLI.
Ingests Medical Guideline layout metadata into ChromaDB Vector Store.
Uses centralized parameters from src/config.py.

Usage:
    python ingest_document.py
"""

import sys
import json
import argparse
import pathlib

from src import config
from src.ingestion.paddle_section_parser import PaddleSectionDetector
from src.retrieval.vector_store import VectorStoreManager


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest Medical Guideline (PaddleOCR JSON layout) into ChromaDB & BM25 index.")
    parser.add_argument("--json", default=config.DEFAULT_OCR_JSON_PATH, help=f"Path to PaddleOCR layout JSON (default: {config.DEFAULT_OCR_JSON_PATH})")
    parser.add_argument("--doc-name", default=config.DEFAULT_DOCUMENT_NAME, help=f"Full title of document (default: '{config.DEFAULT_DOCUMENT_NAME}')")
    parser.add_argument("--doc-slug", default=config.DEFAULT_DOCUMENT_SLUG, help=f"Short slug identifier (default: '{config.DEFAULT_DOCUMENT_SLUG}')")
    parser.add_argument("--page-offset", type=int, default=config.DEFAULT_PAGE_OFFSET, help=f"Front-matter cover page offset (default: {config.DEFAULT_PAGE_OFFSET})")
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL_NAME, help=f"SentenceTransformer embedding model (default: {config.EMBEDDING_MODEL_NAME})")
    parser.add_argument("--chroma-dir", default=config.CHROMA_PERSIST_DIR, help=f"ChromaDB persistence directory (default: {config.CHROMA_PERSIST_DIR})")
    parser.add_argument("--output", default=config.DEFAULT_PROCESSED_JSON_PATH, help=f"Path for output section JSON (default: {config.DEFAULT_PROCESSED_JSON_PATH})")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f"🏥 Medical Guideline Ingestion: {args.doc_name}")
    print("=" * 70)

    json_path = pathlib.Path(args.json)
    if not json_path.exists():
        ocr_path = config.DATA_OCR_DIR / args.json
        if ocr_path.exists():
            json_path = ocr_path
        else:
            print(f"❌ Error: Layout JSON file '{args.json}' not found.", file=sys.stderr)
            sys.exit(1)

    print(f"\n1️⃣ Parsing layout metadata from '{json_path}' (page_offset={args.page_offset})...")
    detector = PaddleSectionDetector(
        document_name=args.doc_name,
        source_url=config.DEFAULT_SOURCE_URL,
        max_chunk_tokens=config.MAX_CHUNK_TOKENS,
        min_chunk_tokens=config.MIN_CHUNK_TOKENS,
        chunk_overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        page_offset=args.page_offset
    )

    with open(json_path, "r", encoding="utf-8") as f:
        raw_pages = json.load(f)

    parsed_payload = detector.parse_from_pages(raw_pages)
    chunks = parsed_payload.get("flat_chunks", [])
    tree = parsed_payload.get("hierarchy_tree", [])

    output_filepath = pathlib.Path(args.output)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(parsed_payload, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Section parsing complete: {len(tree)} root sections, {len(chunks)} chunks.")
    print(f"   💾 Saved section structure to '{output_filepath}'")

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
    print(f"   - Section Output File: {output_filepath}")
    print("=" * 70)


if __name__ == "__main__":
    main()
