"""
Medical RAG Ingestion Pipeline Entrypoint - Day 1 Setup & Verification
Demonstrates layout-aware PDF parsing and section-based chunking for medical guidelines.
"""

import sys
import pathlib
from src.ingestion import MedicalPDFParser, MedicalSectionChunker


def run_ingestion(pdf_path: str, source_url: str = None):
    print(f"=== [Day 1 Ingestion Pipeline] Starting processing for: {pdf_path} ===")
    
    # 1. Initialize Parser and Chunker
    parser = MedicalPDFParser(default_source_url=source_url or "https://www.nice.org.uk/guidance/ng28")
    chunker = MedicalSectionChunker(
        target_chunk_tokens=600,
        max_chunk_tokens=800,
        chunk_overlap_tokens=100
    )

    # 2. Step 1: Parse PDF into page markdown blocks
    print("Step 1: Extracting page-aware Markdown with PyMuPDF4LLM...")
    parsed_pages = parser.parse_pdf(pdf_path)
    print(f"-> Extracted {len(parsed_pages)} valid pages.")

    # 3. Step 2: Perform Section-Aware Chunking
    print("Step 2: Performing Section-Aware Chunking (400-800 tokens)...")
    chunks = chunker.create_chunks(parsed_pages)
    print(f"-> Generated {len(chunks)} compliant document chunks.\n")

    # 4. Verify Metadata Compliance & Token Statistics
    required_keys = {"document_name", "page_number", "section_title", "chunk_id", "source_url"}
    valid_metadata_count = 0
    token_counts = []

    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        missing = required_keys - set(meta.keys())
        if not missing:
            valid_metadata_count += 1
        else:
            print(f"⚠️ Warning: Chunk {i} missing metadata keys: {missing}")

        token_counts.append(meta.get("token_count", 0))

    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0

    print("=== [Pipeline Verification Report] ===")
    print(f"✅ Total Chunks Generated    : {len(chunks)}")
    print(f"✅ Metadata Compliance Rate : {valid_metadata_count} / {len(chunks)} ({valid_metadata_count/len(chunks)*100:.1f}%)")
    print(f"📊 Avg Chunk Size (Tokens)   : {avg_tokens:.1f}")
    print(f"📊 Min Chunk Size (Tokens)   : {min_tokens}")
    print(f"📊 Max Chunk Size (Tokens)   : {max_tokens}")
    print("-" * 50)

    # Sample Output Inspection
    if chunks:
        sample = chunks[0]
        print("🔍 Sample Chunk #1 Inspection:")
        print(f"  - Chunk ID      : {sample.metadata['chunk_id']}")
        print(f"  - Document      : {sample.metadata['document_name']}")
        print(f"  - Page Number   : {sample.metadata['page_number']}")
        print(f"  - Section Title : {sample.metadata['section_title']}")
        print(f"  - Token Count   : {sample.metadata['token_count']}")
        print(f"  - Source URL    : {sample.metadata['source_url']}")
        print("  - Content Snippet:\n")
        snippet = sample.page_content[:300].replace("\n", " ")
        print(f"    \"{snippet}...\"")
        print("=" * 50)

    return chunks


if __name__ == "__main__":
    pdf_filename = "data/raw/type-2-diabetes-in-adults-management.pdf"
    if not pathlib.Path(pdf_filename).exists():
        pdf_filename = "type-2-diabetes-in-adults-management.pdf"
        if not pathlib.Path(pdf_filename).exists():
            print(f"Error: {pdf_filename} does not exist.", file=sys.stderr)
            sys.exit(1)

    run_ingestion(pdf_filename)