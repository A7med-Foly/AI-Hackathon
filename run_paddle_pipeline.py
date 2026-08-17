"""
PaddleOCR Section Detection & JSON Export Pipeline Runner.
Processes extracted PaddleOCR Markdown (.md) and Metadata (.json) files,
generating structured hierarchical section tree and flat RAG chunks in JSON.
"""

import sys
import json
import argparse
import pathlib
from src.ingestion.paddle_section_parser import PaddleSectionDetector


def run_paddle_pipeline(
    md_path: str,
    json_path: str = None,
    output_json_path: str = "paddle_sections_output.json",
    doc_name: str = "type-2-diabetes-in-adults-management.pdf",
    source_url: str = "https://www.nice.org.uk/guidance/ng28",
    max_chunk_tokens: int = 600,
    min_chunk_tokens: int = 30,
    chunk_overlap_tokens: int = 100
):
    print(f"=== [PaddleOCR Section Detection Pipeline] Processing: {md_path} ===")
    
    md_file = pathlib.Path(md_path)
    if not md_file.exists():
        print(f"❌ Error: Markdown file not found at: {md_path}", file=sys.stderr)
        print("Please place your PaddleOCR .md file in the project folder or specify path via --md", file=sys.stderr)
        sys.exit(1)

    # 1. Read Markdown content
    with open(md_file, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 2. Run Section Detection Engine with Configurable Token Bounds & Overlap
    detector = PaddleSectionDetector(
        document_name=doc_name,
        source_url=source_url,
        max_chunk_tokens=max_chunk_tokens,
        min_chunk_tokens=min_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens
    )

    print(f"Running Multi-Pattern Section Detection Engine (Max: {max_chunk_tokens}, Min: {min_chunk_tokens}, Overlap: {chunk_overlap_tokens} tokens)...")
    result = detector.parse(markdown_text=markdown_text, metadata_json_path=json_path)

    # 3. Export output to JSON file
    output_file = pathlib.Path(output_json_path)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully generated output JSON file: {output_file.resolve()}")
    print("-" * 60)

    # 4. Print Summary Report
    info = result["document_info"]
    print("=== [Pipeline Execution Summary] ===")
    print(f"✅ Document Name            : {info['document_name']}")
    print(f"✅ Total Sections Detected  : {info['total_sections_detected']}")
    print(f"✅ Total RAG Chunks Generated: {info['total_chunks_generated']}")
    print(f"✅ Saved JSON Output File   : {output_json_path}")
    print("-" * 60)

    # 5. Display Hierarchy Tree Preview
    tree = result["hierarchy_tree"]
    if tree:
        print("🌲 Hierarchy Tree Preview (Top Sections):")
        for node in tree[:4]:
            sec_num = node["section_number"]
            sec_title = node["section_title"]
            sub_count = len(node.get("subsections", []))
            print(f"  ├── [{sec_num or 'Sec'}] {sec_title} (Page {node['page_number']}) -> {sub_count} subsections")
            for sub in node.get("subsections", [])[:2]:
                print(f"  │    ├── [{sub['section_number']}] {sub['section_title']}")

    # 6. Display Sample Flat Chunk JSON
    chunks = result["flat_chunks"]
    if chunks:
        print("\n📄 Sample RAG Flat Chunk JSON Preview:")
        print(json.dumps(chunks[0], indent=2, ensure_ascii=False))

    return result


def auto_detect_files() -> tuple[str, str]:
    """Finds .md and .json files in current working directory if available."""
    cwd = pathlib.Path.cwd()
    md_files = [f for f in cwd.glob("*.md") if f.name not in ("README.md", "task.md", "implementation_plan.md", "walkthrough.md")]
    json_files = [f for f in cwd.glob("*.json") if f.name not in ("package.json", "paddle_sections_output.json", "tsconfig.json")]

    md_path = str(md_files[0]) if md_files else "extracted_data.md"
    json_path = str(json_files[0]) if json_files else "metadata.json"
    return md_path, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaddleOCR Section Detection & JSON Export Engine")
    parser.add_argument("--md", type=str, help="Path to PaddleOCR extracted Markdown file (.md)")
    parser.add_argument("--json", type=str, help="Path to PaddleOCR metadata JSON file (.json)")
    parser.add_argument("--output", type=str, default="paddle_sections_output.json", help="Path for output JSON file")
    parser.add_argument("--max-tokens", type=int, default=600, help="Maximum allowed tokens per chunk (default: 600)")
    parser.add_argument("--min-tokens", type=int, default=30, help="Minimum tokens per chunk before merging (default: 30)")
    parser.add_argument("--overlap-tokens", type=int, default=100, help="Token overlap for split chunks (default: 100)")

    args = parser.parse_args()

    default_md, default_json = auto_detect_files()
    target_md = args.md or default_md
    target_json = args.json or default_json

    run_paddle_pipeline(
        md_path=target_md,
        json_path=target_json,
        output_json_path=args.output,
        max_chunk_tokens=args.max_tokens,
        min_chunk_tokens=args.min_tokens,
        chunk_overlap_tokens=args.overlap_tokens
    )
