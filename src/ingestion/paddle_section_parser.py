"""
PaddleOCR Section Detector & Metadata Merger for Medical Guidelines.
Extracts hierarchical sections from PaddleOCR .md and .json metadata files using Multiline Regex & Multiline Block Aggregation.
Provides configurable chunk size bounds (max_chunk_tokens, min_chunk_tokens, chunk_overlap_tokens) for RAG embedding optimization.
"""

import json
import re
import tiktoken
from typing import Dict, List, Any, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Pre-compile Regex Patterns with MULTILINE (re.M) & IGNORECASE (re.I) Flags
TOC_LINE_PATTERN = re.compile(r'\.{3,}\s*\d+$', re.MULTILINE)
RUNNING_HEADER_PATTERN = re.compile(r'Type 2 diabetes in adults:\s*management\s*\([^)]*\)', re.IGNORECASE | re.MULTILINE)
COPYRIGHT_FOOTER_PATTERN = re.compile(r'©\s*NICE\s*\d{4}', re.IGNORECASE | re.MULTILINE)

# Section & Recommendation Number Regex (matching start-of-line in multiline mode)
SECTION_NUM_PATTERN = re.compile(r'^(?:Section\s+)?(\d+(?:\.\d+)+|\d+\.)\s+(.+)$', re.IGNORECASE | re.MULTILINE)
RECOMMENDATION_NUM_PATTERN = re.compile(r'^(\d+(?:\.\d+){2,})\s+(.+)$', re.MULTILINE)

# Clinical Keywords
CLINICAL_KEYWORDS = [
    "Rationale and impact",
    "Why the committee made the recommendations",
    "How recommendations might affect practice",
    "Terms used in this guideline",
    "Recommendations for research",
    "Key recommendations for research",
    "Your responsibility",
    "Using this guideline",
    "Contents"
]


class PaddleSectionDetector:
    def __init__(
        self,
        document_name: str = "type-2-diabetes-in-adults-management.pdf",
        source_url: str = "https://www.nice.org.uk/guidance/ng28",
        max_chunk_tokens: int = 600,
        min_chunk_tokens: int = 30,
        chunk_overlap_tokens: int = 100,
        tokenizer_model: str = "cl100k_base"
    ):
        self.document_name = document_name
        self.source_url = source_url
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        except Exception:
            self.tokenizer = None

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_tokens,
            chunk_overlap=chunk_overlap_tokens,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " "]
        )

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())

    def _is_noise_line(self, line: str) -> bool:
        """Determines if a line is a Table of Contents entry or running header/footer using multiline regex."""
        clean = line.strip()
        if not clean:
            return True
        if TOC_LINE_PATTERN.search(clean):
            return True
        if RUNNING_HEADER_PATTERN.search(clean) and len(clean) < 70:
            return True
        if COPYRIGHT_FOOTER_PATTERN.search(clean):
            return True
        return False

    def parse_from_pages(self, raw_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parses page objects directly from PaddleOCR JSON metadata array, aggregating multiline section blocks."""
        sections = []
        current_section = None
        current_l1 = ""
        current_l2 = ""

        for page_idx, page_obj in enumerate(raw_pages):
            page_num = page_idx + 1

            # Extract Page-Level Metadata
            if isinstance(page_obj, dict):
                raw_text = page_obj.get("markdown", {}).get("text", "")
                page_image_url = page_obj.get("inputImage", "")
                layout_image_url = page_obj.get("outputImages", {}).get("layout_det_res", "")
                page_images = page_obj.get("markdown", {}).get("images", {})
                parsing_res_list = page_obj.get("prunedResult", {}).get("parsing_res_list", [])
            else:
                raw_text = str(page_obj)
                page_image_url = ""
                layout_image_url = ""
                page_images = {}
                parsing_res_list = []

            lines = raw_text.split('\n')
            
            for line in lines:
                clean = line.strip()
                if self._is_noise_line(clean):
                    continue

                is_header, sec_num, sec_title, level = self._detect_heading(clean)

                if is_header:
                    if current_section and "\n".join(current_section["content_lines"]).strip():
                        sections.append(current_section)

                    if level == 1:
                        current_l1 = sec_title
                        current_l2 = ""
                        parent = ""
                        h_path = [sec_title]
                    elif level == 2:
                        current_l2 = f"{sec_num} {sec_title}".strip() if sec_num else sec_title
                        parent = current_l1
                        h_path = [p for p in [current_l1, current_l2] if p]
                    else:
                        parent = current_l2 or current_l1
                        curr_str = f"{sec_num} {sec_title}".strip() if sec_num else sec_title
                        h_path = [p for p in [current_l1, current_l2, curr_str] if p]

                    # Start new multiline section block
                    current_section = {
                        "section_number": sec_num,
                        "section_title": sec_title,
                        "parent_section": parent,
                        "hierarchy_path": h_path,
                        "page_number": page_num,
                        "level": level,
                        "content_lines": [clean],
                        "page_image_url": page_image_url,
                        "layout_image_url": layout_image_url,
                        "page_images": page_images,
                        "parsing_res_list": parsing_res_list
                    }
                else:
                    # Append non-header lines to current section's multiline block
                    if current_section:
                        current_section["content_lines"].append(clean)
                    else:
                        # Initial introductory section
                        current_l1 = self.document_name.replace(".pdf", "")
                        current_section = {
                            "section_number": "",
                            "section_title": "Overview",
                            "parent_section": "",
                            "hierarchy_path": [current_l1],
                            "page_number": page_num,
                            "level": 1,
                            "content_lines": [clean],
                            "page_image_url": page_image_url,
                            "layout_image_url": layout_image_url,
                            "page_images": page_images,
                            "parsing_res_list": parsing_res_list
                        }

        if current_section and "\n".join(current_section["content_lines"]).strip():
            sections.append(current_section)

        # Build outputs with merged metadata and configured chunk size bounds
        return self._build_export_payload(sections)

    def parse(self, markdown_text: str, metadata_json_path: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point: parses metadata JSON if provided, merging all metadata into chunks."""
        if metadata_json_path:
            try:
                with open(metadata_json_path, "r", encoding="utf-8") as f:
                    pages = json.load(f)

                if isinstance(pages, list) and len(pages) > 0:
                    return self.parse_from_pages(pages)
            except Exception as e:
                print(f"⚠️ Warning: Could not parse metadata JSON directly ({e}). Fallback to markdown text parser.")

        # Fallback to single text blob parsing
        return self._parse_text_blob(markdown_text)

    def _detect_heading(self, line: str) -> Tuple[bool, str, str, int]:
        """Detects if a line is a heading and determines section number, title, and hierarchy level."""
        clean = line.strip()

        # 1. Markdown Headers (#, ##, ###, ####)
        if clean.startswith('#'):
            hashes = len(clean) - len(clean.lstrip('#'))
            title_part = clean.lstrip('#').strip()
            level = min(hashes, 3)

            # Check if title contains section number (e.g., "## 1.4 Bariatric surgery")
            num_match = SECTION_NUM_PATTERN.match(title_part)
            if num_match:
                return True, num_match.group(1), num_match.group(2), level
            return True, "", title_part, level

        # 2. Numbered Section (e.g. "1.4.1 Lifestyle modifications")
        rec_match = RECOMMENDATION_NUM_PATTERN.match(clean)
        if rec_match:
            return True, rec_match.group(1), rec_match.group(2), 3

        sec_match = SECTION_NUM_PATTERN.match(clean)
        if sec_match and len(clean) < 120:
            return True, sec_match.group(1), sec_match.group(2), 2

        # 3. Clinical Keywords
        for kw in CLINICAL_KEYWORDS:
            if clean.lower() == kw.lower() or clean.lower().startswith(kw.lower() + ":"):
                return True, "", clean, 2

        return False, "", "", 0

    def _parse_text_blob(self, markdown_text: str) -> Dict[str, Any]:
        """Parses a full markdown text string when page-by-page metadata is unavailable."""
        page_texts = markdown_text.split("\n\n---\n\n")
        raw_pages = [{"markdown": {"text": txt}} for txt in page_texts]
        return self.parse_from_pages(raw_pages)

    def _match_layout_metadata(self, sec: Dict[str, Any], content: str) -> Dict[str, Any]:
        """Matches content lines against layout blocks to attach bounding boxes, labels, and image URLs."""
        parsing_list = sec.get("parsing_res_list", [])
        matched_boxes = []
        labels_set = set()

        content_lines = [l.strip() for l in sec.get("content_lines", []) if l.strip()]

        for block in parsing_list:
            b_content = block.get("block_content", "").strip()
            b_label = block.get("block_label", "")
            b_bbox = block.get("block_bbox", [])
            b_polygon = block.get("block_polygon_points", [])

            if not b_content:
                continue

            # Check if block overlaps with chunk content lines
            matched = False
            for line in content_lines[:5]:  # check first few lines
                if line in b_content or b_content in line or line.startswith(b_content[:30]):
                    matched = True
                    break

            if matched:
                if b_label:
                    labels_set.add(b_label)
                matched_boxes.append({
                    "block_id": block.get("block_id"),
                    "label": b_label,
                    "bbox": b_bbox,
                    "polygon": b_polygon
                })

        return {
            "block_labels": list(labels_set) if labels_set else ["text"],
            "bounding_boxes": matched_boxes,
            "page_image_url": sec.get("page_image_url", ""),
            "layout_image_url": sec.get("layout_image_url", ""),
            "images": sec.get("page_images", {})
        }

    def _build_export_payload(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Converts internal sections list into structured recursive tree and flat chunks with configured chunk size limits and token overlap."""
        flat_chunks = []

        for idx, sec in enumerate(sections):
            content = "\n".join(sec["content_lines"]).strip()
            if not content:
                continue

            token_count = self._count_tokens(content)
            layout_meta = self._match_layout_metadata(sec, content)

            # Handle chunk merging if chunk is under min_chunk_tokens
            if token_count < self.min_chunk_tokens and flat_chunks:
                prev = flat_chunks[-1]
                if prev["page_number"] == sec["page_number"]:
                    prev["content"] += "\n" + content
                    prev["token_count"] = self._count_tokens(prev["content"])
                    continue

            # Handle chunk splitting if chunk exceeds max_chunk_tokens (with chunk_overlap_tokens overlap)
            if token_count > self.max_chunk_tokens:
                sub_texts = self.text_splitter.split_text(content)
                for sub_i, sub_txt in enumerate(sub_texts):
                    chunk_id = f"paddle_p{sec['page_number']}_c{idx + 1}_sub{sub_i + 1}"
                    sub_chunk = {
                        "chunk_id": chunk_id,
                        "section_number": sec["section_number"],
                        "section_title": sec["section_title"],
                        "parent_section": sec["parent_section"],
                        "hierarchy_path": sec["hierarchy_path"],
                        "page_number": sec["page_number"],
                        "content": sub_txt,
                        "token_count": self._count_tokens(sub_txt),
                        "document_name": self.document_name,
                        "source_url": self.source_url,
                        "layout_metadata": layout_meta
                    }
                    flat_chunks.append(sub_chunk)
            else:
                chunk_id = f"paddle_p{sec['page_number']}_c{idx + 1}"
                chunk = {
                    "chunk_id": chunk_id,
                    "section_number": sec["section_number"],
                    "section_title": sec["section_title"],
                    "parent_section": sec["parent_section"],
                    "hierarchy_path": sec["hierarchy_path"],
                    "page_number": sec["page_number"],
                    "content": content,
                    "token_count": token_count,
                    "document_name": self.document_name,
                    "source_url": self.source_url,
                    "layout_metadata": layout_meta
                }
                flat_chunks.append(chunk)

        # Build Hierarchical Tree View
        tree = self._build_recursive_tree(sections)

        return {
            "document_info": {
                "document_name": self.document_name,
                "source_url": self.source_url,
                "total_sections_detected": len(flat_chunks),
                "total_chunks_generated": len(flat_chunks)
            },
            "hierarchy_tree": tree,
            "flat_chunks": flat_chunks
        }

    def _build_recursive_tree(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Constructs a nested hierarchy tree from flat section items, attaching layout metadata."""
        root_nodes = []
        last_l1 = None
        last_l2 = None

        for sec in sections:
            content = "\n".join(sec["content_lines"]).strip()
            if not content:
                continue

            layout_meta = self._match_layout_metadata(sec, content)

            node = {
                "section_number": sec["section_number"],
                "section_title": sec["section_title"],
                "level": sec["level"],
                "page_number": sec["page_number"],
                "content": content,
                "layout_metadata": layout_meta,
                "subsections": []
            }

            if sec["level"] == 1:
                root_nodes.append(node)
                last_l1 = node
                last_l2 = None
            elif sec["level"] == 2:
                if last_l1:
                    last_l1["subsections"].append(node)
                else:
                    root_nodes.append(node)
                last_l2 = node
            else:
                if last_l2:
                    last_l2["subsections"].append(node)
                elif last_l1:
                    last_l1["subsections"].append(node)
                else:
                    root_nodes.append(node)

        return root_nodes
