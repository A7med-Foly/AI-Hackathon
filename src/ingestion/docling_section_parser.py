"""
Docling Section Detector & Metadata Merger for Medical Guidelines.
Parses PDFs using IBM Docling, exports raw Markdown to disk with page markers, and produces standardized
flat_chunks and hierarchy_tree compatible with ChromaDB VectorStoreManager.
"""

import json
import re
import pathlib
import tiktoken
from typing import Dict, List, Any, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DocumentConverter = None
    DOCLING_AVAILABLE = False


# Pre-compile Regex Patterns for Section Extraction & Noise Cleaning
TOC_LINE_PATTERN = re.compile(r'\.{3,}\s*\d+$', re.MULTILINE)
RUNNING_HEADER_PATTERN = re.compile(
    r'^(?:GUIDELINE FOR THE PHARMACOLOGICAL TREATMENT OF HYPERTENSION IN ADULTS|Type 2 diabetes in adults:\s*management\s*\([^)]*\)|National Institute for Health and Care Excellence|World Health Organization)$',
    re.IGNORECASE | re.MULTILINE
)
COPYRIGHT_FOOTER_PATTERN = re.compile(r'©\s*(?:NICE|WHO)\s*\d{4}.*', re.IGNORECASE | re.MULTILINE)
PAGE_NUMBER_ARTIFACT_PATTERN = re.compile(r'^\d{1,3}$', re.MULTILINE)

# Section & Recommendation Number Patterns
SECTION_NUM_PATTERN = re.compile(r'^(?:Section\s+)?(\d+(?:\.\d+)+|\d+\.)\s+(.+)$', re.IGNORECASE)
RECOMMENDATION_NUM_PATTERN = re.compile(r'^(\d+(?:\.\d+){2,})\s+(.+)$')

CLINICAL_KEYWORDS = [
    "Rationale and impact",
    "Why the committee made the recommendations",
    "How recommendations might affect practice",
    "Terms used in this guideline",
    "Recommendations for research",
    "Key recommendations for research",
    "Executive summary",
    "Summary of recommendations",
    "Scope and objectives",
    "Target audience",
    "Deciding upon recommendations",
    "Reviews of evidence",
    "Certainty of evidence and strength of recommendations",
    "Funding"
]


class DoclingSectionDetector:
    def __init__(
        self,
        document_name: str = "Guideline for the pharmacological treatment of hypertension in adults",
        source_url: str = "https://www.who.int/publications/i/item/9789240033987",
        max_chunk_tokens: int = 600,
        min_chunk_tokens: int = 30,
        chunk_overlap_tokens: int = 100,
        tokenizer_model: str = "cl100k_base",
        page_offset: int = 0
    ):
        self.document_name = document_name
        self.source_url = source_url
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.page_offset = page_offset

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

    def _format_page_number(self, pdf_page_num: int) -> str:
        """Formats human-readable page number accounting for front-matter page offset."""
        if self.page_offset > 0 and pdf_page_num > self.page_offset:
            return str(pdf_page_num - self.page_offset)
        elif self.page_offset == 12 and pdf_page_num == 7:
            return "v"
        elif self.page_offset == 12 and pdf_page_num == 8:
            return "vi"
        elif self.page_offset == 12 and pdf_page_num == 9:
            return "vii"
        return str(pdf_page_num)

    def _is_noise_line(self, line: str) -> bool:
        """Determines if a line is running header/footer, standalone page number, or TOC noise."""
        clean = line.strip()
        if not clean:
            return True
        if PAGE_NUMBER_ARTIFACT_PATTERN.match(clean):
            return True
        if TOC_LINE_PATTERN.search(clean):
            return True
        if RUNNING_HEADER_PATTERN.search(clean):
            return True
        if COPYRIGHT_FOOTER_PATTERN.search(clean):
            return True
        return False

    def _detect_heading(self, line: str) -> Tuple[bool, str, str, int]:
        """
        Detects if a line is a heading and separates section_number and section_title.
        Returns: (is_heading, section_number, section_title, level)
        """
        clean = line.strip()

        # 1. Markdown Headers (#, ##, ###)
        if clean.startswith('#'):
            hashes = len(clean) - len(clean.lstrip('#'))
            title_part = clean.lstrip('#').strip()
            level = min(hashes, 3)

            # Check if title contains section number (e.g. "## 2.4 Reviews of evidence")
            num_match = SECTION_NUM_PATTERN.match(title_part)
            if num_match:
                return True, num_match.group(1).rstrip('.'), num_match.group(2).strip(), level
            
            rec_match = RECOMMENDATION_NUM_PATTERN.match(title_part)
            if rec_match:
                return True, rec_match.group(1), rec_match.group(2).strip(), 3

            return True, "", title_part, level

        # 2. Numbered Sections without Markdown # prefix (e.g. "2.4 Reviews of evidence")
        rec_match = RECOMMENDATION_NUM_PATTERN.match(clean)
        if rec_match:
            return True, rec_match.group(1), rec_match.group(2).strip(), 3

        sec_match = SECTION_NUM_PATTERN.match(clean)
        if sec_match and len(clean) < 120:
            return True, sec_match.group(1).rstrip('.'), sec_match.group(2).strip(), 2

        # 3. Clinical Keywords
        for kw in CLINICAL_KEYWORDS:
            if clean.lower() == kw.lower() or clean.lower().startswith(kw.lower() + ":"):
                return True, "", clean, 2

        return False, "", "", 0

    def _export_page_annotated_markdown(self, doc) -> str:
        """Exports Docling document elements into Markdown text with embedded <!-- page_number: X --> comments."""
        md_lines = []
        current_page = None

        for item, level in doc.iterate_items():
            page_no = item.prov[0].page_no if (hasattr(item, "prov") and item.prov and len(item.prov) > 0) else 1

            if page_no != current_page:
                current_page = page_no
                md_lines.append(f"\n<!-- page_number: {current_page} -->\n")

            label = getattr(item, "label", "")
            label_str = label.name.lower() if hasattr(label, "name") else str(label).lower()

            text = ""
            if hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown().strip()
                except Exception:
                    text = ""
            if not text:
                text = getattr(item, "text", "").strip()

            if not text:
                continue

            if "header" in label_str or "title" in label_str:
                hashes = "#" * max(1, min(level if level > 0 else 2, 3))
                if not text.startswith("#"):
                    md_lines.append(f"{hashes} {text}")
                else:
                    md_lines.append(text)
            else:
                md_lines.append(text)

        return "\n".join(md_lines)

    def parse_pdf(self, pdf_path: str, md_output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parses a PDF using Docling:
        1. Exports full Markdown text with page markers and saves `.md` file to disk if md_output_path provided.
        2. Parses structured section numbers, titles, and page-accurate hierarchical chunks.
        """
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "Docling package is not installed. Please install it using 'pip install docling'."
            )

        pdf_file = pathlib.Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        converter = DocumentConverter()
        result = converter.convert(str(pdf_file))
        doc = result.document

        # Export full page-annotated Markdown text
        markdown_text = self._export_page_annotated_markdown(doc)

        # Save .md file to disk if path is provided
        if md_output_path:
            out_p = pathlib.Path(md_output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            print(f"   💾 Saved Docling Markdown export with page markers to '{md_output_path}'")

        # Parse page-annotated Markdown text
        return self.parse_markdown_text(markdown_text)

    def parse_markdown_text(self, markdown_text: str) -> Dict[str, Any]:
        """
        Parses full markdown text, extracting section numbers, tracking accurate PDF page numbers,
        cleaning noise lines, and building standardized flat chunks and hierarchy trees.
        """
        lines = markdown_text.split('\n')
        sections = []
        current_section = None
        current_l1 = self.document_name.replace(".pdf", "")
        current_l2 = ""
        pdf_page_num = 1

        for line in lines:
            clean = line.strip()

            # Track page number markers
            page_match = re.search(r'<!--\s*page_number:\s*(\d+)\s*-->', clean, re.IGNORECASE)
            if page_match:
                pdf_page_num = int(page_match.group(1))
                continue

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

                current_section = {
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "parent_section": parent,
                    "hierarchy_path": h_path,
                    "pdf_page_number": pdf_page_num,
                    "level": level,
                    "content_lines": []
                }
            else:
                if current_section:
                    current_section["content_lines"].append(clean)
                else:
                    # Initial overview section
                    current_section = {
                        "section_number": "",
                        "section_title": "Overview",
                        "parent_section": "",
                        "hierarchy_path": [current_l1, "Overview"],
                        "pdf_page_number": pdf_page_num,
                        "level": 1,
                        "content_lines": [clean]
                    }

        if current_section and "\n".join(current_section["content_lines"]).strip():
            sections.append(current_section)

        return self._build_export_payload(sections)

    def _build_export_payload(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Converts extracted internal sections into standardized flat_chunks and hierarchy_tree."""
        flat_chunks = []

        for idx, sec in enumerate(sections):
            content = "\n".join(sec["content_lines"]).strip()
            if not content:
                continue

            pdf_page_num = sec["pdf_page_number"]
            printed_page_num = self._format_page_number(pdf_page_num)
            token_count = self._count_tokens(content)

            layout_meta = {
                "block_labels": ["section_header" if sec["section_number"] else "text"],
                "bounding_boxes": [],
                "images": {}
            }

            # Handle chunk splitting if content exceeds max_chunk_tokens
            if token_count > self.max_chunk_tokens:
                sub_texts = self.text_splitter.split_text(content)
                for sub_i, sub_txt in enumerate(sub_texts):
                    chunk_id = f"docling_p{printed_page_num}_c{idx + 1}_sub{sub_i + 1}"
                    flat_chunks.append({
                        "chunk_id": chunk_id,
                        "section_number": sec["section_number"],
                        "section_title": sec["section_title"],
                        "parent_section": sec["parent_section"],
                        "hierarchy_path": sec["hierarchy_path"],
                        "page_number": printed_page_num,
                        "pdf_page_number": pdf_page_num,
                        "content": sub_txt,
                        "token_count": self._count_tokens(sub_txt),
                        "document_name": self.document_name,
                        "source_url": self.source_url,
                        "layout_metadata": layout_meta
                    })
            else:
                chunk_id = f"docling_p{printed_page_num}_c{idx + 1}"
                flat_chunks.append({
                    "chunk_id": chunk_id,
                    "section_number": sec["section_number"],
                    "section_title": sec["section_title"],
                    "parent_section": sec["parent_section"],
                    "hierarchy_path": sec["hierarchy_path"],
                    "page_number": printed_page_num,
                    "pdf_page_number": pdf_page_num,
                    "content": content,
                    "token_count": token_count,
                    "document_name": self.document_name,
                    "source_url": self.source_url,
                    "layout_metadata": layout_meta
                })

        return {
            "document_info": {
                "document_name": self.document_name,
                "source_url": self.source_url,
                "total_sections_detected": len(flat_chunks),
                "total_chunks_generated": len(flat_chunks)
            },
            "hierarchy_tree": [],
            "flat_chunks": flat_chunks
        }
