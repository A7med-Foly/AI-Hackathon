"""
PDF Parser module using PyMuPDF4LLM for layout-preserved Markdown extraction
with header/footer stripping and strict page number tracking.
"""

import os
import re
import pathlib
from typing import List, Dict, Any, Optional
import pymupdf4llm


class MedicalPDFParser:
    def __init__(self, default_source_url: Optional[str] = None):
        """
        Initialize the PDF parser.
        :param default_source_url: Optional default URL for the medical guideline document.
        """
        self.default_source_url = default_source_url or "https://www.nice.org.uk/guidance/ng28"

    def clean_text(self, text: str) -> str:
        """
        Strips common PDF artifacts, running headers, and footers from medical guidelines.
        """
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            trimmed = line.strip()
            
            # Skip empty lines (preserve markdown structure later)
            if not trimmed:
                cleaned_lines.append("")
                continue
                
            # Filter standard running headers / footers common in NICE/CDC guidelines
            if re.match(r"^©\s*NICE\s*\d{4}\..*", trimmed, re.IGNORECASE):
                continue
            if re.match(r"^Page\s+\d+\s+of\s+\d+$", trimmed, re.IGNORECASE):
                continue
            if re.match(r"^\d+\s+of\s+\d+$", trimmed, re.IGNORECASE):
                continue
            if re.match(r"^National Institute for Health and Care Excellence$", trimmed, re.IGNORECASE):
                continue
            if re.match(r"^Type 2 diabetes in adults: management\s*\(NG28\)$", trimmed, re.IGNORECASE):
                continue
                
            cleaned_lines.append(line)
            
        # Rejoin and clean multiple blank lines
        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        return cleaned_text.strip()

    def parse_pdf(self, pdf_path: str, source_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Converts PDF into page-by-page Markdown chunks with metadata.
        
        :param pdf_path: Path to the input PDF file.
        :param source_url: Optional explicit URL for document provenance.
        :return: List of dicts: [{'text': str, 'page_number': int, 'document_name': str, 'source_url': str}]
        """
        pdf_file = pathlib.Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        document_name = pdf_file.name
        doc_url = source_url or self.default_source_url
        
        # PyMuPDF4LLM converts PDF to per-page markdown dictionary list
        pages_raw = pymupdf4llm.to_markdown(str(pdf_file), page_chunks=True)
        
        parsed_pages = []
        for page_data in pages_raw:
            raw_text = page_data.get("text", "")
            # PyMuPDF metadata uses 1-indexed or 0-indexed page numbers. Ensure 1-indexed int.
            page_num = page_data.get("metadata", {}).get("page", 0)
            if isinstance(page_num, int):
                # PyMuPDF page numbers start at 1 in pymupdf4llm page dict
                page_number = page_num
            else:
                page_number = 1
                
            cleaned_content = self.clean_text(raw_text)
            
            if cleaned_content:
                parsed_pages.append({
                    "text": cleaned_content,
                    "page_number": page_number,
                    "document_name": document_name,
                    "source_url": doc_url
                })
                
        return parsed_pages
