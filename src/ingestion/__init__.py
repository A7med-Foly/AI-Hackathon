"""
Ingestion package for Medical RAG guideline PDFs.
Provides Section-aware chunking and PaddleOCR Markdown section detection engine.
"""

try:
    from .pdf_parser import MedicalPDFParser
except ImportError:
    MedicalPDFParser = None

from .chunker import MedicalSectionChunker
from .paddle_section_parser import PaddleSectionDetector

__all__ = ["MedicalPDFParser", "MedicalSectionChunker", "PaddleSectionDetector"]
