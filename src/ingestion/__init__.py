"""
Ingestion package for Medical RAG guideline PDFs.
Provides PDF parsing with PyMuPDF4LLM, Section-aware chunking,
and PaddleOCR Markdown section detection engine.
"""

from .pdf_parser import MedicalPDFParser
from .chunker import MedicalSectionChunker
from .paddle_section_parser import PaddleSectionDetector

__all__ = ["MedicalPDFParser", "MedicalSectionChunker", "PaddleSectionDetector"]
