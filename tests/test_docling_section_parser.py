"""
Unit tests for DoclingSectionDetector.
Verifies standardized chunk schema parity, section_number extraction, and noise line filtering.
"""

import pytest
from src.ingestion.docling_section_parser import DoclingSectionDetector


def test_docling_section_detector_initialization():
    detector = DoclingSectionDetector(
        document_name="test_guideline.pdf",
        source_url="https://example.com/guideline",
        max_chunk_tokens=500,
        page_offset=2
    )
    assert detector.document_name == "test_guideline.pdf"
    assert detector.page_offset == 2
    assert detector.max_chunk_tokens == 500


def test_docling_markdown_fallback_parsing():
    detector = DoclingSectionDetector(
        document_name="hypertension_guideline.pdf",
        page_offset=0
    )

    sample_md = """# 1. Overview
    
This is an introductory text regarding medical treatment guidelines.

## 2.4 Reviews of evidence

The WHO Steering Group determined the scope of the guideline.
GUIDELINE FOR THE PHARMACOLOGICAL TREATMENT OF HYPERTENSION IN ADULTS
4

## 2.5 Certainty of evidence and strength of recommendations

The GDG rated the certainty of evidence using the GRADE approach.
© WHO 2021
"""
    result = detector.parse_markdown_text(sample_md)

    assert "flat_chunks" in result
    assert "document_info" in result
    assert len(result["flat_chunks"]) >= 3

    chunks = result["flat_chunks"]

    # Verify Chunk 1: "1. Overview"
    assert chunks[0]["section_number"] == "1"
    assert chunks[0]["section_title"] == "Overview"

    # Verify Chunk 2: "2.4 Reviews of evidence"
    assert chunks[1]["section_number"] == "2.4"
    assert chunks[1]["section_title"] == "Reviews of evidence"
    assert "GUIDELINE FOR THE PHARMACOLOGICAL" not in chunks[1]["content"]

    # Verify Chunk 3: "2.5 Certainty of evidence and strength of recommendations"
    assert chunks[2]["section_number"] == "2.5"
    assert chunks[2]["section_title"] == "Certainty of evidence and strength of recommendations"
    assert "© WHO 2021" not in chunks[2]["content"]
