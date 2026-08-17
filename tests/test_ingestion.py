"""
Unit tests for Day 1 PDF Ingestion and Section-Aware Chunking Pipeline.
"""

import pytest
import pathlib
from src.ingestion import MedicalPDFParser, MedicalSectionChunker


@pytest.fixture
def sample_pdf_path():
    path = pathlib.Path("type-2-diabetes-in-adults-management.pdf")
    if not path.exists():
        pytest.skip("Sample PDF not found in workspace.")
    return str(path)


def test_pdf_parser(sample_pdf_path):
    parser = MedicalPDFParser(default_source_url="https://www.nice.org.uk/guidance/ng28")
    pages = parser.parse_pdf(sample_pdf_path)
    
    assert len(pages) > 0
    first_page = pages[0]
    assert "text" in first_page
    assert "page_number" in first_page
    assert first_page["page_number"] >= 1
    assert first_page["document_name"] == "type-2-diabetes-in-adults-management.pdf"
    assert first_page["source_url"] == "https://www.nice.org.uk/guidance/ng28"


def test_section_chunker(sample_pdf_path):
    parser = MedicalPDFParser()
    chunker = MedicalSectionChunker(target_chunk_tokens=600, max_chunk_tokens=800)
    
    pages = parser.parse_pdf(sample_pdf_path)
    chunks = chunker.create_chunks(pages)
    
    assert len(chunks) > 0
    
    required_metadata = {"document_name", "page_number", "section_title", "chunk_id", "source_url", "token_count"}
    
    for chunk in chunks:
        meta = chunk.metadata
        # Check all required keys exist
        for key in required_metadata:
            assert key in meta, f"Missing metadata key '{key}' in chunk {meta.get('chunk_id')}"
        
        # Check token size boundary
        assert meta["token_count"] <= 800, f"Chunk {meta['chunk_id']} exceeds max token limit of 800"
        assert len(chunk.page_content.strip()) > 0
