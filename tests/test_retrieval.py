"""
Unit & Integration Tests for Medical RAG Retrieval Engine (Day 2).
Validates VectorStoreManager, HybridSearchEngine (BM25 + RRF), and ClinicalRetriever.
"""

import pytest
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.retriever import ClinicalRetriever

SAMPLE_CHUNKS = [
    {
        "chunk_id": "test_chunk_1",
        "section_number": "1.4.1",
        "section_title": "Lifestyle modifications",
        "parent_section": "Blood glucose management",
        "hierarchy_path": ["Blood glucose management", "1.4.1 Lifestyle modifications"],
        "page_number": 12,
        "content": "Advise lifestyle changes including dietary advice and physical activity for adults with type 2 diabetes.",
        "token_count": 22,
        "document_name": "test_doc.pdf",
        "source_url": "https://example.com/test",
        "layout_metadata": {"bounding_boxes": [{"block_id": 1, "bbox": [10, 10, 100, 50]}]}
    },
    {
        "chunk_id": "test_chunk_2",
        "section_number": "1.4.2",
        "section_title": "First-line drug treatment",
        "parent_section": "Blood glucose management",
        "hierarchy_path": ["Blood glucose management", "1.4.2 First-line drug treatment"],
        "page_number": 13,
        "content": "Offer standard-release metformin as first-line drug treatment for adults with type 2 diabetes.",
        "token_count": 21,
        "document_name": "test_doc.pdf",
        "source_url": "https://example.com/test",
        "layout_metadata": {"bounding_boxes": [{"block_id": 2, "bbox": [10, 60, 200, 150]}]}
    },
    {
        "chunk_id": "test_chunk_3",
        "section_number": "1.5.7",
        "section_title": "HbA1c targets",
        "parent_section": "Blood glucose management",
        "hierarchy_path": ["Blood glucose management", "1.5.7 HbA1c targets"],
        "page_number": 15,
        "content": "Target HbA1c level of 48 mmol/mol (6.5%) for adults managed with lifestyle and single non-hypoglycaemic drug.",
        "token_count": 25,
        "document_name": "test_doc.pdf",
        "source_url": "https://example.com/test",
        "layout_metadata": {"bounding_boxes": []}
    }
]


def test_bm25_search():
    engine = HybridSearchEngine()
    engine.index_chunks(SAMPLE_CHUNKS)

    # Search for metformin
    res = engine.query_bm25("metformin first-line treatment")
    assert len(res) > 0
    assert res[0]["chunk_id"] == "test_chunk_2"
    assert "metformin" in res[0]["content"].lower()


def test_rrf_fusion():
    engine = HybridSearchEngine()
    
    dense_res = [
        {"chunk_id": "test_chunk_2", "score": 0.9, "content": SAMPLE_CHUNKS[1]["content"]},
        {"chunk_id": "test_chunk_1", "score": 0.7, "content": SAMPLE_CHUNKS[0]["content"]}
    ]
    bm25_res = [
        {"chunk_id": "test_chunk_2", "score": 5.2, "content": SAMPLE_CHUNKS[1]["content"]},
        {"chunk_id": "test_chunk_3", "score": 2.1, "content": SAMPLE_CHUNKS[2]["content"]}
    ]

    fused = engine.reciprocal_rank_fusion(dense_res, bm25_res, top_k=2)
    assert len(fused) == 2
    # test_chunk_2 ranked #1 in both -> highest RRF score
    assert fused[0]["chunk_id"] == "test_chunk_2"
    assert fused[0]["dense_rank"] == 1
    assert fused[0]["bm25_rank"] == 1


def test_vector_store_ingest_and_query(tmp_path):
    store = VectorStoreManager(collection_name="test_coll", persist_dir=str(tmp_path))
    count = store.ingest_chunks(SAMPLE_CHUNKS)
    assert count == 3
    assert store.count() == 3

    # Dense query
    results = store.query_dense("dietary advice physical activity", n_results=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "test_chunk_1"
    assert "layout_metadata" in results[0]["metadata"]
