"""
Unit and Integration Tests for Day 3 Clinical RAG Generation & FastAPI Endpoints using OpenRouter LLM.
"""

import pytest
from src.generation.generator import ClinicalRAGGenerator
from src.api.server import app
from fastapi.testclient import TestClient


def test_generator_initialization():
    gen = ClinicalRAGGenerator()
    assert gen.retriever is not None
    assert "llama" in gen.model_name.lower() or "gpt" in gen.model_name.lower() or "gemini" in gen.model_name.lower()


def test_generator_synthesis():
    gen = ClinicalRAGGenerator()
    gen.retriever.initialize()
    
    query = "What is the first-line treatment for adults with type 2 diabetes?"
    res = gen.generate(query=query, top_k=3)

    assert "query" in res
    assert "answer" in res
    assert "citations" in res
    assert len(res["citations"]) > 0
    assert len(res["answer"]) > 10

    first_cit = res["citations"][0]
    assert "section_number" in first_cit
    assert "page_number" in first_cit
    assert "bounding_boxes" in first_cit


def test_fastapi_endpoints():
    client = TestClient(app)
    
    # Test Health Endpoint
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "ok"

    # Test Query Endpoint
    query_payload = {
        "query": "HbA1c target for adults managed with lifestyle and a single drug",
        "top_k": 3,
        "mode": "hybrid"
    }
    query_res = client.post("/api/query", json=query_payload)
    assert query_res.status_code == 200
    data = query_res.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["citations"]) > 0

    # Test UI Endpoint
    ui_res = client.get("/")
    assert ui_res.status_code == 200
    assert "Medical Guidelines RAG Studio" in ui_res.text
