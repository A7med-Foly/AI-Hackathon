"""
Unit Tests for Medical RAG Evaluation Engine.
Validates metric computations (Hit Rate, MRR, Precision, Recall, Faithfulness).
"""

import pytest
from src.evaluation.evaluator import ClinicalRAGEvaluator


@pytest.fixture
def evaluator():
    return ClinicalRAGEvaluator(dataset_path="data/eval_dataset.json")


def test_is_chunk_relevant(evaluator):
    chunk = {
        "section_number": "4",
        "section_title": "RECOMMENDATION ON DRUG CLASSES TO BE USED AS FIRST-LINE AGENTS",
        "page_number": "10",
        "pdf_page_number": 22
    }
    
    # Test matching expected section
    assert evaluator.is_chunk_relevant(chunk, expected_sections=["4", "4."], expected_pages=["10"])
    
    # Test matching expected page number
    assert evaluator.is_chunk_relevant(chunk, expected_sections=["99"], expected_pages=["10"])
    assert evaluator.is_chunk_relevant(chunk, expected_sections=["99"], expected_pages=["22"])

    # Test non-matching chunk
    assert not evaluator.is_chunk_relevant(chunk, expected_sections=["99"], expected_pages=["99"])


def test_evaluate_retrieval_query(evaluator):
    query_item = {
        "query_id": "test_q1",
        "question": "What are first line drug classes?",
        "expected_sections": ["4"],
        "expected_pages": ["10"]
    }

    retrieved = [
        {"section_number": "1", "section_title": "Other", "page_number": "5"},
        {"section_number": "4", "section_title": "Recommendation 4", "page_number": "10"},
        {"section_number": "5", "section_title": "Combination", "page_number": "13"}
    ]

    metrics = evaluator.evaluate_retrieval_query(query_item, retrieved, top_k=3)
    
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5  # Match at rank 2 -> 1/2 = 0.5
    assert metrics["first_match_rank"] == 2
    assert metrics["precision"] == 1 / 3
    assert metrics["recall"] == 1.0


def test_evaluate_generation_query(evaluator):
    query_item = {
        "query_id": "test_q1",
        "question": "What are recommended first-line drug classes for hypertension?"
    }

    retrieved = [
        {"content": "WHO recommends initial first-line drug treatment for hypertension with thiazide diuretics, ACE inhibitors, and CCBs."}
    ]

    answer = "The recommended first-line drug classes for hypertension are thiazides, ACE inhibitors, and CCBs [Section 4, Page 10]."

    metrics = evaluator.evaluate_generation_query(query_item, answer, retrieved)

    assert metrics["citation_present"] is True
    assert metrics["faithfulness"] > 0.5
    assert metrics["answer_relevance"] > 0.5


def test_summarize_results(evaluator):
    results = [
        {"hit_rate": 1.0, "mrr": 1.0, "precision": 0.5, "recall": 1.0, "faithfulness": 0.9, "answer_relevance": 0.8},
        {"hit_rate": 1.0, "mrr": 0.5, "precision": 0.25, "recall": 1.0, "faithfulness": 0.7, "answer_relevance": 0.8}
    ]

    summary = evaluator.summarize_results(results)

    assert summary["mean_hit_rate"] == 1.0
    assert summary["mean_mrr"] == 0.75
    assert summary["mean_precision"] == 0.375
    assert summary["mean_faithfulness"] == 0.8
