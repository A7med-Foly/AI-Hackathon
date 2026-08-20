"""
Evaluation Package for Medical RAG Framework.
Includes ClinicalRAGEvaluator for calculating retrieval (Hit Rate, MRR, Precision, Recall)
and generation metrics (Faithfulness, Answer Relevance).
"""

from src.evaluation.evaluator import ClinicalRAGEvaluator

__all__ = ["ClinicalRAGEvaluator"]
