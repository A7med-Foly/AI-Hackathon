"""
Clinical RAG Evaluation Engine.
Computes quantitative performance metrics for Retrieval (Hit Rate, MRR, Precision, Recall)
and Generation Quality (Faithfulness, Answer Relevance, Citation Accuracy).
"""

import re
import json
import pathlib
from typing import List, Dict, Any, Optional
from src import config


class ClinicalRAGEvaluator:
    def __init__(self, dataset_path: str = config.EVAL_DATASET_PATH):
        self.dataset_path = dataset_path
        self.dataset: List[Dict[str, Any]] = self.load_dataset()

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads evaluation dataset JSON file."""
        path = pathlib.Path(self.dataset_path)
        if not path.exists():
            print(f"⚠️ Warning: Dataset file '{self.dataset_path}' not found.")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def is_chunk_relevant(chunk: Dict[str, Any], expected_sections: List[str], expected_pages: List[str]) -> bool:
        """Determines if a retrieved chunk matches ground-truth sections or page numbers."""
        meta = chunk.get("metadata", {})
        sec_num = str(chunk.get("section_number") or meta.get("section_number", "")).strip()
        sec_title = str(chunk.get("section_title") or meta.get("section_title", "")).strip()
        page_num = str(chunk.get("page_number") or meta.get("page_number", "")).strip()
        pdf_page_num = str(chunk.get("pdf_page_number") or meta.get("pdf_page_number", "")).strip()

        # Check section number match
        for exp_sec in expected_sections:
            exp_clean = exp_sec.strip().rstrip(".")
            if exp_clean and (
                sec_num == exp_clean or 
                sec_num.startswith(exp_clean + ".") or 
                sec_title.startswith(exp_clean) or
                exp_clean in sec_title
            ):
                return True

        # Check page number match (printed page or pdf page)
        for exp_page in expected_pages:
            exp_p_str = str(exp_page).strip()
            if page_num == exp_p_str or pdf_page_num == exp_p_str:
                return True

        return False

    def evaluate_retrieval_query(
        self,
        query_item: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        top_k: int = config.DEFAULT_EVAL_TOP_K
    ) -> Dict[str, Any]:
        """Calculates retrieval metrics (Hit Rate@K, MRR@K, Precision@K, Recall@K) for a single query."""
        expected_sections = query_item.get("expected_sections", [])
        expected_pages = query_item.get("expected_pages", [])
        
        chunks_to_eval = retrieved_chunks[:top_k]
        
        first_match_rank = None
        matched_chunks_count = 0
        matched_sections = set()

        for idx, chunk in enumerate(chunks_to_eval):
            rank = idx + 1
            if self.is_chunk_relevant(chunk, expected_sections, expected_pages):
                matched_chunks_count += 1
                meta = chunk.get("metadata", {})
                sec_identifier = chunk.get("section_number") or meta.get("section_number") or chunk.get("section_title") or meta.get("section_title") or str(rank)
                if sec_identifier:
                    matched_sections.add(sec_identifier)
                if first_match_rank is None:
                    first_match_rank = rank

        hit_rate = 1.0 if first_match_rank is not None else 0.0
        mrr = (1.0 / first_match_rank) if first_match_rank is not None else 0.0
        precision = matched_chunks_count / top_k if top_k > 0 else 0.0
        recall = matched_chunks_count / max(len(expected_sections), 1)

        return {
            "query_id": query_item.get("query_id"),
            "question": query_item.get("question"),
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision": precision,
            "recall": recall,
            "first_match_rank": first_match_rank,
            "retrieved_count": len(chunks_to_eval)
        }

    def evaluate_generation_query(
        self,
        query_item: Dict[str, Any],
        generated_answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculates RAG triad generation quality metrics (Faithfulness, Citation Accuracy, Answer Relevance)."""
        if not generated_answer or "Error" in generated_answer:
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "citation_present": False
            }

        # 1. Citation presence check
        has_citation = bool(re.search(r"\[Section|Page\s*\d+|Table", generated_answer, re.IGNORECASE))

        # 2. Context Faithfulness (Groundedness)
        # Check token/stem overlap between generated answer key phrases and retrieved chunk contents
        context_text_lower = " ".join([
            str(c.get("content", "")) + " " + str(c.get("section_title", "")) + " " + str(c.get("parent_section", ""))
            for c in retrieved_chunks
        ]).lower()
        context_words = [w[:5] for w in re.findall(r"\b[a-zA-Z]{4,}\b", context_text_lower)]
        answer_words = [w[:5] for w in re.findall(r"\b[a-zA-Z]{4,}\b", generated_answer.lower())]

        # Exclude citation markers and structural artifacts from answer words
        ignore_words = {"secti", "section", "page", "table", "annex", "figure"}
        answer_words_filtered = [w for w in answer_words if w not in ignore_words]

        if answer_words_filtered:
            matched_count = sum(1 for w in answer_words_filtered if w in context_words or w in context_text_lower)
            faithfulness_score = min(1.0, matched_count / len(answer_words_filtered))
        else:
            faithfulness_score = 0.0

        # 3. Answer Relevance (query overlap with generated answer)
        query_stems = [w[:5] for w in re.findall(r"\b[a-zA-Z]{4,}\b", query_item.get("question", "").lower())]
        stop_stems = {"what", "which", "where", "when", "would", "should", "about"}
        query_stems_filtered = [s for s in query_stems if s not in stop_stems]

        if query_stems_filtered:
            matched_q = sum(1 for s in query_stems_filtered if s in context_words or s in answer_words)
            relevance_score = min(1.0, matched_q / len(query_stems_filtered))
        else:
            relevance_score = 0.0

        return {
            "query_id": query_item.get("query_id"),
            "faithfulness": round(faithfulness_score, 4),
            "answer_relevance": round(relevance_score, 4),
            "citation_present": has_citation
        }

    def summarize_results(self, eval_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Computes average summary metrics across all evaluated queries."""
        if not eval_results:
            return {
                "mean_hit_rate": 0.0,
                "mean_mrr": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_faithfulness": 0.0,
                "mean_answer_relevance": 0.0
            }

        n = len(eval_results)
        return {
            "mean_hit_rate": round(sum(r.get("hit_rate", 0.0) for r in eval_results) / n, 4),
            "mean_mrr": round(sum(r.get("mrr", 0.0) for r in eval_results) / n, 4),
            "mean_precision": round(sum(r.get("precision", 0.0) for r in eval_results) / n, 4),
            "mean_recall": round(sum(r.get("recall", 0.0) for r in eval_results) / n, 4),
            "mean_faithfulness": round(sum(r.get("faithfulness", 1.0) for r in eval_results if "faithfulness" in r) / n, 4) if n > 0 else 0.0,
            "mean_answer_relevance": round(sum(r.get("answer_relevance", 1.0) for r in eval_results if "answer_relevance" in r) / n, 4) if n > 0 else 0.0
        }
