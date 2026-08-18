"""
Clinical RAG Generator Engine - Day 3.
Synthesizes evidence-grounded responses from retrieved NICE guideline chunks using OpenRouter / OpenAI LLM,
enforcing strict section & page citations [Section X.Y, Page Z] and visual evidence tracing.
"""

import os
import json
import pathlib
from typing import List, Dict, Any, Optional
from src.retrieval.retriever import ClinicalRetriever


def load_env_file():
    """Loads environment variables from .env if present."""
    env_path = pathlib.Path(".env")
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

load_env_file()

SYSTEM_CLINICAL_PROMPT = """You are an expert Clinical Decision Support Assistant specializing in NICE Medical Guidelines.
Your task is to provide clear, evidence-grounded answers to medical queries based ONLY on the retrieved guideline context provided below.

CRITICAL RULES:
1. ALWAYS ground every clinical claim by citing the exact section and page number in brackets, e.g. [Section 1.5.7, Page 13] or [Section 1.4.2, Page 15].
2. DO NOT make up information or introduce external clinical recommendations not supported by the context.
3. If the provided context does not contain sufficient evidence to answer the query, explicitly state: "Based on the provided NICE guidelines context, there is insufficient evidence to answer this query."
4. Maintain a professional, concise, and structured medical tone suitable for clinicians.
"""


class ClinicalRAGGenerator:
    def __init__(
        self,
        retriever: Optional[ClinicalRetriever] = None,
        model_name: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        self.retriever = retriever or ClinicalRetriever()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY must be set in environment or passed to ClinicalRAGGenerator.")

        from openai import OpenAI
        base_url = "https://openrouter.ai/api/v1" if "openrouter" in self.api_key.lower() or os.environ.get("OPENROUTER_API_KEY") else None
        self.llm_client = OpenAI(api_key=self.api_key, base_url=base_url)

    def generate(self, query: str, top_k: int = 4, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Executes hybrid retrieval and generates an evidence-grounded clinical response with structured citations using LLM.
        """
        if not self.retriever._initialized:
            self.retriever.initialize()

        # 1. Retrieve Evidence Chunks
        retrieved_chunks = self.retriever.retrieve(query=query, top_k=top_k, mode=mode)
        
        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "Based on the provided NICE guidelines context, no relevant evidence was found for this query.",
                "citations": [],
                "evidence_chunks": []
            }

        # 2. Extract Structured Citations & Bounding Boxes
        citations = []
        context_blocks = []

        for idx, item in enumerate(retrieved_chunks, start=1):
            meta = item.get("metadata", item)
            content = item.get("content", meta.get("content", "")).strip()
            sec_num = meta.get("section_number", "")
            sec_title = meta.get("section_title", "")
            page_num = meta.get("page_number", 1)
            layout_meta = meta.get("layout_metadata", {})
            bboxes = layout_meta.get("bounding_boxes", [])
            page_img = layout_meta.get("page_image_url", "")
            layout_img = layout_meta.get("layout_image_url", "")

            citation_label = f"Section {sec_num}" if sec_num else sec_title
            context_blocks.append(f"--- [Evidence Block {idx}] ({citation_label}, Page {page_num}) ---\n{content}\n")

            citations.append({
                "citation_id": idx,
                "section_number": sec_num,
                "section_title": sec_title,
                "page_number": page_num,
                "pdf_page_number": meta.get("pdf_page_number", page_num),
                "chunk_id": item.get("chunk_id", ""),
                "hierarchy_path": meta.get("hierarchy_path", []),
                "bounding_boxes": bboxes,
                "page_image_url": page_img,
                "layout_image_url": layout_img,
                "rrf_score": item.get("rrf_score"),
                "dense_rank": item.get("dense_rank"),
                "bm25_rank": item.get("bm25_rank")
            })

        # 3. Generate Answer using LLM
        context_str = "\n".join(context_blocks)
        answer = self._call_llm(query, context_str)

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "evidence_chunks": retrieved_chunks
        }

    def _call_llm(self, query: str, context_str: str) -> str:
        """Calls OpenAI / OpenRouter LLM API with fallback model strategy."""
        messages = [
            {"role": "system", "content": SYSTEM_CLINICAL_PROMPT},
            {"role": "user", "content": f"CLINICAL GUIDELINE EVIDENCE:\n{context_str}\n\nCLINICAL QUERY: {query}"}
        ]
        
        # Candidate models list for automatic failover
        models_to_try = [
            self.model_name,
            # "openai/gpt-oss-20b:free",
            # "z-ai/glm-5.2:free",
            # "nvidia/nemotron-3.5-lightning:free",
            "google/gemma-4-31b-it:free"
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        models_unique = [m for m in models_to_try if not (m in seen or seen.add(m))]

        last_error = None
        for m in models_unique:
            try:
                response = self.llm_client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024
                )
                if response and response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Model '{m}' call failed ({e}). Trying next fallback model...")
                last_error = e

        raise RuntimeError(f"All LLM models failed. Last error: {last_error}")
