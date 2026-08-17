"""
Clinical RAG Generator Engine - Day 3.
Synthesizes evidence-grounded responses from retrieved NICE guideline chunks,
enforcing strict section & page citations [Section X.Y, Page Z] and visual evidence tracing.
"""

import os
import json
from typing import List, Dict, Any, Optional
from src.retrieval.retriever import ClinicalRetriever


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
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        self.retriever = retriever or ClinicalRetriever()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.llm_client = None

        if self.api_key:
            try:
                from openai import OpenAI
                base_url = "https://openrouter.ai/api/v1" if "openrouter" in self.api_key.lower() or os.environ.get("OPENROUTER_API_KEY") else None
                self.llm_client = OpenAI(api_key=self.api_key, base_url=base_url)
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize OpenAI client ({e}). Using deterministic clinical synthesis engine.")

    def generate(self, query: str, top_k: int = 4, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Executes hybrid retrieval and generates an evidence-grounded clinical response with structured citations.
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
                "chunk_id": item.get("chunk_id", ""),
                "hierarchy_path": meta.get("hierarchy_path", []),
                "bounding_boxes": bboxes,
                "page_image_url": page_img,
                "layout_image_url": layout_img,
                "rrf_score": item.get("rrf_score"),
                "dense_rank": item.get("dense_rank"),
                "bm25_rank": item.get("bm25_rank")
            })

        # 3. Generate Answer (using LLM or High-Precision Synthesis Engine)
        context_str = "\n".join(context_blocks)
        
        if self.llm_client:
            answer = self._call_llm(query, context_str)
        else:
            answer = self._synthesize_response(query, retrieved_chunks)

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "evidence_chunks": retrieved_chunks
        }

    def _call_llm(self, query: str, context_str: str) -> str:
        """Calls OpenAI / OpenRouter LLM API to generate response."""
        messages = [
            {"role": "system", "content": SYSTEM_CLINICAL_PROMPT},
            {"role": "user", "content": f"CLINICAL GUIDELINE EVIDENCE:\n{context_str}\n\nCLINICAL QUERY: {query}"}
        ]
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ LLM API call failed ({e}). Falling back to clinical synthesis engine.")
            return self._synthesize_response(query, [])

    def _synthesize_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Deterministic evidence-grounded clinical synthesis fallback when LLM API key is not present."""
        if not retrieved_chunks:
            return "Based on the provided NICE guidelines context, no relevant evidence was found for this query."

        top_chunk = retrieved_chunks[0]
        meta = top_chunk.get("metadata", top_chunk)
        sec_num = meta.get("section_number", "")
        sec_title = meta.get("section_title", "")
        page_num = meta.get("page_number", 1)
        content = top_chunk.get("content", meta.get("content", "")).strip()

        cit_tag = f"[Section {sec_num}, Page {page_num}]" if sec_num else f"[{sec_title}, Page {page_num}]"
        
        summary_lines = []
        for line in content.split('\n'):
            line_clean = line.strip()
            if line_clean and not line_clean.startswith('#'):
                summary_lines.append(line_clean)

        body_text = " ".join(summary_lines[:3])
        return f"According to the NICE guideline recommendation {cit_tag}:\n\n{body_text} {cit_tag}"
