"""
Section-Aware Chunker for Medical Guidelines.
Uses MarkdownHeaderTextSplitter combined with RecursiveCharacterTextSplitter
to produce 400-800 token chunks enriched with full hackathon metadata.
"""

import tiktoken
from typing import List, Dict, Any, Optional
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_core.documents import Document


class MedicalSectionChunker:
    def __init__(
        self,
        target_chunk_tokens: int = 600,
        max_chunk_tokens: int = 800,
        chunk_overlap_tokens: int = 100,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the section-aware chunker.
        :param target_chunk_tokens: Preferred chunk token size (~600 tokens).
        :param max_chunk_tokens: Upper bound token limit (~800 tokens).
        :param chunk_overlap_tokens: Token overlap (~100 tokens).
        :param encoding_name: Tiktoken encoding name for token counting.
        """
        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

        # 1. Primary Splitter: Markdown Header Hierarchy
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False  # Keep headers inside text for LLM context grounding
        )

        # 2. Secondary Splitter: Recursive Character Splitter for oversized sections
        # Approximating ~4 characters per token
        target_chars = target_chunk_tokens * 4
        overlap_chars = chunk_overlap_tokens * 4
        self.secondary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", "; ", " ", ""]
        )

    def count_tokens(self, text: str) -> int:
        """Counts tokens using tiktoken encoder."""
        return len(self.tokenizer.encode(text))

    def _build_section_title(self, metadata: Dict[str, Any]) -> str:
        """Constructs a breadcrumb section title from hierarchical headers."""
        headers = []
        for _, h_name in self.headers_to_split_on:
            if h_name in metadata:
                headers.append(str(metadata[h_name]))
        
        if headers:
            return " > ".join(headers)
        return "General Content"

    def create_chunks(self, pages: List[Dict[str, Any]]) -> List[Document]:
        """
        Processes list of page dicts and returns a list of LangChain Document objects
        with strict metadata compliance.
        
        Metadata Schema:
          - document_name (str)
          - page_number (int)
          - section_title (str)
          - chunk_id (str)
          - source_url (str)
          - token_count (int)
        """
        all_chunks: List[Document] = []
        global_chunk_idx = 0

        for page in pages:
            page_text = page["text"]
            page_num = page["page_number"]
            doc_name = page["document_name"]
            source_url = page["source_url"]

            if not page_text.strip():
                continue

            # Split page content by Markdown section headers
            section_docs = self.markdown_splitter.split_text(page_text)

            for sec_doc in section_docs:
                content = sec_doc.page_content.strip()
                if not content:
                    continue

                section_title = self._build_section_title(sec_doc.metadata)
                token_count = self.count_tokens(content)

                # Base metadata required by Hackathon Specification
                base_metadata = {
                    "document_name": doc_name,
                    "page_number": page_num,
                    "section_title": section_title,
                    "source_url": source_url,
                    **sec_doc.metadata
                }

                # If section fits within max_chunk_tokens, keep it as a single chunk
                if token_count <= self.max_chunk_tokens:
                    global_chunk_idx += 1
                    chunk_id = f"{doc_name}_p{page_num}_c{global_chunk_idx}"
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            **base_metadata,
                            "chunk_id": chunk_id,
                            "token_count": token_count
                        }
                    )
                    all_chunks.append(doc)
                else:
                    # Overflow section: apply secondary recursive character splitting
                    sub_chunks = self.secondary_splitter.split_text(content)
                    for sub_text in sub_chunks:
                        sub_text_trimmed = sub_text.strip()
                        if not sub_text_trimmed:
                            continue
                            
                        global_chunk_idx += 1
                        chunk_id = f"{doc_name}_p{page_num}_c{global_chunk_idx}"
                        sub_tokens = self.count_tokens(sub_text_trimmed)

                        doc = Document(
                            page_content=sub_text_trimmed,
                            metadata={
                                **base_metadata,
                                "chunk_id": chunk_id,
                                "token_count": sub_tokens
                            }
                        )
                        all_chunks.append(doc)

        return all_chunks
