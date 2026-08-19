"""
High-Level Clinical Retriever Interface.
Coordinates ChromaDB VectorStoreManager (supporting BAAI/bge-small-en-v1.5) and HybridSearchEngine to execute evidence-grounded queries.
"""

import json
import pathlib
from typing import List, Dict, Any, Optional
from src import config
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid_search import HybridSearchEngine


class ClinicalRetriever:
    def __init__(
        self,
        json_chunks_path: str = config.DEFAULT_PROCESSED_JSON_PATH,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        collection_name: Optional[str] = config.DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = config.EMBEDDING_MODEL_NAME
    ):
        self.json_chunks_path = json_chunks_path
        self.vector_store = VectorStoreManager(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_model_name=embedding_model_name
        )
        self.hybrid_engine = HybridSearchEngine()
        self._initialized = False

    def initialize(self, force_reindex: bool = False):
        """Loads chunks, populates ChromaDB vector store, and initializes BM25 keyword index."""
        path = pathlib.Path(self.json_chunks_path)
        if not path.exists():
            alt_path = pathlib.Path("data/processed") / self.json_chunks_path
            if alt_path.exists():
                path = alt_path
            else:
                raise FileNotFoundError(f"Output JSON file not found at: {self.json_chunks_path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        chunks = payload.get("flat_chunks", [])
        
        # 1. Initialize BM25 Keyword Engine
        self.hybrid_engine.index_chunks(chunks)

        # 2. Populate ChromaDB Vector Store if empty or force reindex requested
        if self.vector_store.count() == 0 or force_reindex:
            print(f"Ingesting {len(chunks)} flat RAG chunks into ChromaDB vector store with {self.vector_store.embedding_model_name}...")
            self.vector_store.ingest_chunks(chunks)

        self._initialized = True
        print(f"✅ ClinicalRetriever Initialized! Vector Store ({self.vector_store.embedding_model_name}): {self.vector_store.count()} vectors | BM25 Index: {len(chunks)} chunks.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        section_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes evidence retrieval for clinical queries.
        :param query: Natural language medical question or guideline search query.
        :param top_k: Number of top evidence chunks to return.
        :param mode: Search mode - 'hybrid' (BM25 + Dense RRF), 'dense', or 'bm25'.
        :param section_filter: Optional filter by section_number (e.g. '1.4' or '1.5.1').
        """
        if not self._initialized:
            self.initialize()

        # Build metadata filter for vector search if section_filter provided
        where_filter = None
        if section_filter:
            where_filter = {"section_number": section_filter}

        if mode == "dense":
            return self.vector_store.query_dense(query, n_results=top_k, where_filter=where_filter)

        if mode == "bm25":
            results = self.hybrid_engine.query_bm25(query, top_k=top_k)
            if section_filter:
                results = [r for r in results if r.get("metadata", {}).get("section_number") == section_filter]
            return results[:top_k]

        # Hybrid Search (BM25 + Dense + RRF)
        dense_res = self.vector_store.query_dense(query, n_results=top_k * 2, where_filter=where_filter)
        bm25_res = self.hybrid_engine.query_bm25(query, top_k=top_k * 2)

        if section_filter:
            bm25_res = [r for r in bm25_res if r.get("metadata", {}).get("section_number") == section_filter]

        return self.hybrid_engine.reciprocal_rank_fusion(
            dense_results=dense_res,
            bm25_results=bm25_res,
            top_k=top_k
        )
