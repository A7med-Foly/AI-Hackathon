"""
ChromaDB Vector Store Manager for Medical Guideline RAG Chunks.
Handles dense vector embedding with support for BAAI/bge-small-en-v1.5, local storage persistence, and metadata-filtered similarity search.
"""

import json
import pathlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from src import config


class VectorStoreManager:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        embedding_model_name: str = config.EMBEDDING_MODEL_NAME
    ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Derive isolated collection name based on model to prevent embedding conflicts
        if not collection_name:
            self.collection_name = config.DEFAULT_COLLECTION_NAME
        else:
            self.collection_name = collection_name

        # Initialize Embedding Function
        try:
            print(f"📦 Loading Dense Embedding Model: {embedding_model_name}...")
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize SentenceTransformer '{embedding_model_name}' ({e}). Fallback to Default ONNX embedding function.")
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        except ValueError as e:
            if "embedding function conflict" in str(e).lower():
                print(f"⚠️ Re-creating collection '{self.collection_name}' due to embedding model update...")
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                raise e

    def ingest_chunks_from_json(self, json_path: str = config.DEFAULT_PROCESSED_JSON_PATH) -> int:
        """Ingests flat RAG chunks from paddle_sections_output.json into ChromaDB vector collection."""
        path = pathlib.Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Output JSON file not found at: {json_path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        flat_chunks = payload.get("flat_chunks", [])
        return self.ingest_chunks(flat_chunks)

    def ingest_chunks(self, chunks: List[Dict[str, Any]], clear_existing: bool = True) -> int:
        """Ingests a list of dictionary chunk items into ChromaDB."""
        if not chunks:
            return 0

        if clear_existing and self.collection.count() > 0:
            existing_ids = self.collection.get()['ids']
            if existing_ids:
                self.collection.delete(ids=existing_ids)

        ids = []
        documents = []
        metadatas = []

        for c in chunks:
            cid = c["chunk_id"]
            content = c["content"]
            
            # Prepare metadata (flat primitive fields for ChromaDB query filtering)
            meta = {
                "chunk_id": cid,
                "section_number": c.get("section_number", ""),
                "section_title": c.get("section_title", ""),
                "parent_section": c.get("parent_section", ""),
                "page_number": str(c.get("page_number", "1")),
                "pdf_page_number": int(c.get("pdf_page_number", 1)),
                "token_count": int(c.get("token_count", 0)),
                "document_name": c.get("document_name", ""),
                "source_url": c.get("source_url", ""),
                "hierarchy_path_str": " > ".join(c.get("hierarchy_path", [])),
                "layout_metadata_json": json.dumps(c.get("layout_metadata", {}), ensure_ascii=False)
            }

            ids.append(cid)
            documents.append(content)
            metadatas.append(meta)

        # Upsert in batches of 100 to handle large documents efficiently
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

        return len(ids)

    def query_dense(
        self,
        query: str,
        n_results: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Queries the vector store for top-k semantically similar chunks with optional metadata filtering."""
        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, max(1, self.collection.count()))
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)
        
        parsed_results = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(docs)):
                meta = dict(metas[i])
                # Re-parse layout metadata JSON back to Python dict
                if "layout_metadata_json" in meta:
                    try:
                        meta["layout_metadata"] = json.loads(meta["layout_metadata_json"])
                    except Exception:
                        meta["layout_metadata"] = {}

                parsed_results.append({
                    "chunk_id": meta["chunk_id"],
                    "content": docs[i],
                    "score": 1.0 - distances[i] if i < len(distances) else 0.0,
                    "metadata": meta
                })

        return parsed_results

    def count(self) -> int:
        """Returns total vector count in collection."""
        return self.collection.count()
