"""
Hybrid Search Engine combining BM25 Keyword Search and Dense Vector Search via Reciprocal Rank Fusion (RRF).
Crucial for clinical RAG: ensures exact drug names (e.g. Metformin, SGLT-2) & section numbers match while retrieving semantic context.
"""

import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi


class HybridSearchEngine:
    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Medical-aware tokenization: preserves numbers, section IDs, drug names, and hyphenated terms."""
        clean = text.lower()
        # Extract alphanumeric terms and hyphenated/dotted terms (e.g. sglt-2, 1.4.1, hba1c)
        tokens = re.findall(r'[a-z0-9]+(?:[.\-][a-z0-9]+)*', clean)
        return tokens

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Indexes flat chunks list for BM25 keyword retrieval."""
        self.chunks = chunks
        self.corpus_tokens = [self._tokenize(c.get("content", "")) for c in chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def query_bm25(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Performs BM25 keyword search returning top-k chunk matches."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        
        # Sort indices by descending score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                continue
            chunk = dict(self.chunks[idx])
            results.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "score": float(scores[idx]),
                "bm25_rank": rank,
                "metadata": chunk
            })
        return results

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Merges Dense Vector and BM25 search rankings using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        ranks_info: Dict[str, Dict[str, Optional[int]]] = {}

        # 1. Process Dense Results
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k_rrf + rank))
            chunk_map[cid] = item
            if cid not in ranks_info:
                ranks_info[cid] = {"dense_rank": rank, "bm25_rank": None}
            else:
                ranks_info[cid]["dense_rank"] = rank

        # 2. Process BM25 Results
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k_rrf + rank))
            if cid not in chunk_map:
                chunk_map[cid] = item
            if cid not in ranks_info:
                ranks_info[cid] = {"dense_rank": None, "bm25_rank": rank}
            else:
                ranks_info[cid]["bm25_rank"] = rank

        # 3. Sort by aggregated RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        merged_results = []
        for cid in sorted_ids:
            item = dict(chunk_map[cid])
            item["rrf_score"] = rrf_scores[cid]
            item["dense_rank"] = ranks_info[cid]["dense_rank"]
            item["bm25_rank"] = ranks_info[cid]["bm25_rank"]
            merged_results.append(item)

        return merged_results
