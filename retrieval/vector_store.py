"""Hybrid retrieval over indexed filing chunks.

Pipeline (Step 5 spec):
  1. Dense search in Qdrant (top ~20) + BM25 sparse search (top ~20)
  2. Reciprocal Rank Fusion (RRF) combines the two rankings
  3. Metadata filtering (company/fiscal_period/chunk_type) applied to the
     dense search directly, and to the BM25 candidate pool post-hoc
  4. Reranker cross-encodes the fused candidate set against the query,
     keep top_k
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi

from retrieval.chunker import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class BGEReranker:
    """Real cross-encoder reranker for deployment. Needs network access to
    Hugging Face on first run."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder  # imported lazily

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self.model.predict(pairs)]


class FastEmbedReranker:
    """Light cross-encoder reranker (ONNX, via fastembed). Optional in
    deployment because it adds ~100 MB of RAM; enable with ENABLE_RERANKER=true."""

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # imported lazily

        self.model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(x) for x in self.model.rerank(query, texts)]


class FakeReranker:
    """Test-only stand-in: scores by token overlap (Jaccard similarity)
    between query and candidate text. No model, no network — used to
    verify the surrounding pipeline logic, not retrieval quality."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q_tokens = set(_tokenize(query))
        scores = []
        for t in texts:
            t_tokens = set(_tokenize(t))
            if not q_tokens or not t_tokens:
                scores.append(0.0)
                continue
            overlap = len(q_tokens & t_tokens) / len(q_tokens | t_tokens)
            scores.append(overlap)
        return scores


class HybridIndex:
    def __init__(self, embedder, collection_name: str = "filings", client: QdrantClient | None = None):
        self.embedder = embedder
        self.collection_name = collection_name
        self.client = client or QdrantClient(":memory:")
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._id_map: dict[str, Chunk] = {}

    def build(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        """Index chunks. Pass precomputed `vectors` (same order as chunks) to
        skip embedding at startup; otherwise the embedder is called here."""
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        self._chunks = chunks
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=self.embedder.dim, distance=qmodels.Distance.COSINE),
        )

        if vectors is None:
            vectors = self.embedder.embed([c.text for c in chunks])
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid4())
            self._id_map[point_id] = chunk
            points.append(qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "chunk_type": chunk.chunk_type,
                    "section": chunk.section,
                    "doc_id": chunk.doc_id,
                    "company": chunk.company,
                    "fiscal_period": chunk.fiscal_period,
                },
            ))
        self.client.upsert(collection_name=self.collection_name, points=points)

        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _build_filter(self, company: str | None, fiscal_period: str | None, chunk_type: str | None):
        conditions = []
        if company:
            conditions.append(qmodels.FieldCondition(key="company", match=qmodels.MatchValue(value=company)))
        if fiscal_period:
            conditions.append(qmodels.FieldCondition(key="fiscal_period", match=qmodels.MatchValue(value=fiscal_period)))
        if chunk_type:
            conditions.append(qmodels.FieldCondition(key="chunk_type", match=qmodels.MatchValue(value=chunk_type)))
        return qmodels.Filter(must=conditions) if conditions else None

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 20,
        company: str | None = None,
        fiscal_period: str | None = None,
        chunk_type: str | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        if self._bm25 is None:
            raise RuntimeError("Index not built yet — call build() first.")

        query_vector = (
            self.embedder.embed_query(query)
            if hasattr(self.embedder, "embed_query")
            else self.embedder.embed([query])[0]
        )
        qdrant_filter = self._build_filter(company, fiscal_period, chunk_type)

        dense_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=candidate_pool,
            query_filter=qdrant_filter,
        ).points
        dense_ranking = [hit.id for hit in dense_hits]

        # BM25 over the full corpus, then filtered post-hoc to match the
        # same metadata scope as the dense search (rank_bm25 has no
        # native filtering support).
        allowed_ids = None
        if qdrant_filter is not None:
            allowed_ids = {
                pid for pid, chunk in self._id_map.items()
                if (not company or chunk.company == company)
                and (not fiscal_period or chunk.fiscal_period == fiscal_period)
                and (not chunk_type or chunk.chunk_type == chunk_type)
            }

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        id_order = list(self._id_map.keys())
        scored = list(zip(id_order, bm25_scores))
        if allowed_ids is not None:
            scored = [(pid, s) for pid, s in scored if pid in allowed_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        bm25_ranking = [pid for pid, _ in scored[:candidate_pool]]

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        for rank, pid in enumerate(dense_ranking):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, pid in enumerate(bm25_ranking):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [(pid, score) for pid, score in fused[:candidate_pool]]

        if reranker is not None and candidates:
            texts = [self._id_map[pid].text for pid, _ in candidates]
            rerank_scores = reranker.score(query, texts)
            reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            results = [
                RetrievedChunk(chunk=self._id_map[pid], score=rscore)
                for (pid, _fused_score), rscore in reranked[:top_k]
            ]
        else:
            results = [
                RetrievedChunk(chunk=self._id_map[pid], score=score)
                for pid, score in candidates[:top_k]
            ]

        return results
