"""Embedding interface.

BGEEmbedder wraps sentence-transformers with a BGE model — this is what
runs in deployment, and needs network access to download model weights
the first time.

DeterministicFakeEmbedder produces a stable, content-derived vector with
no model or network dependency at all. It exists purely so the indexing/
retrieval/fusion logic in vector_store.py can be unit-tested in
environments without model access — it is NOT semantically meaningful and
must never be used outside tests.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

EMBEDDING_DIM = 384  # matches BAAI/bge-small-en-v1.5


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class BGEEmbedder:
    """Real embedder for deployment. Requires `sentence-transformers` and
    network access to Hugging Face on first run (weights are cached after)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # imported lazily

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        # BGE models recommend a query instruction prefix for asymmetric
        # search; documents are embedded as-is, queries should use
        # embed_query() instead when calling from the retrieval layer.
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        instruction = "Represent this sentence for searching relevant passages: "
        return self.model.encode([instruction + query], normalize_embeddings=True)[0].tolist()


class FastEmbedder:
    """Same BGE model as BGEEmbedder, run through ONNX Runtime via `fastembed`.

    This is what the deployed web service uses: no PyTorch, so it fits in a
    512 MB instance (sentence-transformers + torch does not). The model
    (~130 MB) is downloaded once and cached; the Dockerfile pre-downloads it
    into the image so cold starts don't wait on Hugging Face.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # imported lazily

        self.model = TextEmbedding(model_name=model_name)
        self.dim = EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, query: str) -> list[float]:
        # fastembed applies BGE's query instruction itself.
        return next(iter(self.model.query_embed(query))).tolist()


class DeterministicFakeEmbedder:
    """Test-only stand-in. Same text -> same vector, unrelated texts ->
    unrelated vectors, but with NO semantic meaning whatsoever."""

    dim = EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_vector(query)

    def _hash_vector(self, text: str) -> list[float]:
        vec = []
        h = text.encode("utf-8")
        for i in range(self.dim):
            digest = hashlib.sha256(h + i.to_bytes(4, "little")).digest()
            # map first 4 bytes of digest to a float in [-1, 1]
            val = int.from_bytes(digest[:4], "little") / 2**32 * 2 - 1
            vec.append(val)
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]
