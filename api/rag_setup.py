"""Builds the filings index for the deployed app, lazily.

The embedding model is only loaded the first time the index is actually
needed (or when the warm-up thread asks for it), so importing the app, running
tests, or hitting /health never loads a model or touches the network.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

DEFAULT_CORPUS_PATH = "data/corpus.jsonl"


class LazyHybridIndex:
    """Quacks like HybridIndex (`hybrid_search`), but builds on first use."""

    def __init__(self, corpus_path: str, embedder_factory=None):
        self.corpus_path = corpus_path
        self._embedder_factory = embedder_factory
        self._index = None
        self._lock = threading.Lock()
        self.chunk_count = 0

    @property
    def ready(self) -> bool:
        return self._index is not None

    def _get(self):
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._index = self._build()
        return self._index

    def _build(self):
        from retrieval.corpus import load_corpus
        from retrieval.vector_store import HybridIndex

        chunks, vectors = load_corpus(self.corpus_path)
        if self._embedder_factory is not None:
            embedder = self._embedder_factory()
        else:
            from retrieval.embeddings import FastEmbedder

            embedder = FastEmbedder()
        index = HybridIndex(embedder=embedder)
        index.build(chunks, vectors)
        self.chunk_count = len(chunks)
        return index

    def warm_up(self) -> None:
        self._get()

    def hybrid_search(self, *args, **kwargs):
        return self._get().hybrid_search(*args, **kwargs)


def build_default_rag() -> tuple[LazyHybridIndex | None, object | None]:
    """Returns (index, reranker). Index is None when no corpus file exists, in
    which case the planner's rag steps fail cleanly with a clear message."""
    corpus_path = os.environ.get("CORPUS_PATH", DEFAULT_CORPUS_PATH)
    if not Path(corpus_path).exists():
        return None, None

    reranker = None
    if os.environ.get("ENABLE_RERANKER", "").lower() in ("1", "true", "yes"):
        from retrieval.vector_store import FastEmbedReranker

        reranker = FastEmbedReranker()
    return LazyHybridIndex(corpus_path), reranker
