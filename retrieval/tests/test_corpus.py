import pytest

from api.rag_setup import LazyHybridIndex
from retrieval.chunker import Chunk
from retrieval.corpus import load_corpus, save_corpus
from retrieval.embeddings import DeterministicFakeEmbedder
from retrieval.vector_store import HybridIndex

CHUNKS = [
    Chunk(text="Revenue grew in fiscal 2024.", chunk_type="text", section="MD&A", doc_id="d", company="Co", fiscal_period="FY2024"),
    Chunk(text="| Item | FY24 |\n| --- | --- |\n| Revenue | 100 |", chunk_type="table", section="Results", doc_id="d", company="Co"),
]


def test_roundtrip_without_vectors(tmp_path):
    path = str(tmp_path / "c.jsonl")
    save_corpus(path, CHUNKS)
    chunks, vectors = load_corpus(path)
    assert [c.text for c in chunks] == [c.text for c in CHUNKS]
    assert chunks[1].chunk_type == "table" and chunks[0].fiscal_period == "FY2024"
    assert vectors is None


def test_roundtrip_with_vectors(tmp_path):
    path = str(tmp_path / "c.jsonl")
    vecs = DeterministicFakeEmbedder().embed([c.text for c in CHUNKS])
    save_corpus(path, CHUNKS, vecs)
    _, loaded = load_corpus(path)
    assert loaded is not None and len(loaded) == 2
    assert loaded[0] == pytest.approx(vecs[0], abs=1e-5)


def test_save_rejects_mismatched_vectors(tmp_path):
    with pytest.raises(ValueError):
        save_corpus(str(tmp_path / "c.jsonl"), CHUNKS, [[0.1]])


def test_index_build_uses_precomputed_vectors_without_embedding():
    class ExplodingEmbedder(DeterministicFakeEmbedder):
        def embed(self, texts):
            raise AssertionError("should not embed when vectors are supplied")

    vecs = DeterministicFakeEmbedder().embed([c.text for c in CHUNKS])
    index = HybridIndex(embedder=ExplodingEmbedder())
    index.build(CHUNKS, vectors=vecs)
    # query embedding still goes through embed_query (deterministic, not exploding)
    assert index.hybrid_search("Revenue grew", top_k=1)


def test_lazy_index_builds_only_on_first_search(tmp_path):
    path = str(tmp_path / "c.jsonl")
    save_corpus(path, CHUNKS)
    built = []

    def factory():
        built.append(1)
        return DeterministicFakeEmbedder()

    lazy = LazyHybridIndex(path, embedder_factory=factory)
    assert not lazy.ready and built == []
    results = lazy.hybrid_search("fiscal 2024 revenue", top_k=2)
    assert lazy.ready and built == [1] and lazy.chunk_count == 2
    assert results
    lazy.hybrid_search("again", top_k=1)
    assert built == [1]  # built once
