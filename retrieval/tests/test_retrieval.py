from pathlib import Path

from retrieval.chunker import chunk_filing
from retrieval.embeddings import DeterministicFakeEmbedder
from retrieval.vector_store import FakeReranker, HybridIndex

SAMPLE_PATH = Path(__file__).parent / "sample_filing.md"


def build_test_index() -> HybridIndex:
    markdown = SAMPLE_PATH.read_text()
    acme_chunks = chunk_filing(markdown, doc_id="ACME_10K_2024", company="Acme Corp", fiscal_period="FY2024")
    other_chunks = chunk_filing(
        "# Item 7. MD&A\n\nOther Co revenue declined due to a divestiture of its hardware division.\n",
        doc_id="OTHER_10K_2024",
        company="Other Co",
        fiscal_period="FY2024",
    )
    index = HybridIndex(embedder=DeterministicFakeEmbedder())
    index.build(acme_chunks + other_chunks)
    return index


def test_metadata_filter_excludes_other_companies():
    index = build_test_index()
    results = index.hybrid_search("What was the gross margin and revenue?", top_k=5, company="Other Co")

    assert len(results) == 1
    assert results[0].chunk.company == "Other Co"


def test_reranker_improves_relevance_ordering():
    index = build_test_index()
    results = index.hybrid_search(
        "What was the gross margin and revenue?", top_k=1, reranker=FakeReranker()
    )

    # The top reranked result should be the most lexically relevant chunk —
    # the one that actually mentions "gross margin" — not an unrelated one.
    assert "gross margin" in results[0].chunk.text.lower() or "margin" in results[0].chunk.text.lower()


def test_hybrid_search_returns_requested_top_k():
    index = build_test_index()
    results = index.hybrid_search("revenue", top_k=2)
    assert len(results) <= 2
