from pathlib import Path

from retrieval.chunker import chunk_filing

SAMPLE_PATH = Path(__file__).parent / "sample_filing.md"


def load_sample_markdown() -> str:
    return SAMPLE_PATH.read_text()


def test_table_is_atomic_and_captioned():
    chunks = chunk_filing(load_sample_markdown(), doc_id="ACME_10K_2024", company="Acme Corp", fiscal_period="FY2024")
    table_chunks = [c for c in chunks if c.chunk_type == "table"]

    assert len(table_chunks) == 1, "expected exactly one table chunk"
    table = table_chunks[0]

    # The whole table must survive as one chunk — no row split across chunks.
    assert "Revenue" in table.text
    assert "Gross profit" in table.text
    assert "Operating income" in table.text

    # Synthetic caption must be present so the table is retrievable by NL queries.
    assert "Results of Operations" in table.text
    assert "Acme Corp" in table.text
    assert "FY2024" in table.text


def test_text_chunks_retain_section_metadata():
    chunks = chunk_filing(load_sample_markdown(), doc_id="ACME_10K_2024", company="Acme Corp", fiscal_period="FY2024")
    text_chunks = [c for c in chunks if c.chunk_type == "text"]

    assert all(c.doc_id == "ACME_10K_2024" for c in text_chunks)
    assert any(c.section == "Item 7. Management's Discussion and Analysis" for c in text_chunks)
    assert any(c.section == "Item 8. Financial Statements" for c in text_chunks)


def test_no_content_lost():
    markdown = load_sample_markdown()
    chunks = chunk_filing(markdown, doc_id="ACME_10K_2024")

    # Every distinctive phrase from the source should appear in some chunk.
    key_phrases = ["cloud segment", "Gross margin improved", "U.S. GAAP", "4,235,102"]
    all_text = "\n".join(c.text for c in chunks)
    for phrase in key_phrases:
        assert phrase in all_text, f"lost content: {phrase!r}"
