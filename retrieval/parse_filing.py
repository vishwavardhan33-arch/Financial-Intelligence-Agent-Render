"""Parse financial filing PDFs into structured Markdown using Docling.

Docling preserves heading hierarchy and renders tables as proper Markdown
tables (rather than flattening them into unstructured text), which is what
chunker.py relies on to do structure-aware + atomic-table chunking.

NOTE: Docling downloads layout/table-structure model weights from Hugging
Face on first run. That requires outbound network access to huggingface.co
at deploy time (a one-time download, cached afterward) — this is expected
to work in the actual deployment environment even though it wasn't
reachable in the sandbox used to build this scaffold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedFiling:
    doc_id: str
    markdown: str
    source_path: str


def parse_filing_pdf(pdf_path: str, doc_id: str | None = None) -> ParsedFiling:
    """Convert a filing PDF to structured Markdown via Docling.

    Args:
        pdf_path: path to the 10-K/10-Q/earnings PDF.
        doc_id: identifier to tag downstream chunks with; defaults to the
            file stem (e.g. "ACME_10K_2024.pdf" -> "ACME_10K_2024").
    """
    from docling.document_converter import DocumentConverter  # imported lazily

    doc_id = doc_id or Path(pdf_path).stem

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()

    return ParsedFiling(doc_id=doc_id, markdown=markdown, source_path=pdf_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m retrieval.parse_filing <path_to_filing.pdf>")
        sys.exit(1)

    parsed = parse_filing_pdf(sys.argv[1])
    print(parsed.markdown)
