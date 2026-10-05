"""End-to-end indexing: PDF filings -> Docling parse -> hybrid chunk -> embed -> Qdrant.

Usage:
    python -m retrieval.index_filings path/to/filings_dir/ --company "Acme Corp" --fiscal-period FY2024

For quick local testing without Docling/BGE/network access, see
retrieval/tests/ for the same pipeline run against a pre-made Markdown
sample with the fake embedder — that's what's actually been verified in
this sandbox. This script wires in the *real* components for deployment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.chunker import chunk_filing
from retrieval.embeddings import BGEEmbedder
from retrieval.parse_filing import parse_filing_pdf
from retrieval.vector_store import HybridIndex


def index_directory(pdf_dir: str, company: str | None, fiscal_period: str | None, collection_name: str = "filings") -> HybridIndex:
    embedder = BGEEmbedder()
    index = HybridIndex(embedder=embedder, collection_name=collection_name)

    all_chunks = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        parsed = parse_filing_pdf(str(pdf_path))
        chunks = chunk_filing(
            parsed.markdown,
            doc_id=parsed.doc_id,
            company=company,
            fiscal_period=fiscal_period,
        )
        print(f"{pdf_path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    index.build(all_chunks)
    print(f"Indexed {len(all_chunks)} total chunks into collection '{collection_name}'.")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", help="Directory containing filing PDFs")
    parser.add_argument("--company", default=None)
    parser.add_argument("--fiscal-period", default=None)
    parser.add_argument("--collection", default="filings")
    args = parser.parse_args()

    index_directory(args.pdf_dir, args.company, args.fiscal_period, args.collection)
