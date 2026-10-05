"""Turn a folder of filings into the corpus file the deployed app loads.

    # PDFs (needs: pip install -r requirements-ingest.txt)
    python -m retrieval.export_corpus path/to/filings/ --company "Acme Corp" \
        --fiscal-period FY2024 --out data/corpus.jsonl

    # Markdown files work too, with no heavy dependencies
    python -m retrieval.export_corpus path/to/markdown/ --company "Acme Corp" --out data/corpus.jsonl

Add --embed to precompute vectors with the same model the server uses, so
the server does not have to embed anything on startup.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.chunker import chunk_filing
from retrieval.corpus import save_corpus


def export(source_dir: str, company: str | None, fiscal_period: str | None, out: str, embed: bool) -> int:
    all_chunks = []
    for path in sorted(Path(source_dir).iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from retrieval.parse_filing import parse_filing_pdf  # heavy import, only when needed

            parsed = parse_filing_pdf(str(path))
            markdown, doc_id = parsed.markdown, parsed.doc_id
        elif suffix in (".md", ".markdown"):
            markdown, doc_id = path.read_text(encoding="utf-8"), path.stem
        else:
            continue
        chunks = chunk_filing(markdown, doc_id=doc_id, company=company, fiscal_period=fiscal_period)
        print(f"{path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    vectors = None
    if embed and all_chunks:
        from retrieval.embeddings import FastEmbedder

        vectors = FastEmbedder().embed([c.text for c in all_chunks])

    save_corpus(out, all_chunks, vectors)
    print(f"Wrote {len(all_chunks)} chunks to {out}" + (" (with vectors)" if vectors else ""))
    return len(all_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("--company", default=None)
    parser.add_argument("--fiscal-period", default=None)
    parser.add_argument("--out", default="data/corpus.jsonl")
    parser.add_argument("--embed", action="store_true", help="precompute embedding vectors")
    args = parser.parse_args()
    export(args.source_dir, args.company, args.fiscal_period, args.out, args.embed)
