"""Prebuilt corpus file: one JSON object per line, one line per chunk.

Why a file: PDF parsing (Docling) needs PyTorch and gigabytes of RAM, which a
small web service does not have. So parsing + chunking (+ optionally
embedding) happens once on your own machine via `retrieval.export_corpus`,
and the deployed app just loads the result.

Each line: the Chunk fields, plus an optional "vector". If every line has a
vector, startup skips embedding entirely.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from retrieval.chunker import Chunk


def save_corpus(path: str, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
    if vectors is not None and len(vectors) != len(chunks):
        raise ValueError("vectors and chunks must be the same length")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            record = asdict(chunk)
            if vectors is not None:
                record["vector"] = [round(v, 6) for v in vectors[i]]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_corpus(path: str) -> tuple[list[Chunk], list[list[float]] | None]:
    """Returns (chunks, vectors). vectors is None unless every line has one."""
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        vector = record.pop("vector", None)
        chunks.append(Chunk(**record))
        if vector is not None:
            vectors.append(vector)
    return chunks, (vectors if chunks and len(vectors) == len(chunks) else None)
