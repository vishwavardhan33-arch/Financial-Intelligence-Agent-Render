"""Hybrid chunking for parsed financial filings.

Input is Markdown text (as produced by Docling: headers preserved, tables
rendered as Markdown tables). Two chunk types come out:

  - "text"  : narrative content, split by header hierarchy first, then
              recursively by paragraph/size within a section.
  - "table" : each Markdown table becomes exactly one chunk, never split,
              with a synthetic caption prepended (pulled from the nearest
              preceding header) so it's retrievable by natural-language
              queries even though raw tables have little self-describing
              text of their own.

This is deliberately dependency-light (regex + simple recursion) so it can
be unit-tested without Docling, embeddings, or a vector DB in the loop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TARGET_CHUNK_TOKENS = 400   # approx, using whitespace-word count as a token proxy
CHUNK_OVERLAP_TOKENS = 50

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.*\|$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    chunk_type: str          # "text" | "table"
    section: str | None = None
    doc_id: str | None = None
    company: str | None = None
    fiscal_period: str | None = None
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


def _word_count(text: str) -> int:
    return len(text.split())


def _split_into_sections(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (header_text, section_body) pairs using header lines.

    Content before the first header is returned under an empty header.
    """
    matches = list(_HEADER_RE.finditer(markdown))
    if not matches:
        return [("", markdown)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        sections.append(("", markdown[: matches[0].start()]))

    for i, m in enumerate(matches):
        header_text = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append((header_text, markdown[body_start:body_end]))

    return sections


def _extract_tables(section_body: str) -> list[tuple[int, int, str]]:
    """Find contiguous blocks of Markdown table rows within a section body.

    Returns a list of (start_offset, end_offset, table_text).
    """
    tables: list[tuple[int, int, str]] = []
    lines = section_body.splitlines(keepends=True)
    offset = 0
    i = 0
    while i < len(lines):
        if _TABLE_ROW_RE.match(lines[i].rstrip("\n")):
            start = offset
            block_lines = []
            while i < len(lines) and (
                _TABLE_ROW_RE.match(lines[i].rstrip("\n")) or lines[i].strip() == ""
                and block_lines
                and i + 1 < len(lines)
                and _TABLE_ROW_RE.match(lines[i + 1].rstrip("\n"))
            ):
                block_lines.append(lines[i])
                offset += len(lines[i])
                i += 1
            table_text = "".join(block_lines).strip("\n")
            end = start + sum(len(l) for l in block_lines)
            tables.append((start, end, table_text))
            continue
        offset += len(lines[i])
        i += 1
    return tables


def _split_paragraphs_to_size(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = _word_count(para)
        if current and current_len + para_len > target_tokens:
            chunks.append("\n\n".join(current))
            # carry the tail of the previous chunk forward as overlap
            overlap_words = " ".join(current[-1].split()[-overlap_tokens:])
            current = [overlap_words, para] if overlap_words else [para]
            current_len = _word_count(overlap_words) + para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_filing(
    markdown: str,
    doc_id: str,
    company: str | None = None,
    fiscal_period: str | None = None,
) -> list[Chunk]:
    """Chunk a Docling-parsed filing into hybrid text/table chunks."""
    chunks: list[Chunk] = []

    for header, body in _split_into_sections(markdown):
        tables = _extract_tables(body)

        if not tables:
            for piece in _split_paragraphs_to_size(body, TARGET_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS):
                chunks.append(Chunk(
                    text=piece,
                    chunk_type="text",
                    section=header or None,
                    doc_id=doc_id,
                    company=company,
                    fiscal_period=fiscal_period,
                ))
            continue

        # Section has one or more tables: emit surrounding text as text chunks,
        # and each table as its own atomic chunk with a synthetic caption.
        cursor = 0
        for start, end, table_text in tables:
            preceding_text = body[cursor:start].strip()
            if preceding_text:
                for piece in _split_paragraphs_to_size(preceding_text, TARGET_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS):
                    chunks.append(Chunk(
                        text=piece,
                        chunk_type="text",
                        section=header or None,
                        doc_id=doc_id,
                        company=company,
                        fiscal_period=fiscal_period,
                    ))

            caption = f"Table from section '{header}'" if header else "Table"
            if company:
                caption += f", {company}"
            if fiscal_period:
                caption += f", {fiscal_period}"

            chunks.append(Chunk(
                text=f"{caption}\n\n{table_text}",
                chunk_type="table",
                section=header or None,
                doc_id=doc_id,
                company=company,
                fiscal_period=fiscal_period,
            ))
            cursor = end

        trailing_text = body[cursor:].strip()
        if trailing_text:
            for piece in _split_paragraphs_to_size(trailing_text, TARGET_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS):
                chunks.append(Chunk(
                    text=piece,
                    chunk_type="text",
                    section=header or None,
                    doc_id=doc_id,
                    company=company,
                    fiscal_period=fiscal_period,
                ))

    return chunks
