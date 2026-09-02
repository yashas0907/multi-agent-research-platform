"""Text processing utilities for the RAG pipeline: cleaning and chunking.

Pipeline: Document → parse → clean → chunk (with overlap) → metadata →
embed → vector store → retriever → evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def clean_text(text: str) -> str:
    """Normalize whitespace and remove obvious boilerplate noise."""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # drop very short junk lines (likely nav/headers) but keep headings/paragraph starts
    lines = text.split("\n")
    kept = [ln if len(ln.strip()) > 2 else "" for ln in lines]
    return "\n".join(kept).strip()


@dataclass
class Chunk:
    index: int
    text: str
    doc_id: str
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def chunk_text(
    text: str,
    doc_id: str,
    *,
    target_words: int = 180,
    overlap_words: int = 40,
    min_words: int = 20,
) -> list[Chunk]:
    """Paragraph-aware sliding-window chunking.

    Splits on paragraph boundaries first, then merges to ~target_words with
    overlap so claims spanning boundaries stay retrievable.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    # split long paragraphs into sentence-ish pieces
    pieces: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= target_words:
            pieces.append(para)
            continue
        for i in range(0, len(words), target_words - overlap_words):
            piece = " ".join(words[i : i + target_words])
            if piece.strip():
                pieces.append(piece)

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_words = 0

    def flush(force: bool = False) -> None:
        nonlocal buffer, buffer_words
        if buffer and (force or buffer_words >= min_words):
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text="\n\n".join(buffer),
                    doc_id=doc_id,
                )
            )
        buffer, buffer_words = [], 0

    for piece in pieces:
        w = len(piece.split())
        if buffer_words + w > target_words and buffer_words >= min_words:
            # start new chunk; keep overlap from previous buffer tail
            tail = " ".join(" ".join(buffer).split()[-overlap_words:])
            flush()
            if tail:
                buffer = [tail]
                buffer_words = len(tail.split())
        buffer.append(piece)
        buffer_words += w
    # final flush with force=True so short documents still produce one chunk
    flush(force=True)

    # enforce a hard cap on chunk count for pathological docs
    return chunks[:500]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used for budget accounting."""
    return max(1, len(text) // 4)
