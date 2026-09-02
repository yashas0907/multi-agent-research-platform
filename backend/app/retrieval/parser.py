"""Document parsing and the retriever (hybrid: vector + keyword, with metadata
filtering and score gating — never blindly trust top-k).

Parser supports: .txt, .md, .pdf (text layer), .html, .json. Binary/unknown
formats are rejected with a clear error (never silently accepted).
"""
from __future__ import annotations

import json
import pathlib
import zipfile
from dataclasses import dataclass

from app.core.errors import DocumentIngestError
from app.retrieval.chunker import Chunk, chunk_text, clean_text


@dataclass
class ParsedDocument:
    text: str
    title: str | None = None
    author: str | None = None
    page_count: int | None = None
    content_type: str = "text/plain"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm", ".json"}


def parse_document(filename: str, raw: bytes) -> ParsedDocument:
    """Parse uploaded bytes into text. Raises DocumentIngestError on failure."""
    ext = pathlib.Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentIngestError(
            f"unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        return _parse_pdf(raw)
    if ext in (".html", ".htm"):
        return _parse_html(raw)
    if ext == ".json":
        return _parse_json(raw)
    return _parse_plain(raw, ext)


def _parse_plain(raw: bytes, ext: str) -> ParsedDocument:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DocumentIngestError("file is not valid UTF-8 text") from exc
    # Front-matter style title extraction for markdown
    title = None
    for line in text.splitlines()[:5]:
        stripped = line.strip()
        if stripped.startswith("#") and len(stripped) > 2:
            title = stripped.lstrip("# ").strip()[:200]
            break
    if not text.strip():
        raise DocumentIngestError("document contains no text")
    return ParsedDocument(
        text=clean_text(text),
        title=title,
        content_type="text/markdown" if ext == ".md" else "text/plain",
    )


def _parse_json(raw: bytes) -> ParsedDocument:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DocumentIngestError("invalid JSON document") from exc

    def _flatten(obj: object, prefix: str = "") -> list[str]:
        parts: list[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    parts.append(f"{key}:")
                    parts.extend(_flatten(v, key))
                else:
                    parts.append(f"{key}: {v}")
        elif isinstance(obj, list):
            for item in obj:
                parts.extend(_flatten(item, prefix))
        else:
            parts.append(str(obj))
        return parts

    text = "\n".join(_flatten(data))
    return ParsedDocument(text=clean_text(text), content_type="application/json")


def _parse_html(raw: bytes) -> ParsedDocument:
    import re

    from app.tools.web_fetch import strip_html

    try:
        html = raw.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover
        raise DocumentIngestError("could not read HTML file") from exc
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    text = strip_html(html)
    if not text.strip():
        raise DocumentIngestError("HTML file contains no extractable text")
    return ParsedDocument(
        text=text,
        title=title_match.group(1).strip()[:200] if title_match else None,
        content_type="text/html",
    )


def _parse_pdf(raw: bytes) -> ParsedDocument:
    """Minimal PDF text extraction (no external deps).

    Supports PDFs with a plain text layer. Encrypted or scanned PDFs raise
    DocumentIngestError with a clear message (use OCR upstream instead).
    """
    if not raw.startswith(b"%PDF"):
        raise DocumentIngestError("not a valid PDF file")
    if b"/Encrypt" in raw:
        raise DocumentIngestError("encrypted PDFs are not supported")

    try:
        text = _extract_pdf_text(raw)
    except Exception as exc:
        raise DocumentIngestError(f"PDF parsing failed: {exc}") from exc
    text = clean_text(text)
    if len(text.strip()) < 20:
        raise DocumentIngestError(
            "PDF has no extractable text layer (likely scanned images)"
        )
    page_count = raw.count(b"/Type /Page") or None
    return ParsedDocument(text=text, page_count=page_count, content_type="application/pdf")


def _extract_pdf_text(raw: bytes) -> str:
    """Extract text from uncompressed and Flate-compressed PDF streams."""
    import re
    import zlib

    parts: list[str] = []
    # find stream objects
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        data = match.group(1)
        try:
            data = zlib.decompress(data)
        except zlib.error:
            pass  # uncompressed stream
        for tmatch in re.finditer(rb"\((?:\\.|[^\\()])*\)", data):
            token = tmatch.group(0)[1:-1]
            token = re.sub(rb"\\([()\\])", rb"\1", token)
            try:
                parts.append(token.decode("latin-1"))
            except UnicodeDecodeError:
                continue
        parts.append("\n")
    return " ".join(parts).strip()
