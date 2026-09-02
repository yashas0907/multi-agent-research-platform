"""Unit tests: RAG pipeline — parsing, chunking, embeddings, retriever, grounding."""
from __future__ import annotations

import pytest

from app.agents.evidence import _grounded
from app.retrieval.chunker import chunk_text, clean_text, estimate_tokens
from app.retrieval.embeddings import MockEmbeddingProvider
from app.retrieval.parser import DocumentIngestError, parse_document
from app.retrieval.vector_store import SQLiteVectorStore


class TestParser:
    def test_markdown_with_title(self):
        doc = parse_document("note.md", b"# My Title\n\nBody text here.")
        assert doc.title == "My Title"
        assert "Body text" in doc.text

    def test_rejects_exe(self):
        with pytest.raises(DocumentIngestError):
            parse_document("evil.exe", b"MZ...")

    def test_rejects_binary_as_text(self):
        with pytest.raises(DocumentIngestError):
            parse_document("bad.txt", bytes([0xFF, 0xFE, 0x00, 0x01]))

    def test_json_flattened(self):
        doc = parse_document("data.json", b'{"name": "RAGAS", "metrics": ["faithfulness"]}')
        assert "RAGAS" in doc.text and "faithfulness" in doc.text

    def test_html_parsed(self):
        html = b"<html><head><title>Paper</title></head><body><p>RAGAS evaluates RAG pipelines with reference-free metrics.</p></body></html>"
        doc = parse_document("page.html", html)
        assert doc.title == "Paper"
        assert "reference-free metrics" in doc.text

    def test_empty_rejected(self):
        with pytest.raises(DocumentIngestError):
            parse_document("empty.txt", b"   \n  \n")


class TestChunker:
    def test_chunks_with_overlap(self):
        text = ("RAGAS evaluates faithfulness. " * 60) + "\n\n" + ("TruLens measures groundedness. " * 60)
        chunks = chunk_text(text, "d1", target_words=40, overlap_words=10, min_words=5)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.word_count >= 5

    def test_short_text_single_chunk(self):
        chunks = chunk_text("One short paragraph only.", "d1")
        assert len(chunks) == 1

    def test_hard_cap(self):
        text = "\n\n".join(f"Paragraph {i} with some words to fill space here." for i in range(2000))
        chunks = chunk_text(text, "d1")
        assert len(chunks) <= 500

    def test_clean_normalizes(self):
        assert "  " not in clean_text("a    b\r\n\r\n\r\n\r\nc")


class TestMockEmbeddings:
    async def test_deterministic_and_normalized(self):
        p = MockEmbeddingProvider(dim=64)
        v1 = await p.embed_query("rag evaluation framework")
        v2 = await p.embed_query("rag evaluation framework")
        assert v1 == v2
        norm = sum(x * x for x in v1) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    async def test_similar_texts_closer_than_unrelated(self):
        p = MockEmbeddingProvider(dim=256)
        from app.retrieval.vector_store import cosine_similarity
        a = await p.embed_query("ragas faithfulness evaluation")
        b = await p.embed_query("ragas evaluation faithfulness metrics")
        c = await p.embed_query("cooking pasta recipes italian")
        assert cosine_similarity(a, b) > cosine_similarity(a, c)


class TestVectorStoreRoundtrip:
    async def test_add_query_filter_delete(self):
        store = SQLiteVectorStore()
        emb = MockEmbeddingProvider(dim=32)
        v_doc = await emb.embed_query("ragas faithfulness")
        v_q = await emb.embed_query("ragas faithfulness metrics")
        await store.add_chunks([
            {"document_id": "dA", "chunk_index": 0, "text": "chunk about ragas", "embedding": v_doc, "metadata": {}},
            {"document_id": "dB", "chunk_index": 0, "text": "chunk about pasta", "embedding": await emb.embed_query("cooking pasta"), "metadata": {}},
        ])
        assert await store.count() == 2
        hits = await store.query(v_q, top_k=2)
        assert hits[0].document_id == "dA"
        # metadata filter
        only_b = await store.query(v_q, top_k=2, document_ids=["dB"])
        assert len(only_b) == 1 and only_b[0].document_id == "dB"
        # empty filter returns nothing
        assert await store.query(v_q, document_ids=[]) == []
        # delete
        assert await store.delete_document("dA") == 1
        assert await store.count() == 1


class TestGroundingGuard:
    def test_exact_match_passes(self):
        text = "RAGAS evaluates faithfulness and answer relevance in RAG pipelines."
        assert _grounded("RAGAS evaluates faithfulness and answer relevance", text) is True

    def test_fabricated_snippet_rejected(self):
        text = "RAGAS evaluates faithfulness and answer relevance in RAG pipelines."
        assert _grounded("GPT-5 was released yesterday with 200k context", text) is False

    def test_whitespace_normalized(self):
        text = "The  framework  measures  groundedness."
        assert _grounded("The framework measures groundedness.", text) is True


class TestTokenEstimate:
    def test_estimate(self):
        assert estimate_tokens("abcd" * 10) == 10
        assert estimate_tokens("") == 1
