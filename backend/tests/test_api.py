"""API tests: research lifecycle, documents, health — against httpx ASGI client."""
from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestHealth:
    async def test_health_ok(self, client):
        r = await client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "degraded")
        assert body["llm_provider"] == "mock"
        assert body["database"] == "ok"


class TestResearchLifecycle:
    async def test_create_status_report_flow(self, client):
        r = await client.post(
            "/api/research",
            json={
                "question": "What are the leading approaches to RAG evaluation and their tradeoffs?",
                "depth": "quick",
                "report_format": "detailed",
            },
        )
        assert r.status_code == 201
        session_id = r.json()["session_id"]
        assert r.json()["status"] == "pending"

        # poll until terminal state (mock providers make this fast)
        for _ in range(120):
            s = await client.get(f"/api/research/{session_id}/status")
            assert s.status_code == 200
            status = s.json()["status"]
            if status in ("report_ready", "failed", "cancelled", "completed_partial"):
                break
            await asyncio.sleep(0.25)
        assert status == "report_ready", f"unexpected status {status}"

        # report endpoint
        rep = await client.get(f"/api/research/{session_id}/report")
        assert rep.status_code == 200
        report = rep.json()
        assert report["executive_summary"]
        assert report["question"].startswith("What are the leading")
        assert isinstance(report["key_findings"], list)
        assert isinstance(report["sources"], list)

        # events endpoint (live trace)
        ev = await client.get(f"/api/research/{session_id}/events")
        assert ev.status_code == 200
        assert len(ev.json()["events"]) > 5

        # subquestions endpoint
        sq = await client.get(f"/api/research/{session_id}/subquestions")
        assert sq.status_code == 200
        assert len(sq.json()["subquestions"]) >= 1

    async def test_report_not_ready_conflict(self, client):
        r = await client.post(
            "/api/research", json={"question": "Compare vector databases for scale and cost"}
        )
        sid = r.json()["session_id"]
        # may still be running right after creation → 409 expected; if the
        # mock pipeline already finished, 200 is also correct.
        rep = await client.get(f"/api/research/{sid}/report")
        assert rep.status_code in (200, 409)
        # drain the background task so teardown is clean
        for _ in range(120):
            s = (await client.get(f"/api/research/{sid}/status")).json()
            if s["status"] in ("report_ready", "failed", "cancelled", "completed_partial"):
                return
            await asyncio.sleep(0.25)
        pytest.fail("research job did not reach terminal state")

    async def test_validation_rejects_short_question(self, client):
        r = await client.post("/api/research", json={"question": "too short"})
        assert r.status_code == 422

    async def test_404_unknown_session(self, client):
        assert (await client.get("/api/research/nope/status")).status_code == 404
        assert (await client.get("/api/research/nope/report")).status_code == 404

    async def test_cancel_unknown_session(self, client):
        r = await client.post("/api/research/nope/cancel")
        assert r.status_code == 404


class TestDocuments:
    async def test_upload_and_flow(self, client):
        content = (
            "# RAG Evaluation Notes\n\n"
            "RAGAS is a reference-free evaluation framework for RAG pipelines. "
            "It measures faithfulness, answer relevance, and context relevance. "
            "TruLens provides the RAG triad feedback functions. DeepEval offers "
            "pytest-style unit testing for LLM applications."
        ).encode()
        r = await client.post(
            "/api/documents",
            files={"file": ("notes.md", content, "text/markdown")},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "ready"
        assert body["chunks"] >= 1
        doc_id = body["document_id"]

        # list
        lst = await client.get("/api/documents")
        assert any(d["id"] == doc_id for d in lst.json()["documents"])

        # fetch with chunks
        got = await client.get(f"/api/documents/{doc_id}?chunks=true")
        assert got.status_code == 200
        assert got.json()["title"] == "RAG Evaluation Notes"

        # delete
        dele = await client.delete(f"/api/documents/{doc_id}")
        assert dele.status_code == 200

    async def test_upload_rejects_bad_type(self, client):
        r = await client.post(
            "/api/documents",
            files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
        )
        assert r.status_code == 422

    async def test_upload_rejects_empty(self, client):
        r = await client.post(
            "/api/documents", files={"file": ("empty.txt", b"", "text/plain")}
        )
        assert r.status_code == 422

    async def test_404_unknown_doc(self, client):
        assert (await client.get("/api/documents/none")).status_code == 404


class TestToolsEndpoint:
    async def test_catalog(self, client):
        r = await client.get("/api/tools")
        assert r.status_code == 200
        names = [t["name"] for t in r.json()["tools"]]
        assert set(names) >= {"web_search", "web_fetch", "calculator", "vector_search"}
