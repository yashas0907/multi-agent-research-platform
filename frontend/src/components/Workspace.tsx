import { useEffect, useState } from "react";
import { api, type DocumentInfo, type Depth, type ReportFormat } from "../lib/api";

interface Props {
  onStart: (sessionId: string) => void;
}

const DEPTHS: { value: string; label: string; desc: string }[] = [
  { value: "quick", label: "Quick", desc: "3 subquestions · 1 query each · 1 critic pass" },
  { value: "standard", label: "Standard", desc: "5 subquestions · 2 queries each · 2 critic passes" },
  { value: "deep", label: "Deep", desc: "8 subquestions · 3 queries each · 3 critic passes" },
];

const EXAMPLES = [
  "Compare RAGAS, TruLens and DeepEval for RAG evaluation and recommend one for a production customer-support system.",
  "What are the leading approaches to evaluating LLM faithfulness, and what are their documented limitations?",
  "Compare SQLite, PostgreSQL with pgvector, and dedicated vector databases for RAG retrieval at moderate scale.",
];

export default function Workspace({ onStart }: Props) {
  const [question, setQuestion] = useState("");
  const [depth, setDepth] = useState("standard");
  const [format, setFormat] = useState("detailed");
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const loadDocs = () => api.listDocuments().then((r) => setDocs(r.documents)).catch(() => {});
  useEffect(() => { loadDocs(); }, []);

  const upload = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      await api.uploadDocument(file);
      await loadDocs();
    } catch (e) {
      setError(e instanceof Error ? e.message : "upload failed");
    } finally {
      setUploading(false);
    }
  };

  const start = async () => {
    if (question.trim().length < 10) {
      setError("Research question must be at least 10 characters.");
      return;
    }
    setStarting(true);
    setError(null);
    try {
      const res = await api.createResearch({
        question: question.trim(),
        depth: depth as Depth,
        report_format: format as ReportFormat,
        document_ids: selectedDocs,
      });
      onStart(res.session_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed to start research");
      setStarting(false);
    }
  };

  return (
    <div className="grid">
      <div>
        <div className="card">
          <h2>Research Question</h2>
          <textarea
            placeholder="Ask a complex research question…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <div style={{ height: 12 }} />
          <div className="row">
            <div style={{ flex: 1 }}>
              <label style={{ color: "var(--muted)", fontSize: 12, display: "block", marginBottom: 6 }}>Research depth</label>
              <select value={depth} onChange={(e) => setDepth(e.target.value)}>
                {DEPTHS.map((d) => (
                  <option key={d.value} value={d.value}>{d.label}</option>
                ))}
              </select>
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ color: "var(--muted)", fontSize: 12, display: "block", marginBottom: 6 }}>Report type</label>
              <select value={format} onChange={(e) => setFormat(e.target.value)}>
                <option value="detailed">Detailed Research Report</option>
                <option value="executive_summary">Executive Summary</option>
                <option value="comparison">Comparison Report</option>
              </select>
            </div>
          </div>
          <p className="hint">{DEPTHS.find((d) => d.value === depth)?.desc}</p>
          <div style={{ height: 10 }} />
          <button className="btn" onClick={start} disabled={starting}>
            {starting ? "Starting…" : "Start Research"}
          </button>
          {error && <div className="error-banner" style={{ marginTop: 12 }}>{error}</div>}
        </div>

        <div className="card">
          <h2>Your Documents (optional)</h2>
          <p className="section-note">
            Uploaded documents are chunked, embedded, and searched alongside web sources — evidence origin is always tracked.
          </p>
          <label className="btn-ghost" style={{ display: "inline-block", cursor: uploading ? "wait" : "pointer" }}>
            {uploading ? "Uploading…" : "Upload PDF / Markdown / Text / HTML / JSON"}
            <input
              type="file"
              accept=".pdf,.md,.txt,.html,.htm,.json"
              style={{ display: "none" }}
              disabled={uploading}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) upload(f);
                e.currentTarget.value = "";
              }}
            />
          </label>
          <div style={{ height: 12 }} />
          {docs.length === 0 ? (
            <div className="empty" style={{ padding: 20 }}>No documents uploaded yet.</div>
          ) : (
            docs.map((d) => (
              <div key={d.id} className="doc-row">
                <div style={{ display: "flex", gap: 10, alignItems: "center", minWidth: 0 }}>
                  <input
                    type="checkbox"
                    checked={selectedDocs.includes(d.id)}
                    onChange={(e) =>
                      setSelectedDocs((prev) =>
                        e.target.checked ? [...prev, d.id] : prev.filter((x) => x !== d.id)
                      )
                    }
                  />
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {d.title || d.filename}
                    </div>
                    <div style={{ color: "var(--muted)", fontSize: 11.5 }}>
                      {d.chunks} chunks · {(d.size_bytes / 1024).toFixed(0)} KB
                    </div>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span className={`doc-status ${d.status}`}>{d.status}</span>
                  <button
                    className="btn-ghost"
                    style={{ padding: "4px 10px", fontSize: 12 }}
                    onClick={async () => { await api.deleteDocument(d.id); loadDocs(); }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div>
        <div className="card">
          <h2>How the pipeline works</h2>
          <p className="section-note" style={{ marginTop: 0 }}>
            Your question is decomposed and researched by specialized agents with
            explicit state, budgets, and verification at every step.
          </p>
          <ol style={{ paddingLeft: 20, lineHeight: 2, fontSize: 13.5, color: "var(--muted)", margin: 0 }}>
            <li><b style={{ color: "var(--text)" }}>Planner</b> — subquestions, evidence requirements, completion criteria</li>
            <li><b style={{ color: "var(--text)" }}>Search</b> — non-redundant queries per subquestion (web + your docs)</li>
            <li><b style={{ color: "var(--text)" }}>Source Evaluator</b> — relevance, authority, recency, primary/secondary</li>
            <li><b style={{ color: "var(--text)" }}>Evidence Extraction</b> — grounded snippets with provenance</li>
            <li><b style={{ color: "var(--text)" }}>Fact-Checker</b> — SUPPORTED / PARTIAL / CONTRADICTED / INSUFFICIENT</li>
            <li><b style={{ color: "var(--text)" }}>Contradiction Detection</b> — disagreements surfaced, never silently resolved</li>
            <li><b style={{ color: "var(--text)" }}>Critic</b> — gaps found → followup research loop (budget-limited)</li>
            <li><b style={{ color: "var(--text)" }}>Synthesis</b> — verified evidence only; facts vs interpretation vs uncertainty</li>
            <li><b style={{ color: "var(--text)" }}>Citation Audit</b> — every claim mapped to real sources</li>
          </ol>
        </div>

        <div className="card">
          <h2>Example questions</h2>
          {EXAMPLES.map((q) => (
            <div
              key={q}
              className="sq"
              style={{ cursor: "pointer" }}
              onClick={() => setQuestion(q)}
            >
              <p className="sq-text" style={{ margin: 0 }}>{q}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
