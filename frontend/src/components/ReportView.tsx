import { useEffect, useState } from "react";
import { api, ResearchReport, Evidence, SourceRecord } from "../lib/api";

interface Props {
  sessionId: string;
}

export default function ReportView({ sessionId }: Props) {
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openEvidence, setOpenEvidence] = useState<string | null>(null);

  useEffect(() => {
    api
      .getReport(sessionId)
      .then(setReport)
      .catch((e) => setError(e instanceof Error ? e.message : "report unavailable"));
  }, [sessionId]);

  if (error) return <div className="error-banner">{error}</div>;
  if (!report) return <div className="empty">Loading report…</div>;

  const evidenceById = new Map(report.evidence.map((e) => [e.id, e]));
  const sourceById = new Map(report.sources.map((s) => [s.id, s]));

  return (
    <div>
      {/* Executive Summary */}
      <div className="card report-section">
        <h2>Executive Summary</h2>
        <p style={{ fontSize: 14.5, lineHeight: 1.7, margin: 0 }}>{report.executive_summary}</p>
        <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
          {Object.entries(report.confidence_summary).map(([k, v]) => (
            <span key={k} className={`conf ${k}`}>{k.replace("_", " ")}: {v}</span>
          ))}
        </div>
      </div>

      {/* Comparison */}
      {report.comparison.length > 0 && (
        <div className="card report-section">
          <h2>Comparison</h2>
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Subject</th>
                <th>Key points</th>
                <th>Advantages</th>
                <th>Disadvantages</th>
              </tr>
            </thead>
            <tbody>
              {report.comparison.map((row) => (
                <tr key={row.subject}>
                  <td><b>{row.subject}</b></td>
                  <td>
                    {Object.entries(row.criteria).map(([k, v]) => (
                      <div key={k} style={{ fontSize: 12.5 }}>{k}: {v}</div>
                    ))}
                  </td>
                  <td>{row.advantages.map((a, i) => <div key={i} className="plus">+ {a}</div>)}</td>
                  <td>{row.disadvantages.map((d, i) => <div key={i} className="minus">− {d}</div>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Findings */}
      <div className="card report-section">
        <h2>Key Findings</h2>
        {report.key_findings.length === 0 ? (
          <div className="empty">No findings could be grounded in retrieved evidence.</div>
        ) : (
          report.key_findings.map((f, i) => (
            <div className="finding" key={i}>
              <p>{f.statement}</p>
              <div className="finding-meta">
                <span className={`conf ${f.confidence}`}>{f.confidence.replace("_", " ")}</span>
                {f.is_interpretation && <span className="interp-tag">INTERPRETATION</span>}
                {f.evidence_ids.map((id) => (
                  <button key={id} className="ev-chip" onClick={() => setOpenEvidence(openEvidence === id ? null : id)}>
                    {id}
                  </button>
                ))}
              </div>
              {f.caveat && <p style={{ color: "var(--muted)", fontSize: 12.5, marginTop: 8 }}>{f.caveat}</p>}
              {openEvidence && f.evidence_ids.includes(openEvidence) && (
                <EvidenceDetail evidence={evidenceById.get(openEvidence)} source={sourceById.get(evidenceById.get(openEvidence)?.source_id || "")} />
              )}
            </div>
          ))
        )}
      </div>

      {/* Contradictions */}
      {report.contradictions.length > 0 && (
        <div className="card report-section">
          <h2>Contradictions Between Sources</h2>
          {report.contradictions.map((c) => (
            <div className="contra" key={c.id}>
              <h4>{c.topic}</h4>
              <p><b>Source A:</b> {c.claim_a}</p>
              <p><b>Source B:</b> {c.claim_b}</p>
              {c.possible_reason && <p><b>Possible reason:</b> {c.possible_reason}</p>}
              {c.resolving_evidence_needed && <p><b>To resolve:</b> {c.resolving_evidence_needed}</p>}
            </div>
          ))}
        </div>
      )}

      {/* Conclusion */}
      <div className="card report-section">
        <h2>Conclusion</h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>{report.conclusion}</p>
        {report.recommendation && (
          <div className="contrast-box">
            <b style={{ color: "var(--green)" }}>Recommendation</b>
            <p style={{ fontSize: 14, lineHeight: 1.7, margin: "8px 0 0" }}>{report.recommendation}</p>
          </div>
        )}
      </div>

      {/* Limitations */}
      <div className="card report-section">
        <h2>Limitations</h2>
        {report.limitations.length === 0 ? (
          <p className="section-note">No limitations recorded.</p>
        ) : (
          report.limitations.map((l, i) => <div className="limitation" key={i}>{l}</div>)
        )}
      </div>

      {/* Methodology */}
      <div className="card report-section">
        <h2>Methodology</h2>
        <p style={{ fontSize: 13, lineHeight: 1.7, color: "var(--muted)" }}>{report.methodology}</p>
      </div>

      {/* Sources */}
      <div className="card report-section">
        <h2>Sources ({report.sources.length})</h2>
        {report.sources.map((s) => (
          <div className="source-row" key={s.id}>
            <div className="source-main">
              <div className="source-title">
                {s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a> : s.title}
              </div>
              <div className="source-meta">
                {s.domain ?? "user document"} · {s.source_type.replace("_", " ")}
                {s.published_at ? ` · published ${s.published_at.slice(0, 10)}` : " · date not available"}
                {s.origin === "user_document" ? " · your document" : ""}
              </div>
            </div>
            <div className="trust-bar" title={`relevance ${s.relevance_score} · authority ${s.authority_score} · recency ${s.recency_score}`}>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>{Math.round(s.trust_score * 100)}</span>
              <div className="trust-track"><div className="trust-fill" style={{ width: `${s.trust_score * 100}%` }} /></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EvidenceDetail({ evidence, source }: { evidence?: Evidence; source?: SourceRecord }) {
  if (!evidence) return null;
  return (
    <div className="evidence-panel" style={{ marginTop: 10 }}>
      <div className="evidence-panel snippet">{evidence.snippet}</div>
      <div className="evidence-meta">
        <span>confidence: {evidence.confidence}</span>
        {evidence.source_location && <span>location: {evidence.source_location}</span>}
        {source && (
          <span>
            source:{" "}
            {source.url ? <a href={source.url} target="_blank" rel="noreferrer">{source.title}</a> : source.title}
          </span>
        )}
        <span>origin: {evidence.origin.replace("_", " ")}</span>
      </div>
    </div>
  );
}
