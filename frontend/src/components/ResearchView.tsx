import { useEffect, useRef, useState } from "react";
import {
  api,
  AgentEvent,
  StatusResponse,
  SubQuestion,
} from "../lib/api";
import { STAGES, stageIndex } from "../lib/stages";
import ReportView from "./ReportView";

interface Props {
  sessionId: string;
  onReset: () => void;
}

const POLL_MS = 900;
const TERMINAL = ["report_ready", "failed", "cancelled", "completed_partial"];

export default function ResearchView({ sessionId, onReset }: Props) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [subquestions, setSubquestions] = useState<SubQuestion[]>([]);
  const [tab, setTab] = useState<"trace" | "report">("trace");
  const seen = useRef(0);

  useEffect(() => {
    let alive = true;
    let es: EventSource | null = null;

    const startPolling = () => {
      const poll = async () => {
        try {
          const s = await api.getStatus(sessionId);
          if (!alive) return;
          setStatus(s);
          const ev = await api.getEvents(sessionId, seen.current);
          if (ev.events.length > 0) {
            seen.current += ev.events.length;
            setEvents((prev) => [...prev, ...ev.events]);
          }
          const sq = await api.getSubQuestions(sessionId);
          if (alive) setSubquestions(sq.subquestions);
          if (TERMINAL.includes(s.status)) {
            if (s.status === "report_ready" || s.status === "completed_partial") {
              setTab("report");
            }
            return; // stop polling
          }
        } catch {
          /* transient poll failure — keep trying */
        }
        if (alive) setTimeout(poll, POLL_MS);
      };
      poll();
    };

    // Real-time first: SSE stream for events + status; polling only as
    // fallback if the stream errors (e.g. proxies without SSE support).
    try {
      es = api.openEventStream(
        sessionId,
        (e) => {
          seen.current += 1;
          setEvents((prev) => [...prev, e]);
        },
        (s) => {
          setStatus((prev) =>
            prev ? { ...prev, status: s.status as StatusResponse["status"], progress_pct: s.progress_pct, stage_label: s.stage_label } : prev
          );
        },
        () => {
          // stream closed → one final full refresh (report, subquestions)
          api.getStatus(sessionId).then((s) => alive && setStatus(s)).catch(() => {});
          api.getSubQuestions(sessionId).then((r) => alive && setSubquestions(r.subquestions)).catch(() => {});
          api
            .getReport(sessionId)
            .then(() => alive && setTab("report"))
            .catch(() => {});
        }
      );
      es.onerror = () => {
        // stream failed — close and fall back to polling
        es?.close();
        es = null;
        if (alive) startPolling();
      };
    } catch {
      startPolling();
    }

    return () => {
      alive = false;
      es?.close();
    };
  }, [sessionId]);

  const cancel = async () => {
    try { await api.cancelResearch(sessionId); } catch { /* already finished */ }
  };

  const currentIdx = status ? stageIndex(status.status) : -1;
  const isTerminal = status ? TERMINAL.includes(status.status) : false;

  return (
    <div className="grid">
      <div>
        <div className="card">
          <h2>Research Progress</h2>
          {status && (
            <>
              <div className="pct">
                <span>{status.stage_label}</span>
                <span>iteration {status.current_iteration}</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${status.progress_pct}%` }} />
              </div>
              <ul className="stage-list">
                {STAGES.map((s, i) => (
                  <li
                    key={s.id}
                    className={
                      i < currentIdx ? "done" : i === currentIdx && !isTerminal ? "active" : ""
                    }
                  >
                    <span className="dot" />
                    {s.label}
                  </li>
                ))}
              </ul>
              {status.error && <div className="error-banner">{status.error}</div>}
              <div style={{ height: 12 }} />
              {!isTerminal ? (
                <button className="btn-danger" onClick={cancel}>Cancel Research</button>
              ) : (
                <button className="btn" onClick={onReset}>New Research</button>
              )}
            </>
          )}
        </div>

        <div className="card">
          <h2>Budget</h2>
          {status ? (
            <div style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 2 }}>
              <BudgetRow label="Searches" used={status.budget.searches} max={status.budget.max_searches} />
              <BudgetRow label="Sources" used={status.budget.sources} max={status.budget.max_sources} />
              <BudgetRow label="Iterations" used={status.budget.iteration} max={status.budget.max_iterations} />
              <div>Tokens used: <b style={{ color: "var(--text)" }}>{status.budget.tokens_used}</b></div>
              <div>LLM calls: <b style={{ color: "var(--text)" }}>{status.budget.llm_calls ?? "—"}</b></div>
            </div>
          ) : (
            <div className="empty" style={{ padding: 14 }}>Loading…</div>
          )}
        </div>

        <div className="card">
          <h2>Subquestions</h2>
          {subquestions.length === 0 ? (
            <div className="empty" style={{ padding: 14 }}>Planner has not run yet.</div>
          ) : (
            subquestions.map((sq) => (
              <div className="sq" key={sq.id}>
                <div className="sq-head">
                  <span className={`badge ${sq.status}`}>{sq.status.replace("_", " ")}</span>
                </div>
                <p className="sq-text">{sq.text}</p>
              </div>
            ))
          )}
        </div>
      </div>

      <div>
        <div className="tabs">
          <button className={`tab ${tab === "trace" ? "active" : ""}`} onClick={() => setTab("trace")}>
            Live Trace ({events.length})
          </button>
          <button
            className={`tab ${tab === "report" ? "active" : ""}`}
            onClick={() => setTab("report")}
            disabled={status ? !TERMINAL.includes(status.status) : true}
            style={{ opacity: status && TERMINAL.includes(status.status) ? 1 : 0.4 }}
          >
            Report
          </button>
        </div>

        {tab === "trace" ? <TraceView events={events} /> : null}
        {tab === "report" && status && TERMINAL.includes(status.status) ? (
          <ReportView sessionId={sessionId} />
        ) : null}
      </div>
    </div>
  );
}

function BudgetRow({ label, used, max }: { label: string; used?: number; max?: number }) {
  if (used === undefined || max === undefined) return null;
  return (
    <div>
      {label}: <b style={{ color: "var(--text)" }}>{used}</b> / {max}
    </div>
  );
}

function TraceView({ events }: { events: AgentEvent[] }) {
  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  if (events.length === 0) {
    return <div className="empty">Waiting for research activity…</div>;
  }
  return (
    <div className="card">
      <h2>Live Research Trace</h2>
      <p className="section-note">
        Safe operational events only — prompts and internal reasoning are never exposed.
      </p>
      <div className="trace">
        {events.map((e) => (
          <div className="trace-item" key={e.id} data-evt={e.event_type}>
            <span className="trace-time">
              {new Date(e.timestamp).toLocaleTimeString()}
            </span>
            <span className={`trace-agent ${e.agent}`}>{e.agent.replace("_", " ")}</span>
            <span className="trace-msg">{e.message}</span>
          </div>
        ))}
        <div ref={bottom} />
      </div>
    </div>
  );
}
