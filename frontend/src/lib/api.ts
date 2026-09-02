// Typed API client + shared types mirroring backend schemas exactly.

export type ResearchStage =
  | "pending" | "planning" | "searching" | "gathering_evidence"
  | "fact_checking" | "contradiction_check" | "critiquing"
  | "synthesizing" | "citing" | "report_ready"
  | "completed_partial" | "cancelled" | "failed";

export type Depth = "quick" | "standard" | "deep";
export type ReportFormat = "detailed" | "executive_summary" | "comparison";

export interface StatusResponse {
  session_id: string;
  status: ResearchStage;
  stage_label: string;
  progress_pct: number;
  current_iteration: number;
  question: string;
  depth: Depth;
  error: string | null;
  budget: Record<string, number>;
}

export interface AgentEvent {
  id: string;
  session_id: string;
  timestamp: string;
  agent: string;
  event_type: string;
  message: string;
  stage: string | null;
  data: Record<string, unknown>;
}

export interface SubQuestion {
  id: string;
  text: string;
  status: "pending" | "in_progress" | "answered" | "insufficient";
  answer: string | null;
  evidence_ids: string[];
}

export interface SourceRecord {
  id: string;
  title: string;
  url: string | null;
  source_type: string;
  origin: "web" | "user_document" | "internal";
  published_at: string | null;
  retrieved_at: string;
  domain: string | null;
  trust_score: number;
  relevance_score: number;
  authority_score: number;
  recency_score: number;
  is_primary: boolean;
  fetched_ok: boolean;
}

export interface Evidence {
  id: string;
  claim_summary: string;
  source_id: string;
  source_location: string | null;
  snippet: string;
  subquestion_id: string | null;
  origin: string;
  confidence: "high" | "moderate" | "low";
}

export interface Claim {
  id: string;
  text: string;
  status: "SUPPORTED" | "PARTIALLY_SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE";
  supporting_evidence_ids: string[];
  verification_rationale: string | null;
}

export interface Contradiction {
  id: string;
  topic: string;
  claim_a: string;
  source_id_a: string;
  claim_b: string;
  source_id_b: string;
  possible_reason: string | null;
  resolving_evidence_needed: string | null;
}

export interface Finding {
  statement: string;
  evidence_ids: string[];
  confidence: "HIGH_EVIDENCE" | "MODERATE_EVIDENCE" | "LOW_EVIDENCE" | "INSUFFICIENT_EVIDENCE";
  is_interpretation: boolean;
  caveat: string | null;
}

export interface ComparisonRow {
  subject: string;
  criteria: Record<string, string>;
  advantages: string[];
  disadvantages: string[];
  evidence_ids: string[];
}

export interface ResearchReport {
  session_id: string;
  question: string;
  format: ReportFormat;
  generated_at: string;
  executive_summary: string;
  methodology: string;
  key_findings: Finding[];
  comparison: ComparisonRow[];
  contradictions: Contradiction[];
  limitations: string[];
  conclusion: string;
  recommendation: string | null;
  sources: SourceRecord[];
  evidence: Evidence[];
  claims: Claim[];
  confidence_summary: Record<string, number>;
  subquestion_answers: Record<string, string>;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  title: string | null;
  status: string;
  chunks: number;
  size_bytes: number;
  error: string | null;
}

const BASE = import.meta.env.VITE_API_URL || "";

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  createResearch: (body: {
    question: string;
    depth: Depth;
    report_format: ReportFormat;
    document_ids?: string[];
  }) =>
    jsonFetch<{ session_id: string }>("/api/research", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getStatus: (id: string) => jsonFetch<StatusResponse>(`/api/research/${id}/status`),

  getEvents: (id: string, after = 0) =>
    jsonFetch<{ events: AgentEvent[] }>(`/api/research/${id}/events?after=${after}`),

  getReport: (id: string) => jsonFetch<ResearchReport>(`/api/research/${id}/report`),

  getSubQuestions: (id: string) =>
    jsonFetch<{ subquestions: SubQuestion[] }>(`/api/research/${id}/subquestions`),

  getSources: (id: string) =>
    jsonFetch<{ sources: SourceRecord[]; total: number }>(`/api/research/${id}/sources`),

  cancelResearch: (id: string) =>
    jsonFetch<{ cancel_requested: boolean }>(`/api/research/${id}/cancel`, {
      method: "POST",
    }),

  listSessions: () => jsonFetch<{ sessions: Array<{ session_id: string; question: string; status: string; depth: string }> }>("/api/research?limit=10"),

  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/api/documents`, { method: "POST", body: form }).then(
      async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(body.detail || "upload failed");
        }
        return res.json();
      }
    );
  },

  listDocuments: () => jsonFetch<{ documents: DocumentInfo[] }>("/api/documents"),

  deleteDocument: (id: string) => jsonFetch<{ chunks_removed: number }>(`/api/documents/${id}`, { method: "DELETE" }),
};
