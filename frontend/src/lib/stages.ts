import type { ResearchStage } from "./api";

export const STAGES: { id: ResearchStage; label: string }[] = [
  { id: "pending", label: "Queued" },
  { id: "planning", label: "Planning" },
  { id: "searching", label: "Searching Sources" },
  { id: "gathering_evidence", label: "Gathering Evidence" },
  { id: "fact_checking", label: "Fact Checking" },
  { id: "contradiction_check", label: "Contradiction Check" },
  { id: "critiquing", label: "Critiquing" },
  { id: "synthesizing", label: "Synthesizing" },
  { id: "citing", label: "Citation Mapping" },
  { id: "report_ready", label: "Report Ready" },
];

export function stageIndex(stage: ResearchStage): number {
  const i = STAGES.findIndex((s) => s.id === stage);
  return i === -1 ? STAGES.length : i;
}
