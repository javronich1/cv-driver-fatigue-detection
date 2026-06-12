// Shared types for the ArchestrAide knowledge base.
// Everything the assistant answers with is traceable to a Source.

export type Topic =
  | "deployment"
  | "templates"
  | "di"
  | "oi"
  | "historian"
  | "alarms"
  | "security"
  | "object-management"
  | "csv"
  | "runtime"
  | "omi"
  | "troubleshooting"
  | "concepts";

export const TOPIC_LABELS: Record<Topic, string> = {
  deployment: "Deployment",
  templates: "Templates",
  di: "Device Integration",
  oi: "OI / OPC",
  historian: "Historian",
  alarms: "Alarms",
  security: "Security",
  "object-management": "Object Management",
  csv: "CSV / Import-Export",
  runtime: "Runtime",
  omi: "OMI / ViewApp",
  troubleshooting: "Troubleshooting",
  concepts: "Concepts",
};

// Where knowledge comes from. We deliberately separate the *kind* of source so
// the UI can show "official" vs "inferred" provenance and earn user trust.
export type SourceKind =
  | "official-doc" // docs.aveva.com or official AVEVA PDF
  | "official-pdf" // official AVEVA manual / PDF
  | "uploaded" // manual the user uploaded in-browser (local to their device)
  | "community" // AVEVA community / vendor tech note (clearly labelled)
  | "runbook" // ArchestrAide-curated runbook based on trusted sources
  | "glossary"; // ArchestrAide-curated concept page

// Product line a piece of knowledge applies to (for filtering / badges).
export type Product = "appserver" | "omi" | "historian" | "general";

export interface Source {
  id: string;
  title: string;
  kind: SourceKind;
  url?: string;
  // Human-friendly location reference (e.g. "Object Viewer User Guide, ScanState")
  reference?: string;
  topics: Topic[];
}

export type Tool =
  | "ArchestrA IDE"
  | "Object Viewer"
  | "OCMC (SMC)"
  | "Platform Manager"
  | "OI Server Manager"
  | "Historian Client Web"
  | "Galaxy Database Manager"
  | "Log Viewer"
  | "Operations Control Management Console";

export interface RunbookStep {
  title: string;
  detail: string;
  tool?: Tool;
  // Optional explicit citation for a single step.
  sourceIds?: string[];
}

export interface Runbook {
  id: string;
  title: string;
  category: string;
  topics: Topic[];
  symptom: string;
  // Likely causes, ordered most → least common.
  likelyCauses: string[];
  firstTool: Tool;
  // Ordered diagnostic checks.
  steps: RunbookStep[];
  confirmResolution: string;
  escalateWhen: string;
  // Severity / blast-radius hint for the UI.
  severity: "low" | "medium" | "high";
  sourceIds: string[];
  // Free-text keywords to improve retrieval recall (error strings etc.).
  keywords?: string[];
}

export interface GlossaryTerm {
  id: string;
  term: string;
  aliases?: string[];
  topics: Topic[];
  short: string; // one-line definition
  explanation: string; // simple explanation
  example?: string; // practical example
  related?: string[]; // related term ids
  sourceIds: string[];
}

export interface KnownIssue {
  id: string;
  title: string;
  topics: Topic[];
  environment: string; // e.g. "Multi-node, redundant AppEngine"
  symptom: string;
  cause: string;
  workaround: string;
  status: "known" | "by-design" | "fixed-in-patch";
  sourceIds: string[];
  keywords?: string[];
}

// A flat, retrievable unit. Everything (runbooks, glossary, known issues,
// official-doc summaries) is projected into Chunks for the retrieval layer.
export interface Chunk {
  id: string;
  kind: "runbook" | "glossary" | "known-issue" | "doc";
  title: string;
  topics: Topic[];
  text: string; // searchable body
  sourceIds: string[];
  href?: string; // in-app link
  ref?: { type: "runbook" | "glossary" | "known-issue"; id: string };
}
