import { retrieveWith } from "./retrieval";
import {
  RUNBOOK_BY_ID,
  GLOSSARY_BY_ID,
  GLOSSARY,
  KNOWN_ISSUE_BY_ID,
  getSources,
} from "./knowledge";
import { Source, Chunk } from "./knowledge/types";

export type AnswerSectionKind = "text" | "list" | "steps";

export interface AnswerStep {
  title: string;
  detail: string;
  tool?: string;
}

export interface AnswerSection {
  heading: string;
  kind: AnswerSectionKind;
  body?: string;
  items?: string[];
  steps?: AnswerStep[];
}

export interface ComposedAnswer {
  query: string;
  mode: "concept" | "troubleshoot" | "general";
  shortAnswer: string;
  sections: AnswerSection[];
  sources: Source[];
  tools: string[];
  confidence: "high" | "medium" | "low";
  confidenceNote?: string;
  relatedTerms: { id: string; term: string }[];
  followUps: string[];
  primaryRef?: { type: "runbook" | "glossary" | "known-issue"; id: string };
  // True when synthesized by an LLM rather than the deterministic composer.
  llm?: boolean;
}

function dedupeSources(ids: string[]): Source[] {
  return getSources(Array.from(new Set(ids)));
}

function relatedTermsFor(ids: string[]): { id: string; term: string }[] {
  return ids
    .map((id) => GLOSSARY_BY_ID[id])
    .filter(Boolean)
    .map((g) => ({ id: g.id, term: g.term }));
}

// Deterministic, fully-grounded answer composition. No external calls.
// `extraChunks` lets user-uploaded manuals participate in retrieval.
export function composeAnswer(
  query: string,
  extraChunks: Chunk[] = []
): ComposedAnswer {
  const results = retrieveWith(extraChunks, query, { limit: 6 });

  if (results.length === 0) {
    return {
      query,
      mode: "general",
      shortAnswer:
        "I couldn't find this in the curated AVEVA knowledge base. I won't guess at AVEVA-specific settings, attribute names, or menu paths that aren't in a trusted source.",
      sections: [
        {
          heading: "What you can do",
          kind: "list",
          items: [
            "Rephrase using AVEVA terms (e.g. object, attribute, AppEngine, OI Server, Galaxy).",
            "Browse Runbooks or the Glossary for the closest topic.",
            "Add the relevant manual/PDF to the knowledge base so this becomes answerable (see README → ingestion).",
          ],
        },
      ],
      sources: [],
      tools: [],
      confidence: "low",
      confidenceNote:
        "No grounded source matched this query, so no AVEVA-specific claim is being made.",
      relatedTerms: [],
      followUps: [
        "What is a DI Object?",
        "Why is my object Bad quality?",
        "Deployment cannot communicate with remote node",
      ],
    };
  }

  const top = results[0];
  const allSourceIds = results.flatMap((r) => r.chunk.sourceIds);

  // Confidence: strong, source-backed top hit → high; weak → low.
  const officialBacked = getSources(top.chunk.sourceIds).some(
    (s) =>
      s.kind === "official-doc" ||
      s.kind === "official-pdf" ||
      s.kind === "uploaded"
  );
  let confidence: ComposedAnswer["confidence"] = "medium";
  if (top.score >= 4 && officialBacked) confidence = "high";
  else if (top.score < 1.5) confidence = "low";

  // ---- Runbook → troubleshooting answer ----
  if (top.chunk.ref?.type === "runbook") {
    const rb = RUNBOOK_BY_ID[top.chunk.ref.id];
    const tools = Array.from(
      new Set([rb.firstTool, ...rb.steps.map((s) => s.tool).filter(Boolean)])
    ) as string[];
    return {
      query,
      mode: "troubleshoot",
      shortAnswer: rb.symptom,
      sections: [
        {
          heading: "Most likely causes",
          kind: "list",
          items: rb.likelyCauses,
        },
        {
          heading: `What to check first — open ${rb.firstTool}`,
          kind: "text",
          body: rb.steps[0]?.detail || "",
        },
        {
          heading: "Step-by-step troubleshooting",
          kind: "steps",
          steps: rb.steps.map((s) => ({
            title: s.title,
            detail: s.detail,
            tool: s.tool,
          })),
        },
        {
          heading: "How to confirm it's resolved",
          kind: "text",
          body: rb.confirmResolution,
        },
        {
          heading: "When to escalate",
          kind: "text",
          body: rb.escalateWhen,
        },
      ],
      sources: dedupeSources(allSourceIds),
      tools,
      confidence,
      confidenceNote:
        "This is a curated troubleshooting heuristic ('most likely / check first'), grounded in the cited sources — not a guarantee of root cause for your specific environment.",
      relatedTerms: relatedTermsFor(
        rb.topics.flatMap((t) =>
          GLOSSARY.filter((g) => g.topics.includes(t)).map((g) => g.id)
        )
      ).slice(0, 6),
      followUps: results
        .slice(1, 4)
        .map((r) => r.chunk.title),
      primaryRef: top.chunk.ref,
    };
  }

  // ---- Glossary → concept answer ----
  if (top.chunk.ref?.type === "glossary") {
    const g = GLOSSARY_BY_ID[top.chunk.ref.id];
    const sections: AnswerSection[] = [
      { heading: "Simple explanation", kind: "text", body: g.explanation },
    ];
    if (g.example)
      sections.push({ heading: "Practical example", kind: "text", body: g.example });
    if (g.related && g.related.length)
      sections.push({
        heading: "Related concepts",
        kind: "list",
        items: g.related
          .map((id) => GLOSSARY_BY_ID[id]?.term)
          .filter(Boolean) as string[],
      });
    return {
      query,
      mode: "concept",
      shortAnswer: g.short,
      sections,
      sources: dedupeSources(allSourceIds),
      tools: [],
      confidence,
      relatedTerms: relatedTermsFor(g.related || []),
      followUps: results.slice(1, 4).map((r) => r.chunk.title),
      primaryRef: top.chunk.ref,
    };
  }

  // ---- Known issue ----
  if (top.chunk.ref?.type === "known-issue") {
    const k = KNOWN_ISSUE_BY_ID[top.chunk.ref.id];
    return {
      query,
      mode: "troubleshoot",
      shortAnswer: k.symptom,
      sections: [
        { heading: "Environment", kind: "text", body: k.environment },
        { heading: "Cause", kind: "text", body: k.cause },
        { heading: "Workaround", kind: "text", body: k.workaround },
        { heading: "Status", kind: "text", body: k.status.replace("-", " ") },
      ],
      sources: dedupeSources(allSourceIds),
      tools: [],
      confidence,
      confidenceNote:
        "Framed as an environment-specific known pattern from the cited readmes/tech notes.",
      relatedTerms: [],
      followUps: results.slice(1, 4).map((r) => r.chunk.title),
      primaryRef: top.chunk.ref,
    };
  }

  // ---- Fallback synthesis from top chunks (incl. uploaded manual passages) ----
  const fromManual = top.chunk.kind === "doc";
  return {
    query,
    mode: "general",
    shortAnswer: fromManual
      ? "Here are the most relevant passages from the uploaded manuals for your question. These are direct excerpts — verify against the full source."
      : "Here is the most relevant grounded material from the AVEVA knowledge base.",
    sections: results.slice(0, 4).map((r) => ({
      heading: r.chunk.title,
      kind: "text" as const,
      body: r.chunk.text.slice(0, 500),
    })),
    sources: dedupeSources(allSourceIds),
    tools: [],
    confidence,
    relatedTerms: [],
    followUps: results.slice(1, 4).map((r) => r.chunk.title),
  };
}
