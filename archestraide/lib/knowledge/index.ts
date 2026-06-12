import { Chunk } from "./types";
import { RUNBOOKS } from "./runbooks";
import { GLOSSARY } from "./glossary";
import { KNOWN_ISSUES } from "./knownIssues";
import { SOURCE_BY_ID } from "./sources";

export * from "./types";
export * from "./sources";
export { RUNBOOKS, RUNBOOK_BY_ID } from "./runbooks";
export { GLOSSARY, GLOSSARY_BY_ID } from "./glossary";
export { KNOWN_ISSUES, KNOWN_ISSUE_BY_ID } from "./knownIssues";

// Project all curated content into a single flat array of retrievable Chunks.
// Each chunk carries a denormalised `text` blob (title + body + keywords + the
// titles of its sources) so the retrieval layer can do hybrid keyword/semantic
// scoring without re-walking the object graph.

function sourceTitles(ids: string[]): string {
  return ids
    .map((id) => SOURCE_BY_ID[id]?.title)
    .filter(Boolean)
    .join(" ");
}

export const CHUNKS: Chunk[] = [
  ...RUNBOOKS.map<Chunk>((r) => ({
    id: `chunk-${r.id}`,
    kind: "runbook",
    title: r.title,
    topics: r.topics,
    sourceIds: r.sourceIds,
    href: `/runbooks#${r.id}`,
    ref: { type: "runbook", id: r.id },
    text: [
      r.title,
      r.category,
      r.symptom,
      r.likelyCauses.join(". "),
      r.steps.map((s) => `${s.title}. ${s.detail}`).join(" "),
      r.confirmResolution,
      r.escalateWhen,
      (r.keywords || []).join(" "),
      sourceTitles(r.sourceIds),
    ].join("  "),
  })),
  ...GLOSSARY.map<Chunk>((g) => ({
    id: `chunk-${g.id}`,
    kind: "glossary",
    title: g.term,
    topics: g.topics,
    sourceIds: g.sourceIds,
    href: `/glossary#${g.id}`,
    ref: { type: "glossary", id: g.id },
    text: [
      g.term,
      (g.aliases || []).join(" "),
      g.short,
      g.explanation,
      g.example || "",
      sourceTitles(g.sourceIds),
    ].join("  "),
  })),
  ...KNOWN_ISSUES.map<Chunk>((k) => ({
    id: `chunk-${k.id}`,
    kind: "known-issue",
    title: k.title,
    topics: k.topics,
    sourceIds: k.sourceIds,
    href: `/known-issues#${k.id}`,
    ref: { type: "known-issue", id: k.id },
    text: [
      k.title,
      k.environment,
      k.symptom,
      k.cause,
      k.workaround,
      (k.keywords || []).join(" "),
      sourceTitles(k.sourceIds),
    ].join("  "),
  })),
];
