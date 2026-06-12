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
        "No encontré esto en la base de conocimiento curada de AVEVA. No voy a inventar settings, nombres de atributos ni rutas de menú específicas de AVEVA que no estén en una fuente confiable.",
      sections: [
        {
          heading: "Qué puedes hacer",
          kind: "list",
          items: [
            "Reformula usando términos de AVEVA (p. ej. object, attribute, AppEngine, OI Server, Galaxy).",
            "Explora los Runbooks o el Glosario para el tema más cercano.",
            "Sube el manual/PDF relevante a la base de conocimiento para que esto sea respondible (ver la página Manuales).",
          ],
        },
      ],
      sources: [],
      tools: [],
      confidence: "low",
      confidenceNote:
        "Ninguna fuente fundamentada coincidió con esta consulta, así que no se hace ninguna afirmación específica de AVEVA.",
      relatedTerms: [],
      followUps: [
        "¿Qué es un DI Object?",
        "¿Por qué mi objeto está en Bad quality?",
        "El despliegue dice cannot communicate with remote node",
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
          heading: "Causas más probables",
          kind: "list",
          items: rb.likelyCauses,
        },
        {
          heading: `Qué revisar primero — abre ${rb.firstTool}`,
          kind: "text",
          body: rb.steps[0]?.detail || "",
        },
        {
          heading: "Troubleshooting paso a paso",
          kind: "steps",
          steps: rb.steps.map((s) => ({
            title: s.title,
            detail: s.detail,
            tool: s.tool,
          })),
        },
        {
          heading: "Cómo confirmar que está resuelto",
          kind: "text",
          body: rb.confirmResolution,
        },
        {
          heading: "Cuándo escalar",
          kind: "text",
          body: rb.escalateWhen,
        },
      ],
      sources: dedupeSources(allSourceIds),
      tools,
      confidence,
      confidenceNote:
        "Esto es una heurística de troubleshooting curada ('más probable / revisa primero'), fundamentada en las fuentes citadas — no una garantía de la causa raíz para tu entorno específico.",
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
      { heading: "Explicación simple", kind: "text", body: g.explanation },
    ];
    if (g.example)
      sections.push({ heading: "Ejemplo práctico", kind: "text", body: g.example });
    if (g.related && g.related.length)
      sections.push({
        heading: "Conceptos relacionados",
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
        { heading: "Entorno", kind: "text", body: k.environment },
        { heading: "Causa", kind: "text", body: k.cause },
        { heading: "Solución alternativa", kind: "text", body: k.workaround },
        { heading: "Estado", kind: "text", body: k.status.replace("-", " ") },
      ],
      sources: dedupeSources(allSourceIds),
      tools: [],
      confidence,
      confidenceNote:
        "Enmarcado como un patrón conocido específico del entorno, de los readmes/tech notes citados.",
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
      ? "Aquí están los pasajes más relevantes de los manuales subidos para tu pregunta. Son extractos directos — verifícalos contra la fuente completa."
      : "Aquí está el material fundamentado más relevante de la base de conocimiento de AVEVA.",
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
