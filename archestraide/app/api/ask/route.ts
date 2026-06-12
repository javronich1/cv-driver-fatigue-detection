import { NextRequest, NextResponse } from "next/server";
import { composeAnswer } from "@/lib/answer";
import { retrieve } from "@/lib/retrieval";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// POST /api/ask  { query: string }
//
// Default behaviour (no API key): returns the deterministic, fully-grounded
// answer composed from the curated knowledge base. This always works and is
// safe to deploy statically-adjacent.
//
// Enhanced behaviour (ANTHROPIC_API_KEY set on the host): the retrieved context
// is passed to Claude with strict grounding instructions to produce a polished,
// support-oriented short answer. Structure, sources, tools and confidence still
// come from the deterministic layer, so citations remain trustworthy.

const MODEL = process.env.ARCHESTRAIDE_MODEL || "claude-sonnet-4-6";

export async function POST(req: NextRequest) {
  let query = "";
  try {
    const body = await req.json();
    query = (body?.query || "").toString().slice(0, 2000);
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }
  if (!query.trim()) {
    return NextResponse.json({ error: "Empty query" }, { status: 400 });
  }

  const composed = composeAnswer(query);
  const apiKey = process.env.ANTHROPIC_API_KEY;

  // No key → grounded deterministic answer.
  if (!apiKey) {
    return NextResponse.json({ ...composed, llm: false });
  }

  try {
    const context = retrieve(query, { limit: 6 })
      .map(
        (r, i) =>
          `[#${i + 1}] (${r.chunk.kind}) ${r.chunk.title}\n${r.chunk.text.slice(0, 900)}`
      )
      .join("\n\n");

    const system = [
      "Eres ArchestrAide, un copilot interno de soporte para AVEVA Application Server / OMI / System Platform.",
      "Responde SIEMPRE en español, pero mantén en inglés los nombres de conceptos y herramientas técnicas (p. ej. Galaxy, Template, AppEngine, OI Server, ViewApp, OnScan, Object Viewer) y los mensajes de error literales.",
      "Responde SOLO usando el CONTEXTO proporcionado, que proviene de documentación oficial de AVEVA y runbooks curados.",
      "NO inventes settings, nombres de atributos, rutas de menú ni nombres de propiedades específicos de AVEVA que no estén en el contexto.",
      "Prefiere frases como 'lo más probable', 'revisa primero', 'según los docs citados'. Distingue hechos oficiales de heurísticas de troubleshooting.",
      "Si el contexto no cubre la pregunta, dilo claramente en vez de adivinar.",
      "Escribe una respuesta corta, concisa y práctica (2-5 frases) como un compañero de soporte con experiencia. Sin encabezados ni listas markdown — solo la respuesta corta en prosa.",
    ].join(" ");

    const resp = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 600,
        system,
        messages: [
          {
            role: "user",
            content: `QUESTION: ${query}\n\nCONTEXT:\n${context}`,
          },
        ],
      }),
    });

    if (!resp.ok) {
      return NextResponse.json({ ...composed, llm: false });
    }

    const data = await resp.json();
    const text: string =
      data?.content?.map((c: any) => c?.text || "").join("").trim() || "";

    if (!text) return NextResponse.json({ ...composed, llm: false });

    // Keep deterministic structure/sources; replace the short answer with the
    // LLM's grounded synthesis.
    return NextResponse.json({
      ...composed,
      shortAnswer: text,
      llm: true,
    });
  } catch {
    return NextResponse.json({ ...composed, llm: false });
  }
}
