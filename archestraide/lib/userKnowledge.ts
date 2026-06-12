"use client";

import { Chunk, Source, Topic, Product } from "./knowledge/types";
import { registerSources } from "./knowledge/sources";

// Client-side ingestion of user-uploaded manuals.
//
// The live site is a static build with no backend, so uploaded manuals are
// parsed in the browser (pdf.js), chunked, and persisted to localStorage on the
// user's device. They are then blended into retrieval (Ask / Docs / Troubleshoot)
// for that browser. This is per-device, not shared across users — see the README
// for the server-backed upgrade path for team-wide shared manuals.

const STORE_KEY = "aa-manuals-v1";

export interface Manual {
  id: string;
  name: string;
  product: Product;
  topic: Topic;
  pages: number;
  addedAt: number;
  source: Source;
  chunks: Chunk[];
}

function safeParse(raw: string | null): Manual[] {
  if (!raw) return [];
  try {
    const v = JSON.parse(raw);
    return Array.isArray(v) ? (v as Manual[]) : [];
  } catch {
    return [];
  }
}

export function listManuals(): Manual[] {
  if (typeof window === "undefined") return [];
  return safeParse(localStorage.getItem(STORE_KEY));
}

function persist(manuals: Manual[]) {
  localStorage.setItem(STORE_KEY, JSON.stringify(manuals));
}

// Register all stored manual sources so citations resolve, and return their
// chunks for retrieval. Call once on mount of pages that use uploaded knowledge.
export function loadUserKnowledge(): { chunks: Chunk[]; manuals: Manual[] } {
  const manuals = listManuals();
  registerSources(manuals.map((m) => m.source));
  return { chunks: manuals.flatMap((m) => m.chunks), manuals };
}

export function getUserChunks(): Chunk[] {
  return loadUserKnowledge().chunks;
}

export function removeManual(id: string): Manual[] {
  const next = listManuals().filter((m) => m.id !== id);
  persist(next);
  return next;
}

// Split raw text into overlapping passages for retrieval.
function chunkText(text: string, size = 750, overlap = 100): string[] {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= size) return clean ? [clean] : [];
  const out: string[] = [];
  let i = 0;
  while (i < clean.length) {
    let end = Math.min(i + size, clean.length);
    // Prefer to break on a sentence/space boundary.
    if (end < clean.length) {
      const dot = clean.lastIndexOf(". ", end);
      const space = clean.lastIndexOf(" ", end);
      const brk = dot > i + size * 0.6 ? dot + 1 : space > i ? space : end;
      end = brk;
    }
    out.push(clean.slice(i, end).trim());
    i = end - overlap;
    if (i < 0) i = 0;
  }
  return out.filter((c) => c.length > 40);
}

export interface IngestResult {
  manual: Manual;
  manuals: Manual[];
}

// Parse a PDF File in the browser and store it as a Manual. `onProgress`
// reports page extraction progress (0..1).
export async function ingestPdf(
  file: File,
  opts: { product: Product; topic: Topic; name?: string },
  onProgress?: (fraction: number, note: string) => void
): Promise<IngestResult> {
  if (typeof window === "undefined") throw new Error("Ingestion is client-only");

  onProgress?.(0.02, "Loading PDF engine…");
  // Dynamic import keeps pdf.js out of the initial bundle.
  const pdfjs: any = await import("pdfjs-dist");
  // Version-matched worker from the jsdelivr npm mirror (1:1 with the installed
  // package, so the path always exists). Avoids bundling the worker file.
  pdfjs.GlobalWorkerOptions.workerSrc = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

  const data = await file.arrayBuffer();
  onProgress?.(0.08, "Reading document…");
  const pdf = await pdfjs.getDocument({ data }).promise;
  const numPages: number = pdf.numPages;

  const id = `manual-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const name = (opts.name || file.name.replace(/\.pdf$/i, "")).slice(0, 120);

  const source: Source = {
    id: `src-${id}`,
    title: `${name} (uploaded)`,
    kind: "uploaded",
    reference: `Uploaded ${opts.product === "omi" ? "OMI" : opts.product === "appserver" ? "Application Server" : ""} manual · ${numPages} pages`.trim(),
    topics: [opts.topic],
  };

  const chunks: Chunk[] = [];
  for (let p = 1; p <= numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    const pageText = content.items
      .map((it: any) => (typeof it.str === "string" ? it.str : ""))
      .join(" ");
    const passages = chunkText(pageText);
    passages.forEach((text, idx) => {
      chunks.push({
        id: `${id}-p${p}-${idx}`,
        kind: "doc",
        title: `${name} — p.${p}`,
        topics: [opts.topic],
        text: `${name} ${text}`,
        sourceIds: [source.id],
      });
    });
    if (p % 2 === 0 || p === numPages) {
      onProgress?.(0.08 + 0.9 * (p / numPages), `Extracting page ${p} / ${numPages}…`);
    }
  }

  if (chunks.length === 0) {
    throw new Error(
      "No extractable text found. This PDF may be scanned images (OCR needed) rather than text."
    );
  }

  const manual: Manual = {
    id,
    name,
    product: opts.product,
    topic: opts.topic,
    pages: numPages,
    addedAt: Date.now(),
    source,
    chunks,
  };

  const manuals = [...listManuals(), manual];
  persist(manuals);
  registerSources([source]);
  onProgress?.(1, "Done");
  return { manual, manuals };
}
