"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import PageHeader from "@/components/PageHeader";
import { search } from "@/lib/retrieval";
import { getSources } from "@/lib/knowledge/sources";
import { Chunk } from "@/lib/knowledge/types";
import { SourceKindBadge } from "@/components/Sources";
import { IconSearch, IconExternal } from "@/components/icons";

const KIND_FILTERS: { id: Chunk["kind"]; label: string }[] = [
  { id: "runbook", label: "Runbooks" },
  { id: "glossary", label: "Glossary" },
  { id: "known-issue", label: "Known issues" },
];

const KIND_LABEL: Record<Chunk["kind"], string> = {
  runbook: "Runbook",
  glossary: "Concept",
  "known-issue": "Known issue",
  doc: "Doc",
};

export default function DocsPage() {
  const [q, setQ] = useState("");
  const [kinds, setKinds] = useState<Chunk["kind"][]>([]);

  const results = useMemo(() => {
    if (!q.trim()) return [];
    return search(q, { kinds: kinds.length ? kinds : undefined, limit: 30 });
  }, [q, kinds]);

  function toggleKind(k: Chunk["kind"]) {
    setKinds((cur) =>
      cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]
    );
  }

  return (
    <div>
      <PageHeader
        eyebrow="Docs / Search"
        title="Search the knowledge base"
        subtitle="Hybrid keyword + concept search across runbooks, glossary concepts, known issues, and the official AVEVA documentation they cite. Use natural language, keywords, or an error string."
      />

      <div className="panel mb-4 flex items-center gap-2 p-2">
        <span className="pl-2 text-slate-400">
          <IconSearch width={18} height={18} />
        </span>
        <input
          autoFocus
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. 'cannot communicate with remote node', 'autobind', 'store and forward'"
          className="flex-1 bg-transparent px-1 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-500"
        />
      </div>

      <div className="mb-6 flex flex-wrap gap-2">
        {KIND_FILTERS.map((f) => (
          <button
            key={f.id}
            onClick={() => toggleKind(f.id)}
            className={`chip card-hover px-3 py-1.5 ${
              kinds.includes(f.id) ? "chip-accent" : ""
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {q.trim() && (
        <p className="mb-4 text-xs text-slate-500">
          {results.length} result{results.length === 1 ? "" : "s"} for “{q}”
        </p>
      )}

      <div className="space-y-3">
        {results.map(({ chunk, matchedTerms }) => {
          const sources = getSources(chunk.sourceIds);
          return (
            <Link
              key={chunk.id}
              href={chunk.href || "#"}
              className="panel card-hover block p-4"
            >
              <div className="mb-1.5 flex flex-wrap items-center gap-2">
                <span className="chip chip-accent">{KIND_LABEL[chunk.kind]}</span>
                {chunk.topics.slice(0, 2).map((t) => (
                  <span key={t} className="chip">{t}</span>
                ))}
              </div>
              <h3 className="font-semibold text-slate-100">{chunk.title}</h3>
              <p className="mt-1 line-clamp-2 text-sm leading-relaxed text-slate-400">
                {snippet(chunk.text, matchedTerms)}
              </p>
              <div className="mt-2.5 flex flex-wrap items-center gap-2">
                {sources.slice(0, 3).map((s) => (
                  <span key={s.id} className="inline-flex items-center gap-1">
                    <SourceKindBadge kind={s.kind} />
                  </span>
                ))}
                {sources.some((s) => s.url) && (
                  <IconExternal width={12} height={12} className="text-slate-600" />
                )}
              </div>
            </Link>
          );
        })}

        {q.trim() && results.length === 0 && (
          <div className="panel p-8 text-center">
            <p className="text-sm text-slate-400">
              No matches. Try AVEVA-specific terms, or browse the{" "}
              <Link href="/glossary" className="text-accent">
                glossary
              </Link>
              .
            </p>
          </div>
        )}

        {!q.trim() && (
          <div className="panel p-8 text-center">
            <p className="text-sm text-slate-400">
              Start typing to search the knowledge base.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// Build a snippet centered on the first matched term for context.
function snippet(text: string, matched: string[]): string {
  if (!matched.length) return text.slice(0, 220);
  const lower = text.toLowerCase();
  let idx = -1;
  for (const m of matched) {
    const i = lower.indexOf(m.toLowerCase());
    if (i >= 0 && (idx === -1 || i < idx)) idx = i;
  }
  if (idx < 0) return text.slice(0, 220);
  const start = Math.max(0, idx - 90);
  return (start > 0 ? "…" : "") + text.slice(start, start + 240).trim() + "…";
}
