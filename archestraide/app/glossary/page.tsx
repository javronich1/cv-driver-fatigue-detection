"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import PageHeader from "@/components/PageHeader";
import { GLOSSARY, GLOSSARY_BY_ID } from "@/lib/knowledge/glossary";
import { getSources } from "@/lib/knowledge/sources";
import { SourceKindBadge } from "@/components/Sources";
import { IconSearch } from "@/components/icons";

export default function GlossaryPage() {
  const [q, setQ] = useState("");
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const hash = window.location.hash.replace("#", "");
    if (hash && GLOSSARY_BY_ID[hash]) {
      setActive(hash);
      setTimeout(
        () =>
          document
            .getElementById(hash)
            ?.scrollIntoView({ behavior: "smooth", block: "center" }),
        80
      );
    }
  }, []);

  const sorted = useMemo(
    () => [...GLOSSARY].sort((a, b) => a.term.localeCompare(b.term)),
    []
  );

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return sorted;
    return sorted.filter(
      (g) =>
        g.term.toLowerCase().includes(s) ||
        (g.aliases || []).some((a) => a.toLowerCase().includes(s)) ||
        g.short.toLowerCase().includes(s)
    );
  }, [q, sorted]);

  return (
    <div>
      <PageHeader
        eyebrow="Glosario"
        title="Conceptos y términos"
        subtitle="Una referencia rápida y curada de los conceptos clave de AVEVA Application Server / OMI / System Platform — definición corta, explicación simple, un ejemplo práctico y términos relacionados."
      />

      <div className="panel mb-6 flex items-center gap-2 p-2">
        <span className="pl-2 text-slate-400">
          <IconSearch width={18} height={18} />
        </span>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Filtra términos — p. ej. 'autobind', 'OnScan', 'galaxy'"
          className="flex-1 bg-transparent px-1 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-500"
        />
      </div>

      <div className="grid gap-3">
        {filtered.map((g) => (
          <div
            key={g.id}
            id={g.id}
            className={`panel scroll-mt-24 p-5 transition-all ${
              active === g.id ? "border-accent/40 shadow-glow" : ""
            }`}
          >
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <h3 className="text-base font-bold tracking-tight text-slate-100">
                {g.term}
              </h3>
              {g.aliases && g.aliases.length > 0 && (
                <span className="font-mono text-xs text-slate-500">
                  {g.aliases.join(" · ")}
                </span>
              )}
            </div>
            <p className="mt-1.5 text-sm font-medium text-accent-soft">{g.short}</p>
            <p className="mt-2 text-sm leading-relaxed text-slate-300">
              {g.explanation}
            </p>
            {g.example && (
              <div className="mt-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  Ejemplo
                </p>
                <p className="mt-1 text-sm leading-relaxed text-slate-300">
                  {g.example}
                </p>
              </div>
            )}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              {g.related?.map((rid) => {
                const r = GLOSSARY_BY_ID[rid];
                if (!r) return null;
                return (
                  <Link
                    key={rid}
                    href={`#${rid}`}
                    onClick={() => setActive(rid)}
                    className="chip card-hover"
                  >
                    {r.term}
                  </Link>
                );
              })}
            </div>
            <div className="mt-3 flex flex-wrap gap-2 border-t border-white/[0.05] pt-3">
              {getSources(g.sourceIds).map((s) =>
                s.url ? (
                  <a key={s.id} href={s.url} target="_blank" rel="noopener noreferrer">
                    <SourceKindBadge kind={s.kind} />
                  </a>
                ) : (
                  <SourceKindBadge key={s.id} kind={s.kind} />
                )
              )}
            </div>
          </div>
        ))}
        {filtered.length === 0 && (
          <div className="panel p-8 text-center text-sm text-slate-400">
            Ningún término coincide con “{q}”.
          </div>
        )}
      </div>
    </div>
  );
}
