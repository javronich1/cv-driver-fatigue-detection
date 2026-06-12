"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/PageHeader";
import { RUNBOOKS } from "@/lib/knowledge/runbooks";
import RunbookDetail, { SeverityChip } from "@/components/RunbookDetail";

const CATEGORIES = ["Todos", ...Array.from(new Set(RUNBOOKS.map((r) => r.category)))];

export default function RunbooksPage() {
  const [cat, setCat] = useState("Todos");
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    const hash = window.location.hash.replace("#", "");
    if (hash && RUNBOOKS.some((r) => r.id === hash)) {
      setOpen(hash);
      setTimeout(() => {
        document
          .getElementById(hash)
          ?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 80);
    }
  }, []);

  const list = RUNBOOKS.filter((r) => cat === "Todos" || r.category === cat);

  return (
    <div>
      <PageHeader
        eyebrow="Runbooks"
        title="Runbooks de soporte"
        subtitle="Playbooks concisos orientados a soporte para los problemas más comunes de AVEVA Application Server y OMI. Cada uno recorre síntoma → causas probables → primera herramienta → revisiones ordenadas → confirmar → escalar."
      />

      <div className="mb-6 flex flex-wrap gap-2">
        {CATEGORIES.map((c) => (
          <button
            key={c}
            onClick={() => setCat(c)}
            className={`chip card-hover px-3 py-1.5 ${cat === c ? "chip-accent" : ""}`}
          >
            {c}
          </button>
        ))}
      </div>

      <div className="space-y-4">
        {list.map((rb) => {
          const isOpen = open === rb.id;
          return (
            <div key={rb.id} id={rb.id} className="panel scroll-mt-24 overflow-hidden">
              <button
                onClick={() => setOpen(isOpen ? null : rb.id)}
                className="flex w-full items-start gap-3 p-5 text-left"
              >
                <div className="min-w-0 flex-1">
                  <div className="mb-1.5 flex flex-wrap items-center gap-2">
                    <span className="chip chip-accent">{rb.category}</span>
                    <SeverityChip severity={rb.severity} />
                  </div>
                  <h3 className="text-base font-bold tracking-tight text-slate-100">
                    {rb.title}
                  </h3>
                  <p className="mt-1 text-sm leading-relaxed text-slate-400">
                    {rb.symptom}
                  </p>
                </div>
                <span className="mt-1 shrink-0 text-lg leading-none text-slate-500">
                  {isOpen ? "−" : "+"}
                </span>
              </button>

              {isOpen && (
                <div className="animate-fade-up border-t border-white/[0.06] p-5">
                  <RunbookDetail rb={rb} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
