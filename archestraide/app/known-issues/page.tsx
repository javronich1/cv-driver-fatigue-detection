"use client";

import { useEffect } from "react";
import PageHeader from "@/components/PageHeader";
import { KNOWN_ISSUES } from "@/lib/knowledge/knownIssues";
import { getSources } from "@/lib/knowledge/sources";
import { SourceKindBadge } from "@/components/Sources";
import { IconAlert } from "@/components/icons";

const STATUS_META: Record<string, { label: string; cls: string }> = {
  known: { label: "Known issue", cls: "border-signal-warn/30 bg-signal-warn/10 text-signal-warn" },
  "by-design": { label: "By design", cls: "border-signal-info/30 bg-signal-info/10 text-signal-info" },
  "fixed-in-patch": { label: "Fixed in patch", cls: "border-signal-ok/30 bg-signal-ok/10 text-signal-ok" },
};

export default function KnownIssuesPage() {
  useEffect(() => {
    const hash = window.location.hash.replace("#", "");
    if (hash) {
      setTimeout(
        () =>
          document
            .getElementById(hash)
            ?.scrollIntoView({ behavior: "smooth", block: "start" }),
        80
      );
    }
  }, []);

  return (
    <div>
      <PageHeader
        eyebrow="Known issues"
        title="Known issues & gotchas"
        subtitle="Environment-specific patterns distilled from official readmes and clearly-labelled community tech notes. These are framed honestly as patterns to check — not universal root causes."
      />

      <div className="space-y-4">
        {KNOWN_ISSUES.map((k) => {
          const status = STATUS_META[k.status];
          return (
            <div key={k.id} id={k.id} className="panel scroll-mt-24 p-5 sm:p-6">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className={`chip ${status.cls}`}>{status.label}</span>
                <span className="chip">{k.environment}</span>
              </div>
              <h3 className="flex items-start gap-2 text-base font-bold tracking-tight text-slate-100">
                <IconAlert width={16} height={16} className="mt-0.5 shrink-0 text-signal-warn" />
                {k.title}
              </h3>

              <dl className="mt-3 space-y-3">
                <Row label="Symptom" value={k.symptom} />
                <Row label="Cause" value={k.cause} />
                <Row label="Workaround" value={k.workaround} />
              </dl>

              <div className="mt-3 flex flex-wrap gap-2 border-t border-white/[0.05] pt-3">
                {getSources(k.sourceIds).map((s) =>
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
          );
        })}
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid gap-1 sm:grid-cols-[110px_1fr] sm:gap-3">
      <dt className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
        {label}
      </dt>
      <dd className="text-sm leading-relaxed text-slate-300">{value}</dd>
    </div>
  );
}
