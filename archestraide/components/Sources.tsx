import { Source, SourceKind } from "@/lib/knowledge/types";
import { IconExternal } from "./icons";

const KIND_META: Record<
  SourceKind,
  { label: string; cls: string; official: boolean }
> = {
  "official-doc": {
    label: "Official doc",
    cls: "border-signal-ok/30 bg-signal-ok/10 text-signal-ok",
    official: true,
  },
  "official-pdf": {
    label: "Official manual",
    cls: "border-signal-ok/30 bg-signal-ok/10 text-signal-ok",
    official: true,
  },
  uploaded: {
    label: "Your manual",
    cls: "border-accent/30 bg-accent/10 text-accent-soft",
    official: true,
  },
  community: {
    label: "Community / vendor",
    cls: "border-signal-warn/30 bg-signal-warn/10 text-signal-warn",
    official: false,
  },
  runbook: {
    label: "Curated runbook",
    cls: "border-signal-info/30 bg-signal-info/10 text-signal-info",
    official: false,
  },
  glossary: {
    label: "Concept page",
    cls: "border-signal-info/30 bg-signal-info/10 text-signal-info",
    official: false,
  },
};

export function SourceKindBadge({ kind }: { kind: SourceKind }) {
  const m = KIND_META[kind];
  return (
    <span className={`chip ${m.cls}`}>
      {m.official ? "✓ " : ""}
      {m.label}
    </span>
  );
}

export function SourceCard({ source }: { source: Source }) {
  const m = KIND_META[source.kind];
  const inner = (
    <div className="panel card-hover group flex items-start gap-3 p-3.5">
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex flex-wrap items-center gap-2">
          <SourceKindBadge kind={source.kind} />
          {source.url && (
            <span className="text-slate-500 transition-colors group-hover:text-accent">
              <IconExternal width={13} height={13} />
            </span>
          )}
        </div>
        <p className="text-sm font-semibold leading-snug text-slate-100">
          {source.title}
        </p>
        {source.reference && (
          <p className="mt-1 text-xs text-slate-500">{source.reference}</p>
        )}
      </div>
    </div>
  );
  if (source.url) {
    return (
      <a href={source.url} target="_blank" rel="noopener noreferrer">
        {inner}
      </a>
    );
  }
  return inner;
}

export function SourceList({ sources }: { sources: Source[] }) {
  if (!sources.length) return null;
  const OFFICIAL = new Set(["official-doc", "official-pdf", "uploaded"]);
  const official = sources.filter((s) => OFFICIAL.has(s.kind));
  const other = sources.filter((s) => !OFFICIAL.has(s.kind));
  return (
    <div className="space-y-3">
      {official.length > 0 && (
        <div className="space-y-2">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Official sources
          </p>
          {official.map((s) => (
            <SourceCard key={s.id} source={s} />
          ))}
        </div>
      )}
      {other.length > 0 && (
        <div className="space-y-2">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Inferred / community guidance
          </p>
          {other.map((s) => (
            <SourceCard key={s.id} source={s} />
          ))}
        </div>
      )}
    </div>
  );
}
