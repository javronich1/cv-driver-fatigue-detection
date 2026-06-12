import { Runbook } from "@/lib/knowledge/types";
import { getSources } from "@/lib/knowledge/sources";
import { SourceList } from "./Sources";
import { IconTool, IconCheck, IconAlert } from "./icons";

export function SeverityChip({ severity }: { severity: Runbook["severity"] }) {
  const cls =
    severity === "high"
      ? "border-signal-danger/30 bg-signal-danger/10 text-signal-danger"
      : severity === "medium"
      ? "border-signal-warn/30 bg-signal-warn/10 text-signal-warn"
      : "border-signal-ok/30 bg-signal-ok/10 text-signal-ok";
  return <span className={`chip ${cls}`}>{severity} severity</span>;
}

function Block({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div>
      <p className="mb-2 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-400">
        {icon} {title}
      </p>
      {children}
    </div>
  );
}

export default function RunbookDetail({
  rb,
  note,
}: {
  rb: Runbook;
  note?: string;
}) {
  return (
    <div className="space-y-5">
      {note && (
        <div className="rounded-xl border border-signal-warn/20 bg-signal-warn/[0.06] p-3 text-sm leading-relaxed text-slate-300">
          {note}
        </div>
      )}

      <Block title="Most likely causes" icon={<IconAlert width={14} height={14} />}>
        <ul className="space-y-1.5">
          {rb.likelyCauses.map((c, i) => (
            <li key={i} className="flex gap-2.5 text-sm text-slate-300">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent/70" />
              <span>{c}</span>
            </li>
          ))}
        </ul>
      </Block>

      <div className="rounded-xl border border-accent/20 bg-accent/[0.05] p-3 text-sm">
        <span className="font-semibold text-accent-soft">
          Open first: {rb.firstTool}
        </span>
      </div>

      <Block title="Step-by-step checks" icon={<IconTool width={14} height={14} />}>
        <ol className="space-y-3">
          {rb.steps.map((s, i) => (
            <li key={i} className="flex gap-3">
              <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-accent/15 text-xs font-bold text-accent">
                {i + 1}
              </span>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-slate-100">{s.title}</p>
                <p className="mt-0.5 text-sm leading-relaxed text-slate-400">
                  {s.detail}
                </p>
                {s.tool && (
                  <span className="mt-1.5 inline-flex chip chip-accent">
                    <IconTool width={12} height={12} /> {s.tool}
                  </span>
                )}
              </div>
            </li>
          ))}
        </ol>
      </Block>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-signal-ok/20 bg-signal-ok/[0.05] p-4">
          <p className="mb-1.5 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-signal-ok">
            <IconCheck width={14} height={14} /> Confirm resolved
          </p>
          <p className="text-sm leading-relaxed text-slate-300">
            {rb.confirmResolution}
          </p>
        </div>
        <div className="rounded-xl border border-signal-danger/20 bg-signal-danger/[0.05] p-4">
          <p className="mb-1.5 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-signal-danger">
            <IconAlert width={14} height={14} /> Escalate when
          </p>
          <p className="text-sm leading-relaxed text-slate-300">
            {rb.escalateWhen}
          </p>
        </div>
      </div>

      <Block title="Sources">
        <SourceList sources={getSources(rb.sourceIds)} />
      </Block>
    </div>
  );
}
