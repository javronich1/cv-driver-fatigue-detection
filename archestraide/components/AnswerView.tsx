import { ComposedAnswer } from "@/lib/answer";
import { SourceList } from "./Sources";
import { IconCheck, IconTool, IconArrow, IconShield, IconSpark } from "./icons";
import Link from "next/link";

const CONFIDENCE_META = {
  high: { label: "Confianza alta", cls: "text-signal-ok", dot: "bg-signal-ok" },
  medium: { label: "Confianza media", cls: "text-signal-warn", dot: "bg-signal-warn" },
  low: { label: "Confianza baja", cls: "text-signal-danger", dot: "bg-signal-danger" },
};

const MODE_META = {
  troubleshoot: { label: "Troubleshooting", cls: "chip-accent" },
  concept: { label: "Concepto", cls: "chip-accent" },
  general: { label: "Referencia", cls: "chip-accent" },
};

export default function AnswerView({
  answer,
  onFollowUp,
}: {
  answer: ComposedAnswer;
  onFollowUp?: (q: string) => void;
}) {
  const conf = CONFIDENCE_META[answer.confidence];
  const mode = MODE_META[answer.mode];

  return (
    <div className="animate-fade-up grid gap-5 lg:grid-cols-[1fr_300px]">
      {/* Main answer column */}
      <div className="min-w-0 space-y-5">
        <div className="panel p-5 sm:p-6">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <span className={`chip ${mode.cls}`}>{mode.label}</span>
            <span className="chip">
              <span className={`h-1.5 w-1.5 rounded-full ${conf.dot}`} />
              <span className={conf.cls}>{conf.label}</span>
            </span>
            {answer.llm ? (
              <span className="chip">
                <IconSpark width={12} height={12} /> Sintetizado por IA
              </span>
            ) : (
              <span className="chip">Compositor fundamentado</span>
            )}
          </div>

          {/* Short answer */}
          <div className="mb-1 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-accent">
            Respuesta corta
          </div>
          <p className="text-[15px] font-medium leading-relaxed text-slate-100">
            {answer.shortAnswer}
          </p>

          {answer.confidenceNote && (
            <div className="mt-4 flex gap-2.5 rounded-xl border border-signal-warn/20 bg-signal-warn/[0.06] p-3 text-xs leading-relaxed text-slate-300">
              <IconShield width={15} height={15} className="mt-0.5 shrink-0 text-signal-warn" />
              <span>{answer.confidenceNote}</span>
            </div>
          )}
        </div>

        {/* Sections */}
        {answer.sections.map((section, i) => (
          <div key={i} className="panel p-5 sm:p-6">
            <h3 className="mb-3 text-sm font-bold tracking-tight text-slate-100">
              {section.heading}
            </h3>
            {section.kind === "text" && section.body && (
              <p className="text-sm leading-relaxed text-slate-300">{section.body}</p>
            )}
            {section.kind === "list" && section.items && (
              <ul className="space-y-2">
                {section.items.map((item, j) => (
                  <li key={j} className="flex gap-2.5 text-sm leading-relaxed text-slate-300">
                    <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent/70" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            )}
            {section.kind === "steps" && section.steps && (
              <ol className="space-y-3">
                {section.steps.map((step, j) => (
                  <li key={j} className="flex gap-3">
                    <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-accent/15 text-xs font-bold text-accent">
                      {j + 1}
                    </span>
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-slate-100">{step.title}</p>
                      <p className="mt-0.5 text-sm leading-relaxed text-slate-400">
                        {step.detail}
                      </p>
                      {step.tool && (
                        <span className="mt-1.5 inline-flex chip chip-accent">
                          <IconTool width={12} height={12} /> {step.tool}
                        </span>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </div>
        ))}
      </div>

      {/* Side panel */}
      <aside className="space-y-5">
        {answer.tools.length > 0 && (
          <div className="panel p-4">
            <h4 className="mb-3 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-400">
              <IconTool width={14} height={14} /> Herramientas a abrir
            </h4>
            <div className="flex flex-wrap gap-2">
              {answer.tools.map((t) => (
                <span key={t} className="chip">{t}</span>
              ))}
            </div>
          </div>
        )}

        <div className="panel p-4">
          <h4 className="mb-3 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-400">
            <IconCheck width={14} height={14} /> Fuentes
          </h4>
          {answer.sources.length ? (
            <SourceList sources={answer.sources} />
          ) : (
            <p className="text-xs text-slate-500">
              Ninguna fuente fundamentada coincidió — no se hace afirmación específica de AVEVA.
            </p>
          )}
        </div>

        {answer.relatedTerms.length > 0 && (
          <div className="panel p-4">
            <h4 className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-400">
              Términos relacionados
            </h4>
            <div className="flex flex-wrap gap-2">
              {answer.relatedTerms.map((t) => (
                <Link key={t.id} href={`/glossary#${t.id}`} className="chip card-hover">
                  {t.term}
                </Link>
              ))}
            </div>
          </div>
        )}

        {answer.followUps.length > 0 && (
          <div className="panel p-4">
            <h4 className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-400">
              Preguntas de seguimiento
            </h4>
            <div className="space-y-2">
              {answer.followUps.map((q) => (
                <button
                  key={q}
                  onClick={() => onFollowUp?.(q)}
                  className="group flex w-full items-center justify-between gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-left text-xs text-slate-300 transition-all hover:border-accent/30 hover:text-white"
                >
                  <span>{q}</span>
                  <IconArrow
                    width={13}
                    height={13}
                    className="shrink-0 text-slate-600 transition-colors group-hover:text-accent"
                  />
                </button>
              ))}
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}
