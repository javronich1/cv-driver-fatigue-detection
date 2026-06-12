import PageHeader from "@/components/PageHeader";
import { SOURCES } from "@/lib/knowledge/sources";
import { RUNBOOKS } from "@/lib/knowledge/runbooks";
import { GLOSSARY } from "@/lib/knowledge/glossary";
import { KNOWN_ISSUES } from "@/lib/knowledge/knownIssues";
import { SourceKindBadge } from "@/components/Sources";
import { IconShield, IconLayers, IconSpark } from "@/components/icons";

export default function AboutPage() {
  const officialCount = SOURCES.filter(
    (s) => s.kind === "official-doc" || s.kind === "official-pdf"
  ).length;

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Settings / About"
        title="About ArchestrAide"
        subtitle="An internal AVEVA Application Server support copilot for onboarding, theory questions, documentation search, and first-line troubleshooting."
      />

      <div className="grid gap-4 sm:grid-cols-4">
        <Stat label="Runbooks" value={RUNBOOKS.length} />
        <Stat label="Concepts" value={GLOSSARY.length} />
        <Stat label="Known issues" value={KNOWN_ISSUES.length} />
        <Stat label="Cited sources" value={SOURCES.length} />
      </div>

      <section className="panel p-5 sm:p-6">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-300">
          <IconLayers width={16} height={16} className="text-accent" /> How it works
        </h2>
        <div className="space-y-3 text-sm leading-relaxed text-slate-300">
          <p>
            ArchestrAide is a <strong>retrieval-grounded</strong> assistant, not a
            generic chatbot. Every runbook, concept, and known issue is stored as
            structured content with explicit source metadata. When you ask a
            question, a hybrid keyword + concept retrieval layer finds the most
            relevant material, and a deterministic composer assembles a structured,
            support-oriented answer with citations and a confidence flag.
          </p>
          <p>
            When an <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">ANTHROPIC_API_KEY</code>{" "}
            is configured on the host, the retrieved context is additionally passed
            to Claude under strict grounding instructions to produce a polished
            short answer — but the structure, sources, and confidence always come
            from the grounded layer, so citations stay trustworthy. With no key, the
            app runs fully on the deterministic composer.
          </p>
        </div>
      </section>

      <section className="panel p-5 sm:p-6">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-300">
          <IconShield width={16} height={16} className="text-signal-warn" /> Trust &
          safety model
        </h2>
        <ul className="space-y-2 text-sm leading-relaxed text-slate-300">
          {[
            "Answers cite their sources and separate official facts from inferred troubleshooting heuristics.",
            "The assistant won't invent AVEVA-specific settings, attribute names, or menu paths that aren't in a trusted source — it says when it's unsure.",
            "Troubleshooting is framed as 'most likely / check first', never 'this is definitely the cause'.",
            "Environment-specific warnings (lab vs production, authentication mode) are surfaced where relevant.",
            "It never claims to have validated live environment state unless you provide it.",
          ].map((t) => (
            <li key={t} className="flex gap-2.5">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent/70" />
              <span>{t}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="panel p-5 sm:p-6">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-300">
          <IconSpark width={16} height={16} className="text-accent" /> Extending the
          knowledge base
        </h2>
        <p className="mb-3 text-sm leading-relaxed text-slate-300">
          The fastest way is the{" "}
          <a href="/manuals" className="text-accent">Manuals</a> page: upload your
          AVEVA Application Server or OMI training PDFs and they&apos;re parsed in
          your browser, indexed, and blended into answers (cited as “Your manual”).
          Stored per-device; for a shared library, ingest server-side.
        </p>
        <p className="text-sm leading-relaxed text-slate-300">
          To extend the built-in knowledge, edit structured content under{" "}
          <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">lib/knowledge/</code>{" "}
          — <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">runbooks.ts</code>,{" "}
          <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">glossary.ts</code>,{" "}
          <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">knownIssues.ts</code>, and{" "}
          <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">sources.ts</code>.
          See the ingestion guide in the README for the PDF→chunks pipeline.
        </p>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-bold uppercase tracking-wider text-slate-400">
          Knowledge sources ({officialCount} official)
        </h2>
        <div className="grid gap-2 sm:grid-cols-2">
          {SOURCES.map((s) => (
            <div key={s.id} className="panel flex items-start gap-3 p-3.5">
              <div className="min-w-0 flex-1">
                <div className="mb-1">
                  <SourceKindBadge kind={s.kind} />
                </div>
                {s.url ? (
                  <a
                    href={s.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium text-slate-100 hover:text-accent"
                  >
                    {s.title}
                  </a>
                ) : (
                  <p className="text-sm font-medium text-slate-100">{s.title}</p>
                )}
                {s.reference && (
                  <p className="mt-0.5 text-xs text-slate-500">{s.reference}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      <p className="text-xs leading-relaxed text-slate-500">
        ArchestrAide is an MVP. It assists with onboarding and first-line
        troubleshooting and is not a substitute for official AVEVA support or
        validated change-management processes in production environments.
      </p>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="panel p-4">
      <p className="text-2xl font-extrabold tracking-tight text-slate-100">
        {value}
      </p>
      <p className="mt-0.5 text-xs font-medium uppercase tracking-wider text-slate-500">
        {label}
      </p>
    </div>
  );
}
