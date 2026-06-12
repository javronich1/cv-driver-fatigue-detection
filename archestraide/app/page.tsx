import Link from "next/link";
import HeroSearch from "@/components/HeroSearch";
import {
  IconChat,
  IconWrench,
  IconBook,
  IconSearch,
  IconArrow,
  IconLayers,
} from "@/components/icons";
import { RUNBOOKS } from "@/lib/knowledge/runbooks";
import { GLOSSARY } from "@/lib/knowledge/glossary";

const QUICK = [
  {
    href: "/ask",
    icon: IconChat,
    title: "Haz una pregunta técnica",
    desc: "Respuestas estructuradas y fundamentadas para preguntas de teoría y runtime.",
  },
  {
    href: "/troubleshoot",
    icon: IconWrench,
    title: "Iniciar troubleshooting",
    desc: "Diagnóstico guiado tipo checklist para las fallas más comunes.",
  },
  {
    href: "/runbooks",
    icon: IconBook,
    title: "Explorar runbooks",
    desc: "Síntoma → causa probable → revisiones → confirmar → escalar.",
  },
  {
    href: "/docs",
    icon: IconSearch,
    title: "Buscar manuales y docs",
    desc: "Búsqueda híbrida en runbooks, conceptos y docs oficiales de AVEVA.",
  },
];

const FEATURED = [
  "rb-deploy-remote-node",
  "rb-bad-quality",
  "rb-historian-no-data",
  "rb-alarm-not-visible",
  "rb-oi-opc-not-updating",
  "rb-security-login",
];

export default function HomePage() {
  const featured = FEATURED.map((id) => RUNBOOKS.find((r) => r.id === id)!);

  return (
    <div className="space-y-10">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-3xl border border-white/[0.06] bg-base-850/60 p-6 shadow-panel sm:p-10">
        <div className="pointer-events-none absolute -right-20 -top-24 h-64 w-64 rounded-full bg-accent/10 blur-3xl" />
        <div className="relative">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/5 px-3 py-1 text-xs font-medium text-accent-soft">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
            Copilot interno de soporte · MVP
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-50 sm:text-5xl">
            Archestr<span className="text-accent">Aide</span>
          </h1>
          <p className="mt-2 text-base font-medium text-slate-300 sm:text-lg">
            AVEVA Application Server &amp; OMI Support Copilot
          </p>
          <p className="mt-4 max-w-2xl text-sm leading-relaxed text-slate-400">
            Un copilot interno de soporte de AVEVA Application Server y OMI para
            onboarding, preguntas de teoría, búsqueda de documentación y
            troubleshooting de primera línea. Fundamentado en documentación
            oficial de AVEVA y runbooks curados, con separación clara entre hechos
            oficiales y guía inferida. Sube tus propios manuales de capacitación
            para ampliar su conocimiento.
          </p>
          <div className="mt-6 max-w-2xl">
            <HeroSearch />
          </div>
          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-500">
            <span>{RUNBOOKS.length} runbooks</span>
            <span>{GLOSSARY.length} conceptos</span>
            <span>Respuestas con fuentes</span>
            <span>Modo oscuro primero</span>
          </div>
        </div>
      </section>

      {/* Quick actions */}
      <section>
        <h2 className="mb-4 text-sm font-bold uppercase tracking-wider text-slate-400">
          Acciones rápidas
        </h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {QUICK.map((q) => {
            const Icon = q.icon;
            return (
              <Link
                key={q.href}
                href={q.href}
                className="panel card-hover group flex items-start gap-4 p-5"
              >
                <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-accent/12 text-accent shadow-[inset_0_0_0_1px_rgba(45,212,191,0.25)]">
                  <Icon width={20} height={20} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold text-slate-100">{q.title}</h3>
                    <IconArrow
                      width={15}
                      height={15}
                      className="text-slate-600 transition-all group-hover:translate-x-0.5 group-hover:text-accent"
                    />
                  </div>
                  <p className="mt-1 text-sm leading-relaxed text-slate-400">
                    {q.desc}
                  </p>
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Featured issues */}
      <section>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400">
            Problemas comunes
          </h2>
          <Link
            href="/runbooks"
            className="text-xs font-medium text-accent hover:text-accent-soft"
          >
            Todos los runbooks →
          </Link>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {featured.map((rb) => (
            <Link
              key={rb.id}
              href={`/runbooks#${rb.id}`}
              className="panel card-hover group flex flex-col p-5"
            >
              <div className="mb-2 flex items-center gap-2">
                <span className="chip chip-accent">{rb.category}</span>
                <SeverityDot severity={rb.severity} />
              </div>
              <h3 className="font-semibold leading-snug text-slate-100 group-hover:text-white">
                {rb.title}
              </h3>
              <p className="mt-2 line-clamp-3 text-sm leading-relaxed text-slate-400">
                {rb.symptom}
              </p>
              <span className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-accent">
                Abrir runbook <IconArrow width={13} height={13} />
              </span>
            </Link>
          ))}
        </div>
      </section>

      {/* Knowledge sources note */}
      <section className="panel flex flex-col gap-3 p-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-signal-info/10 text-signal-info">
            <IconLayers width={18} height={18} />
          </span>
          <div>
            <p className="text-sm font-semibold text-slate-100">
              Recuperación fundamentada, no un chatbot genérico
            </p>
            <p className="mt-0.5 max-w-xl text-xs leading-relaxed text-slate-400">
              Cada respuesta sustancial cita sus fuentes y marca la confianza.
              Sube tus propios manuales de capacitación de Application Server / OMI
              para ampliar la cobertura — se indexan en tu navegador y se citan en
              las respuestas.
            </p>
          </div>
        </div>
        <div className="flex shrink-0 gap-2">
          <Link href="/manuals" className="btn btn-primary">
            Subir manuales
          </Link>
          <Link href="/about" className="btn btn-ghost">
            Cómo funciona
          </Link>
        </div>
      </section>
    </div>
  );
}

function SeverityDot({ severity }: { severity: "low" | "medium" | "high" }) {
  const cls =
    severity === "high"
      ? "bg-signal-danger"
      : severity === "medium"
      ? "bg-signal-warn"
      : "bg-signal-ok";
  return (
    <span className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-500">
      <span className={`h-1.5 w-1.5 rounded-full ${cls}`} /> {severity}
    </span>
  );
}
