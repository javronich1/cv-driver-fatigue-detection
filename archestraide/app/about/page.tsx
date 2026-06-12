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
        eyebrow="Ajustes / Acerca de"
        title="Acerca de ArchestrAide"
        subtitle="Un copilot interno de soporte de AVEVA Application Server y OMI para onboarding, preguntas de teoría, búsqueda de documentación y troubleshooting de primera línea."
      />

      <div className="grid gap-4 sm:grid-cols-4">
        <Stat label="Runbooks" value={RUNBOOKS.length} />
        <Stat label="Conceptos" value={GLOSSARY.length} />
        <Stat label="Problemas conocidos" value={KNOWN_ISSUES.length} />
        <Stat label="Fuentes citadas" value={SOURCES.length} />
      </div>

      <section className="panel p-5 sm:p-6">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-300">
          <IconLayers width={16} height={16} className="text-accent" /> Cómo funciona
        </h2>
        <div className="space-y-3 text-sm leading-relaxed text-slate-300">
          <p>
            ArchestrAide es un asistente <strong>fundamentado en recuperación</strong>,
            no un chatbot genérico. Cada runbook, concepto y problema conocido se
            guarda como contenido estructurado con metadatos de fuente explícitos.
            Cuando preguntas, una capa de recuperación híbrida (palabra clave +
            concepto) encuentra el material más relevante y un compositor
            determinista arma una respuesta estructurada y orientada a soporte, con
            citas y una marca de confianza.
          </p>
          <p>
            Cuando se configura una{" "}
            <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">ANTHROPIC_API_KEY</code>{" "}
            en el host, el contexto recuperado se pasa además a Claude con
            instrucciones estrictas de fundamentación para producir una respuesta
            corta pulida — pero la estructura, las fuentes y la confianza siempre
            vienen de la capa fundamentada, así que las citas siguen siendo
            confiables. Sin clave, la app funciona del todo con el compositor
            determinista.
          </p>
        </div>
      </section>

      <section className="panel p-5 sm:p-6">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-300">
          <IconShield width={16} height={16} className="text-signal-warn" /> Modelo de
          confianza y seguridad
        </h2>
        <ul className="space-y-2 text-sm leading-relaxed text-slate-300">
          {[
            "Las respuestas citan sus fuentes y separan los hechos oficiales de las heurísticas de troubleshooting inferidas.",
            "El asistente no inventa settings, nombres de atributos ni rutas de menú específicas de AVEVA que no estén en una fuente confiable — dice cuándo no está seguro.",
            "El troubleshooting se enmarca como 'lo más probable / revisa primero', nunca como 'esta es definitivamente la causa'.",
            "Se muestran advertencias específicas del entorno (lab vs producción, modo de autenticación) cuando aplican.",
            "Nunca afirma haber validado el estado en vivo del entorno salvo que tú lo proporciones.",
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
          <IconSpark width={16} height={16} className="text-accent" /> Ampliar la base
          de conocimiento
        </h2>
        <p className="mb-3 text-sm leading-relaxed text-slate-300">
          La forma más rápida es la página{" "}
          <a href="/manuals" className="text-accent">Manuales</a>: sube tus PDFs de
          capacitación de AVEVA Application Server u OMI y se procesan en tu
          navegador, se indexan y se integran en las respuestas (citados como “Tu
          manual”). Se guardan por dispositivo; para una biblioteca compartida,
          ingesta en servidor. También puedes aportar problemas a la{" "}
          <a href="/community" className="text-accent">Comunidad</a>.
        </p>
        <p className="text-sm leading-relaxed text-slate-300">
          Para ampliar el conocimiento incorporado, edita el contenido estructurado
          en{" "}
          <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">lib/knowledge/</code>{" "}
          — <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">runbooks.ts</code>,{" "}
          <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">glossary.ts</code>,{" "}
          <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">knownIssues.ts</code> y{" "}
          <code className="rounded bg-base-700 px-1.5 py-0.5 text-xs">sources.ts</code>.
          Mira la guía de ingesta en el README para el pipeline PDF→chunks.
        </p>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-bold uppercase tracking-wider text-slate-400">
          Fuentes de conocimiento ({officialCount} oficiales)
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
        ArchestrAide es un MVP. Asiste en el onboarding y el troubleshooting de
        primera línea y no sustituye al soporte oficial de AVEVA ni a los procesos
        validados de gestión de cambios en entornos de producción.
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
