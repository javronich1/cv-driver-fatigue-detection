"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/PageHeader";
import {
  buildIssueUrl,
  fetchCommunity,
  normalizeSubmission,
  CommunityEntry,
  COMMUNITY_REPO,
} from "@/lib/community";
import {
  IconUsers,
  IconExternal,
  IconPlus,
  IconAlert,
  IconTool,
  IconCheck,
} from "@/components/icons";

const EMPTY = {
  title: "",
  category: "",
  symptom: "",
  likelyCauses: "",
  steps: "",
  firstTool: "",
  confirmResolution: "",
  escalateWhen: "",
  author: "",
};

export default function CommunityPage() {
  const [form, setForm] = useState({ ...EMPTY });
  const [entries, setEntries] = useState<CommunityEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  function load() {
    setError(null);
    setEntries(null);
    fetchCommunity()
      .then(setEntries)
      .catch((e) => {
        setEntries([]);
        setError(e?.message || "No se pudieron cargar los aportes.");
      });
  }

  useEffect(() => {
    load();
  }, []);

  function set<K extends keyof typeof EMPTY>(k: K, v: string) {
    setForm((f) => ({ ...f, [k]: v }));
  }

  function submit() {
    const sub = normalizeSubmission(form);
    if (!sub.title || !sub.symptom) return;
    window.open(buildIssueUrl(sub), "_blank", "noopener,noreferrer");
  }

  const canSubmit = form.title.trim() && form.symptom.trim();

  return (
    <div>
      <PageHeader
        eyebrow="Comunidad"
        title="Problemas de la comunidad"
        subtitle="Aporta problemas y soluciones que vivas en tu entorno. Los aportes se guardan como Issues en el repositorio (compartidos para todos) y se muestran aquí con el mismo formato que los runbooks."
      />

      {/* How it works notice */}
      <div className="panel mb-6 flex items-start gap-3 p-4">
        <span className="mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-signal-info/10 text-signal-info">
          <IconUsers width={17} height={17} />
        </span>
        <div className="text-sm leading-relaxed text-slate-300">
          <p className="font-semibold text-slate-100">Almacenamiento compartido vía GitHub</p>
          <p className="mt-0.5 text-slate-400">
            Al enviar, se abre GitHub con el problema pre-rellenado; haz clic en
            “Submit new issue” para publicarlo. Aparecerá aquí para todos en cuanto
            se cree. Necesitas acceso de GitHub al repositorio{" "}
            <code className="rounded bg-base-700 px-1 py-0.5 text-xs">{COMMUNITY_REPO}</code>{" "}
            para enviar; leer es público.
          </p>
        </div>
      </div>

      <div className="mb-6 flex flex-wrap gap-3">
        <button
          onClick={() => setShowForm((s) => !s)}
          className="btn btn-primary"
        >
          <IconPlus width={16} height={16} />
          {showForm ? "Ocultar formulario" : "Aportar un problema"}
        </button>
        <button onClick={load} className="btn btn-ghost">
          Recargar aportes
        </button>
      </div>

      {/* Submission form */}
      {showForm && (
        <div className="panel animate-fade-up mb-8 space-y-4 p-5 sm:p-6">
          <Field label="Título *" hint="Resumen corto del problema">
            <input
              className="input"
              value={form.title}
              onChange={(e) => set("title", e.target.value)}
              placeholder="p. ej. ViewApp no carga tras cambiar el screen profile"
            />
          </Field>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Categoría">
              <input
                className="input"
                value={form.category}
                onChange={(e) => set("category", e.target.value)}
                placeholder="p. ej. OMI / ViewApp"
              />
            </Field>
            <Field label="Primera herramienta a abrir">
              <input
                className="input"
                value={form.firstTool}
                onChange={(e) => set("firstTool", e.target.value)}
                placeholder="p. ej. Platform Manager"
              />
            </Field>
          </div>
          <Field label="Síntoma *" hint="Qué se observa">
            <textarea
              className="input min-h-[70px]"
              value={form.symptom}
              onChange={(e) => set("symptom", e.target.value)}
              placeholder="Describe el comportamiento observado…"
            />
          </Field>
          <Field label="Causas más probables" hint="Una por línea">
            <textarea
              className="input min-h-[80px]"
              value={form.likelyCauses}
              onChange={(e) => set("likelyCauses", e.target.value)}
              placeholder={"OI Server no corriendo\nReferencia de I/O incorrecta"}
            />
          </Field>
          <Field label="Pasos de troubleshooting" hint="Uno por línea, en orden">
            <textarea
              className="input min-h-[100px]"
              value={form.steps}
              onChange={(e) => set("steps", e.target.value)}
              placeholder={"Revisa el scan state en Platform Manager\nValida la referencia en el IDE"}
            />
          </Field>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Cómo confirmar resolución">
              <textarea
                className="input min-h-[60px]"
                value={form.confirmResolution}
                onChange={(e) => set("confirmResolution", e.target.value)}
              />
            </Field>
            <Field label="Cuándo escalar">
              <textarea
                className="input min-h-[60px]"
                value={form.escalateWhen}
                onChange={(e) => set("escalateWhen", e.target.value)}
              />
            </Field>
          </div>
          <Field label="Autor (opcional)">
            <input
              className="input"
              value={form.author}
              onChange={(e) => set("author", e.target.value)}
              placeholder="Tu nombre o equipo"
            />
          </Field>

          <div className="flex items-center gap-3 pt-1">
            <button
              onClick={submit}
              disabled={!canSubmit}
              className="btn btn-primary"
            >
              <IconExternal width={15} height={15} /> Publicar en GitHub
            </button>
            <span className="text-xs text-slate-500">
              Se abre GitHub con el problema pre-rellenado.
            </span>
          </div>
        </div>
      )}

      {/* Entries */}
      {error && (
        <div className="panel mb-4 flex items-start gap-2 p-4 text-sm text-slate-300">
          <IconAlert width={15} height={15} className="mt-0.5 shrink-0 text-signal-warn" />
          <span>
            {error} (Si nunca se han publicado aportes, esto es normal — sé el
            primero.)
          </span>
        </div>
      )}

      {entries === null ? (
        <div className="panel animate-pulse-soft p-8 text-center text-sm text-slate-400">
          Cargando aportes…
        </div>
      ) : entries.length === 0 ? (
        <div className="panel p-8 text-center text-sm text-slate-400">
          Aún no hay aportes de la comunidad. Usa “Aportar un problema” para añadir
          el primero.
        </div>
      ) : (
        <div className="space-y-4">
          {entries.map((e) => (
            <div key={e.number} className="panel p-5 sm:p-6">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="chip chip-accent">{e.category}</span>
                <span className="chip border-accent/30 bg-accent/10 text-accent-soft">
                  Comunidad
                </span>
              </div>
              <h3 className="text-base font-bold tracking-tight text-slate-100">
                {e.title}
              </h3>
              <p className="mt-1 text-sm leading-relaxed text-slate-400">{e.symptom}</p>

              {e.likelyCauses.length > 0 && (
                <div className="mt-4">
                  <p className="mb-2 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-400">
                    <IconAlert width={14} height={14} /> Causas más probables
                  </p>
                  <ul className="space-y-1.5">
                    {e.likelyCauses.map((c, i) => (
                      <li key={i} className="flex gap-2.5 text-sm text-slate-300">
                        <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent/70" />
                        <span>{c}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {e.firstTool && (
                <div className="mt-4 rounded-xl border border-accent/20 bg-accent/[0.05] p-3 text-sm">
                  <span className="font-semibold text-accent-soft">
                    Abre primero: {e.firstTool}
                  </span>
                </div>
              )}

              {e.steps.length > 0 && (
                <div className="mt-4">
                  <p className="mb-2 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-400">
                    <IconTool width={14} height={14} /> Pasos
                  </p>
                  <ol className="space-y-2">
                    {e.steps.map((s, i) => (
                      <li key={i} className="flex gap-3 text-sm text-slate-300">
                        <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-accent/15 text-xs font-bold text-accent">
                          {i + 1}
                        </span>
                        <span className="pt-0.5">{s}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}

              {(e.confirmResolution || e.escalateWhen) && (
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  {e.confirmResolution && (
                    <div className="rounded-xl border border-signal-ok/20 bg-signal-ok/[0.05] p-3">
                      <p className="mb-1 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-signal-ok">
                        <IconCheck width={13} height={13} /> Confirmar
                      </p>
                      <p className="text-sm text-slate-300">{e.confirmResolution}</p>
                    </div>
                  )}
                  {e.escalateWhen && (
                    <div className="rounded-xl border border-signal-danger/20 bg-signal-danger/[0.05] p-3">
                      <p className="mb-1 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-signal-danger">
                        <IconAlert width={13} height={13} /> Escalar
                      </p>
                      <p className="text-sm text-slate-300">{e.escalateWhen}</p>
                    </div>
                  )}
                </div>
              )}

              <div className="mt-4 flex items-center justify-between border-t border-white/[0.05] pt-3 text-xs text-slate-500">
                <span>{e.author ? `Aporte de ${e.author}` : "Aporte de la comunidad"}</span>
                <a
                  href={e.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-accent hover:text-accent-soft"
                >
                  Ver en GitHub <IconExternal width={12} height={12} />
                </a>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-baseline gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {label}
        </span>
        {hint && <span className="text-[11px] text-slate-500">{hint}</span>}
      </span>
      {children}
    </label>
  );
}
