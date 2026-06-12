"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import PageHeader from "@/components/PageHeader";
import {
  ingestPdf,
  listManuals,
  removeManual,
  Manual,
} from "@/lib/userKnowledge";
import { TOPIC_LABELS } from "@/lib/knowledge/types";
import type { Product, Topic } from "@/lib/knowledge/types";
import { IconLayers, IconChat, IconAlert, IconCheck } from "@/components/icons";

const PRODUCTS: { id: Product; label: string }[] = [
  { id: "appserver", label: "Application Server" },
  { id: "omi", label: "OMI" },
  { id: "historian", label: "Historian" },
  { id: "general", label: "Otro / General" },
];

const TOPICS = Object.entries(TOPIC_LABELS) as [Topic, string][];

interface Busy {
  name: string;
  fraction: number;
  note: string;
}

export default function ManualsPage() {
  const [manuals, setManuals] = useState<Manual[]>([]);
  const [product, setProduct] = useState<Product>("appserver");
  const [topic, setTopic] = useState<Topic>("concepts");
  const [busy, setBusy] = useState<Busy | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [justAdded, setJustAdded] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setManuals(listManuals());
  }, []);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setError(null);
    for (const file of Array.from(files)) {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setError(`${file.name}: only PDF files are supported right now.`);
        continue;
      }
      try {
        const { manuals: next, manual } = await ingestPdf(
          file,
          { product, topic },
          (fraction, note) => setBusy({ name: file.name, fraction, note })
        );
        setManuals(next);
        setJustAdded(manual.id);
        setTimeout(() => setJustAdded(null), 2500);
      } catch (e: any) {
        setError(`${file.name}: ${e?.message || "Falló la ingesta."}`);
      } finally {
        setBusy(null);
      }
    }
    if (inputRef.current) inputRef.current.value = "";
  }

  const totalChunks = manuals.reduce((n, m) => n + m.chunks.length, 0);

  return (
    <div>
      <PageHeader
        eyebrow="Conocimiento / Manuales"
        title="Sube manuales de capacitación"
        subtitle="Agrega tus manuales de capacitación de AVEVA Application Server y OMI (PDF). Se procesan en tu navegador y se integran en Preguntar, Troubleshooting y la búsqueda de Docs — con citas de vuelta a tu manual."
      />

      {/* Local-only notice */}
      <div className="panel mb-6 flex items-start gap-3 p-4">
        <span className="mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-signal-info/10 text-signal-info">
          <IconLayers width={17} height={17} />
        </span>
        <div className="text-sm leading-relaxed text-slate-300">
          <p className="font-semibold text-slate-100">Guardado solo en este dispositivo</p>
          <p className="mt-0.5 text-slate-400">
            Las subidas se procesan localmente y se guardan en el almacenamiento de
            este navegador — nada se envía a un servidor y no se comparten con otros
            usuarios. Borrar los datos del sitio los elimina. Para una biblioteca de
            manuales compartida por todo el equipo, mira la sección de Comunidad o la
            vía de ingesta en servidor del README.
          </p>
        </div>
      </div>

      {/* Uploader */}
      <div className="panel mb-6 p-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-slate-400">
              Producto
            </span>
            <select
              value={product}
              onChange={(e) => setProduct(e.target.value as Product)}
              className="input"
            >
              {PRODUCTS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-slate-400">
              Tema principal
            </span>
            <select
              value={topic}
              onChange={(e) => setTopic(e.target.value as Topic)}
              className="input"
            >
              {TOPICS.map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={!!busy}
          className="mt-4 flex w-full flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/15 bg-white/[0.02] px-4 py-10 text-center transition-all hover:border-accent/40 hover:bg-accent/[0.03] disabled:opacity-60"
        >
          <span className="grid h-12 w-12 place-items-center rounded-xl bg-accent/12 text-accent">
            <IconLayers width={22} height={22} />
          </span>
          <span className="text-sm font-semibold text-slate-100">
            {busy ? "Procesando…" : "Haz clic para elegir manual(es) PDF"}
          </span>
          <span className="text-xs text-slate-500">
            PDFs con texto · procesados en el navegador · varios archivos a la vez
          </span>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />

        {busy && (
          <div className="mt-4">
            <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
              <span className="truncate">{busy.name}</span>
              <span>{Math.round(busy.fraction * 100)}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-base-700">
              <div
                className="h-full rounded-full bg-accent transition-all"
                style={{ width: `${Math.max(4, busy.fraction * 100)}%` }}
              />
            </div>
            <p className="mt-1 text-xs text-slate-500">{busy.note}</p>
          </div>
        )}

        {error && (
          <div className="mt-4 flex items-start gap-2 rounded-xl border border-signal-danger/25 bg-signal-danger/[0.06] p-3 text-sm text-slate-200">
            <IconAlert width={15} height={15} className="mt-0.5 shrink-0 text-signal-danger" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Manual list */}
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400">
          Tus manuales ({manuals.length})
        </h2>
        {totalChunks > 0 && (
          <span className="text-xs text-slate-500">{totalChunks} pasajes indexados</span>
        )}
      </div>

      {manuals.length === 0 ? (
        <div className="panel p-8 text-center text-sm text-slate-400">
          Aún no hay manuales. Sube tu PDF de capacitación de Application Server u
          OMI para empezar — luego pregunta sobre él en{" "}
          <Link href="/ask" className="text-accent">Preguntar</Link>.
        </div>
      ) : (
        <div className="space-y-3">
          {manuals.map((m) => (
            <div
              key={m.id}
              className={`panel flex items-start gap-3 p-4 ${
                justAdded === m.id ? "border-signal-ok/40 shadow-glow" : ""
              }`}
            >
              <div className="min-w-0 flex-1">
                <div className="mb-1 flex flex-wrap items-center gap-2">
                  <span className="chip chip-accent">
                    {PRODUCTS.find((p) => p.id === m.product)?.label || m.product}
                  </span>
                  <span className="chip">{TOPIC_LABELS[m.topic]}</span>
                  {justAdded === m.id && (
                    <span className="chip border-signal-ok/30 bg-signal-ok/10 text-signal-ok">
                      <IconCheck width={12} height={12} /> Agregado
                    </span>
                  )}
                </div>
                <p className="truncate text-sm font-semibold text-slate-100">
                  {m.name}
                </p>
                <p className="mt-0.5 text-xs text-slate-500">
                  {m.pages} páginas · {m.chunks.length} pasajes ·{" "}
                  {new Date(m.addedAt).toLocaleDateString()}
                </p>
              </div>
              <button
                onClick={() => setManuals(removeManual(m.id))}
                className="btn btn-ghost shrink-0 px-3 py-1.5 text-xs"
              >
                Quitar
              </button>
            </div>
          ))}

          <Link href="/ask" className="btn btn-primary mt-2 w-full sm:w-auto">
            <IconChat width={16} height={16} /> Preguntar a tus manuales
          </Link>
        </div>
      )}
    </div>
  );
}
