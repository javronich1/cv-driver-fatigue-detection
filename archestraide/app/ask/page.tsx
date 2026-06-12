"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { askQuestion } from "@/lib/askClient";
import { ComposedAnswer } from "@/lib/answer";
import AnswerView from "@/components/AnswerView";
import PageHeader from "@/components/PageHeader";
import { IconSpark, IconArrow } from "@/components/icons";

const EXAMPLES = [
  "¿Qué es un DI Object?",
  "Diferencia entre Model View y Deployment View",
  "¿Qué es Autobind?",
  "¿Por qué mi objeto está en Bad quality?",
  "¿Qué hace AlarmModeCmd?",
  "¿Por qué no aparecen mis datos historizados?",
  "El despliegue dice cannot communicate with remote node",
];

interface Turn {
  query: string;
  answer: ComposedAnswer | null;
}

function AskInner() {
  const params = useSearchParams();
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const didInit = useRef(false);

  async function ask(query: string) {
    const q = query.trim();
    if (!q || loading) return;
    setInput("");
    setLoading(true);
    setTurns((t) => [...t, { query: q, answer: null }]);
    const answer = await askQuestion(q);
    setTurns((t) =>
      t.map((turn, i) => (i === t.length - 1 ? { ...turn, answer } : turn))
    );
    setLoading(false);
  }

  // Seed from ?q= once.
  useEffect(() => {
    if (didInit.current) return;
    didInit.current = true;
    const q = params.get("q");
    if (q) ask(q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading]);

  return (
    <div>
      <PageHeader
        eyebrow="Modo Preguntar"
        title="Haz una pregunta técnica"
        subtitle="Respuestas estructuradas y fundamentadas sobre AVEVA Application Server / OMI / System Platform — explicaciones de conceptos y troubleshooting de primera línea."
      />

      {turns.length === 0 && (
        <div className="panel mb-6 p-5 sm:p-6">
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Prueba con una de estas
          </p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => ask(ex)}
                className="chip card-hover px-3 py-1.5 text-left"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-8 pb-4">
        {turns.map((turn, i) => (
          <div key={i} className="space-y-4">
            <div className="flex items-start gap-3">
              <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-base-700 text-xs font-bold text-slate-300">
                Tú
              </span>
              <p className="pt-1 text-[15px] font-semibold text-slate-100">
                {turn.query}
              </p>
            </div>
            {turn.answer ? (
              <AnswerView answer={turn.answer} onFollowUp={ask} />
            ) : (
              <LoadingAnswer />
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Composer */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
        className="sticky bottom-4 z-10 mt-4"
      >
        <div className="panel flex items-center gap-2 p-2 shadow-glow">
          <span className="pl-2 text-accent">
            <IconSpark width={18} height={18} />
          </span>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Pregunta sobre objects, deployment, OI/OPC, historian, alarmas, seguridad…"
            className="flex-1 bg-transparent px-1 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-500"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="btn btn-primary px-3 py-2"
            aria-label="Ask"
          >
            {loading ? "…" : <IconArrow width={16} height={16} />}
          </button>
        </div>
      </form>
    </div>
  );
}

function LoadingAnswer() {
  return (
    <div className="panel animate-pulse-soft p-6">
      <div className="mb-3 h-3 w-24 rounded bg-base-700" />
      <div className="space-y-2">
        <div className="h-3 w-full rounded bg-base-700" />
        <div className="h-3 w-5/6 rounded bg-base-700" />
        <div className="h-3 w-3/4 rounded bg-base-700" />
      </div>
    </div>
  );
}

export default function AskPage() {
  return (
    <Suspense fallback={null}>
      <AskInner />
    </Suspense>
  );
}
