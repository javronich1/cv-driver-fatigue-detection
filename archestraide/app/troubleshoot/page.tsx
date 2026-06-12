"use client";

import { useState } from "react";
import PageHeader from "@/components/PageHeader";
import RunbookDetail, { SeverityChip } from "@/components/RunbookDetail";
import { RUNBOOK_BY_ID } from "@/lib/knowledge/runbooks";
import {
  IconWrench,
  IconArrow,
  IconChat,
} from "@/components/icons";
import Link from "next/link";

// Guided categories → runbook mapping.
const CATEGORIES: {
  id: string;
  label: string;
  runbookId: string;
  blurb: string;
}[] = [
  { id: "deploy", label: "Deployment failures", runbookId: "rb-deploy-remote-node", blurb: "Cannot communicate with remote node, deploy timeouts." },
  { id: "quality", label: "Bad quality / no data", runbookId: "rb-bad-quality", blurb: "Attribute shows Bad / Uncertain quality." },
  { id: "ov", label: "No data in Object Viewer", runbookId: "rb-no-data-object-viewer", blurb: "Deployed but values don't update." },
  { id: "oi", label: "OI / OPC / DI communication", runbookId: "rb-oi-opc-not-updating", blurb: "OI Server / OPC client not updating." },
  { id: "historian", label: "Historian / historization missing", runbookId: "rb-historian-no-data", blurb: "Historized data not appearing in trends." },
  { id: "alarms", label: "Alarms not appearing", runbookId: "rb-alarm-not-visible", blurb: "Alarm configured but never goes active." },
  { id: "security", label: "Security / login / auth", runbookId: "rb-security-login", blurb: "Access denied, login or permission problems." },
  { id: "checkin", label: "Check-in / out / version", runbookId: "rb-checkin-version-mismatch", blurb: "Config vs runtime mismatch, locked objects." },
  { id: "scan", label: "Platform / AppEngine / OnScan", runbookId: "rb-onscan-offscan", blurb: "OnScan vs OffScan, nothing executing." },
  { id: "csv", label: "Import / export / CSV", runbookId: "rb-csv-import-conflict", blurb: "CSV / package import conflicts." },
];

type EnvAnswer = "lab" | "prod" | "unsure";
type TriggerAnswer = "reboot" | "change" | "always" | "unsure";

const ENV_Q = [
  { id: "lab", label: "Single-node lab / test" },
  { id: "prod", label: "Multi-node production" },
  { id: "unsure", label: "Not sure" },
];
const TRIGGER_Q = [
  { id: "reboot", label: "After a reboot / restart" },
  { id: "change", label: "After a config change / deploy" },
  { id: "always", label: "It never worked" },
  { id: "unsure", label: "Not sure / intermittent" },
];

function contextNote(
  env: EnvAnswer | null,
  trig: TriggerAnswer | null
): string | undefined {
  const notes: string[] = [];
  if (env === "prod")
    notes.push(
      "Multi-node production: verify name resolution, firewall and time sync between nodes, and avoid disruptive live changes during running operations — make changes in a controlled window."
    );
  if (env === "lab")
    notes.push(
      "Single-node lab: most communication issues here reduce to services, scan state, or references on the one node rather than network/AD."
    );
  if (trig === "reboot")
    notes.push(
      "Started after a reboot: a very common cause is engines left OffScan — set each AppEngine OnScan first, then re-check."
    );
  if (trig === "change")
    notes.push(
      "Started after a change/deploy: confirm the object is checked in and the deployed version matches the configuration before deeper diagnosis."
    );
  return notes.length ? notes.join(" ") : undefined;
}

export default function TroubleshootPage() {
  const [step, setStep] = useState(0);
  const [catId, setCatId] = useState<string | null>(null);
  const [env, setEnv] = useState<EnvAnswer | null>(null);
  const [trig, setTrig] = useState<TriggerAnswer | null>(null);

  const cat = CATEGORIES.find((c) => c.id === catId);
  const rb = cat ? RUNBOOK_BY_ID[cat.runbookId] : null;

  function reset() {
    setStep(0);
    setCatId(null);
    setEnv(null);
    setTrig(null);
  }

  return (
    <div>
      <PageHeader
        eyebrow="Troubleshoot mode"
        title="Guided troubleshooting"
        subtitle="Pick a symptom category and answer two quick scoping questions. ArchestrAide assembles a checklist-style diagnostic path tailored to your environment, with sources and escalation criteria."
      />

      <Stepper step={step} />

      {/* Step 0: category */}
      {step === 0 && (
        <div className="animate-fade-up grid gap-3 sm:grid-cols-2">
          {CATEGORIES.map((c) => (
            <button
              key={c.id}
              onClick={() => {
                setCatId(c.id);
                setStep(1);
              }}
              className="panel card-hover group flex items-start gap-3 p-4 text-left"
            >
              <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-accent/12 text-accent">
                <IconWrench width={18} height={18} />
              </span>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold text-slate-100">{c.label}</h3>
                  <IconArrow
                    width={14}
                    height={14}
                    className="text-slate-600 transition-all group-hover:translate-x-0.5 group-hover:text-accent"
                  />
                </div>
                <p className="mt-0.5 text-sm text-slate-400">{c.blurb}</p>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Step 1: environment */}
      {step === 1 && (
        <Question
          title="What kind of environment is this?"
          options={ENV_Q}
          onPick={(id) => {
            setEnv(id as EnvAnswer);
            setStep(2);
          }}
          onBack={() => setStep(0)}
        />
      )}

      {/* Step 2: trigger */}
      {step === 2 && (
        <Question
          title="When did the problem start?"
          options={TRIGGER_Q}
          onPick={(id) => {
            setTrig(id as TriggerAnswer);
            setStep(3);
          }}
          onBack={() => setStep(1)}
        />
      )}

      {/* Step 3: result */}
      {step === 3 && rb && (
        <div className="animate-fade-up space-y-5">
          <div className="panel p-5 sm:p-6">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className="chip chip-accent">{rb.category}</span>
              <SeverityChip severity={rb.severity} />
            </div>
            <h2 className="text-xl font-bold tracking-tight text-slate-100">
              {rb.title}
            </h2>
            <p className="mt-1 text-sm text-slate-400">{rb.symptom}</p>
          </div>

          <div className="panel p-5 sm:p-6">
            <RunbookDetail rb={rb} note={contextNote(env, trig)} />
          </div>

          <div className="flex flex-wrap gap-3">
            <button onClick={reset} className="btn btn-ghost">
              Start over
            </button>
            <Link
              href={`/ask?q=${encodeURIComponent(rb.title)}`}
              className="btn btn-primary"
            >
              <IconChat width={16} height={16} /> Ask a follow-up
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}

function Stepper({ step }: { step: number }) {
  const labels = ["Symptom", "Environment", "Trigger", "Diagnosis"];
  return (
    <div className="mb-6 flex items-center gap-2">
      {labels.map((l, i) => (
        <div key={l} className="flex items-center gap-2">
          <span
            className={`flex h-6 items-center gap-1.5 rounded-full px-2.5 text-xs font-medium ${
              i <= step
                ? "bg-accent/15 text-accent-soft"
                : "bg-white/[0.03] text-slate-500"
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                i <= step ? "bg-accent" : "bg-slate-600"
              }`}
            />
            {l}
          </span>
          {i < labels.length - 1 && (
            <span className="h-px w-4 bg-white/[0.08]" />
          )}
        </div>
      ))}
    </div>
  );
}

function Question({
  title,
  options,
  onPick,
  onBack,
}: {
  title: string;
  options: { id: string; label: string }[];
  onPick: (id: string) => void;
  onBack: () => void;
}) {
  return (
    <div className="animate-fade-up">
      <div className="panel p-5 sm:p-6">
        <h2 className="mb-4 text-lg font-bold tracking-tight text-slate-100">
          {title}
        </h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {options.map((o) => (
            <button
              key={o.id}
              onClick={() => onPick(o.id)}
              className="panel card-hover flex items-center justify-between gap-2 p-4 text-left text-sm font-medium text-slate-200"
            >
              {o.label}
              <IconArrow width={15} height={15} className="text-slate-500" />
            </button>
          ))}
        </div>
      </div>
      <button onClick={onBack} className="btn btn-ghost mt-4">
        ← Back
      </button>
    </div>
  );
}
