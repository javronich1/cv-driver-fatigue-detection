# ArchestrAide

### AVEVA Application Server & OMI Support Copilot

> An internal AVEVA Application Server & OMI support copilot for onboarding,
> theory questions, documentation search, and first-line troubleshooting.

ArchestrAide is a **retrieval-grounded** AI support assistant for engineering and
support teams working with **AVEVA Application Server, OMI & System Platform**. It
is deliberately *not* a generic chatbot: every substantial answer is assembled
from curated, source-cited content and clearly separates **official documentation
facts** from **inferred troubleshooting heuristics**.

Coverage spans **Application Server** (templates, DI/OI, deployment, historian,
alarms, security, runtime) and **OMI** (ViewApps, layouts, screen profiles, OMI
apps / Content Presenter, web client). Users can also **upload their own training
manuals** (PDF), parsed in-browser and blended into answers.

---

## Product summary — what the MVP supports

| Capability | What it does |
| --- | --- |
| **Ask mode** | Chat-style Q&A that returns structured, support-oriented answers (short answer → causes → what to check → steps → tools → sources → confidence). Routes automatically between *concept* and *troubleshooting* answer shapes based on intent. |
| **Troubleshoot mode** | A guided wizard: pick a symptom category, answer two scoping questions (environment, trigger), and get a tailored, checklist-style diagnostic path with environment-aware warnings. |
| **Runbooks** | 10 curated, support-oriented playbooks (symptom → likely causes → first tool → ordered checks → confirm → escalate), filterable by category and deep-linkable. |
| **Docs / Search** | Hybrid keyword + concept search across runbooks, concepts, and known issues, with source-kind badges and snippet previews. |
| **Glossary** | Curated concept pages spanning Application Server **and OMI** (Galaxy, Template, DI Object, Autobind, OnScan/OffScan, Historian, AlarmModeCmd, ViewApp, Layout, Screen Profile, Content Presenter, OMI Web Client, ViewEngine, …) with definition, explanation, example, related terms, and sources. |
| **Manuals (upload)** | Upload your own AVEVA Application Server / OMI training PDFs. They're parsed **in the browser** (pdf.js), chunked, stored in local storage, and blended into Ask / Troubleshoot / Docs — cited as *Your manual*. Per-device (no backend); see ingestion guide for the shared/server-backed path. |
| **Known issues** | Environment-specific gotchas distilled from official readmes and labelled community tech notes (Application Server + OMI). |
| **Source grounding** | Every answer cites its sources, badged *Official doc / Official manual / Your manual / Community / Curated runbook*. Confidence is flagged (high / medium / low). |

The UI is **dark-mode-first** (with a light mode), responsive, and built for long
technical content.

---

## Architecture

```
Next.js 14 (App Router) + TypeScript + Tailwind CSS
│
├── app/                      # Pages: home, ask, troubleshoot, runbooks,
│   │                         #        docs, glossary, known-issues, about
│   └── api/ask/route.ts      # Optional serverless route for live LLM synthesis
│
├── components/               # Sidebar, AnswerView, RunbookDetail, Sources, icons…
│
└── lib/
    ├── knowledge/            # The knowledge base (structured, typed content)
    │   ├── sources.ts        #   source registry + citations
    │   ├── runbooks.ts       #   curated runbooks
    │   ├── glossary.ts       #   curated concepts
    │   ├── knownIssues.ts    #   known issues
    │   └── index.ts          #   → projects everything into a Chunk[] corpus
    ├── retrieval.ts          # Hybrid TF-IDF + phrase + intent retrieval
    ├── answer.ts             # Deterministic grounded answer composer
    └── askClient.ts          # Client helper (tries API, falls back to local)
```

### How answering works

1. **Retrieve** — `lib/retrieval.ts` scores the `Chunk` corpus against the query
   using normalised **TF-IDF**, **exact-phrase boosting** (precision for error
   strings), **title weighting**, and **intent bias** (troubleshooting-signal
   words favour runbooks; definitional queries favour glossary concepts).
2. **Compose** — `lib/answer.ts` turns the top result into a structured answer:
   troubleshooting shape for runbooks/known-issues, concept shape for glossary,
   each with sources, suggested tools, related terms, follow-ups, and a
   **confidence flag** (high only when a strong hit is backed by official sources).
3. **(Optional) Synthesize** — if `ANTHROPIC_API_KEY` is set, `/api/ask` passes
   the retrieved context to Claude under strict grounding instructions to produce
   a polished short answer. **Structure, sources, and confidence still come from
   the grounded layer**, so citations remain trustworthy. With no key, the app
   runs entirely on the deterministic composer — and the client falls back to it
   automatically if the API route is unavailable (e.g. static hosting).

This “retrieval first, LLM optional” design means the app is **useful and
polished even with zero API keys**, and upgrades cleanly to live AI.

---

## Running locally

```bash
cd archestraide
npm install
npm run dev      # http://localhost:3000
```

Production build:

```bash
npm run build && npm run start
```

### Environment variables (all optional)

Copy `.env.example` → `.env.local`:

| Variable | Purpose | Default |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Enables live LLM synthesis of the short answer. | *(unset → grounded composer only)* |
| `ARCHESTRAIDE_MODEL` | Override the Claude model used for synthesis. | `claude-sonnet-4-6` |

---

## Deployment (Netlify)

This repo includes `netlify.toml`. Netlify auto-detects Next.js and installs
`@netlify/plugin-nextjs`, which makes the optional `/api/ask` function work.

**One-click import:**

1. Go to **Netlify → Add new site → Import an existing project**.
2. Connect GitHub and pick this repository.
3. Build command `npm run build`, publish dir `.next` (auto-detected).
4. *(Optional)* Site settings → Environment → add `ANTHROPIC_API_KEY` to enable
   live AI synthesis.
5. Deploy. You get a live URL.

> The app also deploys cleanly to **Vercel** (zero config — it's a standard
> Next.js app). The client-side fallback keeps Ask working even if serverless
> functions are disabled.

---

## Extending the knowledge base

See **[`docs/INGESTION.md`](docs/INGESTION.md)** for the full guide, including a
PDF → chunks pipeline. In short:

- Add a `Source` in `lib/knowledge/sources.ts`.
- Add a runbook / glossary term / known issue referencing its `sourceIds`.
- Rebuild — content becomes retrievable, searchable, and citable automatically.

To ingest AVEVA manuals, extract text with `pdftotext`/`pypdf`, emit `Chunk`
JSON, and merge it in `lib/knowledge/index.ts`.

---

## Assumptions & limitations

- **Uploaded PDF could not be fetched at build time.** The training manual was
  shared via Google Drive, which is **blocked by this environment's network
  egress policy**. Content was therefore grounded in official `docs.aveva.com`
  pages and official AVEVA/Wonderware product PDFs. The manual can be ingested at
  any time via `docs/INGESTION.md` (a source placeholder already exists).
- **Retrieval is keyword/TF-IDF hybrid, not vector embeddings.** This was a
  deliberate MVP choice for zero-infra, deploy-anywhere operation. The retrieval
  seam is clean — swapping in embeddings only changes `scoreChunk` (see ingestion
  guide).
- **Curated content is a high-value starter set**, not exhaustive AVEVA coverage.
  Runbooks are troubleshooting *heuristics* ("most likely / check first"), framed
  as such — not guarantees of root cause for a specific environment.
- **No live environment introspection.** ArchestrAide never claims to have
  validated live runtime state unless you provide it.
- **Not a substitute** for official AVEVA support or validated production
  change-management.

---

## Tradeoffs

- **Structured TS content over a CMS/DB** → fully type-checked, versioned in git,
  zero runtime infra, trivial to review and extend.
- **Deterministic composer as the default** → trustworthy, instant, free, and
  works offline/statically; LLM is an enhancement, not a dependency.
- **Single accent / dark-first design** → serious engineering-tool aesthetic,
  optimised for long technical reading.

---

## Tech stack

Next.js 14 · React 18 · TypeScript · Tailwind CSS · (optional) Anthropic Claude API.
No UI-component or icon-library dependencies — a small bespoke component system
keeps the bundle lean.
