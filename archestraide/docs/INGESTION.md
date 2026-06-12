# Knowledge Ingestion Guide

ArchestrAide is **retrieval-grounded**: every answer is assembled from structured
content with explicit source citations. This guide explains how the knowledge
base is organised and how to extend it — including ingesting AVEVA PDFs/manuals.

## How the knowledge base is structured

All content lives in `lib/knowledge/` as typed TypeScript (no database required):

| File | Contents |
| --- | --- |
| `types.ts` | Shared types: `Source`, `Runbook`, `GlossaryTerm`, `KnownIssue`, `Chunk`, `Topic`. |
| `sources.ts` | The **source registry**. Every citation references a `Source` by id. Sources are tagged by `kind` (`official-doc`, `official-pdf`, `community`, `runbook`, `glossary`) so the UI can separate *official facts* from *inferred guidance*. |
| `glossary.ts` | Curated concept pages. |
| `runbooks.ts` | Curated troubleshooting runbooks. |
| `knownIssues.ts` | Environment-specific known issues / gotchas. |
| `index.ts` | Projects everything into a flat `Chunk[]` corpus for retrieval. |

At build time, `index.ts` denormalises each runbook / glossary term / known issue
into a `Chunk` (title + searchable body + source ids). The retrieval layer
(`lib/retrieval.ts`) scores chunks with a hybrid TF-IDF + phrase + intent model.

## Adding content by hand (fastest)

1. **Add a source** in `sources.ts` (so the answer can cite it):

   ```ts
   {
     id: "doc-my-topic",
     title: "My AVEVA topic — Official doc",
     kind: "official-doc",
     url: "https://docs.aveva.com/bundle/.../page/XXupXX.html",
     reference: "Help › My topic",
     topics: ["runtime"],
   }
   ```

2. **Add a runbook / glossary term / known issue** referencing that source id via
   `sourceIds`. Types are enforced by TypeScript, so the shape is self-documenting.

3. Rebuild. The new content is automatically retrievable, searchable, citable,
   and appears in Ask / Troubleshoot / Runbooks / Docs / Glossary.

No re-indexing step is needed — the corpus and IDF table are computed at module
load from the structured content.

## Ingesting AVEVA PDFs / manuals

> The original training manual was provided via Google Drive, which is blocked by
> this environment's network egress policy, so it could not be fetched during the
> initial build. Content was instead grounded in official `docs.aveva.com` pages
> and official product PDFs. Use the steps below to ingest the manual once you can
> provide the file.

### 1. Register the manual as a source

Add (or update) an entry in `sources.ts` with `kind: "official-pdf"`:

```ts
{
  id: "pdf-aveva-training",
  title: "AVEVA Application Server Training Manual",
  kind: "official-pdf",
  reference: "Training Manual § <section>",
  topics: ["concepts", "templates", "deployment", "di", "runtime"],
}
```

(A placeholder for this already exists.)

### 2. Extract text chunks from the PDF

A simple, dependency-light pipeline:

```bash
# Option A: pdftotext (poppler-utils)
pdftotext -layout AVEVA_Training.pdf manual.txt

# Option B: Python (pypdf)
pip install pypdf
python - <<'PY'
from pypdf import PdfReader
import json
r = PdfReader("AVEVA_Training.pdf")
chunks = []
for i, page in enumerate(r.pages):
    text = (page.extract_text() or "").strip()
    if len(text) < 40:
        continue
    chunks.append({
        "id": f"manual-p{i+1}",
        "kind": "doc",
        "title": f"Training Manual — p.{i+1}",
        "topics": ["concepts"],
        "text": text,
        "sourceIds": ["pdf-aveva-training"],
        "ref": None
    })
json.dump(chunks, open("manual_chunks.json","w"), indent=2)
print(len(chunks), "chunks")
PY
```

For better retrieval, split long pages into ~500–800 character passages and tag
each with the closest `Topic` (see `Topic` union in `types.ts`).

### 3. Load the extracted chunks into the corpus

Drop `manual_chunks.json` into `lib/knowledge/` and merge it in `index.ts`:

```ts
import manualChunks from "./manual_chunks.json";
// ...
export const CHUNKS: Chunk[] = [
  ...runbookChunks,
  ...glossaryChunks,
  ...knownIssueChunks,
  ...(manualChunks as Chunk[]),
];
```

The new passages are immediately searchable and will be cited as
**Official manual** in answers.

## Upgrading retrieval to real embeddings (optional)

The retrieval seam is intentionally clean: `lib/retrieval.ts` exposes
`retrieve(query, opts)` and scores `Chunk`s. To move from keyword TF-IDF to
vector search:

1. Precompute an embedding per `Chunk` (e.g. with an embeddings API) at build time.
2. Store vectors alongside chunks.
3. Replace `scoreChunk` with cosine similarity (optionally blended with the
   existing keyword score for hybrid retrieval).

Nothing else in the app needs to change — the UI consumes `RetrievalResult`s.
