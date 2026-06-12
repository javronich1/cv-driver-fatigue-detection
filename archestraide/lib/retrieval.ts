import { CHUNKS } from "./knowledge";
import { Chunk, Topic } from "./knowledge/types";

// Lightweight hybrid retrieval that runs entirely in the browser (and on the
// server in the API route). It combines:
//   1. TF-IDF style term weighting (semantic-ish recall over the corpus)
//   2. Exact phrase / substring boosting (precision for error strings)
//   3. Topic / kind metadata filtering
//   4. Intent bias (troubleshooting vs definitional queries)
//
// The corpus is pluggable via buildIndex(), so user-uploaded manuals can be
// blended into the base knowledge base at runtime (see retrieveWith).

const STOP = new Set([
  "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on",
  "for", "and", "or", "but", "with", "as", "at", "by", "it", "this", "that",
  "my", "i", "you", "we", "they", "what", "why", "how", "when", "do", "does",
  "did", "can", "should", "would", "will", "not", "no", "if", "from", "into",
  "about", "me", "your", "our", "so", "up", "out", "get", "got", "im",
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9./_\- ]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP.has(t));
}

export interface RetrievalResult {
  chunk: Chunk;
  score: number;
  matchedTerms: string[];
}

export interface RetrieveOptions {
  topics?: Topic[];
  kinds?: Chunk["kind"][];
  limit?: number;
}

const TROUBLE_SIGNALS =
  /\b(why|won'?t|wont|cannot|can'?t|fail|failed|failing|error|errors|not|no|missing|stuck|broken|issue|problem|troubleshoot|fix|wrong|bad|won|appear|appearing|update|updating|communicate|deploy|deployment)\b/i;

export interface SearchIndex {
  retrieve(query: string, opts?: RetrieveOptions): RetrievalResult[];
  size: number;
}

// Build an immutable retrieval index over a set of chunks. The IDF table is
// computed once per index.
export function buildIndex(chunks: Chunk[]): SearchIndex {
  const docTokens = new Map<string, Map<string, number>>();
  const df = new Map<string, number>();

  for (const chunk of chunks) {
    const counts = new Map<string, number>();
    const tokens = tokenize(chunk.text);
    for (const t of tokens) counts.set(t, (counts.get(t) || 0) + 1);
    docTokens.set(chunk.id, counts);
    for (const t of new Set(tokens)) df.set(t, (df.get(t) || 0) + 1);
  }

  const N = chunks.length || 1;
  const idf = (term: string) =>
    Math.log((N + 1) / ((df.get(term) || 0) + 1)) + 1;

  function scoreChunk(chunk: Chunk, queryTokens: string[], rawQuery: string) {
    const counts = docTokens.get(chunk.id)!;
    const docLen = Array.from(counts.values()).reduce((a, b) => a + b, 0) || 1;
    let score = 0;
    const matched: string[] = [];

    for (const qt of new Set(queryTokens)) {
      const tf = counts.get(qt) || 0;
      if (tf > 0) {
        score += (tf / docLen) * idf(qt) * 10;
        matched.push(qt);
      } else {
        for (const [term, c] of counts) {
          if (term.length > 3 && (term.startsWith(qt) || qt.startsWith(term))) {
            score += (c / docLen) * idf(term) * 3;
            matched.push(qt);
            break;
          }
        }
      }
    }

    const haystack = chunk.text.toLowerCase();
    const q = rawQuery.toLowerCase().trim();
    if (q.length > 6 && haystack.includes(q)) score += 6;

    const titleTokens = new Set(tokenize(chunk.title));
    for (const qt of new Set(queryTokens)) {
      if (titleTokens.has(qt)) score += 1.5;
    }

    return { score, matched: Array.from(new Set(matched)) };
  }

  return {
    size: chunks.length,
    retrieve(query: string, opts: RetrieveOptions = {}): RetrievalResult[] {
      const { topics, kinds, limit = 6 } = opts;
      const queryTokens = tokenize(query);
      if (queryTokens.length === 0) return [];
      const troubleshooting = TROUBLE_SIGNALS.test(query);

      let pool = chunks;
      if (kinds && kinds.length) pool = pool.filter((c) => kinds.includes(c.kind));
      if (topics && topics.length)
        pool = pool.filter((c) => c.topics.some((t) => topics.includes(t)));

      const results: RetrievalResult[] = [];
      for (const chunk of pool) {
        const { score, matched } = scoreChunk(chunk, queryTokens, query);
        let final = score;
        if (troubleshooting) {
          if (chunk.kind === "runbook" || chunk.kind === "known-issue") final *= 1.8;
          else if (chunk.kind === "glossary") final *= 0.8;
        } else if (chunk.kind === "glossary") {
          final *= 1.1;
        }
        if (final > 0) results.push({ chunk, score: final, matchedTerms: matched });
      }

      results.sort((a, b) => b.score - a.score);
      return results.slice(0, limit);
    },
  };
}

// Base index over the curated knowledge base.
const baseIndex = buildIndex(CHUNKS);

export function retrieve(query: string, opts: RetrieveOptions = {}): RetrievalResult[] {
  return baseIndex.retrieve(query, opts);
}

export function search(query: string, opts: RetrieveOptions = {}): RetrievalResult[] {
  return retrieve(query, { ...opts, limit: opts.limit ?? 20 });
}

// Retrieve over the base corpus plus extra (e.g. user-uploaded) chunks. The
// augmented index is memoised by the extra-chunk identity so repeated calls
// (e.g. while typing in search) don't rebuild it every keystroke.
let memoKey = "";
let memoIndex: SearchIndex | null = null;

export function retrieveWith(
  extra: Chunk[],
  query: string,
  opts: RetrieveOptions = {}
): RetrievalResult[] {
  if (!extra || extra.length === 0) return retrieve(query, opts);
  const key = `${extra.length}:${extra.map((c) => c.id).join(",")}`;
  if (key !== memoKey || !memoIndex) {
    memoIndex = buildIndex(CHUNKS.concat(extra));
    memoKey = key;
  }
  return memoIndex.retrieve(query, opts);
}
