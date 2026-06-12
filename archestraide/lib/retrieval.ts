import { CHUNKS } from "./knowledge";
import { Chunk, Topic } from "./knowledge/types";

// Lightweight hybrid retrieval that runs entirely in the browser (and on the
// server in the API route). It combines:
//   1. TF-IDF style term weighting (semantic-ish recall over the corpus)
//   2. Exact phrase / substring boosting (precision for error strings)
//   3. Topic / kind metadata filtering
//
// This deliberately avoids a heavyweight embedding model so the MVP deploys
// anywhere with zero infra. The architecture (Chunk corpus + scorer) is a clean
// seam: swapping in real vector embeddings later only changes `scoreChunk`.

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

// --- Build the IDF table once over the corpus ---
const docTokens: Map<string, Map<string, number>> = new Map();
const df: Map<string, number> = new Map();

for (const chunk of CHUNKS) {
  const counts = new Map<string, number>();
  const tokens = tokenize(chunk.text);
  for (const t of tokens) counts.set(t, (counts.get(t) || 0) + 1);
  docTokens.set(chunk.id, counts);
  for (const t of new Set(tokens)) df.set(t, (df.get(t) || 0) + 1);
}

const N = CHUNKS.length;
function idf(term: string): number {
  const d = df.get(term) || 0;
  return Math.log((N + 1) / (d + 1)) + 1;
}

export interface RetrievalResult {
  chunk: Chunk;
  score: number;
  // Which query terms matched — used to render "why this matched".
  matchedTerms: string[];
}

export interface RetrieveOptions {
  topics?: Topic[];
  kinds?: Chunk["kind"][];
  limit?: number;
}

function scoreChunk(
  chunk: Chunk,
  queryTokens: string[],
  rawQuery: string
): { score: number; matched: string[] } {
  const counts = docTokens.get(chunk.id)!;
  const docLen = Array.from(counts.values()).reduce((a, b) => a + b, 0) || 1;
  let score = 0;
  const matched: string[] = [];

  for (const qt of new Set(queryTokens)) {
    const tf = counts.get(qt) || 0;
    if (tf > 0) {
      // Normalised TF * IDF
      score += (tf / docLen) * idf(qt) * 10;
      matched.push(qt);
    } else {
      // Partial / prefix match (handles plurals, "alarm" vs "alarms")
      for (const [term, c] of counts) {
        if (term.length > 3 && (term.startsWith(qt) || qt.startsWith(term))) {
          score += (c / docLen) * idf(term) * 3;
          matched.push(qt);
          break;
        }
      }
    }
  }

  // Exact phrase / substring boost (precision for multi-word error strings).
  const haystack = chunk.text.toLowerCase();
  const q = rawQuery.toLowerCase().trim();
  if (q.length > 6 && haystack.includes(q)) score += 6;

  // Title hits are strong signals.
  const titleTokens = new Set(tokenize(chunk.title));
  for (const qt of new Set(queryTokens)) {
    if (titleTokens.has(qt)) score += 1.5;
  }

  return { score, matched: Array.from(new Set(matched)) };
}

// Words that signal the user wants to fix something (vs. learn a definition).
// When present, we bias toward runbooks/known-issues over glossary concepts.
const TROUBLE_SIGNALS =
  /\b(why|won'?t|wont|cannot|can'?t|fail|failed|failing|error|errors|not|no|missing|stuck|broken|issue|problem|troubleshoot|fix|wrong|bad|won|appear|appearing|update|updating|communicate|deploy|deployment)\b/i;

export function retrieve(
  query: string,
  opts: RetrieveOptions = {}
): RetrievalResult[] {
  const { topics, kinds, limit = 6 } = opts;
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return [];
  const troubleshooting = TROUBLE_SIGNALS.test(query);

  let pool = CHUNKS;
  if (kinds && kinds.length) pool = pool.filter((c) => kinds.includes(c.kind));
  if (topics && topics.length)
    pool = pool.filter((c) => c.topics.some((t) => topics.includes(t)));

  const results: RetrievalResult[] = [];
  for (const chunk of pool) {
    const { score, matched } = scoreChunk(chunk, queryTokens, query);
    let final = score;
    // Intent bias: troubleshooting queries favour runbooks/known issues;
    // definitional queries lightly favour glossary concepts.
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
}

// Free-text search used by the Docs/Search page (returns more, with kind info).
export function search(
  query: string,
  opts: RetrieveOptions = {}
): RetrievalResult[] {
  return retrieve(query, { ...opts, limit: opts.limit ?? 20 });
}
