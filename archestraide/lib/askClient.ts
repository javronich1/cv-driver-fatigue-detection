import { composeAnswer, ComposedAnswer } from "./answer";
import { getUserChunks } from "./userKnowledge";

// Ask helper used by client components. It tries the server route (which may
// add live LLM synthesis when ANTHROPIC_API_KEY is configured) and gracefully
// falls back to the fully client-side grounded composer — so the app works even
// on static hosting with no serverless functions.
export async function askQuestion(query: string): Promise<ComposedAnswer> {
  // Uploaded manuals live only in the browser, so if the user has any, compose
  // locally with them blended in (the server route can't see them).
  const userChunks = typeof window !== "undefined" ? getUserChunks() : [];
  if (userChunks.length > 0) {
    return composeAnswer(query, userChunks);
  }

  // Static export (e.g. GitHub Pages) has no /api/ask function — compose locally
  // and skip the wasted 404 round-trip.
  if (process.env.NEXT_PUBLIC_BASE_PATH) {
    return composeAnswer(query);
  }
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (res.ok) {
      const data = await res.json();
      if (data && data.shortAnswer) return data as ComposedAnswer;
    }
  } catch {
    /* fall through to local composition */
  }
  return composeAnswer(query);
}
