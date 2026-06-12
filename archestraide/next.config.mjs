/** @type {import('next').NextConfig} */

// The app runs fully client-side (retrieval + grounded answer composition), so
// it can deploy three ways:
//   1. Serverless Next app (Vercel / Netlify) — default; enables the optional
//      /api/ask route for live LLM synthesis when ANTHROPIC_API_KEY is set.
//   2. Static export to GitHub Pages — set STATIC_EXPORT=true. The /api/ask
//      route is removed by the Pages workflow; the client falls back to the
//      deterministic grounded composer, so Ask still works.
//
// BASE_PATH lets the Pages build serve correctly from /<repo-name>/.
const isStatic = process.env.STATIC_EXPORT === "true";
const basePath = process.env.BASE_PATH || "";

const nextConfig = {
  reactStrictMode: true,
  ...(isStatic
    ? {
        output: "export",
        images: { unoptimized: true },
        trailingSlash: true,
        basePath: basePath || undefined,
        env: { NEXT_PUBLIC_BASE_PATH: basePath },
      }
    : {}),
};

export default nextConfig;
