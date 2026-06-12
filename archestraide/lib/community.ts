// Community-submitted troubleshooting problems, stored as GitHub Issues in the
// repo (chosen storage: zero secrets, persisted in the repo, visible to all).
//
// - SUBMIT: we build a pre-filled "new issue" URL. The user (who must have
//   GitHub access to the repo) clicks "Submit new issue" on GitHub. The body
//   embeds a machine-readable JSON block + a marker so we can parse it back.
// - READ: we fetch open issues from the PUBLIC GitHub API (no auth, 60 req/hr/IP)
//   and render those carrying our marker as troubleshooting guides for everyone.

export const COMMUNITY_REPO = "javronich1/cv-driver-fatigue-detection";
const MARKER = "<!-- archestraide-community-v1 -->";
const LABEL = "community-runbook";

export interface CommunitySubmission {
  title: string;
  category: string;
  symptom: string;
  likelyCauses: string[];
  steps: string[];
  firstTool?: string;
  confirmResolution?: string;
  escalateWhen?: string;
  author?: string;
}

export interface CommunityEntry extends CommunitySubmission {
  number: number;
  url: string;
  createdAt: string;
  reactions: number;
}

function lines(s: string): string[] {
  return s
    .split("\n")
    .map((l) => l.replace(/^[-*\d.\s]+/, "").trim())
    .filter(Boolean);
}

export function normalizeSubmission(raw: {
  title: string;
  category: string;
  symptom: string;
  likelyCauses: string;
  steps: string;
  firstTool?: string;
  confirmResolution?: string;
  escalateWhen?: string;
  author?: string;
}): CommunitySubmission {
  return {
    title: raw.title.trim(),
    category: raw.category.trim() || "General",
    symptom: raw.symptom.trim(),
    likelyCauses: lines(raw.likelyCauses),
    steps: lines(raw.steps),
    firstTool: raw.firstTool?.trim() || undefined,
    confirmResolution: raw.confirmResolution?.trim() || undefined,
    escalateWhen: raw.escalateWhen?.trim() || undefined,
    author: raw.author?.trim() || undefined,
  };
}

// Human-readable + machine-readable issue body.
function buildIssueBody(s: CommunitySubmission): string {
  const md = [
    `**Síntoma:** ${s.symptom}`,
    "",
    "**Causas más probables:**",
    ...s.likelyCauses.map((c) => `- ${c}`),
    "",
    "**Pasos de troubleshooting:**",
    ...s.steps.map((c, i) => `${i + 1}. ${c}`),
    s.firstTool ? `\n**Primera herramienta:** ${s.firstTool}` : "",
    s.confirmResolution ? `\n**Confirmar resolución:** ${s.confirmResolution}` : "",
    s.escalateWhen ? `\n**Cuándo escalar:** ${s.escalateWhen}` : "",
    s.author ? `\n_— ${s.author}_` : "",
    "",
    "---",
    "_Enviado vía ArchestrAide · Comunidad. No edites el bloque siguiente._",
    MARKER,
    "```json",
    JSON.stringify(s),
    "```",
  ].join("\n");
  return md;
}

export function buildIssueUrl(s: CommunitySubmission): string {
  const params = new URLSearchParams({
    title: `[Runbook comunidad] ${s.title}`,
    body: buildIssueBody(s),
    labels: LABEL,
  });
  return `https://github.com/${COMMUNITY_REPO}/issues/new?${params.toString()}`;
}

function parseEntry(issue: any): CommunityEntry | null {
  const body: string = issue?.body || "";
  if (!body.includes(MARKER)) return null;
  const m = body.match(/```json\s*([\s\S]*?)```/);
  if (!m) return null;
  let sub: CommunitySubmission;
  try {
    sub = JSON.parse(m[1].trim());
  } catch {
    return null;
  }
  if (!sub.title || !sub.symptom) return null;
  return {
    ...sub,
    likelyCauses: sub.likelyCauses || [],
    steps: sub.steps || [],
    number: issue.number,
    url: issue.html_url,
    createdAt: issue.created_at,
    reactions: issue.reactions?.total_count || 0,
  };
}

export async function fetchCommunity(): Promise<CommunityEntry[]> {
  const res = await fetch(
    `https://api.github.com/repos/${COMMUNITY_REPO}/issues?state=open&per_page=100&sort=created&direction=desc`,
    { headers: { Accept: "application/vnd.github+json" } }
  );
  if (!res.ok) throw new Error(`GitHub API ${res.status}`);
  const issues = await res.json();
  if (!Array.isArray(issues)) return [];
  return issues
    .filter((i) => !i.pull_request) // exclude PRs
    .map(parseEntry)
    .filter((e): e is CommunityEntry => e !== null);
}
