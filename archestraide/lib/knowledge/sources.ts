import { Source } from "./types";

// Source registry. Official AVEVA documentation is prioritised. Community / vendor
// tech notes are included but clearly labelled as non-official guidance.
//
// NOTE ON UPLOADED MANUALS: When AVEVA training PDFs are ingested (see
// scripts/ingest.md), add them here with kind "official-pdf" and a `reference`
// pointing at the section so citations resolve to the manual.

export const SOURCES: Source[] = [
  // ---- Official AVEVA documentation (docs.aveva.com) ----
  {
    id: "doc-deploy-errors",
    title: "Deployment error messages — AVEVA Application Server",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-appserver/page/248679.html",
    reference: "Application Server Help › Deployment error messages",
    topics: ["deployment", "troubleshooting"],
  },
  {
    id: "doc-sp-deployment",
    title: "AVEVA System Platform Deployment Guide",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-deployment/page/1283702.html",
    reference: "System Platform Deployment Guide",
    topics: ["deployment", "security"],
  },
  {
    id: "doc-as-resolved",
    title: "Application Server resolved issues — SP 2023 R2",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-appserver/page/1518384.html",
    reference: "Readme › Resolved issues",
    topics: ["troubleshooting", "runtime"],
  },
  {
    id: "doc-offscan",
    title: "Set an object OffScan — AVEVA Application Server",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-appserver/page/241280.html",
    reference: "Application Server Help › Set an object OffScan",
    topics: ["runtime", "object-management"],
  },
  {
    id: "doc-historian-issues",
    title: "Historian issues — System Platform Installation",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-install/page/495617.html",
    reference: "Install Help › Historian issues",
    topics: ["historian", "troubleshooting"],
  },
  {
    id: "doc-idas-troubleshoot",
    title: "Troubleshoot IDAS connections — AVEVA Historian",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-historian/page/1059730.html",
    reference: "Historian Help › Troubleshoot IDAS connections",
    topics: ["historian", "troubleshooting"],
  },
  {
    id: "doc-idas-sf",
    title: "IDAS store-and-forward capability — AVEVA Historian",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-historian/page/67208.html",
    reference: "Historian Help › IDAS store-and-forward",
    topics: ["historian"],
  },
  {
    id: "doc-alarms-sysobjects",
    title: "Configure alarms for system objects — Application Server",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-appserver/page/248749.html",
    reference: "Application Server Help › Configure alarms for system objects",
    topics: ["alarms"],
  },
  {
    id: "doc-alarms-impl",
    title: "Alarms and events implementation — Application Server",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-appserver/page/248763.html",
    reference: "Application Server Help › Alarms and events",
    topics: ["alarms"],
  },
  {
    id: "doc-alarm-inhibit",
    title: "AlarmInhibit custom property — Situational Awareness Library",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-salibrary/page/76912.html",
    reference: "SA Library › AlarmInhibit",
    topics: ["alarms"],
  },
  {
    id: "doc-opcua-source",
    title: "Configure an OPC UA data source object — Application Server",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-appserver/page/689813.html",
    reference: "Application Server Help › OPC UA data source",
    topics: ["oi", "di"],
  },
  {
    id: "doc-opc-source",
    title: "Configure an OPC data source object — Communication Drivers",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-cdp-drivers/page/GATEWAY4OI4OPC.html",
    reference: "Communication Drivers › OPC data source",
    topics: ["oi", "di"],
  },
  {
    id: "doc-opcua-service",
    title: "Configure and deploy the OPC UA service — OMI",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-omi/page/678498.html",
    reference: "OMI Help › OPC UA service",
    topics: ["oi", "di"],
  },

  // ---- Official AVEVA OMI documentation (docs.aveva.com) ----
  {
    id: "doc-omi-about",
    title: "About Operations Management Interface (OMI)",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-omi/page/257023.html",
    reference: "OMI Help › About OMI",
    topics: ["omi", "concepts"],
  },
  {
    id: "doc-omi-deploy-viewapp",
    title: "Deploy a ViewApp — AVEVA OMI",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-omi/page/270991.html",
    reference: "OMI Help › Deploy a ViewApp",
    topics: ["omi", "deployment"],
  },
  {
    id: "doc-omi-nav",
    title: "About ViewApp navigation hierarchical display — AVEVA OMI",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-omi-awc/page/338129.html",
    reference: "OMI Help › ViewApp navigation hierarchy",
    topics: ["omi"],
  },
  {
    id: "doc-omi-nav-controls",
    title: "Display of controls in the ViewApp navigation hierarchy — AVEVA OMI",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-omi-awc/page/338195.html",
    reference: "OMI Help › Navigation controls display",
    topics: ["omi"],
  },
  {
    id: "doc-omi-webclient-limits",
    title: "General OMI web client limitations — AVEVA OMI",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-omi/page/1220627.html",
    reference: "OMI Help › Web client limitations",
    topics: ["omi", "troubleshooting"],
  },
  {
    id: "doc-omi-webclient-troubleshoot",
    title: "Troubleshoot OMI web client connection issues — AVEVA OMI",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-omi/page/1312461.html",
    reference: "OMI Help › Troubleshoot web client connection",
    topics: ["omi", "troubleshooting", "security"],
  },
  {
    id: "doc-omi-issues",
    title: "AVEVA OMI issues — System Platform Installation",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-install/page/298263.html",
    reference: "Install Help › OMI issues",
    topics: ["omi", "troubleshooting"],
  },
  {
    id: "doc-omi-resolved",
    title: "Resolved issues — AVEVA OMI 2023 R2 Patch 01",
    kind: "official-doc",
    url: "https://docs.aveva.com/bundle/sp-omi/page/1365727.html",
    reference: "OMI Readme › Resolved issues",
    topics: ["omi", "troubleshooting"],
  },
  {
    id: "pdf-omi-workshop",
    title: "AVEVA OMI Workshop — Creating & Running a ViewApp",
    kind: "official-pdf",
    url: "https://cdn.logic-control.com/calendar/AVEVA_OMI_Part03_CreatingRunningViewApp.pdf",
    reference: "OMI Workshop Part 3",
    topics: ["omi", "deployment"],
  },
  {
    id: "pdf-omi-training",
    title: "AVEVA InTouch OMI Training Manual (uploaded)",
    kind: "official-pdf",
    reference: "Uploaded OMI training manual — ingest to resolve section refs",
    topics: ["omi", "concepts"],
  },

  // ---- Official AVEVA / Wonderware product manuals (PDF) ----
  {
    id: "pdf-ide",
    title: "AVEVA Application Server User Guide (ArchestrA IDE)",
    kind: "official-pdf",
    url: "https://cdn.logic-control.com/docs/aveva/hmi-scada/application-server/IDE.pdf",
    reference: "Application Server User Guide (IDE.pdf)",
    topics: ["templates", "object-management", "deployment", "di", "concepts"],
  },
  {
    id: "pdf-object-viewer",
    title: "AVEVA Object Viewer User Guide",
    kind: "official-pdf",
    url: "https://cdn.logic-control.com/docs/aveva/hmi-scada/application-server/ObjectViewer.pdf",
    reference: "Object Viewer User Guide",
    topics: ["runtime", "troubleshooting"],
  },
  {
    id: "pdf-platform-manager",
    title: "AVEVA Platform Manager User Guide",
    kind: "official-pdf",
    url: "https://cdn.logic-control.com/docs/aveva/hmi-scada/application-server/PlatformManager.pdf",
    reference: "Platform Manager User Guide",
    topics: ["runtime", "deployment"],
  },
  {
    id: "pdf-scripting",
    title: "AVEVA Application Server Scripting Guide",
    kind: "official-pdf",
    url: "https://cdn.logic-control.com/docs/aveva/hmi-scada/application-server/Scripting.pdf",
    reference: "Scripting Guide › Execution types",
    topics: ["runtime", "object-management"],
  },
  {
    id: "pdf-alarm-control",
    title: "AVEVA ArchestrA Alarm Control Guide",
    kind: "official-pdf",
    url: "https://cdn.logic-control.com/docs/aveva/hmi-scada/application-server/aaAlarmClientControl.pdf",
    reference: "Alarm Client Control Guide",
    topics: ["alarms"],
  },
  {
    id: "pdf-historian-concepts",
    title: "AVEVA Historian Concepts Guide",
    kind: "official-pdf",
    url: "https://cdn.logic-control.com/docs/aveva/historian/HistorianConcepts.pdf",
    reference: "Historian Concepts Guide",
    topics: ["historian", "concepts"],
  },

  // ---- Uploaded training manual placeholder ----
  {
    id: "pdf-aveva-training",
    title: "AVEVA Application Server Training Manual (uploaded)",
    kind: "official-pdf",
    reference: "Uploaded training manual — ingest to resolve section refs",
    topics: ["concepts", "templates", "deployment", "di", "runtime"],
  },

  // ---- Community / vendor tech notes (clearly labelled non-official) ----
  {
    id: "comm-insource-deploy",
    title: "TN 1283 Troubleshooting Deployment Failures (InSource)",
    kind: "community",
    url: "https://knowledge.insourcess.com/aveva-application-server-tech-notes/tn_-_1283_troubleshooting_deployment_failures",
    reference: "InSource Tech Note 1283 — community guidance",
    topics: ["deployment", "troubleshooting"],
  },
  {
    id: "comm-ddesuitelink",
    title: "Configuring the DDESuiteLink Client Object in the ArchestrA IDE",
    kind: "community",
    url: "https://industrial-software.com/training-support/tech-notes/76-configuring-ddesuitelink-client-object-archestra-ide/",
    reference: "Industrial Software Solutions Tech Note — community guidance",
    topics: ["di", "oi"],
  },
];

export const SOURCE_BY_ID: Record<string, Source> = Object.fromEntries(
  SOURCES.map((s) => [s.id, s])
);

export function getSources(ids: string[] = []): Source[] {
  return ids.map((id) => SOURCE_BY_ID[id]).filter(Boolean) as Source[];
}

// Register runtime sources (e.g. user-uploaded manuals) so citations resolve.
// Idempotent and client-only in practice.
export function registerSources(srcs: Source[]) {
  for (const s of srcs) SOURCE_BY_ID[s.id] = s;
}
