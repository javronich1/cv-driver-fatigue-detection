import { KnownIssue } from "./types";

// Known issues / gotchas distilled from official readmes and community tech notes.
// Framed honestly as environment-specific patterns, not universal truths.

export const KNOWN_ISSUES: KnownIssue[] = [
  {
    id: "ki-reboot-offscan",
    title: "After a platform reboot, engines come back OffScan",
    topics: ["runtime", "deployment"],
    environment: "Any node after an OS reboot / power event",
    symptom:
      "Following a reboot, objects don't update and references fail to resolve even though everything is deployed.",
    cause:
      "Engines are not automatically set OnScan; until they are, reference resolution across objects can fail.",
    workaround:
      "In Platform Manager, set each AppEngine OnScan after the reboot. Consider startup procedures/runbooks so operators do this consistently.",
    status: "by-design",
    sourceIds: ["doc-offscan", "doc-as-resolved"],
    keywords: ["reboot", "offscan", "reference resolution", "after restart"],
  },
  {
    id: "ki-sf-pending-after-upgrade",
    title: "SysStatusSFDataPending true after a Historian upgrade",
    topics: ["historian"],
    environment: "Tier-1 / Tier-2 Historian after version upgrade",
    symptom:
      "After upgrading (e.g. 2023 P03 → 2023 R2), the system tag SysStatusSFDataPending is true and trends show gaps.",
    cause:
      "Store-and-forward backlog left pending after the upgrade; the forward step is not draining buffered data automatically.",
    workaround:
      "Inspect the store-forward folders on the source for pending *.dat files (e.g. original.dat in A000000_001). Stop/start the engine (Platform Manager) to resume forwarding; confirm the SF tag returns to false. Reported via AVEVA community.",
    status: "known",
    sourceIds: ["doc-idas-sf", "doc-historian-issues"],
    keywords: ["sysstatussfdatapending", "store and forward", "upgrade", "trend gap", "historian backlog"],
  },
  {
    id: "ki-dns-slow-deploy",
    title: "Slow DNS resolution causes intermittent deployment failures",
    topics: ["deployment"],
    environment: "Multi-node, especially DHCP / changing IPs",
    symptom:
      "Deployments intermittently fail to reach a remote node; sometimes succeed on retry.",
    cause:
      "Slow or unstable name resolution. Best practice is a DNS reply of 4 seconds or less; flaky DNS makes node communication unreliable.",
    workaround:
      "Use a hosts file so name→IP mappings stay stable if IP addresses change, and ensure DNS is fast and consistent across nodes. Community tech-note guidance (InSource TN 1283).",
    status: "known",
    sourceIds: ["comm-insource-deploy", "doc-deploy-errors"],
    keywords: ["dns", "slow deploy", "intermittent", "hosts file", "remote node"],
  },
  {
    id: "ki-shelve-severity",
    title: "Critical/High alarms cannot be shelved by default",
    topics: ["alarms"],
    environment: "Default Alarm Control configuration",
    symptom:
      "Operators can shelve some alarms but the shelve action does nothing for Critical/High severity alarms.",
    cause:
      "By design, only Medium and Low severity alarms are shelve-enabled by default; Critical and High are excluded to avoid hiding serious conditions.",
    workaround:
      "This is intentional safety behaviour. If a Critical/High alarm is a nuisance, address the root cause or review the alarm rationalisation rather than forcing shelving.",
    status: "by-design",
    sourceIds: ["pdf-alarm-control", "doc-alarms-impl"],
    keywords: ["shelve", "critical alarm", "high alarm", "cannot shelve", "severity"],
  },
  {
    id: "ki-checkout-blocks-deploy",
    title: "Checked-out objects block edits and full deployment",
    topics: ["object-management", "deployment"],
    environment: "Multi-engineer Galaxy",
    symptom:
      "An object can't be edited or fully deployed; deploy options are greyed out.",
    cause:
      "The object is checked out (often by another engineer), locking it. Checked-out objects can't be modified or fully deployed by others.",
    workaround:
      "Have the owner check the object in. An administrator can Undo Check Out, but this discards the owner's in-progress changes — coordinate first.",
    status: "by-design",
    sourceIds: ["pdf-ide"],
    keywords: ["checked out", "locked object", "undo check out", "deploy greyed", "cannot edit"],
  },
  {
    id: "ki-omi-webclient-limits",
    title: "Some OMI apps/features don't work in the web client",
    topics: ["omi"],
    environment: "OMI web (browser) client",
    symptom:
      "A ViewApp that works in the local OMI client renders partially or behaves differently in the browser web client.",
    cause:
      "The OMI web client has documented limitations — not every OMI app or feature is supported in the browser compared with the local client.",
    workaround:
      "Check the official 'General OMI web client limitations' list. For unsupported features, use the local client, or redesign the ViewApp to use web-supported apps. By design per AVEVA docs.",
    status: "by-design",
    sourceIds: ["doc-omi-webclient-limits"],
    keywords: ["omi web client", "not supported", "browser", "partial render", "viewapp web"],
  },
  {
    id: "ki-omi-layout-as-content",
    title: "A layout dropped as content inside another layout misbehaves",
    topics: ["omi"],
    environment: "OMI ViewApp with nested layouts",
    symptom:
      "When a layout is used as content within another layout, panes/navigation don't render or interact as expected.",
    cause:
      "Nested layout-as-content is an advanced pattern with configuration constraints; misconfiguration leads to layout/navigation problems (reported in AVEVA community and addressed across patches).",
    workaround:
      "Review the layout/pane and navigation configuration against the OMI navigation-hierarchy docs; simplify the nesting or follow the supported Content Presenter pattern. Check OMI resolved-issues for your patch level.",
    status: "known",
    sourceIds: ["doc-omi-nav", "doc-omi-resolved"],
    keywords: ["nested layout", "layout as content", "content presenter", "navigation hierarchy", "panes"],
  },
];

export const KNOWN_ISSUE_BY_ID: Record<string, KnownIssue> = Object.fromEntries(
  KNOWN_ISSUES.map((k) => [k.id, k])
);
