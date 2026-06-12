import { Runbook } from "./types";

// Curated, support-oriented runbooks. Each is grounded in official AVEVA docs
// (and clearly-labelled community tech notes where useful). They encode the
// "what to check first → confirm → escalate" flow a senior support engineer
// would follow. They are heuristics, not guarantees — the UI frames them as
// "most likely / check first".

export const RUNBOOKS: Runbook[] = [
  {
    id: "rb-deploy-remote-node",
    title: "Deployment failed: cannot communicate with remote node",
    category: "Deployment",
    topics: ["deployment", "runtime", "security"],
    severity: "high",
    symptom:
      "Deploying a platform/engine fails with an error such as 'cannot communicate with remote node', 'unable to contact', or a timeout reaching the target node.",
    likelyCauses: [
      "Name resolution / DNS or hosts-file mismatch between GR node and target node",
      "ArchestrA / bootstrap services not running on the target node",
      "Firewall blocking ArchestrA communication ports between nodes",
      "Service account credentials wrong, or the node not joined to the same aaAdministrators/aaConfigTools security context",
      "Target WinPlatform not deployed (you tried to deploy an engine before its platform)",
      "Time skew between nodes breaking authentication",
    ],
    firstTool: "OCMC (SMC)",
    steps: [
      {
        title: "Confirm you can resolve and reach the node by name",
        detail:
          "From the GR node, ping the target by the exact node name used in the WinPlatform object. DNS reply should be fast (best practice ≤ 4 s); flaky DNS is a classic cause. Many sites use a hosts file so name→IP stays stable if IPs change.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-deploy-errors", "comm-insource-deploy"],
      },
      {
        title: "Verify the platform is deployed first, then the engine",
        detail:
          "Deploy order matters: the WinPlatform must be deployed and running before its AppEngines. In Deployment View, deploy the platform on its own, confirm it is running, then deploy the engines/objects.",
        tool: "ArchestrA IDE",
        sourceIds: ["doc-sp-deployment"],
      },
      {
        title: "Check ArchestrA / bootstrap services on the target",
        detail:
          "On the target node confirm the ArchestrA services are running and the service logon credentials are correct. Verify NT SERVICE\\aaPIM is in the local Administrators group and the logged-in engineer is in both aaAdministrators and aaConfigTools.",
        tool: "Platform Manager",
        sourceIds: ["comm-insource-deploy", "doc-sp-deployment"],
      },
      {
        title: "Check firewall and ports between nodes",
        detail:
          "Confirm the ArchestrA communication ports are open both directions through any host or network firewall. A one-way rule will let the platform appear reachable but fail to deploy.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-deploy-errors"],
      },
      {
        title: "Read the exact error in Log Viewer on BOTH nodes",
        detail:
          "Open OCMC → Log Viewer on the GR node and the target node at the same time and re-deploy. The target-side log usually contains the real root cause (security, port, or credential message) that the IDE error hides.",
        tool: "Log Viewer",
        sourceIds: ["doc-deploy-errors", "pdf-platform-manager"],
      },
      {
        title: "Verify time sync and security context",
        detail:
          "Significant clock skew between nodes breaks authenticated communication. Confirm both nodes share a time source and belong to the same Galaxy security configuration / domain context.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-sp-deployment"],
      },
    ],
    confirmResolution:
      "The platform deploys and shows running in Platform Manager, engines go OnScan, and a re-deploy of a single object succeeds without the communication error.",
    escalateWhen:
      "Services, ports, credentials, name resolution and time sync all check out but deployment still times out — capture synchronized Log Viewer exports from both nodes and escalate with the network/AD team.",
    sourceIds: ["doc-deploy-errors", "doc-sp-deployment", "comm-insource-deploy"],
    keywords: [
      "cannot communicate with remote node",
      "unable to contact node",
      "deployment timeout",
      "deploy failed",
      "bootstrap",
      "aaConfigTools",
      "aaAdministrators",
    ],
  },
  {
    id: "rb-bad-quality",
    title: "Object attribute shows Bad quality",
    category: "Bad quality / no data",
    topics: ["runtime", "di", "oi", "troubleshooting"],
    severity: "medium",
    symptom:
      "An attribute (e.g. PV) shows Bad (or Uncertain/Initializing) quality in Object Viewer or on a graphic, so the value cannot be trusted.",
    likelyCauses: [
      "Upstream DI/OI link is down (OI Server not running or not connected to the device)",
      "I/O reference string is wrong (typo, wrong item/topic, wrong source object)",
      "Source object or engine is OffScan",
      "Item does not exist in the OI Server / OPC namespace",
      "Security or licensing limiting the OI Server",
    ],
    firstTool: "Object Viewer",
    steps: [
      {
        title: "Confirm where quality goes Bad",
        detail:
          "In Object Viewer, watch the attribute and its InputSource/I-O reference. Bad quality almost always originates upstream — at the DI object or OI Server — not at the consuming object.",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer", "doc-opc-source"],
      },
      {
        title: "Check the I/O reference string",
        detail:
          "Verify the attribute's input source points at the correct DI object item (correct OPCClient/DDESuiteLinkClient instance, topic/group, and item name). A single typo yields Bad quality. Autobound references can drift if the source namespace changed.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide", "doc-opc-source"],
      },
      {
        title: "Verify scan state of the source",
        detail:
          "Ensure the DI object, the consuming object, and their AppEngine are OnScan. An OffScan source produces Bad/last-known quality. After a reboot, engines may be OffScan.",
        tool: "Platform Manager",
        sourceIds: ["doc-offscan", "pdf-platform-manager"],
      },
      {
        title: "Diagnose the OI Server directly",
        detail:
          "In OI Server Manager, check the OI Server is running and connected to the device, and use its diagnostics to confirm the specific item updates with Good quality at the driver level. If it is Bad here, the problem is device/driver-side.",
        tool: "OI Server Manager",
        sourceIds: ["doc-opc-source", "doc-opcua-source"],
      },
      {
        title: "Check the item exists and the device responds",
        detail:
          "Confirm the item/tag actually exists in the OI Server namespace and the PLC/device is reachable and powered. A non-existent item or dead device gives Bad quality even with perfect references.",
        tool: "OI Server Manager",
        sourceIds: ["doc-opc-source"],
      },
    ],
    confirmResolution:
      "The attribute shows Good quality with a fresh timestamp in Object Viewer, and the value tracks the device.",
    escalateWhen:
      "The OI Server diagnostics show the item Good but the App Server attribute stays Bad after references and scan state are verified — escalate with the reference string, OI Server config, and Object Viewer capture.",
    sourceIds: ["pdf-object-viewer", "doc-opc-source", "doc-offscan"],
    keywords: ["bad quality", "uncertain", "initializing", "no value", "i/o reference", "inputsource"],
  },
  {
    id: "rb-no-data-object-viewer",
    title: "No data visible in Object Viewer",
    category: "Bad quality / no data",
    topics: ["runtime", "troubleshooting"],
    severity: "medium",
    symptom:
      "Object Viewer shows the attribute but with no updating value, dashes, or a stale value/timestamp.",
    likelyCauses: [
      "Object or its AppEngine is OffScan",
      "Object is not actually deployed (config-only) or deploy is pending",
      "Wrong attribute path watched (template vs instance, wrong instance)",
      "Engine not running / platform down",
      "Upstream Bad quality (see Bad quality runbook)",
    ],
    firstTool: "Object Viewer",
    steps: [
      {
        title: "Confirm the object is deployed and running",
        detail:
          "In Platform Manager, verify the WinPlatform is running, the AppEngine is started and OnScan, and the object shows a running/deployed state — not 'not deployed' or 'shut down'.",
        tool: "Platform Manager",
        sourceIds: ["pdf-platform-manager", "doc-offscan"],
      },
      {
        title: "Verify scan state",
        detail:
          "Check ScanState on the object and engine. OnScan = processing; OffScan = idle with no live updates. Set OnScan via Platform Manager or ScanStateCmd and watch the value resume.",
        tool: "Platform Manager",
        sourceIds: ["doc-offscan", "pdf-scripting"],
      },
      {
        title: "Confirm you are watching the right path",
        detail:
          "Make sure you added the deployed instance's attribute (not the template) and the correct instance name. Re-add from the running object to be sure.",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer"],
      },
      {
        title: "Check the timestamp and quality",
        detail:
          "A frozen timestamp with Good quality means the source stopped updating; Bad quality means an upstream comms issue — follow the Bad quality runbook.",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer"],
      },
    ],
    confirmResolution:
      "Values update live with a moving timestamp and Good quality in Object Viewer.",
    escalateWhen:
      "Object is confirmed deployed, OnScan, on a running engine, correctly addressed, yet still shows no data — capture Platform Manager state and Object Viewer and escalate.",
    sourceIds: ["pdf-object-viewer", "pdf-platform-manager", "doc-offscan"],
    keywords: ["no data", "no value", "dashes", "stale", "frozen", "not updating", "object viewer"],
  },
  {
    id: "rb-oi-opc-not-updating",
    title: "OI.SIM / OPC client not updating",
    category: "OI / OPC / DI communication",
    topics: ["oi", "di", "troubleshooting"],
    severity: "medium",
    symptom:
      "An OI Server (e.g. OI.SIM or a real driver) or OPC client object is configured but values do not change / show Bad quality.",
    likelyCauses: [
      "OI Server not activated/running, or no client subscriptions",
      "DI object pointing at the wrong server node or program ID / UA endpoint",
      "Topic/group or update interval misconfigured",
      "Item names do not match the OI Server namespace",
      "Protocol mismatch (DDE vs SuiteLink, OPC DA vs UA) or security/cert issue for OPC UA",
    ],
    firstTool: "OI Server Manager",
    steps: [
      {
        title: "Confirm the OI Server is running and has clients",
        detail:
          "In OI Server Manager, verify the server is activated/running and that the DI object's client connection appears. No subscriptions usually means the DI object is not deployed/OnScan or is pointed at the wrong server.",
        tool: "OI Server Manager",
        sourceIds: ["doc-opc-source"],
      },
      {
        title: "Validate items in the server's diagnostics",
        detail:
          "Use the OI Server diagnostics to confirm the specific items update with Good quality. If they update here but not in App Server, the problem is the DI object reference, not the driver.",
        tool: "OI Server Manager",
        sourceIds: ["doc-opc-source", "doc-opcua-source"],
      },
      {
        title: "Check the DI object configuration",
        detail:
          "Confirm the DI object (OPCClient/DDESuiteLinkClient/OPC UA) targets the correct server node and endpoint/ProgID, with the right topic/group, update interval, and protocol (SuiteLink recommended over DDE).",
        tool: "ArchestrA IDE",
        sourceIds: ["comm-ddesuitelink", "doc-opcua-source"],
      },
      {
        title: "For OPC UA, verify endpoint, security and certificates",
        detail:
          "OPC UA needs a reachable endpoint and trusted certificates on both ends. A rejected/untrusted certificate or wrong security policy silently blocks updates — check the UA service and trust lists.",
        tool: "OI Server Manager",
        sourceIds: ["doc-opcua-service", "doc-opcua-source"],
      },
    ],
    confirmResolution:
      "Items update with Good quality in OI Server diagnostics and the App Server attributes track them live.",
    escalateWhen:
      "Driver diagnostics show items Good and the DI object is correctly configured, but App Server still does not update — escalate with the OI Server config export and DI object reference details.",
    sourceIds: ["doc-opc-source", "doc-opcua-source", "comm-ddesuitelink"],
    keywords: ["oi.sim", "opc not updating", "opc ua", "suitelink", "dde", "no subscription", "program id", "endpoint"],
  },
  {
    id: "rb-historian-no-data",
    title: "Historized data not appearing in Historian",
    category: "Historian / historization",
    topics: ["historian", "troubleshooting"],
    severity: "medium",
    symptom:
      "An attribute is configured for history but trends are flat/empty in Historian Client Web, or logging stopped.",
    likelyCauses: [
      "History not actually enabled on the attribute, or the engine has no Historian assigned",
      "Engine/object OffScan so no values are produced to log",
      "Historian stuck in store-and-forward (forward step failing) — SF data pending",
      "IDAS connection problem between source and Historian",
      "Historian storage subsystem not running / disk or licensing issue",
    ],
    firstTool: "Historian Client Web",
    steps: [
      {
        title: "Confirm history is enabled and an Historian is assigned",
        detail:
          "In the IDE, verify the attribute has History enabled and that its AppEngine is configured with the correct Historian. No assigned Historian = nothing to log to.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-historian-concepts", "doc-historian-issues"],
      },
      {
        title: "Verify the source is producing Good values",
        detail:
          "In Object Viewer confirm the attribute is OnScan and Good quality. Historian logs what the engine produces — OffScan/Bad source means nothing meaningful to store.",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer", "doc-offscan"],
      },
      {
        title: "Check for store-and-forward backlog",
        detail:
          "Inspect the store-forward folders for pending *.dat files (e.g. original.dat) and the SysStatusSFDataPending tag. A historian 'stuck in store-forward' often needs the engine stopped and started (via Platform Manager) to resume forwarding.",
        tool: "Platform Manager",
        sourceIds: ["doc-idas-sf", "doc-historian-issues"],
      },
      {
        title: "Troubleshoot the IDAS connection",
        detail:
          "Confirm IDAS is connected and acquiring. Use the Historian IDAS troubleshooting steps to validate the acquisition path from source to storage.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-idas-troubleshoot"],
      },
      {
        title: "Confirm the Historian storage engine is healthy",
        detail:
          "Check the Historian status (storage running, disk space, licensing). If storage is down or out of license, recent data will not be retained.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-historian-issues"],
      },
    ],
    confirmResolution:
      "New values appear in Historian Client Web trends in real time and any store-forward backlog drains.",
    escalateWhen:
      "History is enabled, source is Good/OnScan, IDAS is connected and storage is healthy, but data still does not land — escalate with SF folder contents, IDAS status, and the tag's history config.",
    sourceIds: ["doc-historian-issues", "doc-idas-troubleshoot", "doc-idas-sf", "pdf-historian-concepts"],
    keywords: ["no history", "trend flat", "historian not logging", "store and forward", "sysstatussfdatapending", "idas", "historization missing"],
  },
  {
    id: "rb-alarm-not-visible",
    title: "Alarm configured but not visible/active",
    category: "Alarms",
    topics: ["alarms", "troubleshooting"],
    severity: "medium",
    symptom:
      "An alarm is configured on an attribute but never appears in the Alarm Control / active alarm list when the condition occurs.",
    likelyCauses: [
      "Alarm disabled, inhibited (AlarmInhibit), or shelved",
      "Attribute quality is Bad/Uncertain, so the alarm logic does not evaluate",
      "Alarm limit/condition not actually reached (EU range/scaling wrong)",
      "Alarm Client query filter excludes the object/area/priority",
      "Object/engine OffScan, so the alarm condition is never evaluated",
    ],
    firstTool: "Object Viewer",
    steps: [
      {
        title: "Check enable / inhibit / shelve state",
        detail:
          "Verify the alarm is enabled and not inhibited (AlarmInhibit) or shelved. Inhibited/shelved alarms will not annunciate. Remember default shelving applies to Medium/Low; Critical/High are typically not shelve-enabled.",
        tool: "Object Viewer",
        sourceIds: ["doc-alarm-inhibit", "doc-alarms-impl", "pdf-alarm-control"],
      },
      {
        title: "Confirm the underlying value and quality",
        detail:
          "In Object Viewer confirm the attribute is OnScan, Good quality, and actually crosses the configured limit. Bad quality or an out-of-range EU scaling can prevent the alarm evaluating as expected.",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer", "doc-alarms-impl"],
      },
      {
        title: "Check the Alarm Client query/filter",
        detail:
          "The alarm may be active but filtered out. Verify the Alarm Control query string includes the object's Galaxy/area and the alarm's priority/severity range.",
        tool: "Object Viewer",
        sourceIds: ["pdf-alarm-control"],
      },
      {
        title: "Verify alarm configuration matches a working object",
        detail:
          "Compare against an attribute whose alarm works. SMC logs sometimes show why a specific attribute's alarm description/condition does not surface at runtime.",
        tool: "Log Viewer",
        sourceIds: ["doc-alarms-sysobjects", "doc-alarms-impl"],
      },
    ],
    confirmResolution:
      "Driving the value past the limit produces an active alarm visible in the Alarm Control, and it clears/returns to normal correctly.",
    escalateWhen:
      "The alarm is enabled, not inhibited/shelved, value is Good and over limit, and the query includes it — but it still never activates. Escalate with the alarm config and an SMC log capture.",
    sourceIds: ["doc-alarms-impl", "doc-alarm-inhibit", "pdf-alarm-control", "doc-alarms-sysobjects"],
    keywords: ["alarm not showing", "alarm not active", "alarminhibit", "shelved", "alarmmodecmd", "no alarm", "alarm missing"],
  },
  {
    id: "rb-onscan-offscan",
    title: "AppEngine / object OnScan vs OffScan confusion",
    category: "Platform / AppEngine / scan state",
    topics: ["runtime", "troubleshooting"],
    severity: "low",
    symptom:
      "Objects are deployed but nothing runs/updates; scripts don't execute; references won't resolve — often after a reboot or manual stop.",
    likelyCauses: [
      "AppEngine or object left OffScan (idle, not executing)",
      "Engines not set OnScan after a node reboot, breaking reference resolution",
      "Platform stopped, so all hosted engines are down",
      "Object deployed but never set OnScan",
    ],
    firstTool: "Platform Manager",
    steps: [
      {
        title: "Check platform, engine and object scan states",
        detail:
          "In Platform Manager review the hierarchy: platform running? engine started and OnScan? object OnScan? OnScan means normal processing; OffScan means idle/not executing.",
        tool: "Platform Manager",
        sourceIds: ["pdf-platform-manager", "doc-offscan"],
      },
      {
        title: "Set engines OnScan after a reboot",
        detail:
          "After rebooting a platform you must set each engine OnScan. Failing to do so causes reference-resolution issues across objects. Set the engine OnScan and let objects resume.",
        tool: "Platform Manager",
        sourceIds: ["doc-offscan", "doc-as-resolved"],
      },
      {
        title: "Use ScanStateCmd / OnScan scripts intentionally",
        detail:
          "ScanStateCmd toggles an object OnScan/OffScan. OnScan scripts run the first time an engine executes the object after it goes OnScan — useful for initialisation. Don't leave objects OffScan unless intentionally idled.",
        tool: "Object Viewer",
        sourceIds: ["pdf-scripting", "doc-offscan"],
      },
    ],
    confirmResolution:
      "Platform, engines and objects are all OnScan; values update, scripts run, and references resolve.",
    escalateWhen:
      "Engines are OnScan and the platform is running but objects still won't execute or resolve references — capture Platform Manager state and Log Viewer and escalate.",
    sourceIds: ["doc-offscan", "pdf-platform-manager", "pdf-scripting"],
    keywords: ["onscan", "offscan", "scanstate", "scanstatecmd", "not running", "after reboot", "reference resolution"],
  },
  {
    id: "rb-checkin-version-mismatch",
    title: "Checked out / checked in / deployment version mismatch",
    category: "Check-in / check-out / config vs runtime",
    topics: ["object-management", "deployment", "troubleshooting"],
    severity: "low",
    symptom:
      "Changes don't appear at runtime, deploy is greyed out, an object can't be edited, or runtime behaviour doesn't match the latest configuration.",
    likelyCauses: [
      "Object checked out (by you or someone else) so edits/deploy are blocked",
      "Edited config not checked in, so the deployable version is stale",
      "Object deployed at an older version — config changed but not re-deployed",
      "Pending undeploy/redeploy or a partial cascade deploy",
    ],
    firstTool: "ArchestrA IDE",
    steps: [
      {
        title: "Check the object's check-out state and owner",
        detail:
          "In the IDE, see whether the object is checked out and by whom. A checked-out object can't be fully deployed/edited by others. The owner must check it in (or an admin can Undo Check Out, discarding changes).",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
      {
        title: "Check in pending changes, then deploy",
        detail:
          "Config edits only become deployable after check-in. Check in the object (and any modified templates), then deploy so the runtime version matches the configuration.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
      {
        title: "Compare deployed vs configured version",
        detail:
          "Confirm the deployed version equals the latest checked-in version. If config changed since the last deploy, re-deploy the object (cascade deploy if templates changed).",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide", "doc-sp-deployment"],
      },
    ],
    confirmResolution:
      "Object is checked in, deployed version equals configured version, and runtime behaviour reflects the latest changes.",
    escalateWhen:
      "Versions match and the object is checked in/re-deployed but runtime still differs — capture the object's deployment state and escalate (possible Galaxy/object corruption).",
    sourceIds: ["pdf-ide", "doc-sp-deployment"],
    keywords: ["checked out", "check in", "undo check out", "version mismatch", "config vs runtime", "deploy greyed out", "changes not applied"],
  },
  {
    id: "rb-csv-import-conflict",
    title: "Import from CSV / package conflict handling",
    category: "Import / export / CSV",
    topics: ["csv", "object-management", "troubleshooting"],
    severity: "low",
    symptom:
      "A Galaxy import (CSV/aaPKG/Galaxy dump) fails, partially applies, or raises object/template conflict prompts.",
    likelyCauses: [
      "Template/object already exists with a different definition (version conflict)",
      "CSV column/format or attribute path errors",
      "Required parent template missing (import order/dependency)",
      "Objects checked out, blocking modification during import",
      "Encoding / locale issues in the CSV",
    ],
    firstTool: "ArchestrA IDE",
    steps: [
      {
        title: "Read the conflict prompt carefully and choose intent",
        detail:
          "On import, the IDE prompts how to resolve conflicts (skip / overwrite / create new). Decide deliberately: overwriting a template affects all derived objects. In production, prefer importing into a test Galaxy first.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
      {
        title: "Ensure dependencies/parents exist and order is correct",
        detail:
          "Import base templates before derived templates and instances. A missing parent template causes failures or orphaned objects.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
      {
        title: "Validate the CSV structure",
        detail:
          "Confirm column headers, attribute paths, data types and encoding match what the importer expects. A single malformed row can abort or partially apply the import — check the import log.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
      {
        title: "Make sure target objects are checked in",
        detail:
          "Objects checked out by another user can't be modified by the import. Ensure they're checked in (or Undo Check Out) before re-running.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
    ],
    confirmResolution:
      "Import completes without unresolved conflicts, the import log is clean, and imported objects open and deploy correctly.",
    escalateWhen:
      "Dependencies, format and check-out state are all correct but the import still fails or corrupts objects — back up the Galaxy, capture the import log, and escalate.",
    sourceIds: ["pdf-ide"],
    keywords: ["csv import", "aapkg", "package conflict", "galaxy import", "overwrite template", "import failed", "export import"],
  },
  {
    id: "rb-security-login",
    title: "Security login / authentication confusion",
    category: "Security / login / authentication",
    topics: ["security", "troubleshooting"],
    severity: "medium",
    symptom:
      "Can't log in to the Galaxy/runtime, missing permissions, or 'access denied' deploying or opening the IDE.",
    likelyCauses: [
      "Galaxy security mode (None / Galaxy / OS Group / OS User) not matching how you're authenticating",
      "User not in the right aaAdministrators / aaConfigTools / OS security group",
      "Domain vs local account mismatch, or password/account expired",
      "Security model not deployed after a change",
      "Time skew or domain trust issue breaking authentication between nodes",
    ],
    firstTool: "ArchestrA IDE",
    steps: [
      {
        title: "Identify the Galaxy's security mode",
        detail:
          "Check the configured security mode (None, Galaxy, OS Group based, OS User based). How you must log in — and which accounts have rights — depends entirely on this mode.",
        tool: "ArchestrA IDE",
        sourceIds: ["doc-sp-deployment"],
      },
      {
        title: "Verify group membership for the account",
        detail:
          "Confirm the user is in the required groups: aaAdministrators and aaConfigTools for engineering, plus any OS groups the security model maps to roles. NT SERVICE\\aaPIM should be in local Administrators on each node.",
        tool: "ArchestrA IDE",
        sourceIds: ["comm-insource-deploy", "doc-sp-deployment"],
      },
      {
        title: "Re-deploy the security model after changes",
        detail:
          "Security/role changes must be deployed to take effect at runtime. If you changed the model but didn't deploy it, runtime still enforces the old rules.",
        tool: "ArchestrA IDE",
        sourceIds: ["doc-sp-deployment"],
      },
      {
        title: "Check domain/account health and time sync",
        detail:
          "Confirm the account isn't locked/expired, domain trust is healthy, and node clocks are in sync. Authentication across nodes fails with significant time skew.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-sp-deployment"],
      },
    ],
    confirmResolution:
      "The user logs in with the expected role, can open/edit/deploy as permitted, and no access-denied errors occur.",
    escalateWhen:
      "Mode, group membership and deployment are correct and accounts are healthy but authentication still fails — escalate with the security mode, group memberships, and Log Viewer auth errors to AD/security.",
    sourceIds: ["doc-sp-deployment", "comm-insource-deploy"],
    keywords: ["login failed", "access denied", "authentication", "security mode", "aaadministrators", "aaconfigtools", "permissions", "galaxy security"],
  },
];

export const RUNBOOK_BY_ID: Record<string, Runbook> = Object.fromEntries(
  RUNBOOKS.map((r) => [r.id, r])
);
