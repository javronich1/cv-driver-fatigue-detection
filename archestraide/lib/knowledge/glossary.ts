import { GlossaryTerm } from "./types";

// Curated glossary. Definitions are grounded in official AVEVA documentation
// and the Application Server User Guide. Kept concise-first with practical
// examples and related-term links.

export const GLOSSARY: GlossaryTerm[] = [
  {
    id: "galaxy",
    term: "Galaxy",
    aliases: ["galaxy database", "GR"],
    topics: ["concepts", "object-management"],
    short:
      "The single logical namespace and database that holds an entire Application Server application — all templates, instances, and configuration.",
    explanation:
      "A Galaxy is the whole application: every template, every object instance, security, and deployment topology live inside it. It is hosted by the Galaxy Repository (GR) node and edited through the ArchestrA IDE. You can think of it as the 'project database' for System Platform — one Galaxy = one application namespace.",
    example:
      "A plant might have a Galaxy named 'PLANT_PROD'. Engineers connect the IDE to the GR node and open PLANT_PROD to build and deploy the application.",
    related: ["template", "instance", "ide", "ocmc"],
    sourceIds: ["pdf-ide", "doc-sp-deployment"],
  },
  {
    id: "template",
    term: "Template",
    aliases: ["base template", "$template"],
    topics: ["templates", "object-management"],
    short:
      "A reusable object definition (prefixed with $) used to create instances or derived templates.",
    explanation:
      "Templates define attributes, scripts, and behaviour once so they can be reused. Templates are never deployed to run — they are blueprints. They are shown with a $ prefix (e.g. $Pump). Changing a template propagates to everything derived from or instanced from it (subject to locking).",
    example:
      "$Pump defines Speed, Status, and a Start/Stop script. You create dozens of pump instances from it instead of re-building each one.",
    related: ["derived-template", "instance", "template-toolbox"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "derived-template",
    term: "Derived Template",
    topics: ["templates"],
    short:
      "A template created from another template, inheriting and specialising its parent.",
    explanation:
      "Derivation lets you build a hierarchy of reuse: a base template captures common behaviour, derived templates add or lock specifics. Attribute locking on the parent controls what derived templates and instances are allowed to change.",
    example:
      "$Pump (base) → $Pump_VFD (derived, adds VFD speed control) → instances PMP-101, PMP-102.",
    related: ["template", "instance"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "instance",
    term: "Instance",
    aliases: ["object instance", "automation object"],
    topics: ["object-management", "runtime"],
    short:
      "A concrete, deployable object created from a template — the thing that actually runs.",
    explanation:
      "Instances are the real automation objects that get assigned to an AppEngine and deployed to run. Each instance has its own attribute values but inherits structure from its template. Only instances run at runtime — templates do not.",
    example:
      "PMP-101 is an instance of $Pump assigned to AppEngine1 on WinPlatform_Node1.",
    related: ["template", "appengine", "winplatform", "autobind"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "plant-model",
    term: "Plant Model (Model View)",
    aliases: ["model view"],
    topics: ["object-management", "concepts"],
    short:
      "The logical/physical hierarchy of your plant — what equipment exists and how it is organised (ISA-95 style).",
    explanation:
      "Model View answers 'what is it and where does it belong in the plant?' It organises objects by area / equipment containment (Enterprise → Site → Area → Unit). It is independent of which computer runs each object.",
    example:
      "Site → Area 'Boiler House' → Unit 'Boiler 1' → PMP-101. This says nothing about which node runs PMP-101.",
    related: ["deployment-model", "instance"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "deployment-model",
    term: "Deployment Model (Deployment View)",
    aliases: ["deployment view"],
    topics: ["deployment", "runtime"],
    short:
      "The execution topology — which Platform and AppEngine hosts each object at runtime.",
    explanation:
      "Deployment View answers 'where does it run?' It shows WinPlatforms (nodes), the AppEngines on them, and the objects hosted by each engine. The same object appears in both Model and Deployment views; Model is organisational, Deployment is execution.",
    example:
      "WinPlatform_Node1 → AppEngine1 → Area object → PMP-101. PMP-101 is the same object you see in Model View, just shown by where it executes.",
    related: ["plant-model", "appengine", "winplatform"],
    sourceIds: ["pdf-ide", "doc-sp-deployment"],
  },
  {
    id: "di-object",
    term: "DI Object (Device Integration Object)",
    aliases: ["device integration object", "DDESuiteLinkClient", "OPCClient"],
    topics: ["di", "oi"],
    short:
      "An Application Server object that connects the Galaxy to a data source (OI/DA Server) so field data flows in and out.",
    explanation:
      "DI Objects are the bridge between automation objects and the communication layer. Common DI objects are $OPCClient, $DDESuiteLinkClient, and OPC UA client objects. They define the data source node, topic/group, and the protocol used to talk to an OI Server or DA Server.",
    example:
      "An $OPCClient DI object points at an OI Server on the same node; pump attributes bind to its items via I/O references.",
    related: ["oi-server", "opcclient", "ddesuitelinkclient", "autobind"],
    sourceIds: ["doc-opc-source", "comm-ddesuitelink", "pdf-ide"],
  },
  {
    id: "oi-server",
    term: "OI Server (Operations Integration Server)",
    aliases: ["DA Server", "communication driver"],
    topics: ["oi", "di"],
    short:
      "The communication driver process that talks the device/PLC protocol and exposes data to clients via SuiteLink/OPC.",
    explanation:
      "OI Servers (formerly DAServers) speak protocols like Modbus, OPC UA, Siemens, etc., on one side and present a uniform item namespace to System Platform on the other. They are configured and diagnosed in the OI Server Manager (a snap-in under OCMC/SMC).",
    example:
      "OI.MBTCP connects to a Modbus PLC; a DI Object subscribes to its items and System Platform attributes resolve through it.",
    related: ["di-object", "opcclient", "ocmc"],
    sourceIds: ["doc-opc-source", "doc-opcua-source"],
  },
  {
    id: "opcclient",
    term: "OPCClient",
    aliases: ["$OPCClient", "OPC client object"],
    topics: ["di", "oi"],
    short:
      "A DI Object template that connects the Galaxy to an OPC (DA/UA) server as a client.",
    explanation:
      "The OPCClient object defines the OPC server node and program ID (or UA endpoint), the update interval, and groups/items. Attributes in automation objects reference its items to read/write live data.",
    example:
      "$OPCClient instance 'OPC_PLC1' targets the OI Server's OPC interface; PMP-101.PV uses an I/O reference to OPC_PLC1.Tag.",
    related: ["di-object", "oi-server", "ddesuitelinkclient"],
    sourceIds: ["doc-opc-source", "doc-opcua-source"],
  },
  {
    id: "ddesuitelinkclient",
    term: "DDESuiteLinkClient",
    aliases: ["$DDESuiteLinkClient", "SuiteLink client"],
    topics: ["di", "oi"],
    short:
      "A DI Object that connects to OI/DA Servers using the SuiteLink (or legacy DDE) protocol.",
    explanation:
      "Found under the System / Device Integration templates as $DDESuiteLinkClient. You configure the server node, the communication protocol (SuiteLink recommended over DDE), topics, and optional 'detect connection alarm'. Instances start in the Unassigned Host folder and are then assigned and deployed.",
    example:
      "Create an instance of $DDESuiteLinkClient, set protocol to SuiteLink, point it at the OI Server node, and add a topic mapping to the device.",
    related: ["di-object", "oi-server", "opcclient"],
    sourceIds: ["comm-ddesuitelink", "pdf-ide"],
  },
  {
    id: "autobind",
    term: "Autobind",
    topics: ["di", "object-management"],
    short:
      "A feature that automatically creates and binds attribute references (e.g. I/O) to a DI Object, avoiding manual reference entry.",
    explanation:
      "Autobind speeds up wiring large numbers of attributes to a device by generating the I/O source references automatically based on naming conventions, rather than typing each reference by hand. It is commonly used when standing up a DI object against a structured item namespace.",
    example:
      "Instead of manually setting PV.InputSource for 500 tags, autobind generates the references against the OPCClient item namespace.",
    related: ["di-object", "opcclient", "instance"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "object-viewer",
    term: "Object Viewer",
    topics: ["runtime", "troubleshooting"],
    short:
      "A runtime diagnostic tool that shows live attribute values, quality, and timestamps for deployed objects.",
    explanation:
      "Object Viewer is the first tool for runtime troubleshooting. You add attributes to a watch list and see their live value, data quality (Good/Bad/Uncertain/Initializing), and timestamp. It also lets you set ScanState and write values for testing.",
    example:
      "Open Object Viewer from the IDE on PMP-101, watch PV — if quality is Bad, the problem is upstream (DI/OI), not the display.",
    related: ["quality", "onscan-offscan", "ocmc"],
    sourceIds: ["pdf-object-viewer"],
  },
  {
    id: "ocmc",
    term: "OCMC / SMC",
    aliases: ["System Management Console", "Operations Control Management Console"],
    topics: ["runtime", "troubleshooting"],
    short:
      "The management console that hosts snap-ins for Log Viewer, Platform Manager, OI Server Manager, and Historian administration.",
    explanation:
      "OCMC (Operations Control Management Console; historically the System Management Console / SMC) is the central place for operations and diagnostics. Key snap-ins: Log Viewer (ArchestrA logs), Platform Manager (start/stop platforms & engines, scan state), OI Server Manager, and Historian admin.",
    example:
      "When deployment fails, open OCMC → Log Viewer on both nodes to read the ArchestrA error messages.",
    related: ["object-viewer", "appengine", "winplatform"],
    sourceIds: ["pdf-platform-manager"],
  },
  {
    id: "appengine",
    term: "AppEngine",
    topics: ["runtime", "deployment"],
    short:
      "The runtime host process that executes a group of automation objects on a scan schedule.",
    explanation:
      "An AppEngine runs on a WinPlatform and is the container that actually executes objects (running their scripts and I/O) at a configured scan period. Objects must be assigned to an engine and the engine must be deployed and OnScan for them to run. After a node reboot you must set engines OnScan or you get reference-resolution issues.",
    example:
      "AppEngine1 hosts the Boiler House area objects on a 1000 ms scan; if AppEngine1 is OffScan, none of those objects update.",
    related: ["winplatform", "onscan-offscan", "instance"],
    sourceIds: ["pdf-platform-manager", "doc-offscan"],
  },
  {
    id: "winplatform",
    term: "WinPlatform",
    aliases: ["platform", "platform object"],
    topics: ["runtime", "deployment"],
    short:
      "The object representing a physical/virtual computer node in the Galaxy; it hosts one or more AppEngines.",
    explanation:
      "Every node that participates in the Galaxy runs a WinPlatform object (the bootstrap/PlatformInfo). It must be deployed before the engines on it. Platform-to-platform communication problems are the usual root cause of 'cannot communicate with remote node' deployment errors.",
    example:
      "WinPlatform_Node1 represents server SRV-APP01 and hosts AppEngine1 and a ViewEngine.",
    related: ["appengine", "ocmc", "deployment-model"],
    sourceIds: ["pdf-platform-manager", "doc-deploy-errors"],
  },
  {
    id: "historian",
    term: "Historian",
    aliases: ["AVEVA Historian", "Wonderware Historian"],
    topics: ["historian"],
    short:
      "The time-series database that stores historised attribute values for trending, analysis, and reporting.",
    explanation:
      "AVEVA Historian collects tag data via IDAS (and direct App Server historisation), stores it efficiently, and serves it to clients like Historian Client Web/Trend. App Server attributes are historised by enabling history on the attribute and pointing the engine at a Historian. Store-and-forward buffers data during connection loss.",
    example:
      "PMP-101.PV has History enabled; values flow to the Historian and appear in Historian Client Web trends.",
    related: ["idas", "store-forward", "appengine"],
    sourceIds: ["pdf-historian-concepts", "doc-historian-issues"],
  },
  {
    id: "alarmmodecmd",
    term: "AlarmModeCmd",
    topics: ["alarms"],
    short:
      "An attribute command used to set/clear an alarm's mode (e.g. enable, disable, silence/inhibit) programmatically.",
    explanation:
      "AlarmModeCmd lets scripts or clients change the alarm mode of an object attribute — for example to enable, disable, or inhibit alarming. It works together with related attributes such as AlarmInhibit and the configured alarm enable state. Use with care: disabling/inhibiting can hide genuine alarms.",
    example:
      "A maintenance script writes AlarmModeCmd to inhibit a sensor's alarm during calibration, then restores it afterwards.",
    related: ["alarm-inhibit", "shelving"],
    sourceIds: ["doc-alarms-impl", "doc-alarm-inhibit", "pdf-alarm-control"],
  },
  {
    id: "alarm-inhibit",
    term: "AlarmInhibit",
    topics: ["alarms"],
    short:
      "A property that suppresses alarms for an attribute/object so they do not become active or annunciate.",
    explanation:
      "AlarmInhibit prevents an alarm from activating while set. Unlike shelving (operator-driven, time-bounded), inhibit is typically a configured/engineered suppression. Inhibited alarms will not appear in the active list — a frequent reason an alarm 'is configured but never shows'.",
    example:
      "If TIC-101.AlarmInhibit is true, its HI alarm will not annunciate even when the value exceeds the limit.",
    related: ["alarmmodecmd", "shelving"],
    sourceIds: ["doc-alarm-inhibit", "doc-alarms-impl"],
  },
  {
    id: "shelving",
    term: "Shelving",
    topics: ["alarms"],
    short:
      "Temporarily removing an active alarm from the operator's active list for a defined period; it auto-unshelves when the timer expires.",
    explanation:
      "Shelving is an operator action to reduce nuisance alarms for a bounded time. By default Medium and Low severity alarms are shelve-enabled while Critical and High are not, to avoid hiding serious conditions. When the shelve timer ends, the alarm reappears and resumes its state.",
    example:
      "An operator shelves a chattering Low alarm for 30 minutes; it returns automatically afterwards.",
    related: ["alarm-inhibit", "alarmmodecmd"],
    sourceIds: ["pdf-alarm-control", "doc-alarms-impl"],
  },
  {
    id: "onscan-offscan",
    term: "OnScan / OffScan",
    aliases: ["scan state", "ScanState", "ScanStateCmd"],
    topics: ["runtime"],
    short:
      "An object's scan state: OnScan = executing normally; OffScan = idle and not processing.",
    explanation:
      "ScanState controls whether an object runs. OnScan objects perform their normal processing (scripts, I/O); OffScan objects are idle and not available for execution. You change it with ScanStateCmd or via Platform Manager/Object Viewer. After a reboot, engines and objects may need to be set OnScan. OffScan objects show no live data even though they are deployed.",
    example:
      "PMP-101 is deployed but OffScan, so its PV never updates — set it OnScan to resume processing.",
    related: ["appengine", "object-viewer", "quality"],
    sourceIds: ["doc-offscan", "pdf-scripting", "pdf-platform-manager"],
  },
  {
    id: "quality",
    term: "Data Quality (Good / Bad / Uncertain)",
    aliases: ["bad quality", "OPC quality"],
    topics: ["runtime", "di", "troubleshooting"],
    short:
      "An OPC-style quality flag on every attribute value indicating whether the data can be trusted.",
    explanation:
      "Each attribute value carries a quality: Good (trustworthy), Bad (no valid source — comms down, item not found, OffScan source), Uncertain, or Initializing. Bad quality almost always points upstream: the DI/OI link, the I/O reference, or the source object's scan state — not the consuming object itself.",
    example:
      "PMP-101.PV shows Bad in Object Viewer → check the OPCClient/OI Server connection and the I/O reference string.",
    related: ["onscan-offscan", "di-object", "oi-server", "object-viewer"],
    sourceIds: ["pdf-object-viewer", "doc-opc-source"],
  },
  {
    id: "ide",
    term: "ArchestrA IDE",
    aliases: ["Integrated Development Environment"],
    topics: ["object-management", "templates"],
    short:
      "The engineering tool used to build, configure, and deploy the Galaxy (templates, instances, model & deployment views).",
    explanation:
      "The IDE is where engineers do design-time work: create templates and instances, edit attributes and scripts, organise Model and Deployment views, and check objects in/out. Deployment is launched from the IDE; runtime diagnostics happen in OCMC/Object Viewer.",
    example:
      "Connect the IDE to the GR node, open the Galaxy, build $Pump, create PMP-101, assign it to AppEngine1, and deploy.",
    related: ["galaxy", "template", "checkin-checkout"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "checkin-checkout",
    term: "Check-In / Check-Out",
    topics: ["object-management"],
    short:
      "Source-control style locking of objects in the Galaxy: check out to edit, check in to commit and release the lock.",
    explanation:
      "Objects must be checked out to be edited, which locks them to you. Checking in commits changes and makes them available to deploy and to others. An object checked out (especially by another user) cannot be edited or fully deployed by you — a common source of 'config vs runtime' confusion. Undo Check Out discards changes.",
    example:
      "$Pump is checked out by a colleague, so you cannot modify it — they must check it in (or you Undo Check Out as admin).",
    related: ["ide", "deployment-model"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "idas",
    term: "IDAS",
    aliases: ["InTouch Data Acquisition Service", "data acquisition service"],
    topics: ["historian"],
    short:
      "The Historian data acquisition service that collects tag values and forwards them to the Historian storage engine.",
    explanation:
      "IDAS acquires data (from App Server, OI Servers, etc.) and feeds the Historian. It supports store-and-forward: if the link to the Historian is lost, data is buffered locally (e.g. *.dat files) and forwarded when the connection returns. IDAS connection problems are a common reason history stops logging.",
    example:
      "If trends flat-line, check IDAS status and store-forward folders for pending *.dat files.",
    related: ["historian", "store-forward"],
    sourceIds: ["doc-idas-troubleshoot", "doc-idas-sf"],
  },
  {
    id: "store-forward",
    term: "Store-and-Forward",
    topics: ["historian"],
    short:
      "Local buffering of historian data during a connection outage, forwarded once the Historian is reachable again.",
    explanation:
      "Store-and-forward protects against data loss when the path to the Historian drops. Data accumulates locally and is sent when connectivity returns. A historian stuck 'in store-forward' (e.g. SysStatusSFDataPending true) or full SF folders indicate the forward step is failing — often needing an engine stop/start.",
    example:
      "After a network blip, original.dat files sit in the SF folder; once the link recovers they forward and trends back-fill.",
    related: ["idas", "historian"],
    sourceIds: ["doc-idas-sf", "doc-historian-issues"],
  },
  {
    id: "pv-sp",
    term: "PV / SP",
    aliases: ["process value", "setpoint"],
    topics: ["concepts", "runtime"],
    short:
      "PV = Process Value (the measured value); SP = Setpoint (the target value).",
    explanation:
      "On analog/control objects, PV is the live measured value coming from the field and SP is the desired target. Many UDA/field attributes derive from these. Their quality and EU range determine how values display and alarm.",
    example:
      "TIC-101.PV = 78.2 °C, TIC-101.SP = 80 °C; the controller acts to drive PV toward SP.",
    related: ["eu-range", "quality"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "eu-range",
    term: "EU / Extended EU Range",
    aliases: ["engineering units", "EU range"],
    topics: ["concepts", "runtime"],
    short:
      "The engineering-unit min/max scaling for an analog attribute; Extended EU allows values slightly beyond the nominal range.",
    explanation:
      "EU range defines the meaningful min/max of a value in engineering units (e.g. 0–100 °C). Extended EU range permits readings just outside the nominal range without clamping/flagging, useful for over-range conditions. Wrong EU ranges cause values to display or alarm incorrectly.",
    example:
      "A 4–20 mA temperature has EU 0–150 °C; an Extended EU lets it read 152 °C during an over-temp excursion.",
    related: ["pv-sp", "quality"],
    sourceIds: ["pdf-ide"],
  },
  // ---- AVEVA OMI (Operations Management Interface) ----
  {
    id: "omi",
    term: "OMI (Operations Management Interface)",
    aliases: ["AVEVA OMI", "InTouch OMI", "Operations Management Interface"],
    topics: ["omi", "concepts"],
    short:
      "AVEVA's modern operator visualization layer for System Platform — operators run ViewApps built from reusable apps, layouts, and content driven by the Galaxy.",
    explanation:
      "OMI is the runtime visualization experience for System Platform (the successor/companion to classic InTouch windows). Engineers compose ViewApps from layouts, panes, screen profiles, and OMI apps; the ViewApp binds to Galaxy objects so operators see live data, alarms, trends, and navigation. OMI runs on the local client and increasingly via a web client.",
    example:
      "A control room ViewApp shows a plant overview layout with a navigation pane, an alarm app, and Content Presenter panes — all bound to AppEngine-hosted objects.",
    related: ["viewapp", "omi-layout", "omi-app", "omi-screen-profile", "omi-web-client"],
    sourceIds: ["doc-omi-about", "pdf-omi-workshop"],
  },
  {
    id: "viewapp",
    term: "ViewApp",
    aliases: ["$ViewApp", "view application"],
    topics: ["omi", "deployment"],
    short:
      "The deployable OMI application an operator runs — a composition of layouts, screen profiles, apps, and content, hosted by a ViewEngine.",
    explanation:
      "A ViewApp is the OMI equivalent of a runtime application: it defines which layouts and apps appear, the navigation model, and the screen profile(s) for different stations. Like other objects it is configured in the IDE, assigned to a platform/ViewEngine, and deployed. Operators launch the deployed ViewApp to interact with the plant.",
    example:
      "ViewApp 'Plant_Operations' is assigned to the operator station's ViewEngine and deployed; the operator launches it to monitor and control.",
    related: ["omi", "omi-layout", "omi-screen-profile", "appengine"],
    sourceIds: ["doc-omi-deploy-viewapp", "pdf-omi-workshop"],
  },
  {
    id: "omi-layout",
    term: "Layout / Pane (OMI)",
    aliases: ["OMI layout", "layout editor", "pane"],
    topics: ["omi"],
    short:
      "A reusable arrangement of panes that defines where apps and content appear on screen within a ViewApp.",
    explanation:
      "Layouts (built in the Layout Editor) divide the screen into panes; each pane hosts an app or content (e.g. a graphic, alarm app, or Content Presenter). Layouts are reused across screens and screen profiles to keep a consistent operator experience. A layout dropped as content inside another layout is a common advanced pattern (and a source of issues if misconfigured).",
    example:
      "A 3-pane layout: left navigation, center process graphic, bottom alarm strip — reused on every operating screen.",
    related: ["viewapp", "omi-app", "omi-screen-profile"],
    sourceIds: ["doc-omi-nav", "doc-omi-nav-controls"],
  },
  {
    id: "omi-screen-profile",
    term: "Screen Profile (OMI)",
    topics: ["omi"],
    short:
      "Defines how a ViewApp maps to physical displays/monitors for a given station class.",
    explanation:
      "Screen profiles let one ViewApp target different station configurations (single monitor, multi-monitor, different resolutions) by mapping layouts to screens. This is how the same application adapts to a single operator PC versus a multi-screen control desk.",
    example:
      "A 'Control Desk' screen profile maps the overview layout to monitor 1 and detail layouts to monitors 2–3.",
    related: ["viewapp", "omi-layout"],
    sourceIds: ["doc-omi-about", "pdf-omi-workshop"],
  },
  {
    id: "omi-app",
    term: "OMI App",
    aliases: ["app", "OMI application module"],
    topics: ["omi"],
    short:
      "A pluggable visualization module placed in a pane — e.g. navigation, alarms, trends, or Content Presenter.",
    explanation:
      "OMI apps are reusable building blocks dropped into layout panes to deliver specific functionality. AVEVA ships standard apps (navigation, alarms, trend, Content Presenter, etc.) and partners/users can add more. Apps bind to Galaxy data and react to the ViewApp's selected object/context.",
    example:
      "The Alarm app in the bottom pane shows active alarms filtered to the currently navigated area.",
    related: ["content-presenter", "omi-layout", "viewapp"],
    sourceIds: ["doc-omi-about", "doc-omi-nav-controls"],
  },
  {
    id: "content-presenter",
    term: "Content Presenter (OMI App)",
    topics: ["omi"],
    short:
      "A flexible OMI app that displays content (graphics, documents, web content, embedded apps) with configurable layout, filtering, and sizing.",
    explanation:
      "Content Presenter shows context-driven content in a pane: you configure filter-area properties, layout-area properties (fill, view mode, alignment, padding), and size-area properties (columns/rows, viewport). It's commonly used to surface the right graphic or document for the operator's current selection.",
    example:
      "When an operator selects Pump-101, Content Presenter swaps to that pump's detail graphic automatically.",
    related: ["omi-app", "omi-layout"],
    sourceIds: ["doc-omi-about"],
  },
  {
    id: "omi-web-client",
    term: "OMI Web Client",
    topics: ["omi", "security"],
    short:
      "Browser-based access to OMI ViewApps, subject to documented limitations versus the local client.",
    explanation:
      "The OMI web client lets users open ViewApps in a browser without the full local install. Not every OMI app/feature is supported in the web client, and connectivity depends on the web/OPC UA services, certificates, and authentication. Connection failures usually trace to the web services, certificates, or auth rather than the ViewApp itself.",
    example:
      "An operator opens the plant ViewApp from a tablet browser; if it won't connect, check the OMI web services and certificate trust first.",
    related: ["viewapp", "omi"],
    sourceIds: ["doc-omi-webclient-limits", "doc-omi-webclient-troubleshoot"],
  },
  {
    id: "viewengine",
    term: "ViewEngine",
    topics: ["omi", "runtime"],
    short:
      "The runtime engine (like an AppEngine, but for visualization) that hosts and runs deployed ViewApps on a node.",
    explanation:
      "A ViewEngine runs on a WinPlatform and executes ViewApps, much as an AppEngine executes automation objects. A ViewApp must be assigned to a ViewEngine and deployed (and the engine running) for an operator to launch it.",
    example:
      "The operator station's WinPlatform hosts a ViewEngine that runs the deployed 'Plant_Operations' ViewApp.",
    related: ["viewapp", "appengine", "winplatform"],
    sourceIds: ["doc-omi-deploy-viewapp", "pdf-platform-manager"],
  },

  {
    id: "template-toolbox",
    term: "Template Toolbox",
    topics: ["templates", "object-management"],
    short:
      "The IDE panel that organises all templates by toolset, from which you create instances and derived templates.",
    explanation:
      "The Template Toolbox groups templates into toolsets (e.g. System, Device Integration). You right-click a template and choose New → Instance or New → Derived Template. System DI templates like $OPCClient and $DDESuiteLinkClient live here.",
    example:
      "Under Device Integration, right-click $DDESuiteLinkClient → New → Instance to create a DI object.",
    related: ["template", "instance", "di-object"],
    sourceIds: ["pdf-ide", "comm-ddesuitelink"],
  },
];

export const GLOSSARY_BY_ID: Record<string, GlossaryTerm> = Object.fromEntries(
  GLOSSARY.map((g) => [g.id, g])
);
