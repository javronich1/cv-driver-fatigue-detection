import { KnownIssue } from "./types";

// Problemas conocidos / "gotchas" destilados de readmes oficiales y tech notes de
// comunidad. Enmarcados honestamente como patrones específicos del entorno, no
// como verdades universales. Nombres técnicos y mensajes en inglés.

export const KNOWN_ISSUES: KnownIssue[] = [
  {
    id: "ki-reboot-offscan",
    title: "Tras reiniciar un platform, los engines vuelven OffScan",
    topics: ["runtime", "deployment"],
    environment: "Cualquier nodo tras un reinicio de SO / corte de energía",
    symptom:
      "Tras un reinicio, los objetos no se actualizan y las referencias no resuelven aunque todo esté desplegado.",
    cause:
      "Los engines no se ponen OnScan automáticamente; hasta que lo hagas, la resolución de referencias entre objetos puede fallar.",
    workaround:
      "En Platform Manager, pon cada AppEngine OnScan tras el reinicio. Considera procedimientos/runbooks de arranque para que los operadores lo hagan de forma consistente.",
    status: "by-design",
    sourceIds: ["doc-offscan", "doc-as-resolved"],
    keywords: ["reboot", "offscan", "reference resolution", "after restart"],
  },
  {
    id: "ki-sf-pending-after-upgrade",
    title: "SysStatusSFDataPending true tras un upgrade del Historian",
    topics: ["historian"],
    environment: "Historian Tier-1 / Tier-2 tras un upgrade de versión",
    symptom:
      "Tras un upgrade (p. ej. 2023 P03 → 2023 R2), el system tag SysStatusSFDataPending está en true y las tendencias muestran huecos.",
    cause:
      "Backlog de store-and-forward que quedó pendiente tras el upgrade; el paso de reenvío no está drenando los datos almacenados automáticamente.",
    workaround:
      "Inspecciona las carpetas de store-forward en la fuente por archivos *.dat pendientes (p. ej. original.dat en A000000_001). Para/arranca el engine (Platform Manager) para reanudar el reenvío; confirma que el tag SF vuelve a false. Reportado vía comunidad de AVEVA.",
    status: "known",
    sourceIds: ["doc-idas-sf", "doc-historian-issues"],
    keywords: ["sysstatussfdatapending", "store and forward", "upgrade", "trend gap", "historian backlog"],
  },
  {
    id: "ki-dns-slow-deploy",
    title: "Resolución DNS lenta causa fallos de despliegue intermitentes",
    topics: ["deployment"],
    environment: "Multi-nodo, especialmente DHCP / IPs que cambian",
    symptom:
      "Los despliegues fallan intermitentemente al alcanzar un nodo remoto; a veces tienen éxito al reintentar.",
    cause:
      "Resolución de nombres lenta o inestable. La buena práctica es una respuesta de DNS de 4 segundos o menos; un DNS inestable hace poco confiable la comunicación entre nodos.",
    workaround:
      "Usa un archivo hosts para que los mapeos nombre→IP sean estables si cambian las IP, y asegura que el DNS sea rápido y consistente entre nodos. Guía de tech note de comunidad (InSource TN 1283).",
    status: "known",
    sourceIds: ["comm-insource-deploy", "doc-deploy-errors"],
    keywords: ["dns", "slow deploy", "intermittent", "hosts file", "remote node"],
  },
  {
    id: "ki-shelve-severity",
    title: "Las alarmas Critical/High no se pueden hacer shelving por defecto",
    topics: ["alarms"],
    environment: "Configuración por defecto del Alarm Control",
    symptom:
      "Los operadores pueden hacer shelving de algunas alarmas pero la acción no hace nada para alarmas de severidad Critical/High.",
    cause:
      "Por diseño, solo las alarmas de severidad Medium y Low están habilitadas para shelving por defecto; Critical y High se excluyen para evitar ocultar condiciones serias.",
    workaround:
      "Es un comportamiento de seguridad intencional. Si una alarma Critical/High es molesta, ataca la causa raíz o revisa la racionalización de alarmas en vez de forzar el shelving.",
    status: "by-design",
    sourceIds: ["pdf-alarm-control", "doc-alarms-impl"],
    keywords: ["shelve", "critical alarm", "high alarm", "cannot shelve", "severity"],
  },
  {
    id: "ki-checkout-blocks-deploy",
    title: "Objetos en check out bloquean ediciones y despliegue completo",
    topics: ["object-management", "deployment"],
    environment: "Galaxy con varios ingenieros",
    symptom:
      "Un objeto no se puede editar ni desplegar del todo; las opciones de deploy están deshabilitadas.",
    cause:
      "El objeto está en check out (a menudo por otro ingeniero), bloqueándolo. Los objetos en check out no pueden ser modificados ni desplegados del todo por otros.",
    workaround:
      "Haz que el dueño haga check in del objeto. Un administrador puede hacer Undo Check Out, pero esto descarta los cambios en progreso del dueño: coordina primero.",
    status: "by-design",
    sourceIds: ["pdf-ide"],
    keywords: ["checked out", "locked object", "undo check out", "deploy greyed", "cannot edit"],
  },
  {
    id: "ki-omi-webclient-limits",
    title: "Algunas apps/funciones de OMI no funcionan en el web client",
    topics: ["omi"],
    environment: "OMI web client (navegador)",
    symptom:
      "Una ViewApp que funciona en el cliente local de OMI renderiza parcialmente o se comporta distinto en el web client del navegador.",
    cause:
      "El OMI web client tiene limitaciones documentadas: no todas las apps o funciones de OMI están soportadas en el navegador frente al cliente local.",
    workaround:
      "Revisa la lista oficial 'General OMI web client limitations'. Para funciones no soportadas, usa el cliente local o rediseña la ViewApp con apps soportadas por web. Por diseño según docs de AVEVA.",
    status: "by-design",
    sourceIds: ["doc-omi-webclient-limits"],
    keywords: ["omi web client", "not supported", "browser", "partial render", "viewapp web"],
  },
  {
    id: "ki-omi-layout-as-content",
    title: "Un layout colocado como contenido dentro de otro layout se comporta mal",
    topics: ["omi"],
    environment: "ViewApp de OMI con layouts anidados",
    symptom:
      "Cuando un layout se usa como contenido dentro de otro layout, los panes/navegación no renderizan ni interactúan como se espera.",
    cause:
      "El layout-como-contenido anidado es un patrón avanzado con restricciones de configuración; una mala configuración lleva a problemas de layout/navegación (reportado en comunidad de AVEVA y abordado en varios patches).",
    workaround:
      "Revisa la configuración del layout/pane y la navegación contra las docs de jerarquía de navegación de OMI; simplifica el anidamiento o sigue el patrón soportado de Content Presenter. Revisa los resolved-issues de OMI para tu nivel de patch.",
    status: "known",
    sourceIds: ["doc-omi-nav", "doc-omi-resolved"],
    keywords: ["nested layout", "layout as content", "content presenter", "navigation hierarchy", "panes"],
  },
];

export const KNOWN_ISSUE_BY_ID: Record<string, KnownIssue> = Object.fromEntries(
  KNOWN_ISSUES.map((k) => [k.id, k])
);
