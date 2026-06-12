import { Runbook } from "./types";

// Runbooks curados orientados a soporte. Cada uno está fundamentado en docs
// oficiales de AVEVA (y tech notes de comunidad claramente etiquetadas). Codifican
// el flujo "qué revisar primero → confirmar → escalar" que seguiría un ingeniero
// de soporte sénior. Son heurísticas, no garantías: la UI las enmarca como
// "más probable / revisa primero".
//
// Los nombres de herramientas, conceptos y los mensajes de error literales se
// mantienen en inglés porque así aparecen en el producto.

export const RUNBOOKS: Runbook[] = [
  {
    id: "rb-deploy-remote-node",
    title: "Falla el despliegue: cannot communicate with remote node",
    category: "Despliegue",
    topics: ["deployment", "runtime", "security"],
    severity: "high",
    symptom:
      "Desplegar un platform/engine falla con un error como 'cannot communicate with remote node', 'unable to contact' o un timeout al alcanzar el nodo destino.",
    likelyCauses: [
      "Resolución de nombres / DNS o discrepancia del archivo hosts entre el nodo GR y el nodo destino",
      "Servicios de ArchestrA / bootstrap no corriendo en el nodo destino",
      "Firewall bloqueando la comunicación de ArchestrA entre nodos",
      "Credenciales de la cuenta de servicio incorrectas, o el nodo no está en el mismo contexto de seguridad aaAdministrators/aaConfigTools",
      "El WinPlatform destino no está desplegado (intentaste desplegar un engine antes que su platform)",
      "Desfase horario entre nodos rompiendo la autenticación",
    ],
    firstTool: "OCMC (SMC)",
    steps: [
      {
        title: "Confirma que puedes resolver y alcanzar el nodo por nombre",
        detail:
          "Desde el nodo GR, haz ping al destino con el nombre exacto usado en el objeto WinPlatform. La respuesta de DNS debe ser rápida (buena práctica ≤ 4 s); un DNS inestable es una causa clásica. Muchos sitios usan un archivo hosts para que el mapeo nombre→IP sea estable si cambian las IP.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-deploy-errors", "comm-insource-deploy"],
      },
      {
        title: "Verifica que el platform esté desplegado primero, luego el engine",
        detail:
          "El orden importa: el WinPlatform debe estar desplegado y corriendo antes que sus AppEngines. En la Deployment View, despliega el platform por separado, confirma que está corriendo y luego despliega los engines/objetos.",
        tool: "ArchestrA IDE",
        sourceIds: ["doc-sp-deployment"],
      },
      {
        title: "Revisa los servicios de ArchestrA / bootstrap en el destino",
        detail:
          "En el nodo destino confirma que los servicios de ArchestrA están corriendo y que las credenciales de logon del servicio son correctas. Verifica que NT SERVICE\\aaPIM esté en el grupo local Administrators y que el ingeniero logueado esté en aaAdministrators y aaConfigTools.",
        tool: "Platform Manager",
        sourceIds: ["comm-insource-deploy", "doc-sp-deployment"],
      },
      {
        title: "Revisa el firewall y los puertos entre nodos",
        detail:
          "Confirma que los puertos de comunicación de ArchestrA estén abiertos en ambos sentidos a través de cualquier firewall de host o de red. Una regla unidireccional hará que el platform parezca alcanzable pero falle al desplegar.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-deploy-errors"],
      },
      {
        title: "Lee el error exacto en Log Viewer en AMBOS nodos",
        detail:
          "Abre OCMC → Log Viewer en el nodo GR y en el nodo destino a la vez y vuelve a desplegar. El log del lado destino suele contener la causa raíz real (seguridad, puerto o credenciales) que el error del IDE oculta.",
        tool: "Log Viewer",
        sourceIds: ["doc-deploy-errors", "pdf-platform-manager"],
      },
      {
        title: "Verifica la sincronización horaria y el contexto de seguridad",
        detail:
          "Un desfase de reloj significativo entre nodos rompe la comunicación autenticada. Confirma que ambos nodos comparten una fuente de tiempo y pertenecen a la misma configuración de seguridad de la Galaxy / contexto de dominio.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-sp-deployment"],
      },
    ],
    confirmResolution:
      "El platform despliega y aparece corriendo en Platform Manager, los engines pasan a OnScan y un re-despliegue de un solo objeto tiene éxito sin el error de comunicación.",
    escalateWhen:
      "Servicios, puertos, credenciales, resolución de nombres y sincronización horaria están todos correctos pero el despliegue sigue dando timeout: captura exports sincronizados de Log Viewer de ambos nodos y escala con el equipo de red/AD.",
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
    title: "Un atributo de objeto muestra Bad quality",
    category: "Bad quality / sin datos",
    topics: ["runtime", "di", "oi", "troubleshooting"],
    severity: "medium",
    symptom:
      "Un atributo (p. ej. PV) muestra calidad Bad (o Uncertain/Initializing) en Object Viewer o en un gráfico, por lo que no se puede confiar en el valor.",
    likelyCauses: [
      "El enlace DI/OI aguas arriba está caído (OI Server no corriendo o sin conexión al dispositivo)",
      "La cadena de referencia de I/O es incorrecta (error de tipeo, item/topic equivocado, objeto fuente equivocado)",
      "El objeto fuente o el engine está OffScan",
      "El item no existe en el namespace del OI Server / OPC",
      "Seguridad o licenciamiento limitando el OI Server",
    ],
    firstTool: "Object Viewer",
    steps: [
      {
        title: "Confirma dónde la calidad pasa a Bad",
        detail:
          "En Object Viewer, observa el atributo y su InputSource/referencia de I/O. La calidad Bad casi siempre se origina aguas arriba (en el DI object o el OI Server), no en el objeto que consume.",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer", "doc-opc-source"],
      },
      {
        title: "Revisa la cadena de referencia de I/O",
        detail:
          "Verifica que el input source del atributo apunte al item correcto del DI object (instance de OPCClient/DDESuiteLinkClient, topic/group e item correctos). Un solo error de tipeo da calidad Bad. Las referencias de autobind pueden quedar desfasadas si cambió el namespace fuente.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide", "doc-opc-source"],
      },
      {
        title: "Verifica el scan state de la fuente",
        detail:
          "Asegúrate de que el DI object, el objeto que consume y su AppEngine estén OnScan. Una fuente OffScan produce calidad Bad/último valor conocido. Tras un reinicio, los engines pueden quedar OffScan.",
        tool: "Platform Manager",
        sourceIds: ["doc-offscan", "pdf-platform-manager"],
      },
      {
        title: "Diagnostica el OI Server directamente",
        detail:
          "En OI Server Manager, revisa que el OI Server esté corriendo y conectado al dispositivo, y usa sus diagnósticos para confirmar que el item específico se actualiza con calidad Good a nivel de driver. Si está Bad aquí, el problema es del lado dispositivo/driver.",
        tool: "OI Server Manager",
        sourceIds: ["doc-opc-source", "doc-opcua-source"],
      },
      {
        title: "Revisa que el item exista y que el dispositivo responda",
        detail:
          "Confirma que el item/tag realmente existe en el namespace del OI Server y que el PLC/dispositivo está accesible y encendido. Un item inexistente o un dispositivo muerto da calidad Bad incluso con referencias perfectas.",
        tool: "OI Server Manager",
        sourceIds: ["doc-opc-source"],
      },
    ],
    confirmResolution:
      "El atributo muestra calidad Good con una marca de tiempo fresca en Object Viewer, y el valor sigue al dispositivo.",
    escalateWhen:
      "Los diagnósticos del OI Server muestran el item Good pero el atributo de App Server sigue Bad tras verificar referencias y scan state: escala con la cadena de referencia, la config del OI Server y la captura de Object Viewer.",
    sourceIds: ["pdf-object-viewer", "doc-opc-source", "doc-offscan"],
    keywords: ["bad quality", "uncertain", "initializing", "no value", "i/o reference", "inputsource"],
  },
  {
    id: "rb-no-data-object-viewer",
    title: "No se ven datos en Object Viewer",
    category: "Bad quality / sin datos",
    topics: ["runtime", "troubleshooting"],
    severity: "medium",
    symptom:
      "Object Viewer muestra el atributo pero sin valor que se actualice, con guiones, o con un valor/marca de tiempo congelados.",
    likelyCauses: [
      "El objeto o su AppEngine está OffScan",
      "El objeto no está realmente desplegado (solo config) o el despliegue está pendiente",
      "Ruta de atributo equivocada observada (template vs instance, instance equivocada)",
      "Engine no corriendo / platform caído",
      "Calidad Bad aguas arriba (ver runbook de Bad quality)",
    ],
    firstTool: "Object Viewer",
    steps: [
      {
        title: "Confirma que el objeto está desplegado y corriendo",
        detail:
          "En Platform Manager, verifica que el WinPlatform está corriendo, el AppEngine está iniciado y OnScan, y el objeto muestra un estado corriendo/desplegado, no 'not deployed' ni 'shut down'.",
        tool: "Platform Manager",
        sourceIds: ["pdf-platform-manager", "doc-offscan"],
      },
      {
        title: "Verifica el scan state",
        detail:
          "Revisa el ScanState del objeto y del engine. OnScan = procesando; OffScan = inactivo sin actualizaciones en vivo. Pon OnScan vía Platform Manager o ScanStateCmd y observa cómo se reanuda el valor.",
        tool: "Platform Manager",
        sourceIds: ["doc-offscan", "pdf-scripting"],
      },
      {
        title: "Confirma que observas la ruta correcta",
        detail:
          "Asegúrate de haber añadido el atributo de la instance desplegada (no el template) y el nombre de instance correcto. Vuelve a añadirlo desde el objeto en marcha para estar seguro.",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer"],
      },
      {
        title: "Revisa la marca de tiempo y la calidad",
        detail:
          "Una marca de tiempo congelada con calidad Good significa que la fuente dejó de actualizarse; calidad Bad significa un problema de comunicación aguas arriba: sigue el runbook de Bad quality.",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer"],
      },
    ],
    confirmResolution:
      "Los valores se actualizan en vivo con una marca de tiempo en movimiento y calidad Good en Object Viewer.",
    escalateWhen:
      "El objeto está confirmado como desplegado, OnScan, en un engine corriendo y correctamente direccionado, pero sigue sin mostrar datos: captura el estado de Platform Manager y Object Viewer y escala.",
    sourceIds: ["pdf-object-viewer", "pdf-platform-manager", "doc-offscan"],
    keywords: ["no data", "no value", "dashes", "stale", "frozen", "not updating", "object viewer"],
  },
  {
    id: "rb-oi-opc-not-updating",
    title: "OI.SIM / cliente OPC no se actualiza",
    category: "Comunicación OI / OPC / DI",
    topics: ["oi", "di", "troubleshooting"],
    severity: "medium",
    symptom:
      "Un OI Server (p. ej. OI.SIM o un driver real) o un objeto cliente OPC está configurado pero los valores no cambian / muestran calidad Bad.",
    likelyCauses: [
      "OI Server no activado/corriendo, o sin suscripciones de cliente",
      "El DI object apunta al nodo de servidor equivocado o al program ID / endpoint UA equivocado",
      "Topic/group o intervalo de actualización mal configurado",
      "Los nombres de items no coinciden con el namespace del OI Server",
      "Desajuste de protocolo (DDE vs SuiteLink, OPC DA vs UA) o problema de seguridad/certificado para OPC UA",
    ],
    firstTool: "OI Server Manager",
    steps: [
      {
        title: "Confirma que el OI Server está corriendo y tiene clientes",
        detail:
          "En OI Server Manager, verifica que el servidor está activado/corriendo y que aparece la conexión de cliente del DI object. Sin suscripciones suele significar que el DI object no está desplegado/OnScan o apunta al servidor equivocado.",
        tool: "OI Server Manager",
        sourceIds: ["doc-opc-source"],
      },
      {
        title: "Valida los items en los diagnósticos del servidor",
        detail:
          "Usa los diagnósticos del OI Server para confirmar que los items específicos se actualizan con calidad Good. Si se actualizan aquí pero no en App Server, el problema es la referencia del DI object, no el driver.",
        tool: "OI Server Manager",
        sourceIds: ["doc-opc-source", "doc-opcua-source"],
      },
      {
        title: "Revisa la configuración del DI object",
        detail:
          "Confirma que el DI object (OPCClient/DDESuiteLinkClient/OPC UA) apunta al nodo de servidor y endpoint/ProgID correctos, con el topic/group, intervalo de actualización y protocolo correctos (se recomienda SuiteLink sobre DDE).",
        tool: "ArchestrA IDE",
        sourceIds: ["comm-ddesuitelink", "doc-opcua-source"],
      },
      {
        title: "Para OPC UA, verifica endpoint, seguridad y certificados",
        detail:
          "OPC UA necesita un endpoint accesible y certificados confiables en ambos extremos. Un certificado rechazado/no confiable o una política de seguridad equivocada bloquea silenciosamente las actualizaciones: revisa el servicio UA y las trust lists.",
        tool: "OI Server Manager",
        sourceIds: ["doc-opcua-service", "doc-opcua-source"],
      },
    ],
    confirmResolution:
      "Los items se actualizan con calidad Good en los diagnósticos del OI Server y los atributos de App Server los siguen en vivo.",
    escalateWhen:
      "Los diagnósticos del driver muestran los items Good y el DI object está bien configurado, pero App Server sigue sin actualizarse: escala con el export de config del OI Server y los detalles de referencia del DI object.",
    sourceIds: ["doc-opc-source", "doc-opcua-source", "comm-ddesuitelink"],
    keywords: ["oi.sim", "opc not updating", "opc ua", "suitelink", "dde", "no subscription", "program id", "endpoint"],
  },
  {
    id: "rb-historian-no-data",
    title: "Datos historizados no aparecen en el Historian",
    category: "Historian / historización",
    topics: ["historian", "troubleshooting"],
    severity: "medium",
    symptom:
      "Un atributo está configurado para history pero las tendencias están planas/vacías en Historian Client Web, o el registro se detuvo.",
    likelyCauses: [
      "History no está realmente habilitado en el atributo, o el engine no tiene un Historian asignado",
      "Engine/objeto OffScan, así que no se producen valores que registrar",
      "Historian atascado en store-and-forward (el paso de reenvío falla): SF data pending",
      "Problema de conexión de IDAS entre la fuente y el Historian",
      "Subsistema de almacenamiento del Historian no corriendo / problema de disco o licencia",
    ],
    firstTool: "Historian Client Web",
    steps: [
      {
        title: "Confirma que history está habilitado y hay un Historian asignado",
        detail:
          "En el IDE, verifica que el atributo tiene history habilitado y que su AppEngine está configurado con el Historian correcto. Sin Historian asignado no hay dónde registrar.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-historian-concepts", "doc-historian-issues"],
      },
      {
        title: "Verifica que la fuente produce valores Good",
        detail:
          "En Object Viewer confirma que el atributo está OnScan y con calidad Good. El Historian registra lo que produce el engine: una fuente OffScan/Bad no deja nada significativo que almacenar.",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer", "doc-offscan"],
      },
      {
        title: "Revisa si hay backlog de store-and-forward",
        detail:
          "Inspecciona las carpetas de store-forward por archivos *.dat pendientes (p. ej. original.dat) y el tag SysStatusSFDataPending. Un historian 'atascado en store-forward' suele requerir parar y arrancar el engine (vía Platform Manager) para reanudar el reenvío.",
        tool: "Platform Manager",
        sourceIds: ["doc-idas-sf", "doc-historian-issues"],
      },
      {
        title: "Diagnostica la conexión de IDAS",
        detail:
          "Confirma que IDAS está conectado y adquiriendo. Usa los pasos de troubleshooting de IDAS del Historian para validar la ruta de adquisición desde la fuente hasta el almacenamiento.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-idas-troubleshoot"],
      },
      {
        title: "Confirma que el motor de almacenamiento del Historian está sano",
        detail:
          "Revisa el estado del Historian (almacenamiento corriendo, espacio en disco, licencia). Si el almacenamiento está caído o sin licencia, los datos recientes no se conservarán.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-historian-issues"],
      },
    ],
    confirmResolution:
      "Los nuevos valores aparecen en tiempo real en las tendencias de Historian Client Web y cualquier backlog de store-forward se drena.",
    escalateWhen:
      "History está habilitado, la fuente está Good/OnScan, IDAS conectado y el almacenamiento sano, pero los datos siguen sin llegar: escala con el contenido de la carpeta SF, el estado de IDAS y la config de history del tag.",
    sourceIds: ["doc-historian-issues", "doc-idas-troubleshoot", "doc-idas-sf", "pdf-historian-concepts"],
    keywords: ["no history", "trend flat", "historian not logging", "store and forward", "sysstatussfdatapending", "idas", "historization missing"],
  },
  {
    id: "rb-alarm-not-visible",
    title: "Alarma configurada pero no visible/activa",
    category: "Alarmas",
    topics: ["alarms", "troubleshooting"],
    severity: "medium",
    symptom:
      "Una alarma está configurada en un atributo pero nunca aparece en el Alarm Control / lista de alarmas activas cuando ocurre la condición.",
    likelyCauses: [
      "Alarma deshabilitada, inhibida (AlarmInhibit) o en shelving",
      "La calidad del atributo es Bad/Uncertain, así que la lógica de alarma no evalúa",
      "El límite/condición de alarma no se alcanza realmente (rango EU/escalado incorrecto)",
      "El filtro del Alarm Client excluye el objeto/área/prioridad",
      "Objeto/engine OffScan, así que la condición de alarma nunca se evalúa",
    ],
    firstTool: "Object Viewer",
    steps: [
      {
        title: "Revisa el estado de enable / inhibit / shelve",
        detail:
          "Verifica que la alarma está habilitada y no inhibida (AlarmInhibit) ni en shelving. Las alarmas inhibidas/en shelving no se anuncian. Recuerda que por defecto el shelving aplica a Medium/Low; Critical/High normalmente no están habilitadas para shelving.",
        tool: "Object Viewer",
        sourceIds: ["doc-alarm-inhibit", "doc-alarms-impl", "pdf-alarm-control"],
      },
      {
        title: "Confirma el valor y la calidad subyacentes",
        detail:
          "En Object Viewer confirma que el atributo está OnScan, con calidad Good, y que realmente cruza el límite configurado. Una calidad Bad o un escalado EU fuera de rango puede impedir que la alarma evalúe como se espera.",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer", "doc-alarms-impl"],
      },
      {
        title: "Revisa el query/filtro del Alarm Client",
        detail:
          "La alarma puede estar activa pero filtrada. Verifica que el query string del Alarm Control incluye la Galaxy/área del objeto y el rango de prioridad/severidad de la alarma.",
        tool: "Object Viewer",
        sourceIds: ["pdf-alarm-control"],
      },
      {
        title: "Verifica que la config de alarma coincide con un objeto que funciona",
        detail:
          "Compara con un atributo cuya alarma sí funciona. Los logs del SMC a veces muestran por qué la descripción/condición de alarma de un atributo concreto no aparece en runtime.",
        tool: "Log Viewer",
        sourceIds: ["doc-alarms-sysobjects", "doc-alarms-impl"],
      },
    ],
    confirmResolution:
      "Llevar el valor más allá del límite produce una alarma activa visible en el Alarm Control, y se limpia/vuelve a normal correctamente.",
    escalateWhen:
      "La alarma está habilitada, no inhibida/en shelving, el valor es Good y supera el límite, y el query la incluye, pero aun así nunca se activa: escala con la config de alarma y una captura de los logs del SMC.",
    sourceIds: ["doc-alarms-impl", "doc-alarm-inhibit", "pdf-alarm-control", "doc-alarms-sysobjects"],
    keywords: ["alarm not showing", "alarm not active", "alarminhibit", "shelved", "alarmmodecmd", "no alarm", "alarm missing"],
  },
  {
    id: "rb-onscan-offscan",
    title: "Confusión OnScan vs OffScan de AppEngine / objeto",
    category: "Platform / AppEngine / scan state",
    topics: ["runtime", "troubleshooting"],
    severity: "low",
    symptom:
      "Los objetos están desplegados pero nada corre/actualiza; los scripts no se ejecutan; las referencias no resuelven, a menudo tras un reinicio o una parada manual.",
    likelyCauses: [
      "AppEngine u objeto quedó OffScan (inactivo, sin ejecutar)",
      "Engines no puestos OnScan tras un reinicio de nodo, rompiendo la resolución de referencias",
      "Platform detenido, así que todos los engines alojados están caídos",
      "Objeto desplegado pero nunca puesto OnScan",
    ],
    firstTool: "Platform Manager",
    steps: [
      {
        title: "Revisa los scan states de platform, engine y objeto",
        detail:
          "En Platform Manager revisa la jerarquía: ¿platform corriendo? ¿engine iniciado y OnScan? ¿objeto OnScan? OnScan significa procesamiento normal; OffScan significa inactivo/sin ejecutar.",
        tool: "Platform Manager",
        sourceIds: ["pdf-platform-manager", "doc-offscan"],
      },
      {
        title: "Pon los engines OnScan tras un reinicio",
        detail:
          "Tras reiniciar un platform debes poner cada engine OnScan. No hacerlo causa problemas de resolución de referencias en los objetos. Pon el engine OnScan y deja que los objetos se reanuden.",
        tool: "Platform Manager",
        sourceIds: ["doc-offscan", "doc-as-resolved"],
      },
      {
        title: "Usa ScanStateCmd / scripts OnScan intencionalmente",
        detail:
          "ScanStateCmd alterna un objeto OnScan/OffScan. Los scripts OnScan corren la primera vez que un engine ejecuta el objeto tras pasar a OnScan, útil para inicialización. No dejes objetos OffScan salvo que estén intencionalmente inactivos.",
        tool: "Object Viewer",
        sourceIds: ["pdf-scripting", "doc-offscan"],
      },
    ],
    confirmResolution:
      "Platform, engines y objetos están todos OnScan; los valores se actualizan, los scripts corren y las referencias resuelven.",
    escalateWhen:
      "Los engines están OnScan y el platform corriendo pero los objetos aún no ejecutan ni resuelven referencias: captura el estado de Platform Manager y Log Viewer y escala.",
    sourceIds: ["doc-offscan", "pdf-platform-manager", "pdf-scripting"],
    keywords: ["onscan", "offscan", "scanstate", "scanstatecmd", "not running", "after reboot", "reference resolution"],
  },
  {
    id: "rb-checkin-version-mismatch",
    title: "Desajuste de versión: check out / check in / despliegue",
    category: "Check-in / check-out / config vs runtime",
    topics: ["object-management", "deployment", "troubleshooting"],
    severity: "low",
    symptom:
      "Los cambios no aparecen en runtime, el deploy está deshabilitado, un objeto no se puede editar, o el comportamiento en runtime no coincide con la última configuración.",
    likelyCauses: [
      "Objeto en check out (por ti o por alguien más) bloqueando ediciones/deploy",
      "Config editada sin check in, así que la versión desplegable está obsoleta",
      "Objeto desplegado en una versión anterior: la config cambió pero no se re-desplegó",
      "Undeploy/redeploy pendiente o un cascade deploy parcial",
    ],
    firstTool: "ArchestrA IDE",
    steps: [
      {
        title: "Revisa el estado de check out del objeto y su dueño",
        detail:
          "En el IDE, mira si el objeto está en check out y por quién. Un objeto en check out no puede ser desplegado/editado del todo por otros. El dueño debe hacer check in (o un admin puede hacer Undo Check Out, descartando los cambios).",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
      {
        title: "Haz check in de los cambios pendientes, luego despliega",
        detail:
          "Las ediciones de config solo se vuelven desplegables tras el check in. Haz check in del objeto (y de cualquier template modificado), luego despliega para que la versión de runtime coincida con la configuración.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
      {
        title: "Compara la versión desplegada vs la configurada",
        detail:
          "Confirma que la versión desplegada es igual a la última versión con check in. Si la config cambió desde el último deploy, re-despliega el objeto (cascade deploy si cambiaron templates).",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide", "doc-sp-deployment"],
      },
    ],
    confirmResolution:
      "El objeto tiene check in, la versión desplegada es igual a la configurada, y el comportamiento en runtime refleja los últimos cambios.",
    escalateWhen:
      "Las versiones coinciden y el objeto tiene check in/re-deploy pero el runtime sigue difiriendo: captura el estado de despliegue del objeto y escala (posible corrupción de Galaxy/objeto).",
    sourceIds: ["pdf-ide", "doc-sp-deployment"],
    keywords: ["checked out", "check in", "undo check out", "version mismatch", "config vs runtime", "deploy greyed out", "changes not applied"],
  },
  {
    id: "rb-csv-import-conflict",
    title: "Import desde CSV / manejo de conflictos de package",
    category: "Import / export / CSV",
    topics: ["csv", "object-management", "troubleshooting"],
    severity: "low",
    symptom:
      "Un import a la Galaxy (CSV/aaPKG/Galaxy dump) falla, se aplica parcialmente o lanza avisos de conflicto de objeto/template.",
    likelyCauses: [
      "Template/objeto ya existe con una definición distinta (conflicto de versión)",
      "Errores de columna/formato del CSV o de ruta de atributo",
      "Falta un parent template requerido (orden/dependencia del import)",
      "Objetos en check out, bloqueando la modificación durante el import",
      "Problemas de codificación / locale en el CSV",
    ],
    firstTool: "ArchestrA IDE",
    steps: [
      {
        title: "Lee con cuidado el aviso de conflicto y elige la intención",
        detail:
          "Al importar, el IDE pregunta cómo resolver conflictos (skip / overwrite / create new). Decide deliberadamente: sobrescribir un template afecta a todos los objetos derivados. En producción, prefiere importar primero a una Galaxy de prueba.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
      {
        title: "Asegura que existan dependencias/padres y el orden correcto",
        detail:
          "Importa los base templates antes que los derived templates e instances. Un parent template faltante causa fallos u objetos huérfanos.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
      {
        title: "Valida la estructura del CSV",
        detail:
          "Confirma que los encabezados de columna, rutas de atributos, tipos de datos y codificación coinciden con lo que espera el importador. Una sola fila malformada puede abortar o aplicar parcialmente el import: revisa el log del import.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
      {
        title: "Asegura que los objetos destino tengan check in",
        detail:
          "Los objetos en check out por otro usuario no pueden ser modificados por el import. Asegúrate de que tengan check in (o haz Undo Check Out) antes de reintentar.",
        tool: "ArchestrA IDE",
        sourceIds: ["pdf-ide"],
      },
    ],
    confirmResolution:
      "El import completa sin conflictos sin resolver, el log del import está limpio y los objetos importados abren y despliegan correctamente.",
    escalateWhen:
      "Dependencias, formato y estado de check out están todos correctos pero el import sigue fallando o corrompe objetos: respalda la Galaxy, captura el log del import y escala.",
    sourceIds: ["pdf-ide"],
    keywords: ["csv import", "aapkg", "package conflict", "galaxy import", "overwrite template", "import failed", "export import"],
  },
  {
    id: "rb-security-login",
    title: "Confusión de login / autenticación de seguridad",
    category: "Seguridad / login / autenticación",
    topics: ["security", "troubleshooting"],
    severity: "medium",
    symptom:
      "No se puede iniciar sesión en la Galaxy/runtime, faltan permisos, o 'access denied' al desplegar o abrir el IDE.",
    likelyCauses: [
      "El modo de seguridad de la Galaxy (None / Galaxy / OS Group / OS User) no coincide con cómo te autenticas",
      "Usuario no está en el grupo correcto aaAdministrators / aaConfigTools / grupo de OS",
      "Desajuste de cuenta de dominio vs local, o contraseña/cuenta expirada",
      "Modelo de seguridad no desplegado tras un cambio",
      "Desfase horario o problema de trust de dominio rompiendo la autenticación entre nodos",
    ],
    firstTool: "ArchestrA IDE",
    steps: [
      {
        title: "Identifica el modo de seguridad de la Galaxy",
        detail:
          "Revisa el modo de seguridad configurado (None, Galaxy, OS Group based, OS User based). Cómo debes iniciar sesión, y qué cuentas tienen derechos, depende enteramente de este modo.",
        tool: "ArchestrA IDE",
        sourceIds: ["doc-sp-deployment"],
      },
      {
        title: "Verifica la pertenencia a grupos de la cuenta",
        detail:
          "Confirma que el usuario está en los grupos requeridos: aaAdministrators y aaConfigTools para ingeniería, además de cualquier grupo de OS que el modelo de seguridad mapee a roles. NT SERVICE\\aaPIM debe estar en el grupo local Administrators de cada nodo.",
        tool: "ArchestrA IDE",
        sourceIds: ["comm-insource-deploy", "doc-sp-deployment"],
      },
      {
        title: "Re-despliega el modelo de seguridad tras cambios",
        detail:
          "Los cambios de seguridad/roles deben desplegarse para tener efecto en runtime. Si cambiaste el modelo pero no lo desplegaste, el runtime sigue aplicando las reglas antiguas.",
        tool: "ArchestrA IDE",
        sourceIds: ["doc-sp-deployment"],
      },
      {
        title: "Revisa la salud de cuenta/dominio y la sincronización horaria",
        detail:
          "Confirma que la cuenta no esté bloqueada/expirada, que el trust de dominio esté sano y que los relojes de los nodos estén sincronizados. La autenticación entre nodos falla con un desfase horario significativo.",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-sp-deployment"],
      },
    ],
    confirmResolution:
      "El usuario inicia sesión con el rol esperado, puede abrir/editar/desplegar según lo permitido, y no ocurren errores de access denied.",
    escalateWhen:
      "Modo, pertenencia a grupos y despliegue son correctos y las cuentas están sanas pero la autenticación sigue fallando: escala con el modo de seguridad, las membresías de grupo y los errores de auth de Log Viewer a AD/seguridad.",
    sourceIds: ["doc-sp-deployment", "comm-insource-deploy"],
    keywords: ["login failed", "access denied", "authentication", "security mode", "aaadministrators", "aaconfigtools", "permissions", "galaxy security"],
  },
  {
    id: "rb-omi-viewapp-deploy",
    title: "Una ViewApp de OMI no despliega o los cambios no aparecen",
    category: "OMI / ViewApp",
    topics: ["omi", "deployment", "troubleshooting"],
    severity: "medium",
    symptom:
      "Una ViewApp de OMI falla al desplegar, no se lanza, o los cambios de configuración (layouts, apps, contenido) no aparecen en la estación de operador.",
    likelyCauses: [
      "ViewApp no asignada a un ViewEngine, o el platform/ViewEngine no está desplegado y corriendo",
      "ViewApp en check out o modificada pero sin check in / re-deploy (desajuste de versión)",
      "Layout, screen profile u OMI app referenciado falta, está roto o sin desplegar",
      "Las referencias de gráficos/contenido resuelven a objetos con calidad Bad (problema de datos aguas arriba)",
      "Caché del cliente/versión vieja de la ViewApp aún corriendo en la estación",
    ],
    firstTool: "ArchestrA IDE",
    steps: [
      {
        title: "Confirma la asignación y que el ViewEngine está corriendo",
        detail:
          "En la Deployment View, verifica que la ViewApp está asignada a un ViewEngine en el WinPlatform de la estación de operador, y que el platform y el ViewEngine están desplegados y corriendo (OnScan) en Platform Manager.",
        tool: "Platform Manager",
        sourceIds: ["doc-omi-deploy-viewapp", "pdf-platform-manager"],
      },
      {
        title: "Haz check in de los cambios, luego re-despliega la ViewApp",
        detail:
          "Las ediciones de la ViewApp solo tienen efecto tras el check in y el (re)deploy. Confirma que tiene check in y que la versión desplegada coincide con la configuración; haz cascade deploy si cambiaron layouts/apps compartidos.",
        tool: "ArchestrA IDE",
        sourceIds: ["doc-omi-deploy-viewapp", "pdf-ide"],
      },
      {
        title: "Valida layouts, screen profiles y apps referenciados",
        detail:
          "Abre la ViewApp y confirma que cada layout, screen profile y OMI app referenciado existe y es válido. Un layout colocado como contenido dentro de otro layout es un punto problemático conocido: verifica que esté bien configurado.",
        tool: "ArchestrA IDE",
        sourceIds: ["doc-omi-nav", "doc-omi-nav-controls"],
      },
      {
        title: "Revisa la calidad de datos detrás de los gráficos",
        detail:
          "Si la ViewApp se lanza pero los paneles salen en blanco/rojo, los objetos enlazados pueden estar en calidad Bad u OffScan. Usa Object Viewer para confirmar que los atributos subyacentes están Good/OnScan (ver el runbook de Bad quality).",
        tool: "Object Viewer",
        sourceIds: ["pdf-object-viewer", "doc-offscan"],
      },
      {
        title: "Relanza el cliente para limpiar una ViewApp obsoleta",
        detail:
          "Cierra y relanza la ViewApp en la estación para que cargue la versión recién desplegada en lugar de una en caché. Confirma que la versión/marca de tiempo se actualiza.",
        tool: "Platform Manager",
        sourceIds: ["doc-omi-resolved", "doc-omi-issues"],
      },
    ],
    confirmResolution:
      "La ViewApp despliega, el ViewEngine la ejecuta y los últimos layouts/apps/contenido aparecen con datos en vivo de calidad Good en la estación.",
    escalateWhen:
      "Asignación, check-in/redeploy, referencias y calidad de datos están todos correctos pero la ViewApp sigue sin desplegar o refrescar: captura Log Viewer en la estación y el estado de despliegue de la ViewApp y escala.",
    sourceIds: ["doc-omi-deploy-viewapp", "doc-omi-issues", "doc-omi-resolved"],
    keywords: [
      "viewapp won't deploy",
      "omi changes not showing",
      "viewapp not launching",
      "layout missing",
      "screen profile",
      "viewengine",
    ],
  },
  {
    id: "rb-omi-webclient",
    title: "El OMI web client no conecta o no carga una ViewApp",
    category: "OMI / ViewApp",
    topics: ["omi", "security", "troubleshooting"],
    severity: "medium",
    symptom:
      "El OMI web client falla al conectar, muestra una ViewApp en blanco/parcial, o da error al cargar en el navegador.",
    likelyCauses: [
      "Servicios web de OMI / servicio OPC UA no corriendo o inaccesibles desde el cliente",
      "Certificado no confiable entre el navegador/cliente y el servicio web/UA",
      "Desajuste de autenticación / modo de login para el web client",
      "Uso de una función/app no soportada en el web client (limitaciones documentadas)",
      "Problema de red/firewall o de URL/puerto al endpoint del web client",
    ],
    firstTool: "OCMC (SMC)",
    steps: [
      {
        title: "Confirma que los servicios web de OMI y el servicio OPC UA corren",
        detail:
          "Verifica que los servicios del OMI web client y el servicio OPC UA están desplegados y corriendo en el servidor, y accesibles desde la máquina cliente (URL/puerto).",
        tool: "OCMC (SMC)",
        sourceIds: ["doc-omi-webclient-troubleshoot", "doc-opcua-service"],
      },
      {
        title: "Revisa la confianza del certificado",
        detail:
          "Un certificado rechazado/no confiable bloquea silenciosamente el web client. Confirma que el certificado es válido y confiable para el navegador/cliente y que el certificado del endpoint UA está aceptado.",
        tool: "OI Server Manager",
        sourceIds: ["doc-omi-webclient-troubleshoot", "doc-opcua-service"],
      },
      {
        title: "Verifica la configuración de autenticación / login",
        detail:
          "Confirma que el modo de login del web client y las credenciales coinciden con la configuración de seguridad de la Galaxy. Los desajustes de auth se presentan como fallos de conexión/login más que como errores de la ViewApp.",
        tool: "ArchestrA IDE",
        sourceIds: ["doc-omi-webclient-troubleshoot", "doc-sp-deployment"],
      },
      {
        title: "Descarta funciones no soportadas del web client",
        detail:
          "Si solo carga parte de la ViewApp, revisa las limitaciones documentadas del OMI web client: algunas apps/funciones no están soportadas en el navegador y requieren el cliente local.",
        tool: "ArchestrA IDE",
        sourceIds: ["doc-omi-webclient-limits"],
      },
    ],
    confirmResolution:
      "El navegador conecta, autentica y carga la ViewApp con datos en vivo y las apps soportadas renderizando correctamente.",
    escalateWhen:
      "Los servicios corren, el certificado es confiable, la auth es correcta y las funciones usadas están soportadas, pero el web client sigue fallando: captura el error del navegador, los logs del servidor y escala.",
    sourceIds: ["doc-omi-webclient-troubleshoot", "doc-omi-webclient-limits"],
    keywords: [
      "omi web client",
      "web client won't connect",
      "browser viewapp",
      "certificate",
      "opc ua service",
      "web login failed",
    ],
  },
];

export const RUNBOOK_BY_ID: Record<string, Runbook> = Object.fromEntries(
  RUNBOOKS.map((r) => [r.id, r])
);
