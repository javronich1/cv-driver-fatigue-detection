import { GlossaryTerm } from "./types";

// Glosario curado. Las definiciones están fundamentadas en la documentación
// oficial de AVEVA y en la Guía del Application Server. Conciso primero, con
// ejemplos prácticos y enlaces a términos relacionados.
//
// NOTA: Los nombres de conceptos técnicos se mantienen en inglés (Galaxy,
// Template, AppEngine, ViewApp, OnScan, etc.) porque así aparecen en el producto.

export const GLOSSARY: GlossaryTerm[] = [
  {
    id: "galaxy",
    term: "Galaxy",
    aliases: ["galaxy database", "GR"],
    topics: ["concepts", "object-management"],
    short:
      "El único namespace lógico y la base de datos que contiene toda una aplicación de Application Server: todos los templates, instances y configuración.",
    explanation:
      "Una Galaxy es la aplicación completa: cada template, cada object instance, la seguridad y la topología de despliegue viven dentro de ella. La aloja el nodo Galaxy Repository (GR) y se edita desde el ArchestrA IDE. Piénsala como la 'base de datos del proyecto' de System Platform: una Galaxy = un namespace de aplicación.",
    example:
      "Una planta podría tener una Galaxy llamada 'PLANT_PROD'. Los ingenieros conectan el IDE al nodo GR y abren PLANT_PROD para construir y desplegar la aplicación.",
    related: ["template", "instance", "ide", "ocmc"],
    sourceIds: ["pdf-ide", "doc-sp-deployment"],
  },
  {
    id: "template",
    term: "Template",
    aliases: ["base template", "$template"],
    topics: ["templates", "object-management"],
    short:
      "Una definición de objeto reutilizable (prefijada con $) usada para crear instances o derived templates.",
    explanation:
      "Los templates definen atributos, scripts y comportamiento una sola vez para poder reutilizarlos. Los templates nunca se despliegan para ejecutarse: son planos. Se muestran con el prefijo $ (p. ej. $Pump). Cambiar un template propaga a todo lo derivado o instanciado de él (sujeto a bloqueos).",
    example:
      "$Pump define Speed, Status y un script de Start/Stop. Creas docenas de pump instances a partir de él en lugar de reconstruir cada una.",
    related: ["derived-template", "instance", "template-toolbox"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "derived-template",
    term: "Derived Template",
    topics: ["templates"],
    short:
      "Un template creado a partir de otro template, que hereda y especializa a su padre.",
    explanation:
      "La derivación permite construir una jerarquía de reutilización: un base template captura el comportamiento común y los derived templates añaden o bloquean detalles. El bloqueo de atributos en el padre controla qué pueden cambiar los derived templates e instances.",
    example:
      "$Pump (base) → $Pump_VFD (derivado, añade control de velocidad VFD) → instances PMP-101, PMP-102.",
    related: ["template", "instance"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "instance",
    term: "Instance",
    aliases: ["object instance", "automation object"],
    topics: ["object-management", "runtime"],
    short:
      "Un objeto concreto y desplegable creado a partir de un template: lo que realmente se ejecuta.",
    explanation:
      "Las instances son los automation objects reales que se asignan a un AppEngine y se despliegan para ejecutarse. Cada instance tiene sus propios valores de atributos pero hereda la estructura de su template. Solo las instances se ejecutan en runtime; los templates no.",
    example:
      "PMP-101 es una instance de $Pump asignada a AppEngine1 en WinPlatform_Node1.",
    related: ["template", "appengine", "winplatform", "autobind"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "plant-model",
    term: "Plant Model (Model View)",
    aliases: ["model view"],
    topics: ["object-management", "concepts"],
    short:
      "La jerarquía lógica/física de la planta: qué equipos existen y cómo se organizan (estilo ISA-95).",
    explanation:
      "La Model View responde a '¿qué es y dónde encaja en la planta?'. Organiza los objetos por área / contención de equipos (Enterprise → Site → Area → Unit). Es independiente de qué computador ejecuta cada objeto.",
    example:
      "Site → Area 'Boiler House' → Unit 'Boiler 1' → PMP-101. Esto no dice nada sobre qué nodo ejecuta PMP-101.",
    related: ["deployment-model", "instance"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "deployment-model",
    term: "Deployment Model (Deployment View)",
    aliases: ["deployment view"],
    topics: ["deployment", "runtime"],
    short:
      "La topología de ejecución: qué Platform y AppEngine aloja cada objeto en runtime.",
    explanation:
      "La Deployment View responde a '¿dónde se ejecuta?'. Muestra los WinPlatforms (nodos), los AppEngines en ellos y los objetos alojados por cada engine. El mismo objeto aparece en ambas vistas; la Model View es organizativa y la Deployment View es de ejecución.",
    example:
      "WinPlatform_Node1 → AppEngine1 → objeto Area → PMP-101. PMP-101 es el mismo objeto que ves en la Model View, solo que mostrado por dónde se ejecuta.",
    related: ["plant-model", "appengine", "winplatform"],
    sourceIds: ["pdf-ide", "doc-sp-deployment"],
  },
  {
    id: "di-object",
    term: "DI Object (Device Integration Object)",
    aliases: ["device integration object", "DDESuiteLinkClient", "OPCClient"],
    topics: ["di", "oi"],
    short:
      "Un objeto de Application Server que conecta la Galaxy a una fuente de datos (OI/DA Server) para que los datos de campo fluyan en ambos sentidos.",
    explanation:
      "Los DI Objects son el puente entre los automation objects y la capa de comunicación. Los más comunes son $OPCClient, $DDESuiteLinkClient y los objetos cliente OPC UA. Definen el nodo de la fuente de datos, el topic/group y el protocolo usado para hablar con un OI Server o DA Server.",
    example:
      "Un DI Object $OPCClient apunta a un OI Server en el mismo nodo; los atributos de la bomba se enlazan a sus items mediante referencias de I/O.",
    related: ["oi-server", "opcclient", "ddesuitelinkclient", "autobind"],
    sourceIds: ["doc-opc-source", "comm-ddesuitelink", "pdf-ide"],
  },
  {
    id: "oi-server",
    term: "OI Server (Operations Integration Server)",
    aliases: ["DA Server", "communication driver"],
    topics: ["oi", "di"],
    short:
      "El proceso driver de comunicación que habla el protocolo del dispositivo/PLC y expone los datos a los clientes vía SuiteLink/OPC.",
    explanation:
      "Los OI Servers (antes DAServers) hablan protocolos como Modbus, OPC UA, Siemens, etc. por un lado y presentan un namespace de items uniforme a System Platform por el otro. Se configuran y diagnostican en el OI Server Manager (un snap-in dentro de OCMC/SMC).",
    example:
      "OI.MBTCP se conecta a un PLC Modbus; un DI Object se suscribe a sus items y los atributos de System Platform se resuelven a través de él.",
    related: ["di-object", "opcclient", "ocmc"],
    sourceIds: ["doc-opc-source", "doc-opcua-source"],
  },
  {
    id: "opcclient",
    term: "OPCClient",
    aliases: ["$OPCClient", "OPC client object"],
    topics: ["di", "oi"],
    short:
      "Un template de DI Object que conecta la Galaxy a un servidor OPC (DA/UA) como cliente.",
    explanation:
      "El objeto OPCClient define el nodo del servidor OPC y el program ID (o endpoint UA), el intervalo de actualización y los groups/items. Los atributos de los automation objects referencian sus items para leer/escribir datos en vivo.",
    example:
      "La instance $OPCClient 'OPC_PLC1' apunta a la interfaz OPC del OI Server; PMP-101.PV usa una referencia de I/O a OPC_PLC1.Tag.",
    related: ["di-object", "oi-server", "ddesuitelinkclient"],
    sourceIds: ["doc-opc-source", "doc-opcua-source"],
  },
  {
    id: "ddesuitelinkclient",
    term: "DDESuiteLinkClient",
    aliases: ["$DDESuiteLinkClient", "SuiteLink client"],
    topics: ["di", "oi"],
    short:
      "Un DI Object que se conecta a OI/DA Servers usando el protocolo SuiteLink (o el antiguo DDE).",
    explanation:
      "Se encuentra entre los templates de System / Device Integration como $DDESuiteLinkClient. Configuras el nodo del servidor, el protocolo de comunicación (se recomienda SuiteLink sobre DDE), los topics y la opción 'detect connection alarm'. Las instances comienzan en la carpeta Unassigned Host y luego se asignan y despliegan.",
    example:
      "Crea una instance de $DDESuiteLinkClient, fija el protocolo a SuiteLink, apúntala al nodo del OI Server y añade un mapeo de topic al dispositivo.",
    related: ["di-object", "oi-server", "opcclient"],
    sourceIds: ["comm-ddesuitelink", "pdf-ide"],
  },
  {
    id: "autobind",
    term: "Autobind",
    topics: ["di", "object-management"],
    short:
      "Una función que crea y enlaza automáticamente referencias de atributos (p. ej. I/O) a un DI Object, evitando la entrada manual de referencias.",
    explanation:
      "Autobind acelera el cableado de grandes cantidades de atributos a un dispositivo generando automáticamente las referencias de fuente de I/O según convenciones de nombres, en lugar de escribir cada referencia a mano. Se usa habitualmente al levantar un DI object contra un namespace de items estructurado.",
    example:
      "En vez de fijar manualmente PV.InputSource para 500 tags, autobind genera las referencias contra el namespace de items del OPCClient.",
    related: ["di-object", "opcclient", "instance"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "object-viewer",
    term: "Object Viewer",
    topics: ["runtime", "troubleshooting"],
    short:
      "Una herramienta de diagnóstico de runtime que muestra los valores de atributos en vivo, la calidad (quality) y las marcas de tiempo de los objetos desplegados.",
    explanation:
      "Object Viewer es la primera herramienta para el troubleshooting de runtime. Añades atributos a una watch list y ves su valor en vivo, la calidad del dato (Good/Bad/Uncertain/Initializing) y la marca de tiempo. También permite fijar el ScanState y escribir valores para pruebas.",
    example:
      "Abre Object Viewer desde el IDE sobre PMP-101, observa PV: si la calidad es Bad, el problema está aguas arriba (DI/OI), no en el display.",
    related: ["quality", "onscan-offscan", "ocmc"],
    sourceIds: ["pdf-object-viewer"],
  },
  {
    id: "ocmc",
    term: "OCMC / SMC",
    aliases: ["System Management Console", "Operations Control Management Console"],
    topics: ["runtime", "troubleshooting"],
    short:
      "La consola de gestión que aloja los snap-ins de Log Viewer, Platform Manager, OI Server Manager y administración del Historian.",
    explanation:
      "OCMC (Operations Control Management Console; históricamente System Management Console / SMC) es el lugar central para operaciones y diagnóstico. Snap-ins clave: Log Viewer (logs de ArchestrA), Platform Manager (arrancar/parar platforms y engines, scan state), OI Server Manager y administración del Historian.",
    example:
      "Cuando falla un despliegue, abre OCMC → Log Viewer en ambos nodos para leer los mensajes de error de ArchestrA.",
    related: ["object-viewer", "appengine", "winplatform"],
    sourceIds: ["pdf-platform-manager"],
  },
  {
    id: "appengine",
    term: "AppEngine",
    topics: ["runtime", "deployment"],
    short:
      "El proceso host de runtime que ejecuta un grupo de automation objects según una planificación de scan.",
    explanation:
      "Un AppEngine se ejecuta en un WinPlatform y es el contenedor que realmente ejecuta los objetos (corriendo sus scripts y su I/O) con un periodo de scan configurado. Los objetos deben asignarse a un engine, y el engine debe estar desplegado y OnScan para ejecutarse. Tras reiniciar un nodo debes poner los engines OnScan o tendrás problemas de resolución de referencias.",
    example:
      "AppEngine1 aloja los objetos del área Boiler House con un scan de 1000 ms; si AppEngine1 está OffScan, ninguno de esos objetos se actualiza.",
    related: ["winplatform", "onscan-offscan", "instance"],
    sourceIds: ["pdf-platform-manager", "doc-offscan"],
  },
  {
    id: "winplatform",
    term: "WinPlatform",
    aliases: ["platform", "platform object"],
    topics: ["runtime", "deployment"],
    short:
      "El objeto que representa un nodo (computador físico/virtual) en la Galaxy; aloja uno o más AppEngines.",
    explanation:
      "Cada nodo que participa en la Galaxy ejecuta un objeto WinPlatform (el bootstrap/PlatformInfo). Debe desplegarse antes que los engines que tiene encima. Los problemas de comunicación entre platforms son la causa habitual de los errores de despliegue 'cannot communicate with remote node'.",
    example:
      "WinPlatform_Node1 representa el servidor SRV-APP01 y aloja AppEngine1 y un ViewEngine.",
    related: ["appengine", "ocmc", "deployment-model"],
    sourceIds: ["pdf-platform-manager", "doc-deploy-errors"],
  },
  {
    id: "historian",
    term: "Historian",
    aliases: ["AVEVA Historian", "Wonderware Historian"],
    topics: ["historian"],
    short:
      "La base de datos de series temporales que almacena los valores de atributos historizados para tendencias, análisis y reportes.",
    explanation:
      "AVEVA Historian recopila datos de tags vía IDAS (e historización directa de App Server), los almacena de forma eficiente y los sirve a clientes como Historian Client Web/Trend. Los atributos de App Server se historizan habilitando history en el atributo y apuntando el engine a un Historian. El store-and-forward almacena datos durante pérdidas de conexión.",
    example:
      "PMP-101.PV tiene history habilitado; los valores fluyen al Historian y aparecen en las tendencias de Historian Client Web.",
    related: ["idas", "store-forward", "appengine"],
    sourceIds: ["pdf-historian-concepts", "doc-historian-issues"],
  },
  {
    id: "alarmmodecmd",
    term: "AlarmModeCmd",
    topics: ["alarms"],
    short:
      "Un atributo de comando usado para fijar/limpiar el modo de una alarma (p. ej. habilitar, deshabilitar, silenciar/inhibir) de forma programática.",
    explanation:
      "AlarmModeCmd permite que scripts o clientes cambien el modo de alarma de un atributo de objeto, por ejemplo para habilitar, deshabilitar o inhibir la alarma. Funciona junto con atributos relacionados como AlarmInhibit y el estado de habilitación configurado. Úsalo con cuidado: deshabilitar/inhibir puede ocultar alarmas reales.",
    example:
      "Un script de mantenimiento escribe AlarmModeCmd para inhibir la alarma de un sensor durante la calibración y luego la restaura.",
    related: ["alarm-inhibit", "shelving"],
    sourceIds: ["doc-alarms-impl", "doc-alarm-inhibit", "pdf-alarm-control"],
  },
  {
    id: "alarm-inhibit",
    term: "AlarmInhibit",
    topics: ["alarms"],
    short:
      "Una propiedad que suprime las alarmas de un atributo/objeto para que no se activen ni anuncien.",
    explanation:
      "AlarmInhibit evita que una alarma se active mientras está fijada. A diferencia del shelving (acción del operador, acotada en el tiempo), el inhibit suele ser una supresión configurada/ingenierizada. Las alarmas inhibidas no aparecen en la lista activa: una razón frecuente de que una alarma 'esté configurada pero nunca aparezca'.",
    example:
      "Si TIC-101.AlarmInhibit es true, su alarma HI no se anunciará aunque el valor supere el límite.",
    related: ["alarmmodecmd", "shelving"],
    sourceIds: ["doc-alarm-inhibit", "doc-alarms-impl"],
  },
  {
    id: "shelving",
    term: "Shelving",
    topics: ["alarms"],
    short:
      "Quitar temporalmente una alarma activa de la lista activa del operador durante un periodo definido; se des-archiva automáticamente al expirar el temporizador.",
    explanation:
      "El shelving es una acción del operador para reducir alarmas molestas por un tiempo acotado. Por defecto, las alarmas de severidad Medium y Low están habilitadas para shelving, mientras que Critical y High no, para evitar ocultar condiciones serias. Cuando termina el temporizador, la alarma reaparece y retoma su estado.",
    example:
      "Un operador hace shelving de una alarma Low intermitente durante 30 minutos; vuelve automáticamente después.",
    related: ["alarm-inhibit", "alarmmodecmd"],
    sourceIds: ["pdf-alarm-control", "doc-alarms-impl"],
  },
  {
    id: "onscan-offscan",
    term: "OnScan / OffScan",
    aliases: ["scan state", "ScanState", "ScanStateCmd"],
    topics: ["runtime"],
    short:
      "El scan state de un objeto: OnScan = ejecutándose normalmente; OffScan = inactivo y sin procesar.",
    explanation:
      "El ScanState controla si un objeto se ejecuta. Los objetos OnScan hacen su procesamiento normal (scripts, I/O); los OffScan están inactivos y no disponibles para ejecución. Lo cambias con ScanStateCmd o vía Platform Manager/Object Viewer. Tras un reinicio, engines y objetos pueden necesitar ponerse OnScan. Un objeto OffScan no muestra datos en vivo aunque esté desplegado.",
    example:
      "PMP-101 está desplegado pero OffScan, así que su PV nunca se actualiza: ponlo OnScan para reanudar el procesamiento.",
    related: ["appengine", "object-viewer", "quality"],
    sourceIds: ["doc-offscan", "pdf-scripting", "pdf-platform-manager"],
  },
  {
    id: "quality",
    term: "Data Quality (Good / Bad / Uncertain)",
    aliases: ["bad quality", "OPC quality"],
    topics: ["runtime", "di", "troubleshooting"],
    short:
      "Una bandera de calidad estilo OPC en cada valor de atributo que indica si el dato es confiable.",
    explanation:
      "Cada valor de atributo lleva una calidad: Good (confiable), Bad (sin fuente válida: comunicación caída, item no encontrado, fuente OffScan), Uncertain o Initializing. La calidad Bad casi siempre apunta aguas arriba: el enlace DI/OI, la referencia de I/O o el scan state del objeto fuente, no al objeto que consume.",
    example:
      "PMP-101.PV muestra Bad en Object Viewer → revisa la conexión del OPCClient/OI Server y la cadena de referencia de I/O.",
    related: ["onscan-offscan", "di-object", "oi-server", "object-viewer"],
    sourceIds: ["pdf-object-viewer", "doc-opc-source"],
  },
  {
    id: "ide",
    term: "ArchestrA IDE",
    aliases: ["Integrated Development Environment"],
    topics: ["object-management", "templates"],
    short:
      "La herramienta de ingeniería usada para construir, configurar y desplegar la Galaxy (templates, instances, Model y Deployment views).",
    explanation:
      "El IDE es donde los ingenieros hacen el trabajo de diseño: crear templates e instances, editar atributos y scripts, organizar las Model y Deployment views, y hacer check in/out de los objetos. El despliegue se lanza desde el IDE; el diagnóstico de runtime ocurre en OCMC/Object Viewer.",
    example:
      "Conecta el IDE al nodo GR, abre la Galaxy, construye $Pump, crea PMP-101, asígnala a AppEngine1 y despliega.",
    related: ["galaxy", "template", "checkin-checkout"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "checkin-checkout",
    term: "Check-In / Check-Out",
    topics: ["object-management"],
    short:
      "Bloqueo estilo control de versiones de los objetos en la Galaxy: haces check out para editar y check in para confirmar y liberar el bloqueo.",
    explanation:
      "Los objetos deben estar en check out para editarse, lo que los bloquea para ti. El check in confirma los cambios y los deja disponibles para desplegar y para otros. Un objeto en check out (sobre todo por otro usuario) no puede ser editado ni desplegado del todo por ti: una fuente habitual de confusión 'config vs runtime'. Undo Check Out descarta los cambios.",
    example:
      "$Pump está en check out por un compañero, así que no puedes modificarlo: debe hacer check in (o tú haces Undo Check Out como admin).",
    related: ["ide", "deployment-model"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "idas",
    term: "IDAS",
    aliases: ["InTouch Data Acquisition Service", "data acquisition service"],
    topics: ["historian"],
    short:
      "El servicio de adquisición de datos del Historian que recopila valores de tags y los reenvía al motor de almacenamiento del Historian.",
    explanation:
      "IDAS adquiere datos (de App Server, OI Servers, etc.) y alimenta el Historian. Soporta store-and-forward: si se pierde el enlace con el Historian, los datos se almacenan localmente (p. ej. archivos *.dat) y se reenvían al volver la conexión. Los problemas de conexión de IDAS son una causa común de que el history deje de registrar.",
    example:
      "Si las tendencias se aplanan, revisa el estado de IDAS y las carpetas de store-forward por archivos *.dat pendientes.",
    related: ["historian", "store-forward"],
    sourceIds: ["doc-idas-troubleshoot", "doc-idas-sf"],
  },
  {
    id: "store-forward",
    term: "Store-and-Forward",
    topics: ["historian"],
    short:
      "Almacenamiento local de datos del historian durante un corte de conexión, reenviados cuando el Historian vuelve a estar accesible.",
    explanation:
      "El store-and-forward protege contra la pérdida de datos cuando se cae la ruta al Historian. Los datos se acumulan localmente y se envían al volver la conectividad. Un historian atascado 'en store-forward' (p. ej. SysStatusSFDataPending true) o carpetas SF llenas indican que el paso de reenvío está fallando, a menudo requiriendo parar/arrancar el engine.",
    example:
      "Tras un corte de red, archivos original.dat quedan en la carpeta SF; al recuperarse el enlace se reenvían y las tendencias se rellenan.",
    related: ["idas", "historian"],
    sourceIds: ["doc-idas-sf", "doc-historian-issues"],
  },
  {
    id: "pv-sp",
    term: "PV / SP",
    aliases: ["process value", "setpoint"],
    topics: ["concepts", "runtime"],
    short:
      "PV = Process Value (el valor medido); SP = Setpoint (el valor objetivo).",
    explanation:
      "En objetos analógicos/de control, PV es el valor medido en vivo proveniente del campo y SP es el objetivo deseado. Muchos atributos UDA/de campo derivan de ellos. Su calidad y rango EU determinan cómo se muestran y alarman los valores.",
    example:
      "TIC-101.PV = 78.2 °C, TIC-101.SP = 80 °C; el controlador actúa para llevar PV hacia SP.",
    related: ["eu-range", "quality"],
    sourceIds: ["pdf-ide"],
  },
  {
    id: "eu-range",
    term: "EU / Extended EU Range",
    aliases: ["engineering units", "EU range"],
    topics: ["concepts", "runtime"],
    short:
      "El escalado mín/máx en unidades de ingeniería de un atributo analógico; el Extended EU permite valores ligeramente fuera del rango nominal.",
    explanation:
      "El rango EU define el mín/máx significativo de un valor en unidades de ingeniería (p. ej. 0–100 °C). El Extended EU range permite lecturas justo fuera del rango nominal sin recortar/marcar, útil para condiciones de sobre-rango. Rangos EU incorrectos hacen que los valores se muestren o alarmen mal.",
    example:
      "Una temperatura 4–20 mA tiene EU 0–150 °C; un Extended EU permite leer 152 °C durante una excursión de sobre-temperatura.",
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
      "La capa moderna de visualización para operadores de System Platform: los operadores ejecutan ViewApps construidas con apps, layouts y contenido reutilizables y gobernados por la Galaxy.",
    explanation:
      "OMI es la experiencia de visualización en runtime de System Platform (sucesor/compañero de las ventanas clásicas de InTouch). Los ingenieros componen ViewApps a partir de layouts, panes, screen profiles y OMI apps; la ViewApp se enlaza a los objetos de la Galaxy para que los operadores vean datos en vivo, alarmas, tendencias y navegación. OMI corre en el cliente local y, cada vez más, vía un web client.",
    example:
      "Una ViewApp de sala de control muestra un layout de visión general de planta con un pane de navegación, una alarm app y panes de Content Presenter, todos enlazados a objetos alojados en AppEngine.",
    related: ["viewapp", "omi-layout", "omi-app", "omi-screen-profile", "omi-web-client"],
    sourceIds: ["doc-omi-about", "pdf-omi-workshop"],
  },
  {
    id: "viewapp",
    term: "ViewApp",
    aliases: ["$ViewApp", "view application"],
    topics: ["omi", "deployment"],
    short:
      "La aplicación OMI desplegable que ejecuta un operador: una composición de layouts, screen profiles, apps y contenido, alojada por un ViewEngine.",
    explanation:
      "Una ViewApp es el equivalente OMI de una aplicación de runtime: define qué layouts y apps aparecen, el modelo de navegación y el/los screen profile(s) para distintas estaciones. Como otros objetos, se configura en el IDE, se asigna a un platform/ViewEngine y se despliega. Los operadores lanzan la ViewApp desplegada para interactuar con la planta.",
    example:
      "La ViewApp 'Plant_Operations' se asigna al ViewEngine de la estación de operador y se despliega; el operador la lanza para monitorear y controlar.",
    related: ["omi", "omi-layout", "omi-screen-profile", "appengine"],
    sourceIds: ["doc-omi-deploy-viewapp", "pdf-omi-workshop"],
  },
  {
    id: "omi-layout",
    term: "Layout / Pane (OMI)",
    aliases: ["OMI layout", "layout editor", "pane"],
    topics: ["omi"],
    short:
      "Una disposición reutilizable de panes que define dónde aparecen las apps y el contenido en pantalla dentro de una ViewApp.",
    explanation:
      "Los layouts (creados en el Layout Editor) dividen la pantalla en panes; cada pane aloja una app o contenido (p. ej. un gráfico, una alarm app o un Content Presenter). Los layouts se reutilizan entre pantallas y screen profiles para mantener una experiencia de operador consistente. Un layout colocado como contenido dentro de otro layout es un patrón avanzado común (y una fuente de problemas si se configura mal).",
    example:
      "Un layout de 3 panes: navegación a la izquierda, gráfico de proceso al centro, franja de alarmas abajo; reutilizado en cada pantalla operativa.",
    related: ["viewapp", "omi-app", "omi-screen-profile"],
    sourceIds: ["doc-omi-nav", "doc-omi-nav-controls"],
  },
  {
    id: "omi-screen-profile",
    term: "Screen Profile (OMI)",
    topics: ["omi"],
    short:
      "Define cómo una ViewApp se mapea a los displays/monitores físicos para una clase de estación dada.",
    explanation:
      "Los screen profiles permiten que una sola ViewApp se adapte a distintas configuraciones de estación (un monitor, multi-monitor, distintas resoluciones) mapeando layouts a pantallas. Así la misma aplicación se ajusta a un PC de operador único o a un escritorio de control multi-pantalla.",
    example:
      "Un screen profile 'Control Desk' mapea el layout de visión general al monitor 1 y los layouts de detalle a los monitores 2–3.",
    related: ["viewapp", "omi-layout"],
    sourceIds: ["doc-omi-about", "pdf-omi-workshop"],
  },
  {
    id: "omi-app",
    term: "OMI App",
    aliases: ["app", "OMI application module"],
    topics: ["omi"],
    short:
      "Un módulo de visualización conectable que se coloca en un pane: p. ej. navegación, alarmas, tendencias o Content Presenter.",
    explanation:
      "Las OMI apps son bloques reutilizables que se sueltan en los panes de un layout para entregar funcionalidad específica. AVEVA incluye apps estándar (navegación, alarmas, trend, Content Presenter, etc.) y socios/usuarios pueden añadir más. Las apps se enlazan a datos de la Galaxy y reaccionan al objeto/contexto seleccionado de la ViewApp.",
    example:
      "La Alarm app en el pane inferior muestra las alarmas activas filtradas al área navegada actualmente.",
    related: ["content-presenter", "omi-layout", "viewapp"],
    sourceIds: ["doc-omi-about", "doc-omi-nav-controls"],
  },
  {
    id: "content-presenter",
    term: "Content Presenter (OMI App)",
    topics: ["omi"],
    short:
      "Una OMI app flexible que muestra contenido (gráficos, documentos, contenido web, apps embebidas) con layout, filtrado y dimensionado configurables.",
    explanation:
      "Content Presenter muestra contenido según el contexto en un pane: configuras las propiedades del área de filtro, del área de layout (fill, view mode, alineación, padding) y del área de tamaño (columnas/filas, viewport). Se usa comúnmente para mostrar el gráfico o documento correcto según la selección actual del operador.",
    example:
      "Cuando un operador selecciona Pump-101, Content Presenter cambia automáticamente al gráfico de detalle de esa bomba.",
    related: ["omi-app", "omi-layout"],
    sourceIds: ["doc-omi-about"],
  },
  {
    id: "omi-web-client",
    term: "OMI Web Client",
    topics: ["omi", "security"],
    short:
      "Acceso a las ViewApps de OMI desde el navegador, sujeto a limitaciones documentadas frente al cliente local.",
    explanation:
      "El OMI web client permite abrir ViewApps en un navegador sin la instalación local completa. No todas las apps/funciones de OMI están soportadas en el web client, y la conectividad depende de los servicios web/OPC UA, los certificados y la autenticación. Los fallos de conexión suelen deberse a los servicios web, los certificados o la autenticación más que a la ViewApp en sí.",
    example:
      "Un operador abre la ViewApp de planta desde el navegador de una tablet; si no conecta, revisa primero los servicios web de OMI y la confianza del certificado.",
    related: ["viewapp", "omi"],
    sourceIds: ["doc-omi-webclient-limits", "doc-omi-webclient-troubleshoot"],
  },
  {
    id: "viewengine",
    term: "ViewEngine",
    topics: ["omi", "runtime"],
    short:
      "El motor de runtime (como un AppEngine, pero para visualización) que aloja y ejecuta las ViewApps desplegadas en un nodo.",
    explanation:
      "Un ViewEngine corre en un WinPlatform y ejecuta ViewApps, igual que un AppEngine ejecuta automation objects. Una ViewApp debe asignarse a un ViewEngine y desplegarse (con el engine en marcha) para que un operador pueda lanzarla.",
    example:
      "El WinPlatform de la estación de operador aloja un ViewEngine que ejecuta la ViewApp 'Plant_Operations' desplegada.",
    related: ["viewapp", "appengine", "winplatform"],
    sourceIds: ["doc-omi-deploy-viewapp", "pdf-platform-manager"],
  },

  {
    id: "template-toolbox",
    term: "Template Toolbox",
    topics: ["templates", "object-management"],
    short:
      "El panel del IDE que organiza todos los templates por toolset, desde el cual creas instances y derived templates.",
    explanation:
      "El Template Toolbox agrupa los templates en toolsets (p. ej. System, Device Integration). Haces clic derecho en un template y eliges New → Instance o New → Derived Template. Los templates de DI del sistema como $OPCClient y $DDESuiteLinkClient viven aquí.",
    example:
      "En Device Integration, clic derecho en $DDESuiteLinkClient → New → Instance para crear un DI object.",
    related: ["template", "instance", "di-object"],
    sourceIds: ["pdf-ide", "comm-ddesuitelink"],
  },
];

export const GLOSSARY_BY_ID: Record<string, GlossaryTerm> = Object.fromEntries(
  GLOSSARY.map((g) => [g.id, g])
);
