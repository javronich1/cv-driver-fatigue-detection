# ArchestrAide

### Copilot de Soporte para AVEVA Application Server & OMI

ArchestrAide es un copilot interno de soporte, **fundamentado en recuperación**
(retrieval-grounded), para equipos de ingeniería y soporte que trabajan con
**AVEVA Application Server, OMI y System Platform**. Ofrece respuestas
estructuradas y citadas, troubleshooting guiado, runbooks, búsqueda de
documentación, glosario, carga de manuales propios y aportes de la comunidad.

> Internal, retrieval-grounded AI support copilot for AVEVA Application Server /
> OMI / System Platform. UI in Spanish; AVEVA concept/tool names kept in English.

## 🌐 Demo en vivo

- **Netlify:** https://archestraid.netlify.app

## 📁 La aplicación

Todo el código de la app vive en **[`archestraide/`](./archestraide)** (Next.js 14
+ TypeScript + Tailwind). Consulta **[`archestraide/README.md`](./archestraide/README.md)**
para la arquitectura, cómo correrla localmente, las variables de entorno, el
modelo de fundamentación/seguridad y la guía de ingesta de manuales.

```bash
cd archestraide
npm install
npm run dev     # http://localhost:3000
```

## 🚀 Despliegue

El sitio se despliega como **export estático** en Netlify (config en
[`netlify.toml`](./netlify.toml)). La app funciona 100% del lado del cliente
(recuperación + composición de respuestas fundamentadas), por lo que el deploy
es completamente funcional sin backend. Si se configura una `ANTHROPIC_API_KEY`,
la ruta opcional `/api/ask` añade síntesis de respuestas con Claude.

## ✨ Capacidades

- **Preguntar** — respuestas estructuradas y citadas (concepto vs troubleshooting).
- **Troubleshooting** — asistente guiado tipo checklist con avisos por entorno.
- **Runbooks** — playbooks de soporte para Application Server y OMI.
- **Docs / Buscar** — búsqueda híbrida sobre todo el conocimiento.
- **Manuales** — sube tus PDFs de capacitación (se procesan en el navegador).
- **Comunidad** — aporta problemas/soluciones, guardados como GitHub Issues y
  compartidos con todos.
- **Glosario** y **Problemas conocidos**.
