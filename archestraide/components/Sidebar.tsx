"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  IconSpark,
  IconChat,
  IconWrench,
  IconBook,
  IconSearch,
  IconList,
  IconAlert,
  IconInfo,
  IconSun,
  IconMoon,
  IconLayers,
  IconUsers,
} from "./icons";

const NAV = [
  { href: "/", label: "Inicio", icon: IconSpark },
  { href: "/ask", label: "Preguntar", icon: IconChat },
  { href: "/troubleshoot", label: "Troubleshooting", icon: IconWrench },
  { href: "/runbooks", label: "Runbooks", icon: IconBook },
  { href: "/docs", label: "Docs / Buscar", icon: IconSearch },
  { href: "/manuals", label: "Manuales", icon: IconLayers },
  { href: "/community", label: "Comunidad", icon: IconUsers },
  { href: "/glossary", label: "Glosario", icon: IconList },
  { href: "/known-issues", label: "Problemas conocidos", icon: IconAlert },
  { href: "/about", label: "Ajustes / Acerca de", icon: IconInfo },
];

function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  useEffect(() => {
    const saved = (localStorage.getItem("aa-theme") as "dark" | "light") || "dark";
    setTheme(saved);
    document.documentElement.classList.toggle("light", saved === "light");
  }, []);
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("aa-theme", next);
    document.documentElement.classList.toggle("light", next === "light");
  };
  return (
    <button onClick={toggle} className="nav-link w-full" aria-label="Toggle theme">
      {theme === "dark" ? <IconSun /> : <IconMoon />}
      <span>{theme === "dark" ? "Modo claro" : "Modo oscuro"}</span>
    </button>
  );
}

export default function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Mobile top bar */}
      <div className="sticky top-0 z-30 flex items-center justify-between border-b border-white/[0.06] bg-base-950/80 px-4 py-3 backdrop-blur md:hidden">
        <Brand />
        <button
          onClick={() => setOpen((o) => !o)}
          className="btn-ghost btn px-3 py-1.5"
          aria-label="Menu"
        >
          <IconList />
        </button>
      </div>

      <aside
        className={`${
          open ? "block" : "hidden"
        } fixed inset-x-0 top-[57px] z-20 border-b border-white/[0.06] bg-base-950/95 p-3 backdrop-blur md:static md:top-0 md:block md:w-64 md:shrink-0 md:border-b-0 md:border-r md:bg-transparent md:p-4`}
      >
        <div className="mb-6 hidden px-2 md:block">
          <Brand />
        </div>
        <nav className="space-y-1">
          {NAV.map((item) => {
            const active =
              item.href === "/"
                ? pathname === "/"
                : pathname.startsWith(item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                className={`nav-link ${active ? "nav-link-active" : ""}`}
              >
                <Icon />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="mt-6 border-t border-white/[0.06] pt-3">
          <ThemeToggle />
          <p className="px-3 pt-3 text-[11px] leading-relaxed text-slate-500">
            Fundamentado en docs oficiales de AVEVA + runbooks curados. Verifica
            siempre contra tu entorno.
          </p>
        </div>
      </aside>
    </>
  );
}

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-2.5">
      <span className="grid h-8 w-8 place-items-center rounded-lg bg-accent/15 text-accent shadow-[inset_0_0_0_1px_rgba(45,212,191,0.3)]">
        <IconSpark width={18} height={18} />
      </span>
      <span className="flex flex-col leading-none">
        <span className="text-[15px] font-bold tracking-tight text-slate-100">
          Archestr<span className="text-accent">Aide</span>
        </span>
        <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500">
          Copilot de Soporte AVEVA
        </span>
      </span>
    </Link>
  );
}
