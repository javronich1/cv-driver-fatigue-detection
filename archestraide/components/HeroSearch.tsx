"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { IconSpark, IconArrow } from "./icons";

export default function HeroSearch() {
  const router = useRouter();
  const [q, setQ] = useState("");
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (q.trim()) router.push(`/ask?q=${encodeURIComponent(q.trim())}`);
      }}
      className="panel flex items-center gap-2 p-2 shadow-glow"
    >
      <span className="pl-2 text-accent">
        <IconSpark width={20} height={20} />
      </span>
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Pregunta lo que sea — p. ej. '¿Por qué mi objeto está en Bad quality?'"
        className="flex-1 bg-transparent px-1 py-2.5 text-sm text-slate-100 outline-none placeholder:text-slate-500"
      />
      <button type="submit" className="btn btn-primary px-4 py-2.5">
        Preguntar <IconArrow width={16} height={16} />
      </button>
    </form>
  );
}
