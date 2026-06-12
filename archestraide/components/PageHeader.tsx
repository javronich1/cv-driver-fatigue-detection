export default function PageHeader({
  eyebrow,
  title,
  subtitle,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="mb-7">
      {eyebrow && (
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-accent">
          {eyebrow}
        </p>
      )}
      <h1 className="text-2xl font-bold tracking-tight text-slate-100 sm:text-3xl">
        {title}
      </h1>
      {subtitle && (
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
          {subtitle}
        </p>
      )}
    </div>
  );
}
