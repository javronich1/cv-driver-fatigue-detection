import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark neutral base + a single intelligent accent (signal teal/cyan)
        base: {
          950: "#070a0f",
          900: "#0b0f17",
          850: "#0f1521",
          800: "#141b2a",
          700: "#1c2535",
          600: "#28344a",
          500: "#3a485f",
        },
        accent: {
          DEFAULT: "#2dd4bf",
          soft: "#5eead4",
          deep: "#0d9488",
          glow: "rgba(45, 212, 191, 0.15)",
        },
        signal: {
          info: "#38bdf8",
          warn: "#fbbf24",
          danger: "#f87171",
          ok: "#34d399",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 8px 30px -12px rgba(0,0,0,0.6)",
        glow: "0 0 0 1px rgba(45,212,191,0.25), 0 8px 40px -12px rgba(45,212,191,0.25)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.4s ease-out both",
        "pulse-soft": "pulse-soft 1.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
