"use client";

import { cn } from "@wattsentry/ui/lib/utils";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { DEFAULT_THEME, themeName } from "@/lib/palette";

/**
 * Light/Dark switch for the top bar (Cockpit, #181). Dark is the default; light is for bright rooms and
 * projectors. Labelled with the theme it switches to, so it never relies on the icon alone.
 */
export function ThemeToggle({ className }: { className?: string }) {
  const { resolvedTheme, setTheme } = useTheme();
  // The stored theme is only known on the client: render the default until mounted (no hydration mismatch).
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const current = mounted ? themeName(resolvedTheme) : DEFAULT_THEME;
  const next = current === "dark" ? "light" : "dark";
  const Icon = next === "light" ? Sun : Moon;

  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      aria-label={`Switch to the ${next} theme`}
      title={`Switch to the ${next} theme`}
      className={cn(
        "flex items-center gap-1.5 px-3 text-[13px] font-medium text-text-secondary transition-[color,box-shadow,background-color] duration-fast hover:bg-surface-raised hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
        className,
      )}
    >
      <Icon aria-hidden className="size-4" />
      <span className="capitalize">{next}</span>
    </button>
  );
}
