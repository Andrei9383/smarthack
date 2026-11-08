import { createCn } from "cn/config";

// The Calm type scale (globals.css `--text-*`, team/contracts/ui-style.md §3) adds font sizes Tailwind
// doesn't ship. Registered here so cn() treats `text-kpi` etc. as sizes: without it they read as
// colours, and a size next to a colour (`text-kpi text-foreground`) drops one of the two.
export const cn = createCn({
  extend: { classGroups: { "font-size": [{ text: ["caption", "dense", "body", "title", "page", "kpi"] }] } },
});
