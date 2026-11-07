"use client";

import { Dialog } from "@base-ui/react/dialog";
import { cn } from "@sentinel-app/ui/lib/utils";
import { useQuery } from "@tanstack/react-query";
import { BarChart3, CircleHelp, CornerDownLeft, MapPin, Play, Search, Wind } from "lucide-react";
import { parseAsBoolean, parseAsString, useQueryStates } from "nuqs";
import {
  type KeyboardEvent,
  type ReactNode,
  useCallback,
  useDeferredValue,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

import { StatusMark, type StatusKind } from "@/components/brand/status-mark";
import { useDemandData } from "@/modes/demand/use-demand-data";
import { useOmAssets } from "@/modes/om/use-om-data";
import { useOverviewFeeders } from "@/modes/overview/use-overview-data";
import { MODE_IDS, MODES } from "@/modes/registry";
import { orpc } from "@/utils/orpc";

import { MODE_COPY } from "./mode-copy";

// ⌘K / Ctrl+K "Jump to…" (#124). Searches views, feeders, turbines, counties, parks and the demo
// script with data the app already queries (same query keys, so it's usually a warm cache), and
// navigates through the existing URL keys. Built on Base UI's Dialog (already in packages/ui).

const NAV = {
  mode: parseAsString,
  feeder: parseAsString,
  park: parseAsString,
  turbine: parseAsString,
  site: parseAsString,
  zone: parseAsString,
  omTab: parseAsString,
  summary: parseAsBoolean,
};
type Nav = Partial<{ [K in keyof typeof NAV]: K extends "summary" ? boolean | null : string | null }>;

// A jump never inherits a selection from the view it leaves (the demo jumps follow the same rule).
const CLEAR: Nav = { feeder: null, park: null, turbine: null, site: null, zone: null, omTab: null, summary: null };

// The demo script's six steps (demo/demo-jumps.tsx owns them; pressing 1–6 runs them).
const DEMO_STEPS = [
  "Production: wind over Romania and WP01's 24 h forecast",
  "Demand: where use peaks this evening",
  "Feeders F7: Sentinel keeps F7 Oșorhei under its limit",
  "String failure at home 1042 becomes a repair job",
  "O&M park: WP01 turbines, crews and the review queue",
  "Summary: the week, Sentinel against the baseline",
];

const FOREVER = Number.POSITIVE_INFINITY;

type Item = {
  id: string;
  group: string;
  label: string;
  hint?: string;
  keywords?: string;
  icon: ReactNode;
  run: () => void;
};

const fold = (text: string) =>
  text
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase();

function score(item: Item, tokens: string[]) {
  const label = fold(item.label);
  const rest = fold(`${item.hint ?? ""} ${item.keywords ?? ""} ${item.group}`);
  let total = 0;
  for (const token of tokens) {
    if (label.startsWith(token)) total += 4;
    else if (label.split(/[\s·,()-]+/).some((word) => word.startsWith(token))) total += 3;
    else if (label.includes(token)) total += 2;
    else if (rest.includes(token)) total += 1;
    else return 0;
  }
  return total;
}

const GROUP_ORDER = ["Views", "Feeders", "Counties", "Turbines", "Wind parks", "Demo script", "Help"];
const GROUP_LIMIT: Record<string, number> = { Turbines: 8, Counties: 6 };

const subscribeNothing = () => () => {};

/** "⌘K" on Apple devices, "Ctrl K" elsewhere. */
export function useShortcutLabel() {
  return useSyncExternalStore(
    subscribeNothing,
    () => (/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "⌘K" : "Ctrl K"),
    () => "⌘K",
  );
}

const TURBINE_MARK: Record<string, StatusKind> = {
  eligible: "ok",
  derated: "warn",
  excluded: "critical",
  human_review: "review",
};

function useItems(close: () => void, onTour: () => void): Item[] {
  const [, go] = useQueryStates(NAV);
  const feeders = useOverviewFeeders();
  const { data: assets } = useOmAssets();
  const { counties } = useDemandData();
  const { data: parks } = useQuery(orpc.production.listParks.queryOptions({ staleTime: FOREVER }));

  return useMemo(() => {
    const jump = (state: Nav) => () => {
      close();
      void go({ ...CLEAR, ...state });
    };
    const items: Item[] = [];
    for (const id of MODE_IDS) {
      const { label, icon: Icon } = MODES[id];
      items.push({
        id: `mode-${id}`,
        group: "Views",
        label,
        hint: MODE_COPY[id].descriptor,
        icon: <Icon aria-hidden className="size-4" strokeWidth={1.8} />,
        run: jump({ mode: id }),
      });
    }
    items.push({
      id: "mode-summary",
      group: "Views",
      label: "Summary",
      hint: "The week against the baseline",
      icon: <BarChart3 aria-hidden className="size-4" strokeWidth={1.8} />,
      run: jump({ summary: true }),
    });
    for (const f of feeders) {
      const status: StatusKind =
        f.status === "over" ? "critical" : f.status === "near" ? "warn" : f.status === "ok" ? "ok" : "unknown";
      items.push({
        id: `feeder-${f.id}`,
        group: "Feeders",
        label: f.name,
        hint: `${f.county} · limit ${Math.round(f.limitKw)} kW${f.peakKw === undefined ? "" : ` · peak ${Math.round(f.peakKw)} kW`}`,
        keywords: `${f.id} feeder cable`,
        icon: <StatusMark kind={status} size="sm" />,
        run: jump({ mode: "feeders", feeder: f.id }),
      });
    }
    for (const county of counties) {
      items.push({
        id: `county-${county.id}`,
        group: "Counties",
        label: county.name,
        hint: `Demand · evening stress ${county.peakStress.toFixed(2)}`,
        keywords: "county judet demand",
        icon: <MapPin aria-hidden className="size-4" strokeWidth={1.8} />,
        run: jump({ mode: "demand", zone: county.id }),
      });
    }
    for (const asset of assets ?? []) {
      if (asset.assetKind !== "turbine") continue;
      const short = asset.id.split(".")[1]?.replace("TID", "T") ?? asset.id;
      const park = asset.parkId ?? "WP01";
      items.push({
        id: `turbine-${asset.id}`,
        group: "Turbines",
        label: `Turbine ${short}`,
        hint: `${park} · ${asset.status.replaceAll("_", " ")}`,
        keywords: `${asset.id} ${asset.name} wind turbine`,
        icon: <StatusMark kind={TURBINE_MARK[asset.status] ?? "unknown"} size="sm" />,
        run: jump({ mode: "om", park, turbine: asset.id }),
      });
    }
    for (const park of parks ?? []) {
      items.push({
        id: `park-${park.id}`,
        group: "Wind parks",
        label: `${park.id} wind park`,
        hint: `${park.name} · forecast and turbines`,
        keywords: "park wind production",
        icon: <Wind aria-hidden className="size-4" strokeWidth={1.8} />,
        run: jump({ mode: "production", park: park.id }),
      });
    }
    DEMO_STEPS.forEach((title, index) => {
      items.push({
        id: `demo-${index + 1}`,
        group: "Demo script",
        label: title,
        hint: `Press ${index + 1}`,
        keywords: "demo story step",
        icon: (
          <span className="grid size-4 place-items-center rounded-[4px] bg-muted font-mono text-[10px] font-semibold text-text-secondary">
            {index + 1}
          </span>
        ),
        run: () => {
          close();
          // demo/demo-jumps.tsx listens for 1–6 on window; run the step exactly as the key would.
          setTimeout(() => window.dispatchEvent(new KeyboardEvent("keydown", { key: String(index + 1) })), 0);
        },
      });
    });
    items.push({
      id: "help-tour",
      group: "Help",
      label: "Show the tour",
      hint: "How to read this screen, in 7 steps",
      keywords: "help guide onboarding how it works",
      icon: <CircleHelp aria-hidden className="size-4" strokeWidth={1.8} />,
      run: () => {
        close();
        onTour();
      },
    });
    return items;
  }, [feeders, counties, assets, parks, go, close, onTour]);
}

function Results({ onClose, onTour }: { onClose: () => void; onTour: () => void }) {
  const [query, setQuery] = useState("");
  const deferred = useDeferredValue(query);
  const [active, setActive] = useState(0);
  const listId = useId();
  const listRef = useRef<HTMLDivElement>(null);
  const items = useItems(onClose, onTour);

  const groups = useMemo(() => {
    const tokens = fold(deferred).split(/\s+/).filter(Boolean);
    const shown = tokens.length
      ? items
          .map((item) => ({ item, s: score(item, tokens) }))
          .filter(({ s }) => s > 0)
          .sort((a, b) => b.s - a.s)
          .map(({ item }) => item)
      : items.filter((item) => ["Views", "Feeders", "Demo script"].includes(item.group));
    const byGroup = new Map<string, Item[]>();
    for (const item of shown) {
      const list = byGroup.get(item.group) ?? [];
      if (list.length < (GROUP_LIMIT[item.group] ?? 6)) list.push(item);
      byGroup.set(item.group, list);
    }
    // With a query, the group holding the best match goes first; otherwise the fixed order.
    const order = tokens.length
      ? [...new Set(shown.map((item) => item.group))]
      : GROUP_ORDER.filter((group) => byGroup.has(group));
    return order.map((group) => [group, byGroup.get(group) ?? []] as const);
  }, [items, deferred]);

  const flat = useMemo(() => groups.flatMap(([, list]) => list), [groups]);
  useEffect(() => setActive(0), [deferred]);
  const current = flat[Math.min(active, flat.length - 1)];

  useEffect(() => {
    if (!current) return;
    listRef.current?.querySelector(`[data-id="${current.id}"]`)?.scrollIntoView({ block: "nearest" });
  }, [current]);

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    const last = flat.length - 1;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((i) => (i >= last ? 0 : i + 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((i) => (i <= 0 ? last : i - 1));
    } else if (event.key === "Home" && event.ctrlKey) {
      setActive(0);
    } else if (event.key === "End" && event.ctrlKey) {
      setActive(last);
    } else if (event.key === "Enter") {
      event.preventDefault();
      current?.run();
    }
  };

  return (
    <>
      <div className="flex items-center gap-3 border-b border-border px-4">
        <Search aria-hidden className="size-[18px] shrink-0 text-muted-foreground" />
        <input
          autoFocus
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
          role="combobox"
          aria-expanded
          aria-controls={listId}
          aria-activedescendant={current ? `${listId}-${current.id}` : undefined}
          aria-autocomplete="list"
          aria-label="Jump to"
          placeholder="Jump to a feeder, turbine, county or view…"
          className="h-14 min-w-0 flex-1 bg-transparent text-[15px] text-foreground outline-none placeholder:text-muted-foreground"
        />
        <kbd className="rounded-sm border border-border bg-surface-raised px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
          esc
        </kbd>
      </div>
      <div
        ref={listRef}
        id={listId}
        role="listbox"
        aria-label="Results"
        className="max-h-[min(440px,58vh)] overflow-y-auto overscroll-contain p-2 [scrollbar-width:thin]"
      >
        {flat.length === 0 ? (
          <p className="px-3 py-8 text-center text-[13px] text-text-secondary">
            Nothing called “{deferred}”. Try F7, Oșorhei, T12, Bihor or Demand.
          </p>
        ) : (
          groups.map(([group, list]) => (
            <div key={group} role="group" aria-label={group} className="pb-1">
              <p className="px-3 pt-2 pb-1 text-[11px] font-semibold tracking-[0.06em] text-muted-foreground uppercase">
                {group}
              </p>
              {list.map((item) => {
                const isActive = item.id === current?.id;
                return (
                  <div
                    key={item.id}
                    id={`${listId}-${item.id}`}
                    data-id={item.id}
                    role="option"
                    aria-selected={isActive}
                    onMouseMove={() => {
                      const index = flat.indexOf(item);
                      if (index !== active) setActive(index);
                    }}
                    onClick={item.run}
                    className={cn(
                      "flex min-h-10 cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-[13px] transition-[background-color,box-shadow] duration-100",
                      isActive ? "bg-surface-raised shadow-raised" : "text-foreground",
                    )}
                  >
                    <span
                      className={cn(
                        "grid size-5 shrink-0 place-items-center",
                        isActive ? "text-brand-700" : "text-text-secondary",
                      )}
                    >
                      {item.icon}
                    </span>
                    <span className="min-w-0 flex-1 truncate">
                      <span className="font-medium text-foreground">{item.label}</span>
                      {item.hint ? <span className="ml-2 text-text-secondary">{item.hint}</span> : null}
                    </span>
                    {isActive ? (
                      <CornerDownLeft aria-hidden className="size-3.5 shrink-0 text-muted-foreground" />
                    ) : null}
                  </div>
                );
              })}
            </div>
          ))
        )}
      </div>
      <div className="flex items-center gap-4 border-t border-border bg-surface-raised px-4 py-2 text-[11px] text-muted-foreground">
        <span>
          <kbd className="font-mono">↑↓</kbd> move
        </span>
        <span>
          <kbd className="font-mono">↵</kbd> open
        </span>
        <span className="ml-auto flex items-center gap-1.5">
          <Play aria-hidden className="size-3" /> Demo steps also run with keys 1–6
        </span>
      </div>
    </>
  );
}

/** The palette dialog plus its global ⌘K / Ctrl+K shortcut. Mount once (the app shell). */
export function CommandPalette({
  open,
  onOpenChange,
  onTour,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onTour: () => void;
}) {
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        onOpenChange(!open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onOpenChange]);

  const close = useCallback(() => onOpenChange(false), [onOpenChange]);

  return (
    <Dialog.Root open={open} onOpenChange={(next) => onOpenChange(next)}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-[60] bg-foreground/15 backdrop-blur-[2px] dark:bg-black/50 transition-opacity duration-150 data-[ending-style]:opacity-0 data-[starting-style]:opacity-0 motion-reduce:transition-none" />
        <Dialog.Popup className="fixed top-[12vh] left-1/2 z-[61] flex w-[min(620px,calc(100vw-32px))] -translate-x-1/2 flex-col overflow-hidden rounded-lg bg-card shadow-overlay outline-none transition-[opacity,scale] duration-180 ease-out data-[ending-style]:scale-[0.98] data-[ending-style]:opacity-0 data-[starting-style]:scale-[0.98] data-[starting-style]:opacity-0 motion-reduce:transition-none">
          <Dialog.Title className="sr-only">Jump to</Dialog.Title>
          <Dialog.Description className="sr-only">
            Search views, feeders, turbines, counties, wind parks and the demo script. Enter opens the highlighted result.
          </Dialog.Description>
          <Results onClose={close} onTour={onTour} />
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
