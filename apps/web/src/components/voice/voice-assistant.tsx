"use client";

import {
  ConversationProvider,
  useConversationClientTool,
  useConversationControls,
  useConversationMode,
  useConversationStatus,
} from "@elevenlabs/react";
import { cn } from "@wattsentry/ui/lib/utils";
import { Mic, MicOff, PhoneOff, X } from "lucide-react";
import { parseAsString, useQueryStates } from "nuqs";
import { type ReactNode, useState, useSyncExternalStore } from "react";

import { StatusMark } from "@/components/status-mark";
import { setPlaying } from "@/lib/replay-playing";
import { clampStep, clockLabel, dayOfStep } from "@/lib/replay";
import { usePolicy, useReplayStep } from "@/lib/use-replay";
import { MODE_IDS, MODES, type ModeId } from "@/modes/registry";

import { getDaySummary, getFaults, getFeeders, getWindParks } from "./data-tools";
import { narrator } from "./narrator";
import { useVoiceStatus } from "./use-voice-status";

// "Ask WattSentry" (card #207): a voice conversation with the ElevenLabs agent made by
// scripts/elevenlabs/sync-agent.mjs. It answers questions about the grid and the week on screen
// (feeders, wind parks, faults, results), never about the code. Its facts come from client tools
// that read the app's own API (./data-tools), so answers match the screens.
// The browser never sees the API key: /api/voice/session mints a one-conversation token.

type Said = { id: number; role: "user" | "agent"; text: string };

/* ---------- Small module store: panel open, transcript, last error ---------- */

let state: { open: boolean; said: Said[]; error: string | null; muted: boolean } = {
  open: false,
  said: [],
  error: null,
  muted: false,
};
let nextId = 0;
const listeners = new Set<() => void>();
const update = (patch: Partial<typeof state>) => {
  state = { ...state, ...patch };
  for (const listener of listeners) listener();
};
const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};
const useVoiceState = () => useSyncExternalStore(subscribe, () => state, () => state);

/* ---------- Provider: wraps the shell so the rail button and the card share one session ---------- */

export function VoiceAssistantProvider({ children }: { children: ReactNode }) {
  const { muted } = useVoiceState();
  return (
    <ConversationProvider
      isMuted={muted}
      onMutedChange={(value) => update({ muted: value })}
      onMessage={({ message, role }) => {
        if (!message) return;
        update({ said: [...state.said, { id: nextId++, role, text: message }].slice(-6) });
      }}
      onError={(message) => update({ error: message || "Something went wrong." })}
    >
      <VoiceTools />
      {children}
    </ConversationProvider>
  );
}

const NAV = {
  mode: parseAsString,
  feeder: parseAsString,
  site: parseAsString,
  turbine: parseAsString,
  park: parseAsString,
};

const failed = (error: unknown) =>
  `The lookup failed (${error instanceof Error ? error.message : "unknown error"}). Say you couldn't get the data.`;

/** The agent's client tools: look up the data, open a view, describe what's on screen. */
function VoiceTools() {
  const [nav, setNav] = useQueryStates(NAV);
  const [rawStep] = useReplayStep();
  const [policy] = usePolicy();

  const step = clampStep(rawStep);
  const today = dayOfStep(step);
  useConversationClientTool("get_feeders", (p: { day?: string }) => getFeeders(p.day, today).catch(failed));
  useConversationClientTool("get_wind_parks", () => getWindParks(step).catch(failed));
  useConversationClientTool("get_faults", (p: { day?: string }) => getFaults(p.day, today).catch(failed));
  useConversationClientTool("get_day_summary", (p: { day?: string }) => getDaySummary(p.day, today, step).catch(failed));

  useConversationClientTool("open_view", (params: { mode?: string; feeder?: string }) => {
    const mode = MODE_IDS.find((id) => id === params.mode);
    if (!mode) return `Unknown view "${params.mode}". Views: ${MODE_IDS.join(", ")}.`;
    const feeder = mode === "feeders" && params.feeder ? params.feeder.toUpperCase().replace(/\s+/g, "") : null;
    void setNav({ mode, feeder, site: null, turbine: null });
    return `Opened ${MODES[mode].label}${feeder ? ` with feeder ${feeder} selected` : ""}.`;
  });

  useConversationClientTool("current_view", () => {
    const mode = (MODE_IDS.find((id) => id === nav.mode) ?? "overview") as ModeId;
    const selected = [
      nav.feeder && `feeder ${nav.feeder}`,
      nav.turbine && `turbine ${nav.turbine}`,
      nav.site && `home ${nav.site}`,
      nav.park && `park ${nav.park}`,
    ].filter(Boolean);
    return [
      `View: ${MODES[mode].label} (${MODES[mode].blurb})`,
      `Replay clock: ${clockLabel(clampStep(rawStep))}, week of 2 June 2025.`,
      `Showing: ${policy === "sentinel" ? "with WattSentry" : "the baseline, without WattSentry"}.`,
      selected.length ? `Selected: ${selected.join(", ")}.` : "Nothing selected.",
    ].join(" ");
  });

  return null;
}

/* ---------- Start / stop ---------- */

export function useAskSentinel() {
  const { assistant } = useVoiceStatus();
  const { startSession, endSession } = useConversationControls();
  const { status } = useConversationStatus();
  const [starting, setStarting] = useState(false);
  const live = status === "connected" || status === "connecting" || starting;

  const start = async () => {
    update({ open: true, error: null, said: [] });
    // One voice at a time: the story narrator stops, the replay pauses.
    narrator.stop();
    setPlaying(false);
    setStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      for (const track of stream.getTracks()) track.stop();
    } catch {
      update({ error: "Microphone access is blocked. Allow it in the browser to talk to WattSentry." });
      setStarting(false);
      return;
    }
    try {
      const res = await fetch("/api/voice/session", { cache: "no-store" });
      const body = (await res.json()) as { token?: string; error?: string };
      if (!res.ok || !body.token) throw new Error(body.error ?? "Could not start the conversation.");
      startSession({ conversationToken: body.token, connectionType: "webrtc" });
    } catch (error) {
      update({ error: error instanceof Error ? error.message : "Could not start the conversation." });
    } finally {
      setStarting(false);
    }
  };

  const toggle = () => {
    if (live) endSession();
    else void start();
  };

  return { available: assistant, live, toggle, start, stop: endSession };
}

/* ---------- The conversation card (bottom-left, above the agent toast) ---------- */

export function VoicePanel({ className }: { className?: string }) {
  const { open, said, error, muted } = useVoiceState();
  const { status } = useConversationStatus();
  const { isSpeaking } = useConversationMode();
  const { live, start, stop } = useAskSentinel();
  if (!open) return null;

  const connected = status === "connected";
  const headline = error
    ? "WattSentry can't talk right now"
    : connected
      ? isSpeaking
        ? "WattSentry is speaking"
        : "Listening… ask about the project"
      : live
        ? "Connecting to WattSentry…"
        : "Conversation ended";

  return (
    <section
      aria-label="Talk to WattSentry"
      className={cn(
        "flex w-[380px] max-w-[calc(100%-2rem)] flex-col gap-2 rounded-xl bg-card px-3.5 py-3 shadow-overlay motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2",
        className,
      )}
    >
      <header className="flex items-center gap-2">
        <StatusMark kind={error ? "warn" : connected ? "live" : "unknown"} size="xs" pulse={connected && !isSpeaking} />
        <p role="status" aria-live="polite" className="min-w-0 flex-1 truncate text-[13px] font-semibold text-foreground">
          {headline}
        </p>
        {connected ? (
          <button
            type="button"
            onClick={() => update({ muted: !muted })}
            aria-pressed={muted}
            aria-label={muted ? "Unmute the microphone" : "Mute the microphone"}
            title={muted ? "Unmute" : "Mute"}
            className="grid size-[26px] place-items-center rounded-md text-muted-foreground transition-colors duration-150 hover:bg-surface-raised hover:text-foreground"
          >
            {muted ? <MicOff aria-hidden className="size-3.5" /> : <Mic aria-hidden className="size-3.5" />}
          </button>
        ) : null}
        {live ? (
          <button
            type="button"
            onClick={stop}
            aria-label="End the conversation"
            title="End"
            className="grid size-[26px] place-items-center rounded-md bg-status-critical/10 text-status-critical transition-colors duration-150 hover:bg-status-critical/20"
          >
            <PhoneOff aria-hidden className="size-3.5" />
          </button>
        ) : (
          <button
            type="button"
            onClick={() => update({ open: false, error: null })}
            aria-label="Close"
            className="grid size-[26px] place-items-center rounded-md text-muted-foreground transition-colors duration-150 hover:bg-surface-raised hover:text-foreground"
          >
            <X aria-hidden className="size-3.5" />
          </button>
        )}
      </header>

      {said.length > 0 ? (
        <ol className="flex max-h-56 flex-col gap-1.5 overflow-y-auto text-[13px] leading-snug [scrollbar-width:thin]">
          {said.map((line) => (
            <li
              key={line.id}
              className={cn(
                "max-w-[90%] rounded-lg px-2.5 py-1.5",
                line.role === "user" ? "self-end bg-surface-raised text-text-secondary" : "self-start text-foreground",
              )}
            >
              {line.text}
            </li>
          ))}
        </ol>
      ) : null}

      {error ? <p className="text-xs text-status-critical">{error}</p> : null}
      {!live && !error ? (
        <button
          type="button"
          onClick={() => void start()}
          className="inline-flex h-8 items-center justify-center gap-1.5 self-start rounded-lg bg-primary px-3 text-dense font-semibold text-primary-foreground transition-colors duration-150 hover:bg-primary/85"
        >
          <Mic aria-hidden className="size-3.5" />
          Talk again
        </button>
      ) : null}
      {!said.length && connected ? (
        <p className="text-xs text-text-secondary">
          Try: “How does the agent decide which batteries to charge?” or “Show me the feeders.”
        </p>
      ) : null}
    </section>
  );
}
