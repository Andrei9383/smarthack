import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";

import agent from "./agent.json";

// Server-only helpers for the voice routes (card #207). The key lives in ELEVENLABS_API_KEY
// (.env.local at the repo root, loaded via "varlock.loadPath" in apps/web/package.json) and never
// reaches the browser.

/** The "Ask Sentinel" agent made by scripts/elevenlabs/sync-agent.mjs; the env var overrides it. */
export const AGENT_ID = process.env.ELEVENLABS_AGENT_ID ?? agent.agentId;
/** One voice for the whole app, so the story narrator and the assistant sound the same. */
export const VOICE_ID = process.env.ELEVENLABS_VOICE_ID ?? agent.voiceId;

export const voiceEnabled = () => Boolean(process.env.ELEVENLABS_API_KEY);

let client: ElevenLabsClient | undefined;
/** The SDK reads ELEVENLABS_API_KEY from the environment. */
export function elevenlabs() {
  client ??= new ElevenLabsClient();
  return client;
}
