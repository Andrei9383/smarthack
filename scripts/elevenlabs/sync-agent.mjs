// Creates or updates the "Ask Sentinel" ElevenLabs voice agent.
//
//   node --env-file=.env.local scripts/elevenlabs/sync-agent.mjs
//
// The agent answers questions about the grid and the replay week shown in the app (feeders, wind
// parks, faults, results) and refuses questions about the code. It has no knowledge base: facts come
// from client tools that run in the browser and read the app's own API
// (apps/web/src/components/voice/data-tools.ts). The agent id goes to
// apps/web/src/components/voice/agent.json, which /api/voice/session reads.
// The key comes from ELEVENLABS_API_KEY and never appears in the code or the JSON.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = path.join(ROOT, "apps/web/src/components/voice/agent.json");

/** One voice for the whole app: the story narrator and the assistant sound the same. */
const VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"; // George
const LLM = "claude-haiku-5-5"; // fast enough for a voice turn

const PROMPT = `# Personality
You are Sentinel, the voice of the WattSentry control room for PPC in Romania. You are calm, precise and friendly, like a control-room engineer briefing the shift.

# Environment
You talk to people looking at the WattSentry app: a replay of one week, Monday 2 June to Sunday 8 June 2025, of a Romanian distribution grid and wind fleet. The story day is Wednesday 4 June. "Today" means the day on the replay clock, which current_view tells you.
What is on the grid:
- Five low-voltage feeders with rooftop solar and home batteries: F7 Oșorhei (Bihor), F3 Florești and F10 Apahida (Cluj), F6 Măgura (Bacău), F11 Otopeni (Ilfov). Each has a limit: how much solar power the line can take back.
- One wind park (eolian park) near Fântânele-Cogealac, Constanța county, with 70 turbines.
- About 118 homes with solar.
- Sentinel is the agent: each night it plans which home batteries charge at midday so feeders stay under their limit, it ranks faults and sends crews, and it hands unsure cases to a person. "Without Sentinel" (the baseline) is the same week where batteries only store their own solar.
- All data is synthetic, placed at real Romanian sites.
App views: Overview (the shift briefing), Feeders, O&M (faults, work orders, turbines), Production (wind), Demand (zone demand map), Imbalance (15-minute prices).

# Tools
This step is important: get every number from a tool, never from memory, and never invent one.
- get_feeders(day): feeder limits, minutes over the limit and peaks with and without Sentinel, how many feeders stayed within their limit.
- get_wind_parks(): where the wind parks are, their size, which turbines are stopped right now.
- get_faults(day): work orders, alerts and cases handed to a person on a day.
- get_day_summary(day): one day at a glance (production, wind, availability, open issues) and the week's results with and without Sentinel.
- current_view(): what the person is looking at and the replay clock.
- open_view(mode, feeder): show a view; use it when it helps, and say what you opened.
Pass day as the person said it ("today", "Thursday", "5 June"). If a tool says the day is outside the week, tell them the replay covers 2 to 8 June 2025.

# Scope
Answer only questions about the grid, the assets, the events and the results in this week: where things are, what happened on a day, faults, feeders, wind, batteries, how Sentinel compares with no Sentinel, and what a term on screen means (in one sentence).
This step is important: do not answer questions about the code, the repository, the software, the tech stack, files, functions, how the app or the models were built or trained, the team or the hackathon. Say briefly: "I only answer questions about the grid and this week's data, for example the faults on Thursday or how many feeders stayed within their limit." Off-topic questions get the same short answer.

# Tone
- Short spoken answers: one to three sentences with the key number first, then offer detail.
- Plain words. Say numbers the way a person would and round them.
- No lists, markdown or URLs in speech.
- Speak English. If someone speaks Romanian, answer in simple English and say you speak English only.`;

const dayParam = {
  type: "string",
  description:
    'The day as the person said it: "today", "yesterday", a weekday like "Thursday", a date like "5 June" or "2025-06-05". Leave empty for the replay\'s current day.',
};

const clientTool = (name, description, properties = {}, required = []) => ({
  type: "client",
  name,
  description,
  expects_response: true,
  response_timeout_secs: 10,
  parameters: { type: "object", required, properties },
});

const TOOLS = [
  clientTool(
    "get_feeders",
    "Feeders on one day: location, limit, minutes over the limit and peak with and without Sentinel, and how many stayed within their limit.",
    { day: dayParam },
  ),
  clientTool(
    "get_wind_parks",
    "Wind parks (eolian parks): where they are (county, coordinates), number of turbines, rated MW, and which turbines are stopped now.",
  ),
  clientTool(
    "get_faults",
    "Faults on one day: work orders (asset, fault, crew, confidence, lei lost per day), alerts, and cases handed to a person.",
    { day: dayParam },
  ),
  clientTool(
    "get_day_summary",
    "One day at a glance (homes, feeders, turbines, wind production, wind speed, availability, open issues) and the whole week's results with and without Sentinel.",
    { day: dayParam },
  ),
  clientTool(
    "open_view",
    "Open a view of the app on the person's screen.",
    {
      mode: {
        type: "string",
        description: "Which view to open",
        enum: ["overview", "feeders", "om", "production", "demand", "imbalance"],
      },
      feeder: { type: "string", description: "Optional feeder id to select in the Feeders view, for example F7 or F10" },
    },
    ["mode"],
  ),
  clientTool(
    "current_view",
    "Which view the person is looking at, the replay clock (the current day), whether Sentinel or the baseline is shown, and what is selected.",
  ),
];

function agentBody() {
  // Plain REST JSON (snake_case, as in the ElevenLabs docs); the SDK's typed serializer is strict
  // about nested tool shapes, so agent create/update go through fetch below.
  return {
    name: "Ask Sentinel (WattSentry)",
    tags: ["smarthack", "sentinel"],
    conversation_config: {
      agent: {
        first_message:
          "Hi, I'm Sentinel. Ask me about the grid this week: faults on a day, the feeders, or the wind park.",
        language: "en",
        prompt: {
          prompt: PROMPT,
          llm: LLM,
          temperature: 0.2,
          tools: TOOLS,
          built_in_tools: {
            end_call: { type: "system", name: "end_call", params: { system_tool_type: "end_call" } },
          },
          knowledge_base: [],
          rag: { enabled: false },
        },
      },
      // English agents must use flash/turbo v2.
      tts: { voice_id: VOICE_ID, model_id: "eleven_flash_v2" },
      turn: { turn_eagerness: "normal" },
      conversation: { max_duration_seconds: 600 },
    },
    platform_settings: {
      // Only the app's server can start a conversation (it mints a token with the key).
      auth: { enable_auth: true },
      call_limits: { agent_concurrency_limit: 5, daily_limit: 300 },
      // The prompt keeps it on topic. Both built-in guardrails (focus, prompt injection) silently blocked
      // the polite "I only answer questions about the grid" reply to code questions in agent tests, and
      // the agent holds no secrets and only has read-only tools, so they stay off.
      guardrails: { version: "1", focus: { is_enabled: false }, prompt_injection: { is_enabled: false } },
    },
  };
}

async function api(method, url, body) {
  const res = await fetch(`https://api.elevenlabs.io${url}`, {
    method,
    headers: { "xi-api-key": process.env.ELEVENLABS_API_KEY ?? "", "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`${method} ${url} -> ${res.status}: ${text.slice(0, 2000)}`);
  return text ? JSON.parse(text) : {};
}

const prev = existsSync(OUT) ? JSON.parse(readFileSync(OUT, "utf8")) : {};
let agentId = prev.agentId;
if (agentId) {
  await api("PATCH", `/v1/convai/agents/${agentId}`, agentBody());
  console.log(`updated agent ${agentId}`);
} else {
  const res = await api("POST", "/v1/convai/agents/create", agentBody());
  agentId = res.agent_id;
  console.log(`created agent ${agentId}`);
}

// Earlier versions uploaded the project docs as a knowledge base; the agent no longer uses them.
const client = new ElevenLabsClient();
for (const doc of prev.documents ?? []) {
  try {
    await client.conversationalAi.knowledgeBase.documents.delete(doc.id, { force: true });
    console.log(`deleted old knowledge doc ${doc.file ?? doc.id}`);
  } catch (error) {
    console.warn(`could not delete old doc ${doc.id}: ${error?.message ?? error}`);
  }
}

mkdirSync(path.dirname(OUT), { recursive: true });
writeFileSync(
  OUT,
  `${JSON.stringify({ agentId, voiceId: VOICE_ID, syncedAt: new Date().toISOString() }, null, 2)}\n`,
);
console.log(`wrote ${path.relative(ROOT, OUT)}`);
