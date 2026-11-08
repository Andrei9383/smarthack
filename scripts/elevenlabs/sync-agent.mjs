// Creates or updates the "Ask Sentinel" ElevenLabs voice agent (card #207).
//
//   node --env-file=.env.local scripts/elevenlabs/sync-agent.mjs
//
// Uploads the project docs below as the agent's knowledge base (replacing the previous upload),
// then creates the agent or updates it in place. The agent id and document ids go to
// apps/web/src/components/voice/agent.json, which the app's /api/voice/session route reads.
// The key comes from ELEVENLABS_API_KEY and never appears in the code or the JSON.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = path.join(ROOT, "apps/web/src/components/voice/agent.json");

/** One voice for the whole app: the story narrator and the assistant sound the same. */
export const VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"; // George
const LLM = "claude-haiku-5-5"; // fast enough for a voice turn, good with long context

const DOCS = [
  ["README.md", "Project overview, how it works, headline results"],
  ["team/brief.md", "The Smarthack challenge and how it is judged"],
  ["team/plan.md", "The binding build plan"],
  ["team/decisions.md", "Team decisions and contract changes"],
  ["team/results.md", "Evaluation results and run IDs"],
  ["team/insights.md", "What we learned from experiments"],
  ["team/data-provenance.md", "Data sources and provenance"],
  ["team/judge-design.md", "The Judge: trust filter for model predictions"],
  ["team/features/grid.md", "Feature spec: grid / feeders"],
  ["team/features/imbalance.md", "Feature spec: imbalance and prices"],
  ["team/features/wind.md", "Feature spec: wind O&M"],
];

const PROMPT = `# Personality
You are Sentinel, the voice of "Prosumer Sentinel", an energy agent built for PPC (Romania) by our team in 24 hours at Smarthack XI 2026 (theme "24h to Power PPC"). You are calm, precise and friendly, like a good control-room engineer explaining the shift.

# Environment
You speak with people looking at the Prosumer Sentinel web app: hackathon judges, PPC staff and teammates. The app has these views: Overview (the shift briefing and "Play today's story"), Feeders (forecast export against each low-voltage feeder's limit, baseline vs Sentinel), O&M (wind turbines and home solar sites, work orders, human review), Production (wind parks, expected vs actual), Demand (zone demand map) and Imbalance (15-minute prices and avoided imbalance cost). The replay covers one week of June 2025; the story day is Wednesday 4 June.

# What you know
Everything about the project is in your knowledge base: the challenge brief, the plan, decisions, the models (PVNet, LoadNet, SiteHealth, WindNBM, DemandNet, WindFailure, the Judge), the optimizer (per-feeder LP with P10 commitments), data sources, evaluation results with run IDs, and the feature specs. Answer from it. This step is important: never invent numbers, run IDs or results. If the knowledge base does not say, say you don't know and point to where in the app or repo to look.

# Tone
- Short spoken answers: two to four sentences, then offer more detail.
- Plain words first, jargon only when asked; explain P10, feeder limit or nMAE in one line when you use them.
- Say numbers the way a person would ("about six thousand six hundred minutes").
- No markdown, lists or URLs in speech.
- Speak English. If someone speaks Romanian, answer in simple English and say you speak English only.

# Goal
1. Answer any question about the project: what it does, why, how the agent decides and acts, the models and baselines, results and their caveats, data, architecture, the team's choices, and how to run it.
2. When a view would help, call open_view to show it (for example the Feeders view for a feeder question, O&M for faults and work orders). Tell the person what you opened.
3. Call current_view when the person says "this", "here" or asks what they are looking at.
4. Stay on topic. For questions unrelated to the project, say briefly that you only cover Prosumer Sentinel.`;

const TOOLS = [
  {
    type: "client",
    name: "open_view",
    description:
      "Open a view of the Prosumer Sentinel app on the person's screen. Use it when showing a view helps the answer.",
    expects_response: true,
    response_timeout_secs: 5,
    parameters: {
      type: "object",
      required: ["mode"],
      properties: {
        mode: {
          type: "string",
          description: "Which view to open",
          enum: ["overview", "feeders", "om", "production", "demand", "imbalance"],
        },
        feeder: {
          type: "string",
          description: "Optional feeder id to select in the Feeders view, for example F7 or F10",
        },
      },
    },
  },
  {
    type: "client",
    name: "current_view",
    description:
      "Find out which view the person is looking at, the replay clock, whether Sentinel or the baseline is shown, and what is selected.",
    expects_response: true,
    response_timeout_secs: 5,
    parameters: { type: "object", properties: {} },
  },
];

const client = new ElevenLabsClient();

function previous() {
  if (!existsSync(OUT)) return { agentId: undefined, documents: [] };
  return JSON.parse(readFileSync(OUT, "utf8"));
}

async function uploadDocs() {
  const docs = [];
  for (const [file, label] of DOCS) {
    const text = readFileSync(path.join(ROOT, file), "utf8");
    const name = `Sentinel · ${file} · ${label}`;
    const res = await client.conversationalAi.knowledgeBase.documents.createFromText({ text, name });
    docs.push({ id: res.id, name, file });
    console.log(`uploaded ${file} (${text.length} chars)`);
  }
  return docs;
}

function agentBody(docs) {
  // Plain REST JSON (snake_case, as in the ElevenLabs docs); the SDK's typed serializer is strict
  // about nested tool shapes, so agent create/update go through fetch below.
  return {
    name: "Ask Sentinel (Prosumer Sentinel)",
    tags: ["smarthack", "sentinel"],
    conversation_config: {
      agent: {
        first_message:
          "Hi, I'm Sentinel. Ask me anything about the project: the agent, the models, the results or the demo.",
        language: "en",
        prompt: {
          prompt: PROMPT,
          llm: LLM,
          temperature: 0.3,
          tools: TOOLS,
          built_in_tools: {
            end_call: { type: "system", name: "end_call", params: { system_tool_type: "end_call" } },
          },
          knowledge_base: docs.map((d) => ({ type: "text", id: d.id, name: d.name, usage_mode: "auto" })),
          rag: { enabled: true, embedding_model: "multilingual_e5_large_instruct", max_retrieved_rag_chunks_count: 20 },
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
      guardrails: { version: "1", focus: { is_enabled: true }, prompt_injection: { is_enabled: true } },
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

const prev = previous();
const docs = await uploadDocs();
const body = agentBody(docs);

async function deleteDocs(list) {
  for (const doc of list) {
    try {
      await client.conversationalAi.knowledgeBase.documents.delete(doc.id, { force: true });
    } catch (error) {
      console.warn(`could not delete doc ${doc.id}: ${error?.message ?? error}`);
    }
  }
}

let agentId = prev.agentId;
try {
  if (agentId) {
    await api("PATCH", `/v1/convai/agents/${agentId}`, body);
    console.log(`updated agent ${agentId}`);
  } else {
    const res = await api("POST", "/v1/convai/agents/create", body);
    agentId = res.agent_id;
    console.log(`created agent ${agentId}`);
  }
} catch (error) {
  // Don't leave this run's upload behind when the agent step fails.
  await deleteDocs(docs);
  throw error;
}

// The agent now points at the new docs; the old upload can go.
await deleteDocs(prev.documents ?? []);

mkdirSync(path.dirname(OUT), { recursive: true });
writeFileSync(
  OUT,
  `${JSON.stringify({ agentId, voiceId: VOICE_ID, documents: docs, syncedAt: new Date().toISOString() }, null, 2)}\n`,
);
console.log(`wrote ${path.relative(ROOT, OUT)}`);
