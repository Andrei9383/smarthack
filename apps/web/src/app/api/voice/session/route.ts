import { AGENT_ID, elevenlabs, voiceEnabled } from "@/components/voice/server";

export const dynamic = "force-dynamic";

/**
 * A short-lived WebRTC token for one conversation with the "Ask WattSentry" agent. The agent has
 * auth on, so the browser can only connect with a token minted here.
 */
export async function GET() {
  if (!voiceEnabled() || !AGENT_ID) {
    return Response.json({ error: "Voice is not configured on this server." }, { status: 503 });
  }
  try {
    const { token } = await elevenlabs().conversationalAi.conversations.getWebrtcToken({ agentId: AGENT_ID });
    return Response.json({ token }, { headers: { "cache-control": "no-store" } });
  } catch (error) {
    console.error("[voice] could not get a conversation token", error);
    return Response.json({ error: "Could not reach ElevenLabs." }, { status: 502 });
  }
}
