import { AGENT_ID, voiceEnabled } from "@/components/voice/server";

export const dynamic = "force-dynamic";

/** Which voice features this server can offer; the app hides the controls when they're off. */
export function GET() {
  const enabled = voiceEnabled();
  return Response.json({ narration: enabled, assistant: enabled && Boolean(AGENT_ID) });
}
