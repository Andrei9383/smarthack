import { elevenlabs, VOICE_ID, voiceEnabled } from "@/components/voice/server";

export const dynamic = "force-dynamic";

const MAX_CHARS = 700;
const CACHE_LIMIT = 64;

// The story repeats the same lines on every play, so keep the audio and pay for each line once.
const cache = new Map<string, Uint8Array>();

async function speak(text: string) {
  const hit = cache.get(text);
  if (hit) return hit;
  const stream = await elevenlabs().textToSpeech.convert(VOICE_ID, {
    text,
    modelId: "eleven_flash_v2_5",
    outputFormat: "mp3_44100_128",
    applyTextNormalization: "on",
  });
  const audio = new Uint8Array(await new Response(stream).arrayBuffer());
  if (cache.size >= CACHE_LIMIT) cache.delete(cache.keys().next().value!);
  cache.set(text, audio);
  return audio;
}

/** Story narration: one line of text in, MP3 out. */
export async function POST(request: Request) {
  if (!voiceEnabled()) return Response.json({ error: "Voice is not configured on this server." }, { status: 503 });
  const body = (await request.json().catch(() => null)) as { text?: unknown } | null;
  const text = typeof body?.text === "string" ? body.text.trim() : "";
  if (!text || text.length > MAX_CHARS) {
    return Response.json({ error: `Send 1 to ${MAX_CHARS} characters of text.` }, { status: 400 });
  }
  try {
    const audio = await speak(text);
    return new Response(audio as BodyInit, {
      headers: { "content-type": "audio/mpeg", "cache-control": "private, max-age=3600" },
    });
  } catch (error) {
    console.error("[voice] text to speech failed", error);
    return Response.json({ error: "Could not reach ElevenLabs." }, { status: 502 });
  }
}
