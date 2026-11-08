"use client";

import { useQuery } from "@tanstack/react-query";

// What the server can do: both are false on a checkout without ELEVENLABS_API_KEY,
// and the voice controls stay hidden.
type VoiceStatus = { narration: boolean; assistant: boolean };

const OFF: VoiceStatus = { narration: false, assistant: false };

export function useVoiceStatus(): VoiceStatus {
  const { data } = useQuery({
    queryKey: ["voice-status"],
    queryFn: async () => {
      const res = await fetch("/api/voice");
      return res.ok ? ((await res.json()) as VoiceStatus) : OFF;
    },
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });
  return data ?? OFF;
}
