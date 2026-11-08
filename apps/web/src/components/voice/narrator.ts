"use client";

import { setReplayHeld } from "@/lib/replay-playing";

// Speaks story lines one after another through /api/voice/tts (ElevenLabs).
// While a line is waiting or playing, the replay clock is held so the voice and the screen
// stay on the same moment. Pausing the story pauses the voice; stop() drops everything.

const audioUrls = new Map<string, Promise<string | null>>();

/** Fetches (once) the MP3 for a line; null when the server can't speak. */
export function loadLine(text: string): Promise<string | null> {
  let url = audioUrls.get(text);
  if (!url) {
    url = fetch("/api/voice/tts", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    })
      .then((res) => (res.ok ? res.blob() : null))
      .then((blob) => (blob ? URL.createObjectURL(blob) : null))
      .catch(() => null);
    audioUrls.set(text, url);
    // A failed line can be tried again on the next play.
    void url.then((value) => {
      if (!value) audioUrls.delete(text);
    });
  }
  return url;
}

class Narrator {
  private queue: string[] = [];
  private audio: HTMLAudioElement | null = null;
  private finish: (() => void) | null = null;
  private running = false;
  private generation = 0;
  private paused = false;

  say(text: string) {
    this.queue.push(text);
    setReplayHeld(true);
    void this.pump();
  }

  pause() {
    this.paused = true;
    this.audio?.pause();
  }

  resume() {
    this.paused = false;
    this.audio?.play().catch(() => this.finish?.());
  }

  stop() {
    this.generation += 1;
    this.queue = [];
    this.audio?.pause();
    this.finish?.();
    this.audio = null;
    this.running = false;
    setReplayHeld(false);
  }

  private async pump() {
    if (this.running) return;
    this.running = true;
    const generation = this.generation;
    while (this.queue.length > 0 && generation === this.generation) {
      const text = this.queue.shift()!;
      const url = await loadLine(text);
      if (generation !== this.generation) return;
      if (!url) continue;
      await this.play(url);
    }
    if (generation !== this.generation) return;
    this.running = false;
    setReplayHeld(false);
  }

  private play(url: string) {
    return new Promise<void>((resolve) => {
      const audio = new Audio(url);
      this.audio = audio;
      const done = () => {
        audio.onended = null;
        audio.onerror = null;
        if (this.audio === audio) this.audio = null;
        this.finish = null;
        resolve();
      };
      this.finish = done;
      audio.onended = done;
      audio.onerror = done;
      if (!this.paused) audio.play().catch(done);
    });
  }
}

export const narrator = new Narrator();
