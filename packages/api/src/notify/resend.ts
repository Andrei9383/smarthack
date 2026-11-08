// Resend API (https://resend.com/docs/api-reference/emails/send-batch-emails), plain fetch, no SDK.
// The key only ever comes from the server environment (RESEND_API_KEY); it is never committed.

export type OutgoingEmail = { to: string[]; subject: string; html: string; text: string; tags?: { name: string; value: string }[] };

const DEFAULT_FROM = "WattSentry grid alerts <alerts@devsolution.ro>";
const BATCH_MAX = 100; // Resend batch limit

export function notifyConfig() {
  const to = (process.env.NOTIFY_DEMO_TO ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(s));
  return {
    apiKey: process.env.RESEND_API_KEY ?? "",
    from: process.env.RESEND_FROM ?? DEFAULT_FROM,
    demoTo: to,
    /** emails per process per hour: the demo endpoint is public, so cap what it can send */
    hourlyCap: Number(process.env.NOTIFY_HOURLY_CAP ?? 300),
  };
}

/** "yolo@example.com" -> "y***@example.com", so the public page never shows the inbox in full. */
export function maskEmail(email: string) {
  const [user = "", domain = ""] = email.split("@");
  return `${user.slice(0, 1)}***@${domain}`;
}

const sentAt: number[] = [];
function takeQuota(n: number, cap: number) {
  const hourAgo = Date.now() - 60 * 60 * 1000;
  while (sentAt.length && (sentAt[0] ?? 0) < hourAgo) sentAt.shift();
  if (sentAt.length + n > cap) return false;
  for (let i = 0; i < n; i++) sentAt.push(Date.now());
  return true;
}

export type SendResult = { ok: true; ids: string[] } | { ok: false; error: string };

export async function sendBatch(emails: OutgoingEmail[]): Promise<SendResult> {
  const cfg = notifyConfig();
  if (!cfg.apiKey) return { ok: false, error: "RESEND_API_KEY is not set on the server" };
  if (emails.length === 0) return { ok: true, ids: [] };
  if (!takeQuota(emails.length, cfg.hourlyCap))
    return { ok: false, error: `hourly cap of ${cfg.hourlyCap} emails reached (NOTIFY_HOURLY_CAP)` };

  const ids: string[] = [];
  for (let i = 0; i < emails.length; i += BATCH_MAX) {
    const chunk = emails.slice(i, i + BATCH_MAX).map((e) => ({ from: cfg.from, ...e }));
    const res = await fetch("https://api.resend.com/emails/batch", {
      method: "POST",
      headers: { Authorization: `Bearer ${cfg.apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify(chunk),
    });
    const body = (await res.json().catch(() => ({}))) as {
      data?: { id: string }[];
      message?: string;
      name?: string;
    };
    if (!res.ok) return { ok: false, error: `Resend ${res.status}: ${body.message ?? body.name ?? "error"}` };
    ids.push(...(body.data ?? []).map((d) => d.id));
  }
  return { ok: true, ids };
}
