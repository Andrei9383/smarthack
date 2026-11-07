// Household congestion notice: subject, HTML and plain text (card #120). Content depends on the
// congestion level and on the home: its solar in the window, its Sentinel battery plan, its appliances.
import type { CongestionLevel, FeederForecast } from "./forecast";

export type Household = {
  assetId: string;
  locality: string;
  county: string;
  feederId: string;
  kwp: number;
  hasBattery: boolean;
  batteryKwh: number;
  comfortReserveKwh: number;
  hasEv: boolean;
  hasAc: boolean;
  hasBoiler: boolean;
  worksFromHome: boolean;
  /** M1 P50 solar in the congestion window */
  pvWindowKwh: number;
  /** P50 solar minus yesterday's load, summed over the window where positive */
  exportWindowKwh: number;
  /** Sentinel battery schedule: charge planned inside the window */
  batteryPlanKwh: number;
};

export type NoticeContext = {
  feederId: string;
  feederName: string;
  limitKw: number;
  forecast: FeederForecast;
  modelVersion: string;
};

export type RenderedEmail = { subject: string; html: string; text: string };

const TZ = "Europe/Bucharest";
const hm = (ts: string) =>
  new Intl.DateTimeFormat("en-GB", { timeZone: TZ, hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(new Date(ts));
const dayName = (ts: string) =>
  new Intl.DateTimeFormat("en-GB", { timeZone: TZ, weekday: "short", day: "numeric", month: "short" }).format(new Date(ts));

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const kwh = (v: number) => (v >= 10 ? Math.round(v).toString() : v.toFixed(1));

const LEVEL_COPY: Record<Exclude<CongestionLevel, "none">, { tag: string; color: string; bg: string; lead: string }> = {
  severe: {
    tag: "Action needed",
    color: "#b42318",
    bg: "#fef3f2",
    lead: "is forecast to receive far more solar power than it can carry",
  },
  congested: {
    tag: "Solar peak",
    color: "#b54708",
    bg: "#fffaeb",
    lead: "is forecast to receive more solar power than it can carry",
  },
  watch: {
    tag: "Heads-up",
    color: "#175cd3",
    bg: "#eff8ff",
    lead: "may get close to its limit if the day turns out sunnier than expected",
  },
};

function tips(h: Household, window: string): string[] {
  const out: string[] = [];
  if (h.hasEv) out.push(`Plug in your electric car between ${window} instead of in the evening.`);
  if (h.hasBoiler) out.push(`Heat your hot water with the electric boiler during ${window}.`);
  if (h.hasAc) out.push(`Pre-cool the house a little during ${window}; it stays cool into the evening.`);
  out.push(
    h.worksFromHome
      ? `Run the washing machine, dishwasher or dryer during ${window}.`
      : `Set the washing machine or dishwasher timer to start inside ${window}.`,
  );
  return out.slice(0, 3);
}

export function renderNotice(h: Household, ctx: NoticeContext): RenderedEmail | null {
  const f = ctx.forecast;
  if (f.level === "none" || !f.windowFrom || !f.windowTo) return null;
  const copy = LEVEL_COPY[f.level];
  const day = dayName(f.windowFrom);
  const window = `${hm(f.windowFrom)}–${hm(f.windowTo)}`;
  const subject =
    f.level === "severe"
      ? `Action needed ${day}: your street's grid is full ${window}`
      : f.level === "congested"
        ? `${day}, ${window}: use your solar power at home`
        : `Heads-up: possible solar peak on ${day}, ${window}`;

  const facts = [
    `Your ${kwh(h.kwp)} kWp panels are expected to make about ${kwh(h.pvWindowKwh)} kWh in that window` +
      (h.exportWindowKwh > 0.05 ? `, about ${kwh(h.exportWindowKwh)} kWh more than your home normally uses then.` : "."),
    "When the line is full, inverters on the street switch off on high voltage and that solar energy is lost.",
  ];
  const plan =
    h.hasBattery && h.batteryPlanKwh > 0.05
      ? `Your home battery is scheduled to store about ${kwh(h.batteryPlanKwh)} kWh during the window. Your ${kwh(h.comfortReserveKwh)} kWh comfort reserve is never used.`
      : h.hasBattery
        ? "Your home battery is kept ready; no extra charging is planned for it in this window."
        : null;
  const todo = tips(h, window);
  const why =
    `Level "${f.level}": ${f.reason}. Forecast peak ${Math.round(f.peakP50Kw)} kW (P50) / ${Math.round(f.peakP90Kw)} kW (P90) ` +
    `against a ${Math.round(ctx.limitKw)} kW limit on feeder ${ctx.feederId}. Solar forecast by Sentinel's PVNet model ` +
    `(${ctx.modelVersion}); load forecast = ${f.loadSource === "yesterday" ? "yesterday's measured load" : f.loadSource === "weekly_profile" ? "typical load profile (no data for yesterday)" : "none available"}.`;

  const text = [
    `${copy.tag}: ${day}, ${window}`,
    "",
    `Hello, household ${h.assetId} in ${h.locality}.`,
    `On ${day} between ${window} the local power line (feeder ${ctx.feederId}, ${ctx.feederName}) ${copy.lead}.`,
    "",
    ...facts,
    ...(plan ? ["", plan] : []),
    "",
    "How you can help (every kWh you use in the window stays on your street):",
    ...todo.map((t) => `- ${t}`),
    "",
    `Why you got this: ${why}`,
    "",
    `Demo notice for a synthetic household (site ${h.assetId}), Smarthack XI / Sentinel.`,
  ].join("\n");

  const li = (items: string[]) =>
    items.map((t) => `<li style="margin:0 0 6px">${esc(t)}</li>`).join("");
  const html = `<!doctype html><html><body style="margin:0;background:#f5f6f7;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#101828">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:24px 12px">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:560px;background:#ffffff;border:1px solid #e4e7ec;border-radius:10px">
<tr><td style="padding:20px 24px 0">
  <div style="font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#667085">Sentinel · grid notice</div>
  <div style="display:inline-block;margin-top:12px;padding:4px 10px;border-radius:999px;background:${copy.bg};color:${copy.color};font-size:13px;font-weight:600">${esc(copy.tag)} · ${esc(day)}, ${esc(window)}</div>
  <h1 style="font-size:20px;line-height:1.3;margin:14px 0 8px">${esc(subject)}</h1>
  <p style="font-size:15px;line-height:1.5;margin:0 0 12px">Hello, household ${esc(h.assetId)} in ${esc(h.locality)}. On <b>${esc(day)}</b> between <b>${esc(window)}</b> the local power line (feeder ${esc(ctx.feederId)}, ${esc(ctx.feederName)}) ${esc(copy.lead)}.</p>
  <p style="font-size:15px;line-height:1.5;margin:0 0 12px">${facts.map(esc).join(" ")}</p>
  ${plan ? `<p style="font-size:15px;line-height:1.5;margin:0 0 12px;padding:10px 12px;background:#f0fdf4;border-radius:8px;color:#05603a">${esc(plan)}</p>` : ""}
  <p style="font-size:15px;line-height:1.5;margin:16px 0 6px;font-weight:600">How you can help</p>
  <ul style="font-size:15px;line-height:1.5;margin:0 0 6px;padding-left:20px">${li(todo)}</ul>
  <p style="font-size:13px;color:#667085;margin:0 0 16px">Every kWh you use in the window stays on your street instead of overloading the line.</p>
</td></tr>
<tr><td style="padding:14px 24px 20px;border-top:1px solid #e4e7ec;font-size:12px;line-height:1.5;color:#667085">
  <b>Why you got this.</b> ${esc(why)}<br><br>
  Demo notice for a synthetic household (site ${esc(h.assetId)}), Smarthack XI · Sentinel.
</td></tr></table></td></tr></table></body></html>`;

  return { subject, html, text };
}
