import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

import { Term } from "./term";

export type HowItDecidesMode = "overview" | "feeders" | "om" | "production" | "demand" | "imbalance";

type Step = { verb: string; text: ReactNode };

// The agent's loop per mode: sees → predicts → decides → does. Specifics come from team/plan.md §3.5,
// team/contracts/{opt,actions}.md and the replay (team/insights.md). Keep each step to one sentence.
const LOOPS: Record<HowItDecidesMode, { aside: string; steps: Step[] }> = {
  overview: {
    aside: "every 15 min",
    steps: [
      {
        verb: "Sees",
        text: (
          <>
            118 solar homes on 5 <Term id="feeder">feeders</Term>, 70 turbines at park WP01, the weather and OPCOM
            prices.
          </>
        ),
      },
      {
        verb: "Predicts",
        text: (
          <>
            Solar, household load, county demand and turbine output with its own trained models, each as a{" "}
            <Term id="forecast-band">P10–P90 range</Term>.
          </>
        ),
      },
      {
        verb: "Decides",
        text: (
          <>
            For every asset: count on it, count on less (<Term id="derated">derated</Term>), leave it out, or hand it
            to a <Term id="human-review">person</Term>. Then it plans each feeder against its limit.
          </>
        ),
      },
      {
        verb: "Does",
        text: (
          <>
            Schedules batteries, sends <Term id="work-order">work orders</Term> and alerts, and logs each action with
            the forecast that caused it.
          </>
        ),
      },
    ],
  },
  feeders: {
    aside: "per feeder, day ahead",
    steps: [
      {
        verb: "Sees",
        text: (
          <>
            Every home&apos;s solar, <Term id="home-battery">battery</Term> and meter on the{" "}
            <Term id="feeder">feeder</Term>, plus tomorrow&apos;s weather.
          </>
        ),
      },
      {
        verb: "Predicts",
        text: (
          <>
            Each feeder&apos;s midday <Term id="export">export</Term> a day ahead, as a{" "}
            <Term id="forecast-band">P10–P90 band</Term>.
          </>
        ),
      },
      {
        verb: "Decides",
        text: (
          <>
            If <Term id="p90">P90</Term> export would cross the <Term id="feeder-limit">limit</Term> (62 kW on F7
            Oșorhei), it promises only <Term id="committed-flex">P10 flexibility</Term> and never touches the{" "}
            <Term id="comfort-reserve">comfort reserve</Term>.
          </>
        ),
      },
      {
        verb: "Does",
        text: (
          <>
            Charges home batteries from the midday surplus, releases it into the evening, then measures what was{" "}
            <Term id="realised">realised</Term>.
          </>
        ),
      },
    ],
  },
  om: {
    aside: "every 15 min",
    steps: [
      {
        verb: "Sees",
        text: <>Output of every solar home, and each turbine&apos;s power, wind speed and temperatures.</>,
      },
      {
        verb: "Predicts",
        text: (
          <>
            What each asset should be producing (<Term id="p50">P50</Term>) and, when it falls short, the most likely
            fault.
          </>
        ),
      },
      {
        verb: "Decides",
        text: (
          <>
            Real fault: a <Term id="work-order">work order</Term>. Sensor fault: a{" "}
            <Term id="remote-check">remote check</Term>. <Term id="curtailment">Curtailment</Term>: logged only. Not
            sure: <Term id="human-review">a person decides</Term>.
          </>
        ),
      },
      {
        verb: "Does",
        text: (
          <>
            Gives each crew its <Term id="crew-top-k">top-K</Term> jobs for the day, ranked by chance of fault × kWh
            and lei lost × days to the next visit.
          </>
        ),
      },
    ],
  },
  production: {
    aside: "day ahead",
    steps: [
      {
        verb: "Sees",
        text: <>The 100 m wind forecast over Romania, the solar forecast, and each park&apos;s live output.</>,
      },
      {
        verb: "Predicts",
        text: (
          <>
            Each park&apos;s output for the next day as <Term id="p10">P10</Term> / <Term id="p50">P50</Term> /{" "}
            <Term id="p90">P90</Term>, turning wind speed into MW through the turbines&apos; power curve.
          </>
        ),
      },
      {
        verb: "Decides",
        text: (
          <>
            Which turbines count at full strength, <Term id="derated">derated</Term> or not at all, and which shortfall
            is weather versus <Term id="curtailment">curtailment</Term>.
          </>
        ),
      },
      {
        verb: "Does",
        text: (
          <>
            Writes a daily park report (MWh vs expected, <Term id="availability">availability</Term>, curtailed MWh)
            and hands lost output to <Term id="om">O&amp;M</Term>.
          </>
        ),
      },
    ],
  },
  demand: {
    aside: "per zone, 15 min",
    steps: [
      {
        verb: "Sees",
        text: (
          <>
            Household meters, weather, the calendar and national load (<Term id="real-ro">real RO</Term>).
          </>
        ),
      },
      {
        verb: "Predicts",
        text: (
          <>
            Demand for every county, locality and feeder: <Term id="p50">P50</Term> with a{" "}
            <Term id="forecast-band">P10–P90 band</Term>.
          </>
        ),
      },
      {
        verb: "Decides",
        text: (
          <>
            Scores each zone&apos;s <Term id="stress">stress</Term> (P90 demand ÷ capacity) and ranks where the evening
            peak bites hardest.
          </>
        ),
      },
      {
        verb: "Does",
        text: (
          <>
            Reports the peak hour and MW per zone; on its feeders, batteries discharge into the{" "}
            <span className="whitespace-nowrap">17:00–22:00</span> peak.
          </>
        ),
      },
    ],
  },
  imbalance: {
    aside: "per portfolio, 15 min",
    steps: [
      { verb: "Sees", text: "Smart and profile meters, weather, OPCOM day-ahead and intraday prices (real RO)." },
      {
        verb: "Predicts",
        text: (
          <>
            The portfolio's net purchase with a calibrated <Term id="forecast-band">P10–P90 band</Term>, including what
            each home's battery will do.
          </>
        ),
      },
      { verb: "Decides", text: "How much to buy a day ahead: the quantile that balances shortfall against surplus." },
      { verb: "Does", text: "Buys on PZU at D-1 and corrects at 15:00 on PI, so 18–21 is no longer bought twice." },
    ],
  },
};

/**
 * "How WattSentry decides": the agent's loop for one mode in 4 short steps (sees → predicts → decides → does),
 * with jargon wrapped in <Term>. Renders as a plain section (hairlines, numbered markers, no card), so drop it
 * into an inspector section as-is. `children` render under the steps, e.g. a "See the decision on F7 ›" link.
 */
export function HowItDecides({
  mode,
  title = "How WattSentry decides",
  className,
  children,
}: {
  mode: HowItDecidesMode;
  title?: string;
  className?: string;
  children?: ReactNode;
}) {
  const loop = LOOPS[mode];
  return (
    <section className={cn("text-[13px]", className)} aria-label={title} data-help="how-it-decides">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h3 className="text-[11px] font-semibold tracking-[0.06em] whitespace-nowrap text-muted-foreground uppercase">
          {title}
        </h3>
        <span className="font-mono text-[11px] whitespace-nowrap text-muted-foreground">{loop.aside}</span>
      </div>
      <ol>
        {loop.steps.map((step, i) => (
          <li
            key={step.verb}
            className="relative grid grid-cols-[22px_1fr] gap-x-2.5 border-t border-border py-2.5 first:border-t-0 first:pt-1"
          >
            <span aria-hidden className="relative flex justify-center">
              <span className="relative z-[1] mt-px flex size-[18px] items-center justify-center rounded-full border border-border bg-background font-mono text-[10px] text-text-secondary tabular-nums">
                {i + 1}
              </span>
            </span>
            <p className="leading-[1.5] text-text-secondary">
              <span className="font-semibold text-foreground">{step.verb}</span> {step.text}
            </p>
          </li>
        ))}
      </ol>
      {children ? <div className="mt-1 border-t border-border pt-2.5">{children}</div> : null}
    </section>
  );
}
