import type { ReactNode } from "react";

import type { ModeId } from "@/modes/registry";

import { Term } from "@/components/help";

export type Provenance = "synthetic" | "real" | "non";

export type ModeCopy = {
  /** Rail and top-bar descriptor: one plain line, no jargon. */
  descriptor: string;
  /** Inspector title suffix, e.g. "Feeders · peak shaving". */
  heading: string;
  /** The three questions every inspector answers first. Overview answers them in its own panel. */
  what?: ReactNode;
  doing?: ReactNode;
  look?: ReactNode;
  /** Where this view's numbers come from, for the provenance popover. */
  sources: [Provenance, string][];
};

export const MODE_COPY: Record<ModeId, ModeCopy> = {
  overview: {
    descriptor: "Alerts and today's results",
    heading: "Overview",
    sources: [
      ["synthetic", "Homes, feeders, batteries, turbines, alarms and work orders"],
      ["real", "Romanian county shapes and the 100 m wind field"],
      ["non", "Turbine fault model, trained on SCADA from outside Romania"],
    ],
  },
  feeders: {
    descriptor: "Keep cables under their limit",
    heading: "Feeders · peak shaving",
    what: (
      <>
        Rooftop solar can overload street cables (<Term id="feeder">feeders</Term>) at midday. Sentinel charges
        home batteries to keep each feeder under its <Term id="feeder-limit">limit</Term>.
      </>
    ),
    doing: "Plans every home battery a day ahead, then checks the plan held.",
    look: "Red lines on the map and the minutes-over-limit row below.",
    sources: [
      ["synthetic", "Feeder loads, homes and batteries"],
      ["real", "Romanian county shapes"],
    ],
  },
  om: {
    descriptor: "Alarms become repair jobs",
    heading: "O&M · repairs",
    what: "Turbine and solar alarms become a short, ranked to-do list for repair crews.",
    doing: (
      <>
        Ranks faults by lost energy and lei a day; unsure cases go to <Term id="human-review">human review</Term>.
      </>
    ),
    look: "Red and purple dots on the map, then the top of the work-order list.",
    sources: [
      ["non", "Turbine fault model, trained on SCADA from outside Romania"],
      ["synthetic", "Alarms and work orders, mapped to wind park WP01"],
    ],
  },
  production: {
    descriptor: "Wind and solar coming in",
    heading: "Production · forecast",
    what: (
      <>
        How much wind and solar power is coming, with a likely range (<Term id="forecast-band">P10–P90</Term>).
      </>
    ),
    doing: "Forecasts the next 24 hours for each park and each county's rooftop solar.",
    look: "The wind arrows on the map and the shaded band on the park chart.",
    sources: [
      ["real", "Weather reanalysis over Romania drives the wind field"],
      ["synthetic", "Park output and rooftop solar"],
    ],
  },
  demand: {
    descriptor: "Where use peaks this evening",
    heading: "Demand · evening peak",
    what: "Where electricity use will peak this evening, county by county.",
    doing: (
      <>
        Forecasts demand per county and feeder and scores its <Term id="stress">stress</Term>.
      </>
    ),
    look: "The darkest counties on the map and the peak hour beside each.",
    sources: [
      ["synthetic", "Household demand and feeder capacity"],
      ["real", "Romanian county shapes"],
    ],
  },
};

export const PROVENANCE_TEXT: Record<Provenance, { label: string; detail: string }> = {
  synthetic: { label: "synthetic RO", detail: "Simulated, calibrated to Romanian homes and weather." },
  real: { label: "real RO", detail: "Measured or reanalysed Romanian data." },
  non: { label: "non-RO", detail: "Real data, but from outside Romania." },
};
