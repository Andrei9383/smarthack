import type { ReactNode } from "react";

import type { ModeId } from "@/modes/registry";

import { Term } from "@/components/help";

export type ModeCopy = {
  /** Rail and top-bar descriptor: one plain line, no jargon. */
  descriptor: string;
  /** Inspector title suffix, e.g. "Feeders · peak shaving". */
  heading: string;
  /** The three questions every inspector answers first. Overview answers them in its own panel. */
  what?: ReactNode;
  doing?: ReactNode;
  look?: ReactNode;
};

export const MODE_COPY: Record<ModeId, ModeCopy> = {
  overview: {
    descriptor: "Alerts and today's results",
    heading: "Overview",
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
  },
  imbalance: {
    descriptor: "Stop paying for prosumer imbalance",
    heading: "Imbalance · supplier position",
  },
};
