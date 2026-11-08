import type { Metadata } from "next";

import { OnePager } from "./_components/one-pager";
import "./onepager.css";

export const metadata: Metadata = {
  title: "WattSentry · Every feeder, kept under its limit",
  description:
    "Forecasting and control for distributed energy: rooftop solar, home batteries, wind farms and the low-voltage grid. Built for PPC at Smarthack XI.",
};

export default function OnePagerPage() {
  return <OnePager />;
}
