import { parseAsString, useQueryState } from "nuqs";

// `?site=1042` opens the site drawer from any mode; the demo jump (#33) sets it.
export function useSelectedSite() {
  return useQueryState("site", parseAsString);
}
