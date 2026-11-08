import { useMap } from "@wattsentry/ui/components/map";
import { useEffect, useState } from "react";

// Keeps the map's focus clear of a right-side panel; resets when the mode unmounts.
export function useMapRightPadding(pixels: number) {
  const { map, isLoaded } = useMap();
  useEffect(() => {
    if (!map || !isLoaded) return;
    map.easeTo({ padding: { right: pixels }, duration: 300 });
    return () => {
      map.easeTo({ padding: { right: 0 }, duration: 300 });
    };
  }, [map, isLoaded, pixels]);
}

/** True while the map's zoom is at or above `threshold` (updated on zoomend). */
export function useZoomAtLeast(threshold: number) {
  const { map, isLoaded } = useMap();
  const [above, setAbove] = useState(false);
  useEffect(() => {
    if (!map || !isLoaded) return;
    const sync = () => setAbove(map.getZoom() >= threshold);
    sync();
    map.on("zoomend", sync);
    return () => {
      map.off("zoomend", sync);
    };
  }, [map, isLoaded, threshold]);
  return above;
}
