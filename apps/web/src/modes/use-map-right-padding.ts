import { useMap } from "@sentinel-app/ui/components/map";
import { useEffect } from "react";

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
