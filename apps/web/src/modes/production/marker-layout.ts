export type MarkerCircle = {
  id: string;
  kind: "park" | "county";
  x: number;
  y: number;
  radius: number;
};

export type MarkerOffset = { dx: number; dy: number };

const GAP_PX = 4;
const MAX_PASSES = 60;

// Moves circles apart on screen; data positions never change. Parks push each other equally and
// counties yield to parks, so a park only ever moves for another park.
export function resolveCollisions(circles: MarkerCircle[]): Map<string, MarkerOffset> {
  const placed = circles.map((circle) => ({ ...circle, dx: 0, dy: 0 }));
  for (let pass = 0; pass < MAX_PASSES; pass++) {
    let moved = false;
    placed.forEach((a, i) => {
      for (const b of placed.slice(i + 1)) {
        const deltaX = b.x + b.dx - (a.x + a.dx);
        const deltaY = b.y + b.dy - (a.y + a.dy);
        const distance = Math.hypot(deltaX, deltaY);
        const overlap = a.radius + b.radius + GAP_PX - distance;
        if (overlap <= 0) continue;
        moved = true;
        // Same point: pick a direction from the index so the result is stable.
        const unitX = distance > 0 ? deltaX / distance : Math.cos(i + 1);
        const unitY = distance > 0 ? deltaY / distance : Math.sin(i + 1);
        const aShare = a.kind === b.kind ? 0.5 : a.kind === "county" ? 1 : 0;
        a.dx -= unitX * overlap * aShare;
        a.dy -= unitY * overlap * aShare;
        b.dx += unitX * overlap * (1 - aShare);
        b.dy += unitY * overlap * (1 - aShare);
      }
    });
    if (!moved) break;
  }
  return new Map(placed.map(({ id, dx, dy }) => [id, { dx, dy }]));
}
