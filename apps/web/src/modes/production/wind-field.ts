export type WindMeta = {
  source: string;
  bounds: [number, number, number, number];
  imageUnscale: [number, number];
  speedMax: number;
  hours: { ts: string }[];
};

export const WIND_BASE_URL = "/windfield";

export const textureUrl = (hour: number) => `${WIND_BASE_URL}/${String(hour).padStart(2, "0")}.png`;
export const speedUrl = (hour: number) => `${WIND_BASE_URL}/${String(hour).padStart(2, "0")}-speed.png`;

export async function fetchWindMeta(): Promise<WindMeta> {
  const response = await fetch(`${WIND_BASE_URL}/meta.json`);
  if (!response.ok) throw new Error(`windfield meta: ${response.status}`);
  return response.json();
}
