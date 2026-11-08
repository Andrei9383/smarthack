import path from "node:path";

import { loadEnvConfig } from "@next/env";
import type { NextConfig } from "next";

// Secrets such as ELEVENLABS_API_KEY (#207) live in .env.local at the repo root; Next only reads
// this app's folder, so load the root too. Values already set (or set in apps/web) win.
loadEnvConfig(path.resolve(process.cwd(), "../.."));

const nextConfig: NextConfig = {
  typedRoutes: true,
  reactCompiler: true,
  serverExternalPackages: ["libsql", "@libsql/client"],
};

export default nextConfig;
