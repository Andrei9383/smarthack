import type { Metadata } from "next";

import "../index.css";
import Providers from "@/components/providers";

export const metadata: Metadata = {
  title: "Prosumer Sentinel",
  description: "Feeder peak shaving and fault detection for Romanian prosumers",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
