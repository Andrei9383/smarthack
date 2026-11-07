"use client";

import { Popover as Base } from "@base-ui/react/popover";
import { cn } from "@sentinel-app/ui/lib/utils";
import type { ComponentProps, ReactNode } from "react";

// A small white popover for progressive disclosure (ui-style.md §7): opens on click or Enter/Space,
// closes on Esc and outside click, scales in from its trigger in 180 ms. Built on Base UI, which
// packages/ui already ships (its Sheet and Tooltip use it), so there is no new dependency.

export const Popover = Base.Root;
export const PopoverClose = Base.Close;
export const PopoverTitle = Base.Title;

export function PopoverTrigger({ className, ...props }: ComponentProps<typeof Base.Trigger>) {
  return <Base.Trigger className={className} {...props} />;
}

export function PopoverContent({
  children,
  className,
  side = "bottom",
  align = "start",
  sideOffset = 8,
}: {
  children: ReactNode;
  className?: string;
  side?: "top" | "bottom" | "left" | "right";
  align?: "start" | "center" | "end";
  sideOffset?: number;
}) {
  return (
    <Base.Portal>
      <Base.Positioner side={side} align={align} sideOffset={sideOffset} collisionPadding={12} className="z-50">
        <Base.Popup
          className={cn(
            "w-[300px] origin-[var(--transform-origin)] rounded-xl bg-card p-4 text-[13px] leading-snug text-foreground shadow-overlay outline-none",
            "transition-[opacity,scale] duration-180 ease-out data-[ending-style]:scale-[0.97] data-[ending-style]:opacity-0 data-[starting-style]:scale-[0.97] data-[starting-style]:opacity-0 motion-reduce:transition-none",
            className,
          )}
        >
          {children}
        </Base.Popup>
      </Base.Positioner>
    </Base.Portal>
  );
}
