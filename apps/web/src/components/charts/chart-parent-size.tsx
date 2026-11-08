"use client";

import { type ReactNode, type RefObject, useEffect, useLayoutEffect, useRef, useState } from "react";

const useIsomorphicLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

interface ElementSize {
  width: number;
  height: number;
}

/** Layout size (unaffected by CSS transforms), fractional like `ResizeObserver`'s `contentRect`. */
function readElementSize(element: HTMLElement): ElementSize {
  const style = getComputedStyle(element);
  const width = Number.parseFloat(style.width);
  const height = Number.parseFloat(style.height);
  return {
    width: Number.isFinite(width) ? width : element.clientWidth,
    height: Number.isFinite(height) ? height : element.clientHeight,
  };
}

/**
 * Element size, measured first in a layout effect (the sized render commits before the first
 * paint), then from a `ResizeObserver` throttled to one update per `debounceTime` plus a trailing one.
 */
function useElementSize(ref: RefObject<HTMLElement | null>, debounceTime: number): ElementSize {
  const [size, setSize] = useState<ElementSize>({ width: 0, height: 0 });

  useIsomorphicLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const apply = () => {
      const next = readElementSize(element);
      setSize((prev) => (prev.width === next.width && prev.height === next.height ? prev : next));
    };
    apply();
    if (typeof ResizeObserver === "undefined") return;

    let timeout: ReturnType<typeof setTimeout> | null = null;
    let trailing = false;
    const flush = () => {
      if (trailing) {
        trailing = false;
        apply();
        timeout = setTimeout(flush, debounceTime);
      } else {
        timeout = null;
      }
    };
    const observer = new ResizeObserver(() => {
      if (timeout !== null) {
        trailing = true;
        return;
      }
      apply();
      timeout = setTimeout(flush, debounceTime);
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      if (timeout !== null) clearTimeout(timeout);
    };
  }, [ref, debounceTime]);

  return size;
}

/**
 * Like visx `<ParentSize>` without its `overflow: hidden` measuring layer, so tooltips and strokes
 * at the chart edge aren't clipped.
 */
export function ParentSize({
  children,
  debounceTime,
}: {
  children: (size: ElementSize) => ReactNode;
  debounceTime: number;
}) {
  const parentRef = useRef<HTMLDivElement>(null);
  const size = useElementSize(parentRef, debounceTime);
  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div ref={parentRef} style={{ position: "absolute", inset: 0 }}>
        {children(size)}
      </div>
    </div>
  );
}
