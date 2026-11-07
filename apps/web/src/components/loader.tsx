import { Loader2 } from "lucide-react";

/** Loading state (Cockpit): a spinner plus a line that says what is loading and that it will appear here. */
export default function Loader() {
  return (
    <div role="status" className="flex h-full flex-col items-center justify-center gap-2 pt-8 text-center">
      <Loader2 aria-hidden className="size-5 animate-spin text-primary motion-reduce:animate-none" />
      <p className="text-xs text-text-secondary">Loading this view's data. It appears here in a moment.</p>
    </div>
  );
}
