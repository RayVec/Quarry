import { useLayoutEffect, useRef, useState } from "react";

export function useDockedComposerOffset(enabled: boolean) {
  const workspaceColumnRef = useRef<HTMLElement | null>(null);
  const dockedComposerRef = useRef<HTMLFormElement | null>(null);
  const [dockedComposerOffset, setDockedComposerOffset] = useState(0);

  useLayoutEffect(() => {
    if (!enabled) {
      setDockedComposerOffset(0);
      return;
    }

    const measure = () => {
      const workspaceColumn = workspaceColumnRef.current;
      const dockedComposer = dockedComposerRef.current;
      if (!workspaceColumn || !dockedComposer) {
        setDockedComposerOffset(0);
        return;
      }

      const dockedComposerRect = dockedComposer.getBoundingClientRect();
      const clearance = 16;
      const nextOffset = Math.ceil(dockedComposerRect.height + clearance);
      setDockedComposerOffset(nextOffset);
    };

    measure();

    const resizeObserver = new ResizeObserver(() => measure());
    if (workspaceColumnRef.current) {
      resizeObserver.observe(workspaceColumnRef.current);
    }
    if (dockedComposerRef.current) {
      resizeObserver.observe(dockedComposerRef.current);
    }

    window.addEventListener("resize", measure);
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [enabled]);

  return {
    workspaceColumnRef,
    dockedComposerRef,
    dockedComposerOffset,
  };
}
