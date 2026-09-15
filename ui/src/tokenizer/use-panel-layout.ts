import { useLayoutEffect, useRef, useState } from 'react';

const dividerHeight = 12;
const promptMinimum = 160;
const embeddingsMinimum = 180;
// Header (40), status (21), their gap (6), and editor borders (2).
const promptChrome = 69;

/** Only workspace and editor geometry allocate panels; matrix state never does. */
export function usePanelLayout() {
  const workspace = useRef<HTMLDivElement>(null);
  const [available, setAvailable] = useState(0);
  const [contentHeight, setContentHeight] = useState(100);
  const [manualHeight, setManualHeight] = useState<number | null>(null);
  useLayoutEffect(() => {
    const node = workspace.current!;
    const observer = new ResizeObserver(() => setAvailable(node.clientHeight));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const usable = Math.max(0, available - dividerHeight);
  // At very short window heights both minima yield proportionally, keeping
  // the split inside the shell instead of forcing document overflow.
  const minimum = Math.min(promptMinimum, Math.floor(usable * promptMinimum / (promptMinimum + embeddingsMinimum)));
  const maximum = Math.max(minimum, usable - Math.min(embeddingsMinimum, usable - minimum));
  const autoMaximum = Math.max(minimum, Math.min(maximum, Math.floor(usable * 0.45)));
  const clamp = (height: number) => Math.max(minimum, Math.min(maximum, Math.round(height)));
  const height = manualHeight === null
    ? Math.max(minimum, Math.min(autoMaximum, Math.ceil(contentHeight) + promptChrome))
    : clamp(manualHeight);

  return { workspace, height, minimum, maximum, automatic: manualHeight === null,
    onContentHeight: setContentHeight,
    resize: (next: number) => setManualHeight(clamp(next)),
    reset: () => setManualHeight(null),
  };
}
