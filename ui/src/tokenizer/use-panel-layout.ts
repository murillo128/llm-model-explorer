import { useLayoutEffect, useRef, useState } from 'react';

const dividerHeight = 12;
const promptMinimum = 160;
const embeddingsMinimum = 180;

/** Only workspace and editor geometry allocate panels; matrix state never does. */
export function usePanelLayout() {
  const workspace = useRef<HTMLDivElement>(null);
  const [available, setAvailable] = useState(0);
  const [contentHeight, setContentHeight] = useState(100);
  const [manualHeight, setManualHeight] = useState<number | null>(null);
  const [promptChrome, setPromptChrome] = useState(95);
  useLayoutEffect(() => {
    const node = workspace.current!;
    const panel = node.querySelector<HTMLElement>('.prompt-panel')!;
    const editor = panel.querySelector<HTMLElement>('.tokenizer-editor')!;
    const observer = new ResizeObserver(() => {
      setAvailable(node.clientHeight);
      // Actual title, card padding/borders, status and editor border geometry.
      // Scientific content is deliberately outside this measurement boundary.
      setPromptChrome(panel.getBoundingClientRect().height - editor.clientHeight);
    });
    observer.observe(node);
    observer.observe(panel);
    observer.observe(editor);
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
