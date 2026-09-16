import { useLayoutEffect, useMemo } from 'react';

type Callback<Args extends unknown[]> = (...args: Args) => void;

// Create the slot outside the hook/render scope. Its stable proxy must retain
// only the current handler, never an earlier Canvas render's closure context.
function createCallback<Args extends unknown[]>() {
  let current: Callback<Args> | null = null;
  return {
    handler: (...args: Args) => { current?.(...args); },
    commit: (callback: Callback<Args>) => { current = callback; },
    clear: () => { current = null; },
  };
}

/** Canvas-owned event handler, updated only when its render commits. */
export function useCanvasCallback<Args extends unknown[]>(callback: Callback<Args>): Callback<Args> {
  const event = useMemo(() => createCallback<Args>(), []);
  useLayoutEffect(() => {
    event.commit(callback);
    return event.clear;
  }, [callback, event]);
  return event.handler;
}
