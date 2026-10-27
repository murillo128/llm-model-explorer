import { useLayoutEffect, useState } from 'react';
import { Lifetime } from '../app/lifetime';

/** Publish only effect-owned generations, including a fresh one after StrictMode replay. */
export function useChildLifetime(parent: Lifetime) {
  const [lifetime, setLifetime] = useState<Lifetime | null>(null);
  useLayoutEffect(() => {
    const current = new Lifetime();
    const detach = parent.onDispose(current.dispose);
    // Children may subscribe only after their effect-owned lifetime is installed.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLifetime(current);
    return () => { current.dispose(); detach(); };
  }, [parent]);
  return lifetime;
}
