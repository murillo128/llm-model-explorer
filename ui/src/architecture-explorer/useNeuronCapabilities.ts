import { useEffect, useState } from 'react';
import type { components } from '../api/generated/types';
import type { ExplorerContextValue } from '../app/explorer-context';
import { Lifetime } from '../app/lifetime';

export function useNeuronCapabilities({ client, session, selection }: ExplorerContextValue) {
  const [retry, setRetry] = useState(0);
  const [result, setResult] = useState<{ capabilities?: components['schemas']['NeuronCapabilities']; error?: string }>({});
  useEffect(() => {
    const request = new Lifetime();
    const detach = selection.onDispose(request.dispose);
    const timeout = setTimeout(() => {
      if (request.isCurrent()) { request.dispose(); setResult({ error: 'Neuron capabilities timed out. Retry retrieval.' }); }
    }, 120_000);
    void client.getNeuronCapabilities(session, request.signal).then(request.guard(capabilities => {
      setResult({ capabilities });
    }), request.guard(() => setResult({ error: 'Neuron capabilities are unavailable. Retry retrieval.' })))
      .finally(() => clearTimeout(timeout));
    return () => { clearTimeout(timeout); request.dispose(); detach(); };
  }, [client, session, selection, retry]);
  return { ...result, retry: () => { setResult({}); setRetry(value => value + 1); } };
}
