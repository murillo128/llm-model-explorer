import { useEffect, useState } from 'react';
import { loadRuntimeConfig } from '../api/runtime-config';
import type { RuntimeConfig } from '../api/runtime-config';
import { Button } from '../components/Button';
import { StatusText } from '../components/StatusText';
import { WorkingSurface } from '../components/WorkingSurface';
import { App } from './App';

type StartupState = { kind: 'loading' } | { kind: 'error'; message: string } |
  { kind: 'ready'; config: RuntimeConfig };

function ConfigurationGate({ retry }: { retry: () => void }) {
  const [state, setState] = useState<StartupState>({ kind: 'loading' });
  useEffect(() => {
    const controller = new AbortController();
    void loadRuntimeConfig(controller.signal).then(
      (config) => { if (!controller.signal.aborted) setState({ kind: 'ready', config }); },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({
          kind: 'error', message: error instanceof Error ? error.message : 'Could not load configuration. Retry.',
        });
      },
    );
    return () => controller.abort();
  }, []);

  if (state.kind === 'ready') return <App config={state.config} />;
  return (
    <div className="app-shell">
      <header className="app-bar"><span className="product-name">LLM Model Explorer</span></header>
      <main className="startup-workspace">
        <WorkingSurface title="Startup">
          {state.kind === 'loading' ? (
            <StatusText kind="loading">Loading application configuration…</StatusText>
          ) : (
            <>
              <h2>Configuration needs attention</h2>
              <StatusText kind="error">{state.message}</StatusText>
              <Button onClick={retry}>Retry configuration</Button>
            </>
          )}
        </WorkingSurface>
      </main>
      <footer className="app-status-bar">Application setup</footer>
    </div>
  );
}

export function Bootstrap() {
  const [attempt, setAttempt] = useState(0);
  return <ConfigurationGate key={attempt} retry={() => setAttempt((value) => value + 1)} />;
}
