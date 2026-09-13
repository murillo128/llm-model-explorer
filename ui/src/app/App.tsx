import { useState } from 'react';
import type { RuntimeConfig } from '../api/runtime-config';
import { ScreenHeader } from '../components/ScreenHeader';
import { StatusText } from '../components/StatusText';
import { WorkingSurface } from '../components/WorkingSurface';

const explorers = ['Tensor Explorer', 'Tokenizer Explorer'] as const;
type Explorer = typeof explorers[number];

/** Explorer slots only. Future API initialization receives the validated config here. */
export function App({ config }: { config: RuntimeConfig }) {
  const [explorer, setExplorer] = useState<Explorer>('Tensor Explorer');
  return (
    <>
      <a className="skip-link" href="#workspace">Skip to workspace</a>
      <div className="app-shell">
        <ScreenHeader title={explorer}>No model selected</ScreenHeader>
        <nav className="explorer-nav" aria-label="Explorers">
          {explorers.map((name) => (
            <button key={name} type="button" className="explorer-link"
              aria-current={explorer === name ? 'page' : undefined}
              onClick={() => setExplorer(name)}>{name}</button>
          ))}
        </nav>
        <main id="workspace" tabIndex={-1}>
          <WorkingSurface title={`${explorer} workspace`}>
            <div className="empty-state">
              <h2>Workspace ready for implementation</h2>
              <StatusText kind="empty">{explorer} is not implemented yet.</StatusText>
              <p>This surface is reserved for the explorer. No model data has been loaded.</p>
            </div>
          </WorkingSurface>
        </main>
        <footer className="connection-context">
          <span className="section-label">Backend configuration</span>
          <span className="metadata" data-testid="backend-url">{config.backendBaseUrl}</span>
          <span className="connection-note">Configured · connection not checked</span>
        </footer>
      </div>
    </>
  );
}
