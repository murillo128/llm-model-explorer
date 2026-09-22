import { useRef, useState } from 'react';
import { Button } from '../components/Button';
import type { ModelSummary, SessionController, ShellState } from './session-controller';

function formatModelSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const unit = bytes > 0 ? Math.min(Math.floor(Math.log10(bytes) / 3), units.length - 1) : 0;
  return `${Number((bytes / 1000 ** unit).toFixed(unit === 0 ? 0 : 1))} ${units[unit]}`;
}

export function AppBar({ state, controller, model, backend }: {
  state: ShellState; controller: SessionController; model: ModelSummary | undefined; backend: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  return <header className="app-bar">
    <span className="product-name" title="LLM Model Explorer">LLM Model Explorer</span>
    <nav className="explorer-nav" aria-label="Explorers">
      {(['Tensor Explorer', 'Tokenizer Explorer', 'Architecture Explorer'] as const).map((name) =>
        <button key={name} type="button" className="explorer-link" aria-label={name}
          aria-current={state.explorer === name ? 'page' : undefined}
          onClick={() => controller.switchExplorer(name)}>
          {name === 'Architecture Explorer' ? <><span className="architecture-nav-label">Architecture</span><span className="architecture-nav-short">Graph</span></> : name.replace(' Explorer', '')}<span className="explorer-suffix"> Explorer</span>
        </button>)}
    </nav>
    <label className="model-label"><span className="visually-hidden">Model</span>
      <select value={state.session?.model_id ?? ''} disabled={state.catalogue !== 'complete' || !state.models.length}
        onChange={(event) => controller.chooseModel(event.target.value)}>
        <option value="" disabled>Select a model</option>
        {state.session && !model && <option value={state.session.model_id}>{state.session.model_id}</option>}
        {state.models.map((entry) => <option key={entry.id} value={entry.id}>{entry.display_name} · {entry.id}</option>)}
      </select>
    </label>
    <button type="button" className="chrome-button" aria-label="Refresh models" title="Refresh models"
      disabled={state.catalogue === 'loading'} onClick={controller.loadModels}><span aria-hidden="true">↻</span></button>
    <span className="session-indicator" aria-label={state.session ? 'Session active' : 'Session inactive'}
      title={state.session ? 'Session active' : 'Session inactive'} data-active={Boolean(state.session)} />
    <div className="session-overflow" onBlur={(event) => {
      if (!event.currentTarget.contains(event.relatedTarget)) setExpanded(false);
    }} onKeyDown={(event) => {
      if (event.key === 'Escape') { setExpanded(false); trigger.current?.focus(); }
    }}>
      <button ref={trigger} type="button" className="chrome-button" aria-label="Session options" title="Session options"
        aria-expanded={expanded} aria-controls="session-options" onClick={() => setExpanded(!expanded)}>
        <span aria-hidden="true">⋯</span>
      </button>
      <section id="session-options" className="session-options" aria-label="Session options" hidden={!expanded}>
        <p className="section-label">LLM Model Explorer</p>
        {state.session && <Button disabled={state.sessionStatus === 'closing'} onClick={() => {
          controller.closeSession(); setExpanded(false); trigger.current?.focus();
        }}>Close session</Button>}
        <dl className="model-metadata metadata">
          <dt>Backend</dt><dd data-testid="backend-url">{backend}</dd>
          <dt>Session</dt><dd>{state.session?.id ?? 'Inactive'}</dd>
          {model && <>
            <dt>Architecture / type</dt><dd>{[...model.architectures, model.model_type].filter(Boolean).join(' · ') || 'Not supplied'}</dd>
            <dt>Tokenizer</dt><dd>{model.tokenizer_available ? 'Available' : 'Unavailable'}</dd>
          </>}
          {model?.parameter_count !== undefined && <><dt>Parameters</dt><dd>{model.parameter_count.toLocaleString()}</dd></>}
          {model?.size_bytes !== undefined && <><dt>Size (bytes)</dt><dd>{model.size_bytes.toLocaleString()}</dd></>}
        </dl>
      </section>
    </div>
  </header>;
}

export function AppStatusBar({ state, model, onRetry }: { state: ShellState; model: ModelSummary | undefined; onRetry: () => void }) {
  const feedback = state.sessionStatus === 'loading' ? 'Loading session…'
    : state.sessionStatus === 'closing' ? 'Closing session…'
    : state.sessionStatus === 'failed' ? 'Session unavailable.'
    : state.session ? 'Session active.' : 'Session inactive.';
  return <footer className="app-status-bar" aria-label="Application status">
    <div className="connection-slot">
      <span role="status" aria-live="polite" aria-atomic="true" data-state={state.connection}>
        <span className="connection-icon" aria-hidden="true">{state.connection === 'connected' ? '●' : '○'}</span>
        {{ connecting: 'Connecting…', connected: 'Connected', reconnecting: 'Reconnecting…', disconnected: 'Disconnected' }[state.connection]}
      </span>
      {(state.connection === 'disconnected' || state.connection === 'reconnecting') && <button type="button"
        className="connection-retry" aria-label="Retry connection" disabled={state.catalogue === 'loading'} onClick={onRetry}>Retry</button>}
    </div>
    <span className="session-status" role="status" title={feedback} data-state={state.sessionStatus}>{feedback}</span>
    {model && <div className="status-metadata metadata">
      {(model.architectures.length > 0 || model.model_type) &&
        <span title={[...model.architectures, model.model_type].filter(Boolean).join(' · ')}>
          {[...model.architectures, model.model_type].filter(Boolean).join(' · ')}
        </span>}
      {model.size_bytes !== undefined && <span title={`${model.size_bytes.toLocaleString()} bytes`}>{formatModelSize(model.size_bytes)}</span>}
      <span>Tokenizer {model.tokenizer_available ? 'available' : 'unavailable'}</span>
    </div>}
  </footer>;
}
