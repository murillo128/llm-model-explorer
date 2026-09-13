import { useEffect, useMemo, useState, useSyncExternalStore } from 'react';
import { ApiClient } from '../api/client';
import type { RuntimeConfig } from '../api/runtime-config';
import { Button } from '../components/Button';
import { ScreenHeader } from '../components/ScreenHeader';
import { TensorTree } from '../components/TensorTree';
import { WorkingSurface } from '../components/WorkingSurface';
import { ExplorerContext } from './explorer-context';
import type { ExplorerContextValue, ExplorerSlots } from './explorer-context';
import { SessionController } from './session-controller';
import type { Explorer, SessionStorage } from './session-controller';

const explorers: Explorer[] = ['Tensor Explorer', 'Tokenizer Explorer'];
interface AppProps { config: RuntimeConfig; slots?: ExplorerSlots }
function tabStorage(): SessionStorage | null {
  try { return window.sessionStorage; } catch { return null; }
}

/** Backend changes remount every session/resource owner, even if IDs coincide. */
export function App({ config, slots = {} }: AppProps) {
  return <BackendApp key={config.backendBaseUrl} config={config} slots={slots} />;
}

function BackendApp({ config, slots }: Required<AppProps>) {
  const [controller] = useState(() => new SessionController(new ApiClient(config), config.backendBaseUrl, tabStorage()));
  const state = useSyncExternalStore(controller.subscribe, controller.getSnapshot);
  useEffect(() => { controller.start(); return controller.dispose; }, [controller]);
  const model = state.models.find((entry) => entry.id === state.session?.model_id);
  const tensor = state.selected;
  const supported = tensor?.rank === 1 || tensor?.rank === 2;
  const Slot = state.explorer === 'Tensor Explorer' ? slots.tensor : slots.tokenizer;
  const canCompose = state.session && state.sessionStatus === 'ready' &&
    (state.explorer === 'Tokenizer Explorer' || supported);
  const context = useMemo<ExplorerContextValue | null>(() => canCompose && state.session ? {
    client: controller.client, session: state.session, sessionId: state.session.id,
    selectedTensor: tensor, selection: state.view,
    reportStatus: (status) => controller.reportStatus(state.view, status),
  } : null, [canCompose, controller, state.session, state.view, tensor]);
  const sessionFeedback = state.sessionStatus === 'loading' ? 'Loading session…'
    : state.sessionStatus === 'closing' ? 'Closing session…'
    : state.message || (state.session ? 'Session active.' : 'Select a model to start a session.');
  return <>
    <a className="skip-link" href="#workspace">Skip to workspace</a>
    <div className="app-shell">
      <ScreenHeader title={state.explorer}>
        {state.session?.model_id ?? 'No model selected'}
        {state.explorer === 'Tensor Explorer' && tensor && <> · {tensor.path.join(' › ')} · [{tensor.shape.join(', ')}]</>}
      </ScreenHeader>
      <nav className="explorer-nav" aria-label="Explorers">
        {explorers.map((name) => <button key={name} type="button" className="explorer-link"
          aria-current={state.explorer === name ? 'page' : undefined}
          onClick={() => controller.switchExplorer(name)}>{name}</button>)}
      </nav>
      <section className="session-controls" aria-label="Model and session">
        <label className="model-label">Model
          <select value={state.session?.model_id ?? ''} disabled={state.catalogue !== 'complete' || !state.models.length}
            onChange={(event) => controller.chooseModel(event.target.value)}>
            <option value="" disabled>Select a model</option>
            {state.session && !model && <option value={state.session.model_id}>{state.session.model_id}</option>}
            {state.models.map((entry) => <option key={entry.id} value={entry.id}>{entry.display_name} · {entry.id}</option>)}
          </select>
        </label>
        {state.session && <Button disabled={state.sessionStatus === 'closing'} onClick={controller.closeSession}>Close session</Button>}
        {state.catalogue === 'loading' && <p role="status">Loading models…</p>}
        {state.catalogue === 'complete' && !state.models.length && <p role="status">No models available on this backend.</p>}
        {state.catalogue === 'failed' && <p role="alert">Could not load models. The backend is unreachable or returned an invalid response.</p>}
        {state.catalogue !== 'loading' && <Button onClick={controller.loadModels}>Refresh models</Button>}
        <p className="session-feedback" role="status" data-state={state.sessionStatus}>{sessionFeedback}</p>
        {state.sessionStatus === 'failed' && <Button onClick={() => controller.retrySession()}>Retry session</Button>}
        {!state.storageAvailable && <p>Tab storage is unavailable. This session cannot be recovered after refresh.</p>}
        {model && <dl className="model-metadata metadata">
          <dt>Architectures</dt><dd>{model.architectures.length ? model.architectures.join(', ') : 'Not supplied'}</dd>
          {model.model_type !== undefined && <><dt>Model type</dt><dd>{model.model_type}</dd></>}
          {model.parameter_count !== undefined && <><dt>Parameters</dt><dd>{model.parameter_count.toLocaleString()}</dd></>}
          {model.size_bytes !== undefined && <><dt>Size (bytes)</dt><dd>{model.size_bytes.toLocaleString()}</dd></>}
          <dt>Tokenizer</dt><dd>{model.tokenizer_available ? 'Available' : 'Unavailable'}</dd>
        </dl>}
      </section>
      <main id="workspace" tabIndex={-1} className={state.explorer === 'Tensor Explorer' && state.session ? 'tensor-layout' : undefined}>
        {state.explorer === 'Tensor Explorer' && state.session && <aside aria-label="Tensor inventory">
          <h2>Tensors</h2>
          {state.inventory === 'loading' && <p role="status">Loading tensor inventory…</p>}
          {state.inventory === 'failed' && <Button onClick={controller.loadInventory}>Retry tensor inventory</Button>}
          {state.inventory === 'complete' && !state.tensors.length && <p>No tensors available.</p>}
          <TensorTree tensors={state.tensors} selectedId={tensor?.id} onSelect={controller.selectTensor} />
        </aside>}
        <WorkingSurface title={`${state.explorer} workspace`}>
          {state.explorer === 'Tensor Explorer' && tensor && <>
            <h2>{tensor.name}</h2>
            <dl className="tensor-metadata metadata">
              <dt>Logical path</dt><dd>{tensor.path.join(' › ')}</dd>
              <dt>Shape</dt><dd>[{tensor.shape.join(', ')}]</dd>
              <dt>Rank</dt><dd>{tensor.rank}</dd>
              <dt>Elements</dt><dd>{tensor.numel.toLocaleString()}</dd>
              <dt>Storage dtype</dt><dd>{tensor.storage_dtype}</dd>
              {tensor.storage_format !== undefined && <><dt>Storage format</dt><dd>{tensor.storage_format}</dd></>}
              <dt>Logical dtype</dt><dd>{tensor.logical_dtype}</dd>
            </dl>
            {!supported && <p>Direct viewing supports complete rank-1 and rank-2 tensors. This rank-{tensor.rank} tensor is available for metadata inspection only.</p>}
          </>}
          {context ? <ExplorerContext.Provider key={state.viewRevision} value={context}>
            {Slot ? <Slot {...context} /> : <p>{state.explorer} view is not connected yet. No computation has started.</p>}
            {state.viewStatus !== 'idle' && <p role="status" data-state={state.viewStatus}>{state.viewStatus === 'failed' ? 'Operation failed.' : `Operation ${state.viewStatus}.`}</p>}
          </ExplorerContext.Provider> : !tensor || state.explorer === 'Tokenizer Explorer' ?
            <p>{state.session ? 'Select a tensor to inspect.' : 'Open a model session to use this explorer.'}</p> : null}
        </WorkingSurface>
      </main>
      <footer className="connection-context">
        <span className="section-label">Backend</span>
        <span className="metadata" data-testid="backend-url">{config.backendBaseUrl}</span>
      </footer>
    </div>
  </>;
}
