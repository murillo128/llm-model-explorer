import { lazy, Suspense, useEffect, useMemo, useState, useSyncExternalStore } from 'react';
import { GraphViews } from '../architecture-explorer/graph';
import { ArchitectureWorkspace } from '../architecture-explorer/ArchitectureWorkspace';
import { TensorExplorer } from '../explorers/TensorExplorer';
import { ApiClient } from '../api/client';
import type { RuntimeConfig } from '../api/runtime-config';
import { Button } from '../components/Button';
import { ToastHost } from './ToastHost';
import { AppBar, AppStatusBar } from './AppChrome';
import { TensorHeader } from '../components/TensorHeader';
import { ViewerPanel } from '../matrix-explorer';
import { TensorTree } from '../components/TensorTree';
import { TensorWorkspace } from '../components/TensorWorkspace';
import { WorkingSurface } from '../components/WorkingSurface';
import { ExplorerContext } from './explorer-context';
import type { ExplorerContextValue, ExplorerSlots } from './explorer-context';
import { SessionController } from './session-controller';
import type { SessionStorage } from './session-controller';

const ArchitectureExplorer = lazy(() => import('../architecture-explorer/ArchitectureExplorer').then((module) => ({ default: module.ArchitectureExplorer })));
const TokenizerExplorer = lazy(() => import('../tokenizer/TokenizerExplorer').then((module) => ({ default: module.TokenizerExplorer })));
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
  const [graphViews] = useState(() => new GraphViews());
  const state = useSyncExternalStore(controller.subscribe, controller.getSnapshot);
  useEffect(() => { controller.start(); return controller.dispose; }, [controller]);
  const model = state.models.find((entry) => entry.id === state.session?.model_id);
  const tensor = state.selected;
  const supported = tensor?.rank === 1 || tensor?.rank === 2;
  const Slot = state.explorer === 'Tensor Explorer' ? (slots.tensor ?? TensorExplorer) : state.explorer === 'Tokenizer Explorer' ? slots.tokenizer : slots.architecture;
  const canCompose = state.session && state.sessionStatus === 'ready' &&
    (state.explorer !== 'Tensor Explorer' || supported);
  const context = useMemo<ExplorerContextValue | null>(() => canCompose && state.session ? {
    client: controller.explorerClient(state.view), session: state.session, sessionId: state.session.id,
    selectedTensor: tensor, selection: state.view,
    reportStatus: (status) => controller.reportStatus(state.view, status),
  } : null, [canCompose, controller, state.session, state.view, tensor]);
  const architectureLoading = state.sessionStatus === 'loading' ? 'Loading model session…' :
    state.sessionStatus === 'failed' ? 'Session unavailable. Retry the session to load the architecture.' :
      state.sessionStatus === 'ready' ? 'Retrieving prepared architecture…' : 'Open a model session to use this explorer.';
  const surface = <WorkingSurface title={`${state.explorer} workspace`} showTitle={false}>
    {state.explorer === 'Tensor Explorer' && tensor && <>
      {!supported ? <section className="tensor-explorer" aria-label="Tensor metadata view">
        <ViewerPanel header={<TensorHeader key={tensor.id} tensor={tensor} />}>
          <p>Direct viewing supports complete rank-1 and rank-2 tensors. This rank-{tensor.rank} tensor is available for metadata inspection only.</p>
        </ViewerPanel>
      </section> : slots.tensor && <TensorHeader key={tensor.id} tensor={tensor} />}
    </>}
    {context ? <ExplorerContext.Provider key={state.viewRevision} value={context}>
      {Slot ? <Slot {...context} /> : state.explorer === 'Tokenizer Explorer' ?
        <Suspense fallback={<p role="status">Loading prompt editor…</p>}><TokenizerExplorer {...context} tokenizerAvailable={model?.tokenizer_available ?? true} /></Suspense> :
        <Suspense fallback={<div className="architecture-explorer explorer-card architecture-empty" aria-label="Architecture capability"><header className="architecture-empty-heading">Architecture</header><p className="architecture-capability-state" role="status">Retrieving prepared architecture…</p></div>}><ArchitectureExplorer {...context} views={graphViews} diagnostics={controller.diagnostics} tokenizerAvailable={model?.tokenizer_available ?? false} onInspect={slots.inspectArchitecture} /></Suspense>}
      {state.viewStatus !== 'idle' && <p role="status" data-state={state.viewStatus}>{state.viewStatus === 'failed' ? 'Operation failed.' : `Operation ${state.viewStatus}.`}</p>}
    </ExplorerContext.Provider> : state.explorer === 'Architecture Explorer' ?
      <div className="architecture-explorer explorer-card architecture-empty" aria-label="Architecture capability"><header className="architecture-empty-heading">Architecture</header><p className="architecture-capability-state" role="status">{architectureLoading}</p></div> : !tensor || state.explorer !== 'Tensor Explorer' ?
        <p>{state.session ? 'Select a tensor to inspect.' : 'Open a model session to use this explorer.'}</p> : null}
  </WorkingSurface>;
  return <>
    <a className="skip-link" href="#workspace">Skip to workspace</a>
    <div className="app-shell">
      <AppBar state={state} controller={controller} model={model} backend={config.backendBaseUrl} />
      <div className="workspace-frame">
        <div className="workspace-notices">
          {!state.session && state.catalogue === 'loading' && <p role="status">Loading models…</p>}
          {!state.session && state.catalogue === 'complete' && !state.models.length && <p role="status">No models available on this backend.</p>}
          {!state.session && state.catalogue === 'failed' && <p role="alert">Could not load models. The backend is unreachable or returned an invalid response.</p>}
          {!state.session && state.message && <p role="status">{state.message}</p>}
          {state.sessionStatus === 'failed' && <Button onClick={() => controller.retrySession()}>Retry session</Button>}
          {!state.storageAvailable && <p>Tab storage is unavailable. This session cannot be recovered after refresh.</p>}
        </div>
        <TensorWorkspace enabled={state.explorer === 'Tensor Explorer'} inventory={<>
            {state.sessionStatus === 'loading' && <p role="status">Loading model session…</p>}
            {!state.session && state.sessionStatus !== 'loading' && <p role="status">Open a model session to browse tensors.</p>}
            {state.inventory === 'loading' && <p role="status">Loading tensor inventory…</p>}
            {state.inventory === 'failed' && <><p role="status">{state.message}</p><Button onClick={controller.loadInventory}>Retry tensor inventory</Button></>}
            {state.inventoryCoverage === 'partial' && <p role="status">Partial tensor inventory: some parameters cannot be inspected numerically.</p>}
            {state.inventoryDiagnostics.map((d, i) => <p key={i}>{d.message}</p>)}
            {state.inventory === 'complete' && !state.tensors.length && <p>No tensors available.</p>}
            <TensorTree tensors={state.tensors} selectedId={tensor?.id} onSelect={controller.selectTensor} />
          </>}>
          {state.explorer === 'Architecture Explorer' ? <ArchitectureWorkspace browserPlaceholder browser={<p role="status" className="architecture-browser-status">{architectureLoading}</p>}>
            {surface}
          </ArchitectureWorkspace> : surface}
        </TensorWorkspace>
      </div>
      <AppStatusBar state={state} model={model} onRetry={controller.loadModels} />
      <ToastHost toasts={state.toasts} onDismiss={controller.feedback.dismiss} />
    </div>
  </>;
}
