import { lazy, Suspense, useEffect, useMemo, useState, useSyncExternalStore } from 'react';
import { GraphViews } from '../architecture-explorer/graph';
import { TensorExplorer } from '../explorers/TensorExplorer';
import { ApiClient } from '../api/client';
import type { RuntimeConfig } from '../api/runtime-config';
import { Button } from '../components/Button';
import { AppBar, AppStatusBar } from './AppChrome';
import { TensorHeader } from '../components/TensorHeader';
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
    client: controller.client, session: state.session, sessionId: state.session.id,
    selectedTensor: tensor, selection: state.view,
    reportStatus: (status) => controller.reportStatus(state.view, status),
  } : null, [canCompose, controller, state.session, state.view, tensor]);
  return <>
    <a className="skip-link" href="#workspace">Skip to workspace</a>
    <div className="app-shell">
      <AppBar state={state} controller={controller} model={model} backend={config.backendBaseUrl} />
      <div className="workspace-frame">
        <div className="workspace-notices">
          {state.catalogue === 'loading' && <p role="status">Loading models…</p>}
          {state.catalogue === 'complete' && !state.models.length && <p role="status">No models available on this backend.</p>}
          {state.catalogue === 'failed' && <p role="alert">Could not load models. The backend is unreachable or returned an invalid response.</p>}
          {state.message && <p role="status">{state.message}</p>}
          {state.sessionStatus === 'failed' && <Button onClick={() => controller.retrySession()}>Retry session</Button>}
          {!state.storageAvailable && <p>Tab storage is unavailable. This session cannot be recovered after refresh.</p>}
        </div>
        <TensorWorkspace enabled={state.explorer === 'Tensor Explorer' && !!state.session} inventory={<>
            {state.inventory === 'loading' && <p role="status">Loading tensor inventory…</p>}
            {state.inventory === 'failed' && <Button onClick={controller.loadInventory}>Retry tensor inventory</Button>}
            {state.inventoryCoverage === 'partial' && <p role="status">Partial tensor inventory: some parameters cannot be inspected numerically.</p>}
            {state.inventoryDiagnostics.map((d, i) => <p key={i}>{d.message}</p>)}
            {state.inventory === 'complete' && !state.tensors.length && <p>No tensors available.</p>}
            <TensorTree tensors={state.tensors} selectedId={tensor?.id} onSelect={controller.selectTensor} />
          </>}>
          <WorkingSurface title={`${state.explorer} workspace`} showTitle={false}>
            {state.explorer === 'Tensor Explorer' && tensor && <>
              {(!supported || slots.tensor) && <TensorHeader key={tensor.id} tensor={tensor} />}
              {!supported && <p>Direct viewing supports complete rank-1 and rank-2 tensors. This rank-{tensor.rank} tensor is available for metadata inspection only.</p>}
            </>}
            {context ? <ExplorerContext.Provider key={state.viewRevision} value={context}>
              {Slot ? <Slot {...context} /> : state.explorer === 'Tokenizer Explorer' ?
                <Suspense fallback={<p role="status">Loading prompt editor…</p>}><TokenizerExplorer {...context} tokenizerAvailable={model?.tokenizer_available ?? true} /></Suspense> :
                <Suspense fallback={<p role="status">Loading architecture canvas…</p>}><ArchitectureExplorer {...context} views={graphViews} tokenizerAvailable={model?.tokenizer_available ?? false} onInspect={slots.inspectArchitecture} /></Suspense>}
              {state.viewStatus !== 'idle' && <p role="status" data-state={state.viewStatus}>{state.viewStatus === 'failed' ? 'Operation failed.' : `Operation ${state.viewStatus}.`}</p>}
            </ExplorerContext.Provider> : !tensor || state.explorer !== 'Tensor Explorer' ?
              <p>{state.session ? 'Select a tensor to inspect.' : 'Open a model session to use this explorer.'}</p> : null}
          </WorkingSurface>
        </TensorWorkspace>
      </div>
      <AppStatusBar state={state} model={model} />
    </div>
  </>;
}
