import { captureBookmark, restoreInspection } from './navigation-bookmark';
import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { NeuronInspector } from './NeuronInspector';
import { useNeuronCapabilities } from './useNeuronCapabilities';
import { neuronForInspection } from './neuron-selection';
import type { NeuronModule, NeuronSelectionContext, NeuronViewState } from './neuron-selection';
import { createPortal } from 'react-dom';
import type { ExplorerContextValue } from '../app/explorer-context';
import { Lifetime } from '../app/lifetime';
import { ApiFailure } from '../api/errors';
import type { components } from '../api/generated/types';
import { ArchitectureCanvas } from './ArchitectureCanvas';
import type { ArchitectureSelection } from './ArchitectureCanvas';
import { GraphViews } from './graph';
import { DiagnosticBand } from '../app/ModelDiagnostics';
import { ModelDiagnostics, finding } from '../app/model-diagnostics';
import { interfaceIndex } from './interfaces';
import { ArchitectureInspection } from './ArchitectureInspection';
import { useArchitectureBrowserTarget } from './architecture-browser-target';

type Response = components['schemas']['ArchitectureResponse'];
const unavailable = {
  unsupported_architecture: 'Architecture is not supported for this model.',
  analysis_failed: 'Architecture preparation failed for this model.',
  restart_required: 'The model is new or changed. Restart the backend to prepare its architecture.',
  unsupported_size: 'This architecture exceeds the supported response size.',
  cache_unavailable: 'The prepared architecture cache is unavailable.',
};
type Props = ExplorerContextValue & {
  diagnostics?: ModelDiagnostics;
  neuronExperiment?: (context: NeuronSelectionContext) => ReactNode;
  views: GraphViews; tokenizerAvailable: boolean; onInspect?: ((selection: ArchitectureSelection) => void) | undefined;
};
export function ArchitectureExplorer(props: Props) {
  return <SessionArchitectureExplorer key={JSON.stringify([props.session.id, props.session.model_id, props.session.model_revision])} {...props} />;
}
function SessionArchitectureExplorer(props: Props) {
  const browserTarget = useArchitectureBrowserTarget();
  const { client, session, selection, views, onInspect, refreshKey, refreshPresentation } = props;
  const [localDiagnostics] = useState(() => { const store = new ModelDiagnostics(); store.activate(session); return store; });
  const diagnostics = props.diagnostics ?? localDiagnostics;
  const [inspected, setInspected] = useState<ArchitectureSelection | null>(null);
  const [result, setResult] = useState<{ response?: Response; inventory?: components['schemas']['TensorInventory'] | undefined; error?: string }>({});
  const [retry, setRetry] = useState(0);
  const capabilities = useNeuronCapabilities(props);
  const [experimentTarget, setExperimentTarget] = useState<HTMLDivElement | null>(null);
  const [neuron, setNeuron] = useState(() => ({ ...views.neuronView(session) }));
  const neuronTrigger = useRef<HTMLElement | null>(null);
  const changeNeuron = (patch: Partial<NeuronViewState>) => {
    Object.assign(views.neuronView(session), patch);
    setNeuron(previous => ({ ...previous, ...patch }));
  };
  const openNeuron = (trigger: HTMLElement, module?: NeuronModule) => {
    neuronTrigger.current = trigger;
    setInspected(null);
    changeNeuron({ open: true, ...(module && module.module_id !== neuron.moduleId ? { moduleId: module.module_id, index: 0, scroll: 0 } : {}) });
  };
  const closeNeuron = () => {
    changeNeuron({ open: false });
    if (neuronTrigger.current?.isConnected) neuronTrigger.current.focus({ preventScroll: true });
  };
  useEffect(() => {
    const request = new Lifetime();
    const fail = (message: string) => {
      diagnostics.observe(session, 'Architecture', [finding(session.model_id, session.id, 'Architecture', { code: 'retrieval_failed', message }, 'error')]);
      setResult(previous => ({ inventory: previous.inventory, error: message }));
    };
    const detach = selection.onDispose(request.dispose);
    const timeout = setTimeout(() => {
      if (request.isCurrent()) { request.dispose(); fail('Architecture retrieval timed out. Retry retrieval.'); }
    }, 120_000);
    // Inventory is required to validate actionable parameter bindings; no tensor bytes or tokenization.
    void client.listTensors(session.id, request.signal).then(request.guard(async (inventory) => {
      setResult(previous => ({ ...previous, inventory }));
      diagnostics.observe(session, 'Tensor inventory', inventory.diagnostics.map((d) => finding(session.model_id, session.id, 'Tensor inventory', d, 'warning')));
      try {
        const response = await client.getArchitecture(session.id, { modelId: session.model_id }, request.signal);
        if (request.isCurrent()) {
          const graph = response.status === 'available' ? response.graph : undefined;
          const generation = graph?.graph_id ?? session.id;
          diagnostics.graph(session, session.model_id, generation, graph);
          const severity = response.status === 'unavailable' && ['analysis_failed', 'cache_unavailable'].includes(response.reason) ? 'error' : 'warning';
          const records = [...response.diagnostics, ...(graph?.diagnostics ?? [])].map((d) => {
            const label = d.node_id ? graph?.nodes.find((n) => n.id === d.node_id)?.label ?? 'Component' : undefined;
            return finding(session.model_id, generation, 'Architecture', d, severity,
              [d.node_id ? `${label} · ${d.node_id}` : '', d.parameter_id].filter(Boolean).join(' · ') || undefined, label);
          });
          if (graph) for (const [id, message] of interfaceIndex(graph).notices) records.push(finding(session.model_id, generation, 'Architecture',
            { code: 'interface_mapping_ambiguous', node_id: id, message }, 'warning', `${graph.nodes.find((n) => n.id === id)!.label} · ${id}`,
            graph.nodes.find((n) => n.id === id)!.label));
          if (response.status === 'unavailable' && !records.length) records.push(finding(session.model_id, generation, 'Architecture',
            { code: response.reason, message: unavailable[response.reason] }, severity));
          diagnostics.observe(session, 'Architecture', records, graph?.scope === 'model_defined');
          const bookmark = refreshKey && refreshPresentation?.consume(refreshKey);
          if (graph && bookmark) {
            const restored = views.get(session.model_id, graph, bookmark);
            const inspection = restoreInspection(graph, bookmark, inventory, session.id, session.model_id, document.body);
            if (inspection) setInspected(inspection);
            else if (bookmark.inspection) restored.notice ??= 'The previous inspection is no longer available.';
          }
          setResult({ response, inventory });
        }
      } catch (error) {
        if (request.isCurrent()) fail(error instanceof ApiFailure && error.detail?.code === 'model_content_changed'
          ? 'Updating model…'
          : error instanceof ApiFailure && error.kind === 'protocol'
            ? error.message.length <= 240 ? error.message : 'Architecture response failed validation.'
            : 'Architecture retrieval failed or returned an invalid graph. Retry retrieval.');
      } finally { clearTimeout(timeout); }
    }), request.guard(() => { clearTimeout(timeout); fail('Could not validate the model inventory. Retry retrieval.'); }));
    return () => { clearTimeout(timeout); request.dispose(); detach(); };
  }, [client, session, selection, retry, diagnostics, views, refreshKey, refreshPresentation]);
  const response = result.response;
  const graph = response?.status === 'available' ? response.graph : undefined;
  useEffect(() => {
    if (!graph || !props.refreshPresentation) return;
    return props.refreshPresentation.register(selection, () => captureBookmark(graph, views.get(session.model_id, graph), inspected, result.inventory));
  }, [graph, props.refreshPresentation, selection, views, session.model_id, inspected, result.inventory]);
  const band = <><DiagnosticBand store={diagnostics} /><div className="neuron-entry">
    <button aria-expanded={neuron.open} onClick={event => openNeuron(event.currentTarget)}>Neurons</button>
    <span>{capabilities.capabilities?.observation.status === 'unavailable' ? 'Neuron execution unavailable' : 'Concrete MLP coordinates and linked weights'}</span>
  </div></>;
  const panel = neuron.open && <NeuronInspector context={props} capabilities={capabilities} inventory={result.inventory} state={neuron}
    onChange={changeNeuron} onBack={closeNeuron} tokenizerAvailable={props.tokenizerAvailable} experiment={props.neuronExperiment} experimentTarget={experimentTarget} />;
  const experiment = <div ref={setExperimentTarget} className="neuron-experiment" aria-label="Neuron experiment" hidden={!neuron.open || !props.neuronExperiment} />;
  const matchedModule = inspected && neuronForInspection(capabilities.capabilities, inspected);
  // Graph retrieval replaces only this first pane. The neuron panel and its
  // independent weight consumers must retain their positions and lifetimes.
  return <>
    {browserTarget && (!response || response.status === 'unavailable') && createPortal(<p className="architecture-browser-state" role={result.error ? 'alert' : 'status'}>
      {result.error ? 'Architecture browser unavailable. Retry retrieval.' : !response ? 'Retrieving prepared architecture…' : unavailable[response.reason]}
    </p>, browserTarget)}
    <div className={`architecture-neuron-layout${neuron.open ? ' neuron-open' : ''}${neuron.open && props.neuronExperiment ? ' has-neuron-experiment' : ''}`}>
    {response?.status === 'available' ? <ArchitectureCanvas key={JSON.stringify([session.id, response.model_id, response.graph.graph_id])} graph={response.graph} notices={band}
      modelId={response.model_id} sessionId={session.id} view={views.get(response.model_id, response.graph)} onDismissInspection={() => setInspected(null)} onInspect={(value) => { setInspected(value); onInspect?.(value); }} />
      : <div tabIndex={-1} className="architecture-explorer explorer-card architecture-empty" aria-label="Architecture capability">
    <header className="architecture-empty-heading">Architecture</header>
    {band}
    <div className="architecture-capability-state">
      {result.error ? <div role="alert">{result.error} <button onClick={() => { setResult({}); setRetry(retry + 1); }}>Retry retrieval</button></div>
        : !response ? <p role="status">Retrieving prepared architecture…</p>
        : <div role="status"><p>{unavailable[response.reason]}</p>
          {response.diagnostics.length > 0 && <ul className="architecture-failure-details">
            {response.diagnostics.map((diagnostic, index) => <li key={`${diagnostic.code}-${index}`}>{diagnostic.message}</li>)}
          </ul>}
          {response.requires_restart && response.reason !== 'restart_required' && <p>Restart the backend to prepare this model again.</p>}</div>}
    </div>
    </div>}
    {panel}{experiment}</div>
    {response?.status === 'available' && inspected && result.inventory && inspected.sessionId === session.id && inspected.modelId === response.model_id && inspected.graphId === response.graph.graph_id &&
      <ArchitectureInspection key={JSON.stringify([inspected.sessionId, inspected.graphId, inspected.node?.id, inspected.boundary, inspected.parameterId, Boolean(inspected.structureOnly)])} context={props}
        onExploreNeurons={matchedModule ? () => openNeuron(inspected.trigger, matchedModule) : undefined}
        graph={response.graph} diagnostics={diagnostics.getSnapshot().records} inventory={result.inventory} selected={inspected} onParameterChange={(parameterId) => setInspected((previous) => previous ? { ...previous, parameterId } : null)} onClose={() => setInspected(null)} />}
  </>;
}
