import { useEffect, useState } from 'react';
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
  views: GraphViews; tokenizerAvailable: boolean; onInspect?: ((selection: ArchitectureSelection) => void) | undefined;
};
export function ArchitectureExplorer(props: Props) {
  return <SessionArchitectureExplorer key={JSON.stringify([props.session.id, props.session.model_id])} {...props} />;
}
function SessionArchitectureExplorer(props: Props) {
  const { client, session, selection, views, tokenizerAvailable, onInspect } = props;
  const [localDiagnostics] = useState(() => { const store = new ModelDiagnostics(); store.activate(session); return store; });
  const diagnostics = props.diagnostics ?? localDiagnostics;
  const [inspected, setInspected] = useState<ArchitectureSelection | null>(null);
  const [result, setResult] = useState<{ response?: Response; inventory?: components['schemas']['TensorInventory']; error?: string }>({});
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const request = new Lifetime();
    const fail = (message: string) => {
      diagnostics.observe(session, 'Architecture', [finding(session.model_id, session.id, 'Architecture', { code: 'retrieval_failed', message }, 'error')]);
      setResult({ error: message });
    };
    const detach = selection.onDispose(request.dispose);
    const timeout = setTimeout(() => {
      if (request.isCurrent()) { request.dispose(); fail('Architecture retrieval timed out. Retry retrieval.'); }
    }, 15_000);
    // Inventory is required to validate actionable parameter bindings; no tensor bytes or tokenization.
    void client.listTensors(session.id, request.signal).then(request.guard(async (inventory) => {
      diagnostics.observe(session, 'Tensor inventory', inventory.diagnostics.map((d) => finding(session.model_id, session.id, 'Tensor inventory', d, 'warning')));
      try {
        const response = await client.getArchitecture(session.id, { modelId: session.model_id, inventory, tokenizerAvailable }, request.signal);
        if (request.isCurrent()) {
          const graph = response.status === 'available' ? response.graph : undefined;
          const generation = graph?.graph_id ?? session.id;
          diagnostics.graph(session, session.model_id, generation);
          const severity = response.status === 'unavailable' && ['analysis_failed', 'cache_unavailable'].includes(response.reason) ? 'error' : 'warning';
          const records = [...response.diagnostics, ...(graph?.diagnostics ?? [])].map((d) => finding(session.model_id, generation, 'Architecture', d, severity,
            [d.node_id ? `${graph?.nodes.find((n) => n.id === d.node_id)?.label ?? 'Component'} · ${d.node_id}` : '', d.parameter_id].filter(Boolean).join(' · ') || undefined));
          if (graph) for (const [id, message] of interfaceIndex(graph).notices) records.push(finding(session.model_id, generation, 'Architecture',
            { code: 'interface_mapping_ambiguous', node_id: id, message }, 'warning', `${graph.nodes.find((n) => n.id === id)!.label} · ${id}`));
          if (response.status === 'unavailable' && !records.length) records.push(finding(session.model_id, generation, 'Architecture',
            { code: response.reason, message: unavailable[response.reason] }, severity));
          diagnostics.observe(session, 'Architecture', records, graph?.scope === 'model_defined');
          setResult({ response, inventory });
        }
      } catch (error) {
        if (request.isCurrent()) fail(error instanceof ApiFailure && error.detail?.code === 'model_content_changed'
          ? 'Model content changed. Close this session and open a fresh session.' : 'Architecture retrieval failed or returned an invalid graph. Retry retrieval.');
      } finally { clearTimeout(timeout); }
    }), request.guard(() => { clearTimeout(timeout); fail('Could not validate the model inventory. Retry retrieval.'); }));
    return () => { clearTimeout(timeout); request.dispose(); detach(); };
  }, [client, session, selection, tokenizerAvailable, retry, diagnostics]);
  const response = result.response;
  const band = <DiagnosticBand store={diagnostics} />;
  if (!response || response.status === 'unavailable') return <div tabIndex={-1} className="architecture-explorer architecture-empty" aria-label="Architecture capability">
    <header className="architecture-empty-heading">Architecture</header>
    {band}
    <div className="architecture-capability-state">
      {result.error ? <div role="alert">{result.error} <button onClick={() => { setResult({}); setRetry(retry + 1); }}>Retry retrieval</button></div>
        : !response ? <p role="status">Retrieving prepared architecture…</p>
        : <div role="status"><p>{unavailable[response.reason]}</p>
          {response.requires_restart && response.reason !== 'restart_required' && <p>Restart the backend to prepare this model again.</p>}</div>}
    </div>
  </div>;
  return <>
    <ArchitectureCanvas key={JSON.stringify([session.id, response.model_id, response.graph.graph_id])} graph={response.graph} notices={band}
      modelId={response.model_id} sessionId={session.id} view={views.get(response.model_id, response.graph)} onDismissInspection={() => setInspected(null)} onInspect={(value) => { setInspected(value); onInspect?.(value); }} />
    {inspected && result.inventory && inspected.sessionId === session.id && inspected.modelId === response.model_id && inspected.graphId === response.graph.graph_id &&
      <ArchitectureInspection key={JSON.stringify([inspected.sessionId, inspected.graphId, inspected.node?.id, inspected.boundary, inspected.parameterId, Boolean(inspected.structureOnly)])} context={props}
        graph={response.graph} diagnostics={response.diagnostics} inventory={result.inventory} selected={inspected} onClose={() => setInspected(null)} />}
  </>;
}
