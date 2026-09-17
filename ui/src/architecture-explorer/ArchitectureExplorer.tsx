import { useEffect, useState } from 'react';
import type { ExplorerContextValue } from '../app/explorer-context';
import { Lifetime } from '../app/lifetime';
import { ApiFailure } from '../api/errors';
import type { components } from '../api/generated/types';
import { ArchitectureCanvas } from './ArchitectureCanvas';
import type { ArchitectureSelection } from './ArchitectureCanvas';
import { GraphViews } from './graph';
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
  views: GraphViews; tokenizerAvailable: boolean; onInspect?: ((selection: ArchitectureSelection) => void) | undefined;
};
export function ArchitectureExplorer(props: Props) {
  return <SessionArchitectureExplorer key={JSON.stringify([props.session.id, props.session.model_id])} {...props} />;
}
function SessionArchitectureExplorer(props: Props) {
  const { client, session, selection, views, tokenizerAvailable, onInspect } = props;
  const [inspected, setInspected] = useState<ArchitectureSelection | null>(null);
  const [result, setResult] = useState<{ response?: Response; inventory?: components['schemas']['TensorInventory']; error?: string }>({});
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const request = new Lifetime();
    const detach = selection.onDispose(request.dispose);
    const timeout = setTimeout(() => {
      if (request.isCurrent()) { request.dispose(); setResult({ error: 'Architecture retrieval timed out. Retry retrieval.' }); }
    }, 15_000);
    // Inventory is required to validate actionable parameter bindings; no tensor bytes or tokenization.
    void client.listTensors(session.id, request.signal).then(request.guard(async (inventory) => {
      try {
        const response = await client.getArchitecture(session.id, { modelId: session.model_id, inventory, tokenizerAvailable }, request.signal);
        if (request.isCurrent()) setResult({ response, inventory });
      } catch (error) {
        if (request.isCurrent()) setResult({ error: error instanceof ApiFailure && error.detail?.code === 'model_content_changed'
          ? 'Model content changed. Close this session and open a fresh session.' : 'Architecture retrieval failed or returned an invalid graph. Retry retrieval.' });
      } finally { clearTimeout(timeout); }
    }), request.guard(() => { clearTimeout(timeout); setResult({ error: 'Could not validate the model inventory. Retry retrieval.' }); }));
    return () => { clearTimeout(timeout); request.dispose(); detach(); };
  }, [client, session, selection, tokenizerAvailable, retry]);
  const response = result.response;
  if (result.error) return <div role="alert">{result.error} <button onClick={() => { setResult({}); setRetry(retry + 1); }}>Retry retrieval</button></div>;
  if (!response) return <p role="status">Retrieving prepared architecture…</p>;
  if (response.status === 'unavailable') return <div role="status"><p>{unavailable[response.reason]}</p>
    {response.requires_restart && response.reason !== 'restart_required' && <p>Restart the backend to prepare this model again.</p>}
    {response.diagnostics.map((d, i) => <p key={i}>{d.message}</p>)}
  </div>;
  return <>
    {response.graph.scope === 'model_defined' && <p role="note">Model-supplied architecture. Structure and weight bindings are validated; equivalence to model code is not verified.</p>}
    {response.diagnostics.map((d, i) => <p key={i} role="status">{d.message}</p>)}
    <ArchitectureCanvas key={JSON.stringify([session.id, response.model_id, response.graph.graph_id])} graph={response.graph}
      modelId={response.model_id} sessionId={session.id} view={views.get(response.model_id, response.graph)} onDismissInspection={() => setInspected(null)} onInspect={(value) => { setInspected(value); onInspect?.(value); }} />
    {inspected && result.inventory && inspected.sessionId === session.id && inspected.modelId === response.model_id && inspected.graphId === response.graph.graph_id &&
      <ArchitectureInspection key={JSON.stringify([inspected.sessionId, inspected.graphId, inspected.node?.id, inspected.boundary, inspected.parameterId, Boolean(inspected.structureOnly)])} context={props}
        graph={response.graph} diagnostics={response.diagnostics} inventory={result.inventory} selected={inspected} onClose={() => setInspected(null)} />}
  </>;
}
