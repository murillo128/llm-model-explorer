import { useCallback, useRef, useState } from 'react';
import type { ExplorerContextValue } from '../app/explorer-context';
import { Button } from '../components/Button';
import { TensorExplorerController } from './tensor-explorer-controller';
import type { ExplorerStatus, ResultState } from './tensor-explorer-controller';

const stateText: Record<ResultState, string> = {
  loading: 'Waiting for data — incomplete', streaming: 'Receiving data — incomplete',
  complete: 'Complete', failed: 'Failed — incomplete', cancelled: 'Cancelled — incomplete', unneeded: 'Not required',
};

export function TensorExplorer(context: ExplorerContextValue) {
  const tensor = context.selectedTensor;
  if (!tensor || (tensor.rank !== 1 && tensor.rank !== 2)) return <p>Direct viewing supports rank-1 and rank-2 tensors.</p>;
  if (tensor.numel === 0) return <p role="status">Empty tensor — no values to render.</p>;
  return <LoadedTensorExplorer {...context} />;
}

function LoadedTensorExplorer(context: ExplorerContextValue) {
  const controller = useRef<TensorExplorerController | null>(null);
  const [status, setStatus] = useState<ExplorerStatus>({ tensor: 'loading', statistics: 'loading', distributions: context.selectedTensor!.rank === 2 ? 'loading' : 'unneeded', rendering: 'ready' });
  const [allocationFailed, setAllocationFailed] = useState(false);
  const mount = useCallback((host: HTMLDivElement | null) => {
    if (!host) return;
    try {
      controller.current = new TensorExplorerController(host, context, setStatus);
    } catch {
      setAllocationFailed(true);
    }
    return () => { controller.current?.dispose(); controller.current = null; };
  }, [context]);
  const active = [status.tensor, status.statistics, status.distributions].some((state) => state === 'loading' || state === 'streaming');
  return <section className="tensor-explorer" aria-label="Tensor scientific view">
    <p className="metadata">One value per device pixel · columns → · rows ↓</p>
    {allocationFailed || status.rendering === 'failed' ? <p role="alert">Exact rendering is unavailable. WebGL2 resources could not be allocated or were lost. Select the tensor again to retry.</p> : null}
    {!allocationFailed && <div className="tensor-results" aria-label="Result status">
      {(['tensor', 'statistics', 'distributions'] as const).filter((result) => status[result] !== 'unneeded').map((result) =>
        <p key={result} role="status" data-result={result} data-state={status[result]}><span className="section-label">{result}</span> · {stateText[status[result]]}</p>)}
    </div>}
    <div ref={mount} />
    {active && !allocationFailed && <Button onClick={() => controller.current?.cancel()}>Cancel loading</Button>}
  </section>;
}
