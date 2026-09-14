import { useMemo, useRef, useState } from 'react';
import { MatrixExplorer } from '../matrix-explorer';
import type { MatrixSource } from '../matrix-explorer';
import type { ExplorerContextValue } from '../app/explorer-context';
import { TensorHeader } from '../components/TensorHeader';
import { TensorExplorerController } from './tensor-explorer-controller';
import type { ExplorerStatus, ResultState } from './tensor-explorer-controller';

const stateText: Record<ResultState, string> = {
  loading: 'Waiting', streaming: 'Receiving',
  complete: 'Complete', failed: 'Failed — incomplete', cancelled: 'Cancelled — incomplete', unneeded: 'Not required',
};

export function TensorExplorer(context: ExplorerContextValue) {
  const tensor = context.selectedTensor;
  if (!tensor || (tensor.rank !== 1 && tensor.rank !== 2)) return <p>Direct viewing supports rank-1 and rank-2 tensors.</p>;
  if (tensor.numel === 0) return <><TensorHeader tensor={tensor} /><p role="status">Empty tensor — no values to render.</p></>;
  return <LoadedTensorExplorer {...context} />;
}

function LoadedTensorExplorer({ client, sessionId, selectedTensor, selection }: ExplorerContextValue) {
  const controller = useRef<TensorExplorerController | null>(null);
  const [status, setStatus] = useState<ExplorerStatus>({ tensor: 'loading', statistics: 'loading', distributions: selectedTensor!.rank === 2 ? 'loading' : 'unneeded' });
  const [allocationFailed, setAllocationFailed] = useState(false);
  const source = useMemo<MatrixSource>(() => ({
    descriptor: selectedTensor!,
    distributions: selectedTensor!.rank === 2,
    subscribe(updates) {
      try {
        const current = new TensorExplorerController(updates, { client, sessionId, selectedTensor, selection }, setStatus);
        controller.current = current;
        return () => {
          current.dispose();
          if (controller.current === current) controller.current = null;
        };
      } catch {
        // The controller cancels any partially started operations before throwing.
        setStatus({ tensor: 'failed', statistics: 'failed', distributions: selectedTensor!.rank === 2 ? 'failed' : 'unneeded' });
        return () => {};
      }
    },
  }), [client, sessionId, selectedTensor, selection]);
  const active = [status.tensor, status.statistics, status.distributions].some((state) => state === 'loading' || state === 'streaming');
  return <section className="tensor-explorer" aria-label="Tensor scientific view">
    <MatrixExplorer source={source} label="Tensor matrix; scroll to inspect all values"
      header={<TensorHeader tensor={selectedTensor!} status={!allocationFailed && <div className="tensor-results" aria-label="Result status">
        {(['tensor', 'statistics', 'distributions'] as const).filter((result) => status[result] !== 'unneeded' && status[result] !== 'complete').map((result) =>
          <p key={result} role={status[result] === 'failed' ? 'alert' : 'status'} data-result={result} data-state={status[result]}>
            {(status[result] === 'loading' || status[result] === 'streaming') && <span className="operation-spinner" aria-hidden="true" />}
            <span>{result}</span> · {stateText[status[result]]}</p>)}
      </div>} actions={!allocationFailed && active && <button type="button" className="matrix-header-cancel"
        aria-label="Cancel loading" title="Cancel loading" onClick={() => controller.current?.cancel()}>×</button>} />}
      onRenderingStateChange={(state) => setAllocationFailed(state === 'failed' && controller.current === null)} />
  </section>;
}
