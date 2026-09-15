import type { ReactNode } from 'react';
import { PanelHeader } from '../matrix-explorer';
import { InfoPopover } from '../matrix-explorer/InfoPopover';
import type { DistributionDomain } from '../rendering/distribution-scale';
import type { EmbeddingResult, EmbeddingState } from './embedding-controller';

export function EmbeddingHeader({ state, status, domain, actions, cancel, metadataState = state }: {
  state?: EmbeddingState | undefined;
  metadataState?: EmbeddingState | undefined;
  status: string;
  domain?: DistributionDomain | undefined;
  actions?: ReactNode;
  cancel?: ((result: EmbeddingResult) => void) | undefined;
}) {
  const descriptor = metadataState?.source?.descriptor;
  const statistics = metadataState?.statisticsMetadata;
  const auxiliary = state && (['statistics', 'distributions'] as const).filter(result => state[result] !== 'complete');
  const pending = state && (['values', 'statistics', 'distributions'] as const).filter(result =>
    ['loading', 'streaming'].includes(result === 'values' ? state.status : state[result]));
  return <PanelHeader identity={<h2>Input Embeddings</h2>}
    information={descriptor && <InfoPopover label="Input embeddings information">
      <dl className="embedding-metadata metadata">
        <dt>Shape</dt><dd>[{descriptor.shape.join(' × ')}]</dd>
        <dt>Logical dtype</dt><dd>{descriptor.logical_dtype}</dd>
        <dt>Elements</dt><dd>{descriptor.numel.toLocaleString()}</dd>
        <dt>Rows</dt><dd>Token sequence positions, including repeated IDs</dd>
        <dt>Statistics</dt><dd>{statistics ? 'Exact requested input embeddings' : `${metadataState?.statistics ?? 'loading'} · statistics unavailable`}</dd>
        {statistics && <>
          <dt>Finite elements</dt><dd>{statistics.finite_count.toLocaleString()}</dd>
          <dt>Nonfinite elements</dt><dd>{statistics.non_finite_count.toLocaleString()}</dd>
          <dt>True finite minimum</dt><dd>{statistics.minimum ?? 'Unavailable'}</dd>
          <dt>True finite maximum</dt><dd>{statistics.maximum ?? 'Unavailable'}</dd>
          <dt>Mean</dt><dd>{statistics.mean ?? 'Unavailable'}</dd>
          <dt>Population standard deviation</dt><dd>{statistics.stddev ?? 'Unavailable'}</dd>
          <dt>Luminosity anchors (p01 / p99)</dt><dd>{statistics.percentiles.p01 ?? 'Unavailable'} / {statistics.percentiles.p99 ?? 'Unavailable'}</dd>
        </>}
        <dt>Distribution domain</dt><dd>{!domain ? 'Bin domain unavailable' : domain.minimum === null
          ? 'No finite values · bin domain unavailable' : <>Full-range bins · {domain.minimum} to {domain.maximum}
            {domain.minimum === domain.maximum && ' · Constant · samples in bin 50; no value span'}</>}</dd>
        <dt>Distributions</dt><dd>{metadataState?.distributions === 'complete' ? 'Complete' : `${metadataState?.distributions ?? 'loading'} · incomplete`}</dd>
      </dl>
    </InfoPopover>}
    summary={descriptor && <span className="embedding-shape">[{descriptor.shape.join(' × ')}] · {descriptor.logical_dtype}</span>}
    status={(status || !!auxiliary?.length) && <div className="embedding-results" aria-label="Embedding result status">
      {status && <span role={state?.status === 'failed' ? 'alert' : 'status'}>{status}</span>}
      {auxiliary?.map(result => <span key={result} role={state![result] === 'failed' ? 'alert' : 'status'}
        data-result={result} data-state={state![result]}>
        {['loading', 'streaming'].includes(state![result]) && <span className="operation-spinner" aria-hidden="true" />}
        {result} · {state![result] === 'loading' ? 'Waiting' : state![result] === 'streaming' ? 'Receiving'
          : state![result] === 'failed' ? 'Failed — incomplete' : 'Cancelled — incomplete'}
      </span>)}
    </div>}
    actions={<>{cancel && pending?.map(result => <button key={result} type="button" className="matrix-header-cancel"
      aria-label={`Cancel embedding ${result}`} title={`Cancel embedding ${result}`} onClick={() => cancel(result)}>×</button>)}{actions}</>} />;
}
