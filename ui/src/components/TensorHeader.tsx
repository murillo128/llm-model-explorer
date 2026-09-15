import type { ReactNode } from 'react';
import type { TensorDescriptor } from '../app/session-controller';
import { PanelHeader } from '../matrix-explorer/PanelHeader';
import { InfoPopover } from '../matrix-explorer/InfoPopover';
import { compactValue } from '../rendering/distribution-scale';
import type { DistributionDomain } from '../rendering/distribution-scale';

export function TensorHeader({ tensor, status, actions, domain, showInformation = true }: {
  tensor: TensorDescriptor; status?: ReactNode; actions?: ReactNode;
  domain?: DistributionDomain | undefined; showInformation?: boolean | undefined;
}) {
  const path = tensor.path.join(' › ') || tensor.name;
  const finite = domain && domain.minimum !== null && domain.maximum !== null;
  return <PanelHeader
    identity={<h2 className="tensor-identity" title={path}>{path}</h2>}
    information={showInformation && <InfoPopover label="Tensor information">
      <dl className="tensor-metadata metadata">
        <dt>Logical path</dt><dd>{path}</dd>
        <dt>Rank</dt><dd>{tensor.rank}</dd>
        <dt>Elements</dt><dd>{tensor.numel.toLocaleString()}</dd>
        <dt>Storage dtype</dt><dd>{tensor.storage_dtype}</dd>
        {tensor.storage_format !== undefined && <><dt>Storage format</dt><dd>{tensor.storage_format}</dd></>}
        <dt>Logical dtype</dt><dd>{tensor.logical_dtype}</dd>
        {tensor.rank === 2 && <>
          <dt>Distribution domain</dt><dd>{!domain ? 'Bin domain unavailable' : !finite
            ? 'No finite values · bin domain and true min/max unavailable'
            : <>Full-range bins{domain.minimum === domain.maximum && <> · Constant · samples in bin 50; no value span</>}</>}</dd>
          <dt>True finite minimum</dt><dd title={finite ? `True finite minimum: ${domain.minimum}` : undefined}>
            {finite ? compactValue(domain.minimum!) : 'Unavailable'}</dd>
          <dt>True finite maximum</dt><dd title={finite ? `True finite maximum: ${domain.maximum}` : undefined}>
            {finite ? compactValue(domain.maximum!) : 'Unavailable'}</dd>
        </>}
      </dl>
    </InfoPopover>}
    summary={<span className="tensor-summary metadata">[{tensor.shape.join(' × ')}] · {tensor.storage_dtype}</span>}
    status={status} actions={actions} />;
}
