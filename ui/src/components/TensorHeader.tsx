import type { ReactNode } from 'react';
import type { TensorDescriptor } from '../app/session-controller';
import { PanelHeader } from '../matrix-explorer/PanelHeader';
import { InfoPopover } from '../matrix-explorer/InfoPopover';

export function TensorHeader({ tensor, status, actions, showInformation = true }: { tensor: TensorDescriptor; status?: ReactNode; actions?: ReactNode; showInformation?: boolean | undefined }) {
  const path = tensor.path.join(' › ') || tensor.name;
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
      </dl>
    </InfoPopover>}
    summary={<span className="tensor-summary metadata">[{tensor.shape.join(' × ')}] · {tensor.storage_dtype}</span>}
    status={status} actions={actions} />;
}
