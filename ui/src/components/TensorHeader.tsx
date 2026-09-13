import { useEffect, useId, useRef, useState } from 'react';
import type { TensorDescriptor } from '../app/session-controller';

export function TensorHeader({ tensor }: { tensor: TensorDescriptor }) {
  const [expanded, setExpanded] = useState(false);
  const id = useId();
  const host = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  const path = tensor.path.join(' › ') || tensor.name;
  useEffect(() => {
    if (!expanded) return;
    close.current?.focus();
    const dismiss = (event: PointerEvent) => {
      if (!host.current?.contains(event.target as Node)) setExpanded(false);
    };
    document.addEventListener('pointerdown', dismiss);
    return () => document.removeEventListener('pointerdown', dismiss);
  }, [expanded]);
  return <div className="tensor-header" ref={host} onBlur={(event) => {
    if (!event.currentTarget.contains(event.relatedTarget)) setExpanded(false);
  }} onKeyDown={(event) => {
    if (event.key === 'Escape' && expanded) {
      event.stopPropagation(); setExpanded(false); trigger.current?.focus();
    }
  }}>
    <h2 className="tensor-identity" title={path}>{path}</h2>
    <button ref={trigger} type="button" className="tensor-info" aria-label="Tensor information and help"
      title="Tensor information and help" aria-expanded={expanded} aria-controls={id} aria-haspopup="dialog"
      onClick={() => setExpanded(!expanded)}><span aria-hidden="true">ⓘ</span></button>
    <span className="tensor-summary metadata">[{tensor.shape.join(' × ')}] · {tensor.storage_dtype}</span>
    {expanded && <section id={id} className="tensor-details" role="dialog" aria-label="Tensor information and help">
      <button ref={close} type="button" className="button" onClick={() => {
        setExpanded(false); trigger.current?.focus();
      }}>Close tensor information</button>
      <dl className="tensor-metadata metadata">
        <dt>Logical path</dt><dd>{path}</dd>
        <dt>Rank</dt><dd>{tensor.rank}</dd>
        <dt>Elements</dt><dd>{tensor.numel.toLocaleString()}</dd>
        <dt>Storage dtype</dt><dd>{tensor.storage_dtype}</dd>
        {tensor.storage_format !== undefined && <><dt>Storage format</dt><dd>{tensor.storage_format}</dd></>}
        <dt>Logical dtype</dt><dd>{tensor.logical_dtype}</dd>
      </dl>
      <p>One value per device pixel. Columns run horizontally; rows run vertically. Oversized tensors scroll at their native size.</p>
      {tensor.rank === 2 && <p>Focus the matrix and use arrow keys to inspect cells. Escape clears inspection.</p>}
    </section>}
  </div>;
}
