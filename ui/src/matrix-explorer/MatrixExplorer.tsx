import { useEffectEvent, useLayoutEffect, useRef, useState } from 'react';
import { MatrixViewport } from '../rendering/matrix-viewport';
import type { Inspection } from '../rendering/matrix-inspection';
import type { RendererState } from '../rendering/tensor-renderer';
import { compactValue } from '../rendering/distribution-scale';
import type { DistributionDomain } from '../rendering/distribution-scale';
import { InspectionCard } from './InspectionCard';
import type { MatrixExplorerProps, MatrixUpdates } from './types';
import './matrix-explorer.css';

/** Transport-free scientific composition. The parent owns operation status/errors. */
export function MatrixExplorer({ source, header, label = 'Matrix; scroll to inspect all values',
  onCellSelect, onRowSelect, onColumnSelect, onRenderingStateChange, highlightedRow = null }: MatrixExplorerProps) {
  const host = useRef<HTMLDivElement>(null);
  const currentViewport = useRef<MatrixViewport | null>(null);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [failed, setFailed] = useState(false);
  const [domainState, setDomainState] = useState<{ source: typeof source; domain: DistributionDomain }>();
  const domain = domainState?.source === source ? domainState.domain : undefined;
  const selected = useRef<{ row: number; column: number } | null>(null);
  const inspect = useEffectEvent((value: Inspection | null) => {
    setInspection(value);
    const previous = selected.current;
    const cell = value ? { row: value.row, column: value.column } : null;
    selected.current = cell;
    if (previous?.row !== cell?.row || previous?.column !== cell?.column) onCellSelect?.(cell);
    if (previous?.row !== cell?.row) onRowSelect?.(cell?.row ?? null);
    if (previous?.column !== cell?.column) onColumnSelect?.(cell?.column ?? null);
  });
  const rendering = useEffectEvent((state: RendererState) => {
    setFailed(['failed', 'lost', 'needs-reconstruction'].includes(state));
    onRenderingStateChange?.(state);
  });
  useLayoutEffect(() => {
    let active = true;
    // A reused source object still starts a fresh delivery after A → B → A.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setDomainState(undefined);
    let allocated: MatrixViewport | undefined;
    let viewport: MatrixViewport;
    try {
      allocated = viewport = new MatrixViewport(host.current!, source.descriptor, {
        distributions: source.distributions ?? false,
        onInspection: inspect,
        onStateChange: rendering,
      });
      if (source.transfer) viewport.matrix.renderer.setTransfer(source.transfer);
      currentViewport.current = viewport;
    } catch {
      allocated?.dispose();
      // Report synchronous allocation failure before paint, like renderer state callbacks.
      rendering('failed');
      return;
    }
    // Every callback closes over this allocation, never a mutable current-view ref.
    const updates: MatrixUpdates = {
      values(values, offset) {
        if (!active) return;
        viewport.matrix.renderer.upload(values, offset);
        viewport.refresh();
      },
      transfer(parameters) {
        if (!active) return;
        viewport.matrix.renderer.setTransfer(parameters);
        viewport.refresh();
      },
      distribution(axis, counts, offset) {
        if (!active) return;
        const target = axis === 'rows' ? viewport.rows : viewport.columns;
        if (!target) throw new Error('This source has no distribution surfaces.');
        target.upload(counts, offset);
        viewport.refresh();
      },
      distributionDomain(domain) {
        if (!active) return;
        viewport.setDistributionDomain(domain);
        setDomainState({ source, domain });
      },
    };
    let detach: () => void;
    try { detach = source.subscribe(updates); }
    catch (error) {
      active = false;
      viewport.dispose();
      throw error; // Domain/subscription failures belong to the parent.
    }
    return () => {
      active = false;
      currentViewport.current = null;
      try { detach(); } finally { viewport.dispose(); }
    };
  }, [source]);
  useLayoutEffect(() => { currentViewport.current?.setLinkedRow(highlightedRow); }, [highlightedRow, source]);
  // Context/slot/callback changes do not restart subscriptions or upload weights.
  useLayoutEffect(() => {
    host.current?.querySelector('.matrix-scroll')?.setAttribute('aria-label', label);
  }, [label, source]);
  return <div className="matrix-explorer">
    {header && <div className="matrix-explorer-header">{header}</div>}
    {failed && <p role="alert">Exact rendering is unavailable. WebGL2 resources could not be allocated or were lost. Reopen this view to retry.</p>}
    {source.distributions && <div className="distribution-range" aria-label="Distribution range">
      {!domain ? 'Bin domain unavailable' : domain.minimum === null || domain.maximum === null
        ? 'No finite values · bin domain and true min/max unavailable'
        : <><span>Full-range bins</span><span title={`True finite minimum: ${domain.minimum}`}>min {compactValue(domain.minimum)}</span>
          <span title={`True finite maximum: ${domain.maximum}`}>max {compactValue(domain.maximum)}</span>
          {domain.minimum === domain.maximum && <span>Constant · samples in bin 50; no value span</span>}</>}
    </div>}
    <div ref={host} />
    {inspection && <InspectionCard inspection={inspection} />}
  </div>;
}
