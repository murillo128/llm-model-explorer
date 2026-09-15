import { useEffectEvent, useLayoutEffect, useRef, useState } from 'react';
import { MatrixViewport } from '../rendering/matrix-viewport';
import type { Inspection } from '../rendering/matrix-inspection';
import type { RendererState } from '../rendering/tensor-renderer';
import type { DistributionDomain } from '../rendering/distribution-scale';
import { PanelHeader } from './PanelHeader';
import { ViewerPanel } from './ViewerPanel';
import { InspectionCard } from './InspectionCard';
import type { MatrixExplorerProps, MatrixUpdates } from './types';
import './matrix-explorer.css';

/** Transport-free scientific composition. The parent owns operation status/errors. */
export function MatrixExplorer({ source, header, label = 'Matrix; scroll to inspect all values',
  onCellSelect, onRowSelect, onColumnSelect, onRenderingStateChange, highlightedRow = null, revealRow }: MatrixExplorerProps) {
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
    let frame: number | undefined;
    let firstValues = true;
    const cancelRefresh = () => {
      if (frame !== undefined) cancelAnimationFrame(frame);
      frame = undefined;
    };
    const refresh = () => { cancelRefresh(); if (active) viewport.refresh(); };
    const scheduleRefresh = () => {
      if (frame !== undefined) return;
      const scheduled = requestAnimationFrame(() => {
        if (frame !== scheduled) return;
        frame = undefined;
        if (active) viewport.refresh();
      });
      frame = scheduled;
    };
    // Uploads remain immediate and exact. Present the first values progressively;
    // later rows share one frame instead of redrawing three surfaces per row.
    // Every callback closes over this allocation, never a mutable current-view ref.
    const updates: MatrixUpdates = {
      values(values, offset) {
        if (!active) return;
        viewport.matrix.renderer.upload(values, offset);
        if (firstValues) { firstValues = false; refresh(); }
        else scheduleRefresh();
      },
      transfer(parameters) {
        if (!active) return;
        viewport.matrix.renderer.setTransfer(parameters);
        refresh();
      },
      distribution(axis, counts, offset) {
        if (!active) return;
        const target = axis === 'rows' ? viewport.rows : viewport.columns;
        if (!target) throw new Error('This source has no distribution surfaces.');
        target.upload(counts, offset);
        if (offset === 0) refresh();
        else scheduleRefresh();
      },
      distributionDomain(domain) {
        if (!active) return;
        viewport.setDistributionDomain(domain);
        setDomainState({ source, domain });
      },
      flush() { if (active && frame !== undefined) refresh(); },
    };
    let detach: () => void;
    try { detach = source.subscribe(updates); }
    catch (error) {
      active = false;
      cancelRefresh();
      viewport.dispose();
      throw error; // Domain/subscription failures belong to the parent.
    }
    return () => {
      active = false;
      cancelRefresh();
      currentViewport.current = null;
      try { detach(); } finally { viewport.dispose(); }
    };
  }, [source]);
  useLayoutEffect(() => { currentViewport.current?.setLinkedRow(highlightedRow); }, [highlightedRow, source]);
  useLayoutEffect(() => {
    if (revealRow) currentViewport.current?.revealRow(revealRow.row);
  }, [revealRow, source]);
  // Context/slot/callback changes do not restart subscriptions or upload weights.
  useLayoutEffect(() => {
    host.current?.querySelector('.matrix-scroll')?.setAttribute('aria-label', label);
  }, [label, source]);
  const controls = source.descriptor.rank === 2 && source.descriptor.numel > 0
    ? <button type="button" className="matrix-fit-width" disabled={failed}
      title="Fit width (minimum 1:1). Wheel or pinch to zoom; scrollbars to navigate."
      onClick={() => currentViewport.current?.fitWidth()}>Fit width</button> : null;
  const toolbar = typeof header === 'function' ? header(controls, domain)
    : controls ? <PanelHeader identity={header} actions={controls} /> : header;
  return <ViewerPanel className="matrix-explorer" header={toolbar}>
    {failed && <p role="alert">Exact rendering is unavailable. WebGL2 resources could not be allocated or were lost. Reopen this view to retry.</p>}
    <div ref={host} />
    {inspection && <InspectionCard inspection={inspection} host={host} />}
  </ViewerPanel>;
}
