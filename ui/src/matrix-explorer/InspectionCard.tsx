import { createPortal } from 'react-dom';
import { useLayoutEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { Inspection } from '../rendering/matrix-inspection';

export function InspectionCard({ inspection, host }: { inspection: Inspection; host: RefObject<HTMLElement | null> }) {
  const [container, setContainer] = useState<Element>(document.body);
  useLayoutEffect(() => {
    // Keep the readout in the native modal top layer when composed there.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setContainer(host.current?.closest('dialog') ?? document.body);
  }, [host]);
  const canvas = useRef<HTMLCanvasElement>(null);
  useLayoutEffect(() => {
    if (canvas.current) inspection.draw(canvas.current);
  }, [inspection, container]);
  return createPortal(<aside className="matrix-inspection" style={{ left: inspection.left, top: inspection.top, width: inspection.width }} aria-label="Cell inspection">
      {inspection.magnifier && <div className="magnifier-card">
        <canvas width={9} height={9} ref={canvas} aria-label="9 by 9 matrix neighborhood" />
        <span className="magnifier-guide magnifier-guide-horizontal" aria-hidden="true" />
        <span className="magnifier-guide magnifier-guide-vertical" aria-hidden="true" />
        <span className="magnifier-center" aria-hidden="true" />
      </div>}
      <output className="inspection-readout" role="status" aria-live="polite" aria-atomic="true">
        <span>row {inspection.row} · column {inspection.column}</span>
        <span>{inspection.value}</span>
      </output>
    </aside>, container);
}
