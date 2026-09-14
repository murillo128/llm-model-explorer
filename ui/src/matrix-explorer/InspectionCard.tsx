import { createPortal } from 'react-dom';
import { useLayoutEffect, useRef } from 'react';
import type { Inspection } from '../rendering/matrix-inspection';

export function InspectionCard({ inspection }: { inspection: Inspection }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useLayoutEffect(() => {
    if (canvas.current) inspection.draw(canvas.current);
  }, [inspection]);
  return createPortal(<aside className="matrix-inspection" style={{ left: inspection.left, top: inspection.top }} aria-label="Cell inspection">
      <div className="magnifier-card">
        <canvas width={9} height={9} ref={canvas} aria-label="9 by 9 matrix neighborhood" />
        <span className="magnifier-center" aria-hidden="true" />
      </div>
      <output className="inspection-readout" role="status" aria-live="polite" aria-atomic="true">
        <span>row {inspection.row} · column {inspection.column}</span>
        <span>{inspection.value}</span>
      </output>
    </aside>, document.body);
}
