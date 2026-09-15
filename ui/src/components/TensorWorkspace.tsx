import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { readInventoryPreference, writeInventoryPreference } from './inventory-preferences';

const preferenceKey = 'lmex.tensor-inventory.pane';
const minimum = 200;
const maximum = 480;
function initialPreference() {
  const saved = readInventoryPreference(preferenceKey) as { visible?: unknown; width?: unknown } | null;
  return {
    visible: typeof saved?.visible === 'boolean' ? saved.visible : true,
    width: typeof saved?.width === 'number' && Number.isFinite(saved.width)
      ? Math.max(minimum, Math.min(maximum, saved.width)) : 280,
  };
}

/** Inventory changes leave the scientific subtree and its consumers mounted. */
export function TensorWorkspace({ enabled, inventory, children }: {
  enabled: boolean; inventory: ReactNode; children: ReactNode;
}) {
  const [preference, setPreference] = useState(initialPreference);
  const [available, setAvailable] = useState(0);
  const main = useRef<HTMLElement>(null);
  const hide = useRef<HTMLButtonElement>(null);
  const restore = useRef<HTMLButtonElement>(null);
  const moveFocus = useRef(false);
  const drag = useRef<{ pointer: number; x: number; width: number } | null>(null);
  const limit = Math.max(minimum, Math.min(maximum, available - 360 - 16));
  const width = Math.min(preference.width, limit);
  useEffect(() => {
    const node = main.current!;
    const observer = new ResizeObserver(() => setAvailable(node.clientWidth));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  useEffect(() => writeInventoryPreference(preferenceKey, preference), [preference]);
  useLayoutEffect(() => {
    if (moveFocus.current) {
      (preference.visible ? hide : restore).current?.focus();
      moveFocus.current = false;
    }
  }, [preference.visible]);
  function show(visible: boolean) {
    moveFocus.current = true;
    setPreference((old) => ({ ...old, visible }));
  }
  function resize(next: number) {
    setPreference((old) => ({ ...old, width: Math.max(minimum, Math.min(limit, Math.round(next))) }));
  }
  const hidden = !preference.visible;
  return <main id="workspace" tabIndex={-1} ref={main}
    className={enabled ? `tensor-layout${hidden ? ' inventory-hidden' : ''}` : undefined}
    style={{ '--inventory-width': `${width}px` } as CSSProperties}>
    {enabled && <>
      <aside id="tensor-inventory" aria-label="Tensor inventory" hidden={hidden}>
        <div className="inventory-header">
          <h2>Inventory</h2>
          <button type="button" className="inventory-toggle" ref={hide} aria-expanded="true"
            aria-controls="tensor-inventory" aria-label="Collapse inventory" title="Collapse inventory" onClick={() => show(false)}>
            <svg aria-hidden="true" viewBox="0 0 16 16"><path d="m10 3-5 5 5 5" /></svg>
          </button>
        </div>
        {inventory}
      </aside>
      <div role="separator" aria-label="Resize tensor inventory" aria-orientation="vertical"
        aria-controls="tensor-inventory" aria-valuemin={minimum} aria-valuemax={limit} aria-valuenow={width}
        aria-valuetext={`${width} pixels`} tabIndex={0} className="inventory-resizer" hidden={hidden}
        onKeyDown={(event) => {
          const next = { ArrowLeft: width - 16, ArrowRight: width + 16, Home: minimum, End: limit }[event.key];
          if (next !== undefined) { event.preventDefault(); resize(next); }
        }}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          event.currentTarget.focus();
          drag.current = { pointer: event.pointerId, x: event.clientX, width };
          event.preventDefault();
        }}
        onPointerMove={(event) => {
          if (drag.current?.pointer === event.pointerId) resize(drag.current.width + event.clientX - drag.current.x);
        }}
        onPointerUp={(event) => {
          if (drag.current?.pointer === event.pointerId) {
            event.currentTarget.releasePointerCapture(event.pointerId);
            drag.current = null;
          }
        }}
        onLostPointerCapture={() => { drag.current = null; }} />
    </>}
    {enabled && hidden && <nav className="inventory-rail" aria-label="Inventory navigation">
      <button type="button" className="inventory-toggle inventory-restore" ref={restore}
        aria-label="Expand inventory" aria-describedby="inventory-restore-tooltip"
        aria-expanded="false" aria-controls="tensor-inventory" onClick={() => show(true)}>
        <svg aria-hidden="true" viewBox="0 0 16 16"><rect x="2" y="2" width="12" height="12" rx="1" /><path d="M6 2v12M9 5h3M9 8h3M9 11h3" /></svg>
      </button>
      <span role="tooltip" id="inventory-restore-tooltip">Expand inventory</span>
    </nav>}
    {children}
  </main>;
}
