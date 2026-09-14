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
  enabled: boolean; inventory: ReactNode; children: (restore: ReactNode) => ReactNode;
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
          <h2>Tensors</h2>
          <button type="button" className="inventory-toggle" ref={hide} aria-expanded="true"
            aria-controls="tensor-inventory" onClick={() => show(false)}>Hide inventory</button>
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
    {children(enabled && hidden ? <button type="button" className="inventory-toggle inventory-restore" ref={restore}
      aria-expanded="false" aria-controls="tensor-inventory" onClick={() => show(true)}>Show inventory</button> : null)}
  </main>;
}
