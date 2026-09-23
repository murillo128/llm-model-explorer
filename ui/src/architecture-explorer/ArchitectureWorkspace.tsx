import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { readInventoryPreference, writeInventoryPreference } from '../components/inventory-preferences';

const preferenceKey = 'lmex.architecture-browser.pane';
const minimum = 200, maximum = 480;
function initialPreference() {
  const saved = readInventoryPreference(preferenceKey) as { visible?: unknown; width?: unknown } | null;
  return { visible: typeof saved?.visible === 'boolean' ? saved.visible : true,
    width: typeof saved?.width === 'number' && Number.isFinite(saved.width) ? Math.max(minimum, Math.min(maximum, saved.width)) : 280 };
}

/** Mirrors the inventory's bounded splitter/rail without replacing the canvas subtree. */
export function ArchitectureWorkspace({ browser, children }: { browser: ReactNode; children: ReactNode }) {
  const [preference, setPreference] = useState(initialPreference);
  const [available, setAvailable] = useState(0);
  const main = useRef<HTMLDivElement>(null), hide = useRef<HTMLButtonElement>(null), restore = useRef<HTMLButtonElement>(null);
  const focusPending = useRef(false), drag = useRef<{ pointer: number; x: number; width: number } | null>(null);
  const [resizing, setResizing] = useState(false);
  const limit = Math.max(minimum, Math.min(maximum, available - 360 - 16));
  const width = Math.min(preference.width, limit), hidden = !preference.visible;
  useEffect(() => {
    const node = main.current!;
    const observer = new ResizeObserver(() => setAvailable(node.clientWidth));
    observer.observe(node); return () => observer.disconnect();
  }, []);
  useEffect(() => writeInventoryPreference(preferenceKey, preference), [preference]);
  useLayoutEffect(() => {
    if (focusPending.current) { (hidden ? restore : hide).current?.focus(); focusPending.current = false; }
  }, [hidden]);
  const show = (visible: boolean) => { focusPending.current = true; setPreference((old) => ({ ...old, visible })); };
  const resize = (next: number) => setPreference((old) => ({ ...old, width: Math.max(minimum, Math.min(limit, Math.round(next))) }));
  return <div ref={main} className={`architecture-workspace${hidden ? ' browser-hidden' : ''}`}
    style={{ '--browser-width': `${width}px` } as CSSProperties}>
    <aside id="architecture-browser" className="explorer-card" aria-label="Architecture browser" hidden={hidden}>
      <header className="inventory-header"><h2>Browser</h2>
        <button ref={hide} className="inventory-toggle" aria-label="Collapse browser" title="Collapse browser" aria-expanded="true"
          aria-controls="architecture-browser" onClick={() => show(false)}>
          <svg aria-hidden="true" viewBox="0 0 16 16"><path d="m10 3-5 5 5 5" /></svg>
        </button>
      </header>
      {browser}
    </aside>
    <div role="separator" aria-label="Resize architecture browser" aria-orientation="vertical" aria-controls="architecture-browser"
      aria-valuemin={minimum} aria-valuemax={limit} aria-valuenow={width} aria-valuetext={`${width} pixels`} tabIndex={0}
      className="inventory-resizer architecture-browser-resizer" data-resizing={resizing || undefined} hidden={hidden}
      onKeyDown={(event) => {
        const next = { ArrowLeft: width - 16, ArrowRight: width + 16, Home: minimum, End: limit }[event.key];
        if (next !== undefined) { event.preventDefault(); resize(next); }
      }} onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.currentTarget.setPointerCapture(event.pointerId); event.currentTarget.focus();
        drag.current = { pointer: event.pointerId, x: event.clientX, width }; setResizing(true); event.preventDefault();
      }} onPointerMove={(event) => {
        if (drag.current?.pointer === event.pointerId) resize(drag.current.width + event.clientX - drag.current.x);
      }} onPointerUp={(event) => {
        if (drag.current?.pointer === event.pointerId) { event.currentTarget.releasePointerCapture(event.pointerId); drag.current = null; setResizing(false); }
      }} onPointerCancel={() => { drag.current = null; setResizing(false); }}
      onLostPointerCapture={() => { drag.current = null; setResizing(false); }} />
    {hidden && <nav className="inventory-rail" aria-label="Browser navigation">
      <button ref={restore} className="inventory-toggle inventory-restore" aria-label="Expand browser" title="Expand browser"
        aria-expanded="false" aria-controls="architecture-browser" onClick={() => show(true)}>
        <svg aria-hidden="true" viewBox="0 0 16 16"><rect x="2" y="2" width="12" height="12" rx="1" /><path d="M6 2v12M9 5h3M9 8h3M9 11h3" /></svg>
      </button>
    </nav>}
    {children}
  </div>;
}
