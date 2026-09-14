import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';

/** Hover/focus previews never steal focus; activation pins an interactive dialog. */
export function InfoPopover({ label, children }: { label: string; children: ReactNode }) {
  const [mode, setMode] = useState<'closed' | 'preview' | 'pinned'>('closed');
  const id = useId();
  const host = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const dialog = useRef<HTMLElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  const suppressed = useRef(false);
  const expanded = mode !== 'closed';
  function dismiss(restore: boolean) {
    suppressed.current = true;
    setMode('closed');
    if (restore) trigger.current?.focus();
  }
  useLayoutEffect(() => {
    if (!expanded) return;
    const position = () => {
      const anchor = trigger.current!.getBoundingClientRect();
      const pane = host.current!.closest('.working-surface')?.getBoundingClientRect();
      const left = Math.max(8, pane?.left ?? 8);
      const right = Math.min(window.innerWidth - 8, pane?.right ?? window.innerWidth - 8);
      const bottom = Math.min(window.innerHeight - 8, pane?.bottom ?? window.innerHeight - 8);
      const width = Math.min(360, right - left);
      Object.assign(dialog.current!.style, {
        top: `${anchor.bottom}px`, left: `${Math.max(left, Math.min(anchor.left, right - width))}px`,
        width: `${width}px`, maxHeight: `${Math.max(0, bottom - anchor.bottom)}px`,
      });
    };
    position();
    const observer = new ResizeObserver(position);
    observer.observe(host.current!.closest('.matrix-panel-context') ?? host.current!);
    window.addEventListener('resize', position);
    window.addEventListener('scroll', position, true);
    return () => { observer.disconnect(); window.removeEventListener('resize', position); window.removeEventListener('scroll', position, true); };
  }, [expanded]);
  useEffect(() => {
    if (mode === 'pinned') close.current?.focus();
    if (!expanded) return;
    const outside = (event: PointerEvent) => {
      if (!host.current?.contains(event.target as Node)) dismiss(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); dismiss(true); }
    };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', escape); };
  }, [expanded, mode]);
  return <div className="matrix-info" ref={host}
    onPointerEnter={(event) => {
      if (event.pointerType === 'touch') return;
      suppressed.current = false;
      setMode((old) => old === 'pinned' ? old : 'preview');
    }}
    onPointerLeave={() => {
      suppressed.current = false;
      if (!host.current?.contains(document.activeElement)) setMode((old) => old === 'preview' ? 'closed' : old);
    }}
    onFocus={() => { if (!suppressed.current) setMode((old) => old === 'closed' ? 'preview' : old); }}
    onBlur={(event) => {
      if (!event.currentTarget.contains(event.relatedTarget)) {
        suppressed.current = false;
        setMode((old) => old === 'preview' ? 'closed' : old);
      }
    }}>
    <button ref={trigger} type="button" className="matrix-info-trigger" aria-label={label}
      aria-expanded={expanded} aria-controls={expanded ? id : undefined} aria-haspopup="dialog"
      onClick={() => { if (mode === 'pinned') dismiss(true); else { suppressed.current = false; setMode('pinned'); } }}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" /><path d="M12 11v6" /><circle cx="12" cy="7" r=".75" /></svg>
    </button>
    {expanded && <section ref={dialog} id={id} className="matrix-info-popover" role="dialog" aria-label={label}>
      {mode === 'pinned' && <button ref={close} type="button" className="matrix-info-close"
        aria-label={`Close ${label.toLowerCase()}`} onClick={() => dismiss(true)}>×</button>}
      {children}
    </section>}
  </div>;
}
