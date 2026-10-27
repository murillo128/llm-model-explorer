import { useId, useLayoutEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { createPortal } from 'react-dom';

/** Shared architecture/weight modal, outside all graph transforms. */
export function InspectionDialog({ heading, trigger, onClose, children }: {
  heading: ReactNode; trigger: HTMLElement; onClose: () => void; children: ReactNode;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const title = useId();
  useLayoutEffect(() => {
    const element = dialog.current!;
    element.showModal();
    return () => {
      element.close();
      if (trigger.isConnected) trigger.focus({ preventScroll: true });
    };
  }, [trigger]);
  return createPortal(<dialog ref={dialog} className="architecture-inspection" aria-labelledby={title}
    onCancel={(event) => { event.preventDefault(); onClose(); }}
    onKeyDownCapture={(event) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onClose(); }
      if (event.key === 'Tab') {
        const controls = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('button, select, input, textarea, a[href], [tabindex]'))
          .filter((element) => element.tabIndex >= 0 && !element.matches(':disabled') && element.getClientRects().length > 0);
        const first = controls[0], last = controls.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    }}>
    <header className="architecture-inspection-heading"><h2 id={title}>{heading}</h2><button autoFocus onClick={onClose} aria-label="Close inspection">×</button></header>
    {children}
  </dialog>, document.body);
}
