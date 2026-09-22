import { useEffect, useState } from 'react';
import type { Toast } from './feedback';

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: number) => void }) {
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (toast.severity !== 'info' || hovered || focused) return;
    const timer = window.setTimeout(() => onDismiss(toast.id), 5000);
    return () => window.clearTimeout(timer);
  }, [toast, hovered, focused, onDismiss]);
  return <li className="app-toast" data-severity={toast.severity}
    onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}
    onFocus={() => setFocused(true)} onBlur={(event) => {
      if (!event.currentTarget.contains(event.relatedTarget)) setFocused(false);
    }}>
    <p role={toast.severity === 'error' ? 'alert' : 'status'}>{toast.message}</p>
    <div className="toast-actions">
      {toast.action && <button type="button" onClick={() => { onDismiss(toast.id); toast.action!.run(); }}>{toast.action.label}</button>}
      <button type="button" aria-label="Dismiss notification" onClick={() => onDismiss(toast.id)}>Dismiss</button>
    </div>
  </li>;
}

export function ToastHost({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return <ol className="toast-host" aria-label="Application notifications">
    {toasts.map((toast) => <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />)}
  </ol>;
}
