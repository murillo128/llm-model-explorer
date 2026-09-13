import type { ReactNode } from 'react';

export function StatusText({ kind, children }: {
  kind: 'loading' | 'error' | 'empty';
  children: ReactNode;
}) {
  return (
    <p className={`status-text status-text--${kind}`} role={kind === 'error' ? 'alert' : 'status'}>
      <span className="status-label">{kind === 'loading' ? 'Loading' : kind === 'error' ? 'Error' : 'Empty'}</span>
      {children}
    </p>
  );
}
