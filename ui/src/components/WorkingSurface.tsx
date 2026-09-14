import type { ReactNode } from 'react';

export function WorkingSurface({ title, children, showTitle = true }: { title: string; children: ReactNode; showTitle?: boolean }) {
  return (
    <section className="working-surface" aria-label={title}>
      {showTitle && <div className="surface-label">{title}</div>}
      <div className="surface-content">{children}</div>
    </section>
  );
}
