import type { ReactNode } from 'react';

export function WorkingSurface({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="working-surface" aria-label={title}>
      <div className="surface-label">{title}</div>
      <div className="surface-content">{children}</div>
    </section>
  );
}
