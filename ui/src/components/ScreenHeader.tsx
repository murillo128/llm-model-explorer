import type { ReactNode } from 'react';

export function MetadataLine({ children }: { children: ReactNode }) {
  return <p className="metadata">{children}</p>;
}

export function ScreenHeader({ title, children }: { title: string; children: ReactNode }) {
  return (
    <header className="screen-header">
      <p className="product-eyebrow">LLM Model Explorer</p>
      <h1>{title}</h1>
      <MetadataLine>{children}</MetadataLine>
    </header>
  );
}
