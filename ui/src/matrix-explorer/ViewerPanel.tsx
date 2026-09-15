import type { ReactNode } from 'react';
import './matrix-explorer.css';

/** The title is a sibling of the padded scientific card, including empty states. */
export function ViewerPanel({ header, children, className = '' }: {
  header?: ReactNode; children?: ReactNode; className?: string;
}) {
  return <div className={`viewer-panel ${className}`}>
    {header && <div className="matrix-explorer-header">{header}</div>}
    <div className="viewer-panel-body">{children}</div>
  </div>;
}
