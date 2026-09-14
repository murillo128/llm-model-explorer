import type { ReactNode } from 'react';
import './matrix-explorer.css';

/** Fixed scientific chrome; operation state never changes the viewport origin. */
export function PanelHeader({ identity, information, summary, status, actions }: {
  identity: ReactNode; information?: ReactNode; summary?: ReactNode;
  status?: ReactNode; actions?: ReactNode;
}) {
  return <header className="matrix-panel-header">
    <div className="matrix-panel-context">{identity}{information}{summary}</div>
    <div className="matrix-panel-status">{status}</div>
    <div className="matrix-panel-actions">{actions}</div>
  </header>;
}
