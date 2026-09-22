import { useId, useState } from 'react';
import { createPortal } from 'react-dom';
import { Button } from '../components/Button';

const paths = {
  plus: 'M12 5v14M5 12h14', minus: 'M5 12h14',
  fit: 'M9 4H4v5M15 4h5v5M4 15v5h5M20 15v5h-5',
  expand: 'M4 9V4h5M15 4h5v5M4 4l6 6M20 4l-6 6M4 15v5h5M15 20h5v-5M4 20l6-6M20 20l-6-6',
  collapse: 'M4 4l6 6M10 5v5H5M20 4l-6 6M14 5v5h5M4 20l6-6M10 19v-5H5M20 20l-6-6M14 19v-5h5',
};

export function GraphAction({ icon, label, tooltip = label, onClick, iconOnly = false }: {
  icon: keyof typeof paths; label: string; tooltip?: string; onClick: () => void; iconOnly?: boolean;
}) {
  const id = useId();
  const [anchor, setAnchor] = useState<{ left: number; top: number }>();
  const show = (element: HTMLButtonElement) => {
    const box = element.getBoundingClientRect();
    setAnchor({ left: Math.max(8, Math.min(box.left, window.innerWidth - 248)), top: box.top > 70 ? box.top - 62 : box.bottom + 6 });
  };
  return <>
    <Button className="architecture-action" aria-label={label} title={tooltip} aria-describedby={anchor ? id : undefined}
      onClick={onClick} onFocus={(event) => show(event.currentTarget)} onBlur={() => setAnchor(undefined)}
      onMouseEnter={(event) => show(event.currentTarget)} onMouseLeave={() => setAnchor(undefined)}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d={paths[icon]} /></svg>
      {!iconOnly && <span>{label}</span>}
    </Button>
    {anchor && createPortal(<span id={id} role="tooltip" className="architecture-action-tooltip" style={anchor}>{tooltip}</span>, document.body)}
  </>;
}
