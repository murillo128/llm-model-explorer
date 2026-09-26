import { useLayoutEffect, useRef, useState } from 'react';
import { GraphAction } from './GraphAction';

/** Keep camera controls clear of shell notifications without resizing the graph. */
export function CameraDock({ zoomIn, zoomOut, fit }: {
  zoomIn: () => void; zoomOut: () => void; fit: () => void;
}) {
  const dock = useRef<HTMLDivElement>(null);
  const [bottom, setBottom] = useState(10);
  useLayoutEffect(() => {
    const element = dock.current;
    const viewport = element?.parentElement;
    const notifications = document.querySelector('.toast-host');
    if (!element || !viewport || !notifications) return;
    const place = () => {
      const area = viewport.getBoundingClientRect();
      const controls = element.getBoundingClientRect();
      const toast = notifications.getBoundingClientRect();
      const intersects = toast.height > 0 && controls.left < toast.right && controls.right > toast.left
        && area.bottom - 10 > toast.top && area.bottom - 10 - controls.height < toast.bottom;
      setBottom(intersects
        ? Math.max(10, Math.min(area.height - controls.height - 10, area.bottom - toast.top + 10))
        : 10);
    };
    const observer = new ResizeObserver(place);
    observer.observe(viewport);
    observer.observe(notifications);
    observer.observe(element);
    window.addEventListener('resize', place);
    place();
    return () => { observer.disconnect(); window.removeEventListener('resize', place); };
  }, []);
  return <div ref={dock} className="architecture-camera-dock" role="group" aria-label="Graph camera" style={{ bottom }}>
    <GraphAction icon="plus" label="Zoom in" onClick={zoomIn} iconOnly />
    <GraphAction icon="minus" label="Zoom out" onClick={zoomOut} iconOnly />
    <GraphAction icon="fit" label="Fit view" onClick={fit} iconOnly />
  </div>;
}
