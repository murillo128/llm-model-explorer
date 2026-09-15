import { useContext, useId, useLayoutEffect, useRef } from 'react';
import type { Edge, EdgeProps } from '@xyflow/react';
import type { Graph, Route } from './graph';
import { formatShape } from './graph';
import type { Endpoint, ProjectedEdge, ProjectedNode, Projection } from './projection';
import { endpointKey } from './projection';
import { ConnectionContext, routePath } from './connection-context';

export type ConnectionEdge = Edge<{ connection: ProjectedEdge; route: Route; projection: Projection; dimensions: boolean }, 'connection'>;


export function Connection({ data }: EdgeProps<ConnectionEdge>) {
  const interaction = useContext(ConnectionContext);
  const markerId = useId().replaceAll(':', '');
  if (!data) return null;
  const { connection, route, projection, dimensions } = data;
  const active = interaction.emphasized.has(connection.id);
  const sourceNode = projection.nodes.find((n) => n.id === connection.source.node_id)!;
  const targetNode = projection.nodes.find((n) => n.id === connection.target.node_id)!;
  const sourcePort = sourceNode.ports.find((p) => p.id === connection.source.port_id)!;
  const targetPort = targetNode.ports.find((p) => p.id === connection.target.port_id)!;
  const label = `${sourceNode.label}.${sourcePort.label} → ${targetNode.label}.${targetPort.label}`;
  return <g className="architecture-connection" data-edge-id={connection.id} data-emphasized={active}
    data-source-node={connection.source.node_id} data-source-port={connection.source.port_id}
    data-target-node={connection.target.node_id} data-target-port={connection.target.port_id}
    data-original-edge-ids={JSON.stringify(connection.originalEdgeIds)} data-kind={connection.kind}
    role="button" tabIndex={0} aria-label={`${connection.kind} connection: ${label}`}
    onPointerEnter={() => interaction.hover({ edgeId: connection.id })} onPointerLeave={() => interaction.hover(null)}
    onFocus={() => interaction.focus({ edgeId: connection.id })} onBlur={() => interaction.focus(null)}
    onClick={(event) => { event.stopPropagation(); interaction.pin(connection.id, event.currentTarget as unknown as HTMLElement); }}
    onKeyDown={(event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); interaction.pin(connection.id, event.currentTarget as unknown as HTMLElement); }
    }}>
    <title>{label}</title>
    <defs><marker id={markerId} viewBox="0 0 12 12" refX="11" refY="6" markerWidth="10" markerHeight="10" orient="auto-start-reverse" markerUnits="userSpaceOnUse">
      <path d="M1,1 L11,6 L1,11 Z" className="architecture-arrow" />
    </marker></defs>
    {route.sections.map((points, i) => <g key={i}>
      {active && <path d={routePath(points)} className="architecture-edge-halo" vectorEffect="non-scaling-stroke" />}
      <path d={routePath(points)} className="architecture-edge-line" vectorEffect="non-scaling-stroke"
        markerEnd={i === route.sections.length - 1 ? `url(#${markerId})` : undefined} />
      <path d={routePath(points)} className="architecture-edge-hit" vectorEffect="non-scaling-stroke" />
    </g>)}
    {route.junctions.map((point, i) => <circle key={i} cx={point.x} cy={point.y} r={3} className="architecture-junction" />)}
    {dimensions && route.labels?.map((label, i) => <g key={i} className="architecture-edge-label" pointerEvents="none">
      <rect x={label.x} y={label.y} width={label.width} height={label.height} rx={2} fill="var(--ui-surface-soft)" />
      <text className="react-flow__edge-text architecture-shape-label" x={label.x + label.width / 2} y={label.y + 12} textAnchor="middle">
        {label.lines.map((line, index) => <tspan key={index} x={label.x + label.width / 2} dy={index ? 14 : 0}>{line}</tspan>)}
      </text>
    </g>)}
  </g>;
}

/** A single graph-local floating inspection surface. Native weight inspection
 * continues to use the existing modal, outside this graph transform. */
export function ConnectionInspection({ graph, edge, node, trigger, onClose, inspectNode }: {
  graph: Graph; edge?: ProjectedEdge | undefined; node?: ProjectedNode | undefined; trigger: HTMLElement;
  onClose: () => void; inspectNode: (id: string) => void;
}) {
  const close = useRef<HTMLButtonElement>(null);
  useLayoutEffect(() => {
    close.current?.focus({ preventScroll: true });
    return () => { if (trigger.isConnected) trigger.focus({ preventScroll: true }); };
  }, [trigger]);
  const records = new Map(graph.nodes.map((n) => [n.id, n]));
  const endpointText = (ep: Endpoint) => `${records.get(ep.node_id)?.label ?? ep.node_id} · ${ep.port_id}`;
  return <section className="architecture-connection-inspection" role="dialog" aria-label={edge ? 'Connection inspection' : 'Group inspection'}
    onKeyDown={(event) => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onClose(); }
      if (event.key === 'Tab') {
        const controls = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('button, summary, [tabindex]'))
          .filter((element) => element.tabIndex >= 0 && !element.matches(':disabled') && element.getClientRects().length > 0);
        const first = controls[0], last = controls.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      } }}>
    <header><strong>{edge ? 'Connection' : node?.label}</strong><button ref={close} onClick={onClose}
      aria-label={edge ? 'Close connection inspection' : 'Close group inspection'}>×</button></header>
    <div className="architecture-connection-details">
      {edge && <>
        <p>{edge.kind} · {edge.paths.length} source {edge.paths.length === 1 ? 'path' : 'paths'} · direction →</p>
        {edge.paths.map((path, i) => <details key={i} open={edge.paths.length === 1}>
          <summary>{endpointText(path[0]!.source)} → {endpointText(path.at(-1)!.target)}</summary>
          {path.map((segment) => <div key={segment.id}>
            <p><strong>Source:</strong> {endpointText(segment.source)}<br /><code>{endpointKey(segment.source)}</code></p>
            <p><strong>Destination:</strong> {endpointText(segment.target)}<br /><code>{endpointKey(segment.target)}</code></p>
            <p>{formatShape(records.get(segment.source.node_id)!.ports.find((p) => p.id === segment.source.port_id)!.shape)} → {formatShape(records.get(segment.target.node_id)!.ports.find((p) => p.id === segment.target.port_id)!.shape)}</p>
            <p>Original edge: <code>{segment.id}</code> · {segment.kind}</p>
          </div>)}
        </details>)}
      </>}
      {node && <>
        <p>{node.summary}</p>
        {node.presentation === 'mlp' && <p>Derived visual group, reversible. These are five existing source operations.</p>}
        {node.repetitionId && <p>Repetition: <code>{node.repetitionId}</code>. Structural repetition does not share weights or states.</p>}
        {node.instances?.map((i) => <p key={i.node_id}>Instance {i.index}: {i.variant} · <code>{i.node_id}</code></p>)}
        {node.sourceIds.map((id) => <p key={id}><button onClick={() => inspectNode(id)}>{records.get(id)?.label ?? id}</button></p>)}
      </>}
    </div>
  </section>;
}
