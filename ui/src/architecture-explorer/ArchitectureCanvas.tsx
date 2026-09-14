import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Handle, Position, ReactFlow, ReactFlowProvider, useReactFlow } from '@xyflow/react';
import type { Edge, Node, NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './architecture.css';
import type { Graph, GraphNode, GraphView, Layout } from './graph';
import { formatShape } from './graph';
import { requestLayout } from './layout';

export interface ArchitectureSelection {
  modelId: string;
  sessionId: string;
  graphId: string;
  node: GraphNode;
  /** Restores the activating element when an inspection consumer closes. */
  trigger: HTMLElement;
}
interface CanvasProps {
  graph: Graph;
  modelId: string;
  sessionId: string;
  view: GraphView;
  onInspect?: ((selection: ArchitectureSelection) => void) | undefined;
}
type Data = { record: GraphNode; expanded: boolean; variant: string | undefined; diagnostic: boolean;
  toggle: (id: string) => void; select: (id: string) => void; inspect: ((node: GraphNode, trigger: HTMLElement) => void) | undefined };
type CanvasNode = Node<Data, 'architecture'>;
const OperationNode = memo(function OperationNode({ data, selected }: NodeProps<CanvasNode>) {
  const node = data.record;
  return <div className="architecture-node" data-kind={node.kind} data-selected={selected}>
    <div className="architecture-node-heading">
      <button className="nodrag nopan architecture-node-label" title={node.label} aria-label={`Select ${node.label}`}
        aria-pressed={selected} onClick={(event) => { event.stopPropagation(); data.select(node.id); data.inspect?.(node, event.currentTarget); }}
        onKeyDown={(event) => {
          if (event.key === 'ArrowRight' && node.kind === 'group' && !data.expanded || event.key === 'ArrowLeft' && node.kind === 'group' && data.expanded) {
            event.preventDefault(); data.toggle(node.id);
          }
        }}>{node.label}</button>
      {node.kind === 'group' && <button className="nodrag nopan" aria-label={`${data.expanded ? 'Collapse' : 'Expand'} ${node.label}`}
        aria-expanded={data.expanded} onClick={(event) => { event.stopPropagation(); data.toggle(node.id); }}>{data.expanded ? '−' : '+'}</button>}
      {data.inspect && <button className="nodrag nopan" aria-label={`Inspect ${node.label}`} onClick={(event) => { event.stopPropagation(); data.inspect!(node, event.currentTarget); }}>ⓘ</button>}
    </div>
    <div className="architecture-node-type">{data.variant ?? node.operation ?? node.kind}{data.diagnostic ? ' · diagnostic' : ''}</div>
    {node.ports.map((port, i) => <div key={port.id} className="architecture-port-label" style={{ top: 62 + i * 20, [port.direction === 'input' ? 'left' : 'right']: 8 }} title={port.label}>
      {port.label}
      {/* Boundary forwarding can use an input as source, or output as target. Both retain the same port identity. */}
      {(['source', 'target'] as const).map((type) => <Handle key={type} type={type} id={`${type}:${port.id}`}
        position={port.direction === 'input' ? Position.Left : Position.Right} isConnectable={false}
        style={{ top: 8, [port.direction === 'input' ? 'left' : 'right']: -12 }} />)}
    </div>)}
  </div>;
});
const nodeTypes = { architecture: OperationNode };

export function ArchitectureCanvas(props: CanvasProps) {
  return <ReactFlowProvider><Canvas {...props} /></ReactFlowProvider>;
}
function Canvas({ graph, modelId, sessionId, view, onInspect }: CanvasProps) {
  const flow = useReactFlow<CanvasNode>();
  const [expanded, setExpanded] = useState(() => new Set(view.expanded));
  const [selected, setSelected] = useState(view.selected);
  const [dimensions, setDimensions] = useState(view.dimensions);
  const [result, setResult] = useState<{ layout?: Layout; error?: string }>({});
  const [retry, setRetry] = useState(0);
  const anchor = useRef<{ id: string; x: number; y: number } | null>(null);
  const centerPending = useRef<string | null>(null);
  const initialized = useRef(false);
  const records = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph]);
  const ports = useMemo(() => new Map(graph.nodes.map((n) => [n.id, new Map(n.ports.map((p) => [p.id, p]))])), [graph]);
  const boxes = useMemo(() => new Map(result.layout?.boxes.map((b) => [b.id, b])), [result.layout]);
  const variants = useMemo(() => new Map(graph.repetitions.flatMap((r) => r.instances.map((i) => [i.node_id, `${r.label} · instance ${i.index} · ${i.variant}`] as const))), [graph]);
  const diagnosed = useMemo(() => new Set(graph.diagnostics.map((d) => d.node_id)), [graph.diagnostics]);
  const select = useCallback((id: string) => { view.update({ selected: id }); setSelected(id); }, [view]);
  const inspect = useMemo(() => onInspect ? (node: GraphNode, trigger: HTMLElement) => {
    select(node.id); onInspect({ modelId, sessionId, graphId: graph.graph_id, node, trigger });
  } : undefined, [onInspect, modelId, sessionId, graph.graph_id, select]);
  const replaceExpanded = useCallback((next: Set<string>) => {
    view.update({ expanded: [...next] }); setExpanded(next);
  }, [view]);
  const toggle = useCallback((id: string) => {
    const box = boxes.get(id);
    if (box) {
      const camera = flow.getViewport();
      anchor.current = { id, x: box.absoluteX * camera.zoom + camera.x, y: box.absoluteY * camera.zoom + camera.y };
    }
    const next = new Set(expanded);
    if (next.has(id)) next.delete(id); else next.add(id);
    replaceExpanded(next);
  }, [boxes, expanded, flow, replaceExpanded]);
  const reveal = useCallback((id: string) => {
    select(id);
    const next = new Set(expanded);
    let parent = records.get(id)?.parent_id;
    while (parent) { next.add(parent); parent = records.get(parent)?.parent_id; }
    centerPending.current = id; replaceExpanded(next);
  }, [expanded, records, replaceExpanded, select]);
  useEffect(() => {
    const controller = new AbortController();
    void requestLayout(graph, [...expanded], controller.signal).then((layout) => {
      if (!controller.signal.aborted) setResult({ layout });
    }, () => {
      if (!controller.signal.aborted) setResult({ error: 'Layout failed or exceeded 10 seconds. Retry or collapse groups.' });
    });
    return () => controller.abort();
  }, [graph, expanded, retry]);
  useEffect(() => {
    const layout = result.layout;
    if (!layout) return;
    const frame = requestAnimationFrame(() => {
      const camera = flow.getViewport();
      const center = centerPending.current && boxes.get(centerPending.current);
      const at = anchor.current && boxes.get(anchor.current.id);
      if (center) {
        void flow.setCenter(center.absoluteX + center.width / 2, center.absoluteY + Math.min(center.height / 2, 60), { zoom: Math.max(camera.zoom, 0.8) });
      } else if (at && anchor.current) {
        void flow.setViewport({ ...camera, x: anchor.current.x - at.absoluteX * camera.zoom, y: anchor.current.y - at.absoluteY * camera.zoom });
      } else if (!initialized.current) {
        if (view.viewport) void flow.setViewport(view.viewport);
        else void flow.fitView({ padding: 0.12, minZoom: 0.00001, maxZoom: 1 });
      }
      initialized.current = true; anchor.current = null; centerPending.current = null;
    });
    return () => cancelAnimationFrame(frame);
  }, [result.layout, boxes, flow, view]);
  const nodes = useMemo<CanvasNode[]>(() => (result.layout?.boxes ?? []).map((box) => ({
    id: box.id, type: 'architecture', position: { x: box.x, y: box.y },
    ...(box.parentId ? { parentId: box.parentId } : {}),
    width: box.width, height: box.height, style: { width: box.width, height: box.height },
    selected: selected === box.id,
    data: { record: records.get(box.id)!, expanded: expanded.has(box.id), variant: variants.get(box.id),
      diagnostic: diagnosed.has(box.id), toggle, select, inspect },
  })), [result.layout, selected, records, expanded, variants, diagnosed, toggle, select, inspect]);
  const edges = useMemo<Edge[]>(() => {
    const ids = new Set(result.layout?.edgeIds);
    return graph.edges.filter((e) => ids.has(e.id)).map((edge) => {
      const source = ports.get(edge.source.node_id)!.get(edge.source.port_id)!;
      const target = ports.get(edge.target.node_id)!.get(edge.target.port_id)!;
      return { id: edge.id, source: edge.source.node_id, target: edge.target.node_id,
        sourceHandle: `source:${edge.source.port_id}`, targetHandle: `target:${edge.target.port_id}`,
        label: [edge.label, dimensions ? `${formatShape(source.shape)} → ${formatShape(target.shape)}` : undefined].filter(Boolean).join(' · '),
        ariaLabel: `${edge.kind}: ${edge.source.node_id}.${source.label} to ${edge.target.node_id}.${target.label}`,
        type: 'smoothstep', zIndex: 2, ...(edge.kind === 'data' ? {} : { style: { strokeDasharray: '6 4' } }),
        markerEnd: { type: 'arrowclosed' as const },
      };
    });
  }, [graph.edges, result.layout, ports, dimensions]);
  return <div className="architecture-explorer" aria-label="Architecture graph" data-graph-id={graph.graph_id}
    data-node-count={graph.nodes.length} data-edge-count={graph.edges.length} data-visible-nodes={nodes.length}
    data-layout-ms={result.layout?.milliseconds}>
    <div className="architecture-toolbar" aria-label="Graph navigation">
      <button onClick={() => replaceExpanded(new Set(graph.nodes.filter((n) => n.kind === 'group').map((n) => n.id)))}>Expand all</button>
      <button onClick={() => replaceExpanded(new Set())}>Collapse all</button>
      <button onClick={() => { void flow.fitView({ padding: 0.12, minZoom: 0.00001, maxZoom: 1 }); }}>Fit graph</button>
      <button disabled={!selected} onClick={() => selected && reveal(selected)}>Center selected</button>
      <button aria-label="Zoom graph in" onClick={() => { void flow.zoomIn(); }}>+</button>
      <button aria-label="Zoom graph out" onClick={() => { void flow.zoomOut(); }}>−</button>
      <label><input type="checkbox" checked={dimensions} onChange={(event) => {
        view.update({ dimensions: event.target.checked }); setDimensions(event.target.checked);
      }} /> Show dimensions</label>
      <label>Component <select aria-label="Select graph component" value={selected ?? ''} onChange={(event) => reveal(event.target.value)}>
        <option value="" disabled>Select…</option>
        {graph.nodes.map((n) => <option key={n.id} value={n.id}>{n.label} · {variants.get(n.id) ?? n.id}</option>)}
      </select></label>
      {selected && records.get(selected)?.kind === 'group' && <button onClick={() => toggle(selected)}>Toggle selected group</button>}
      {selected && inspect && <button onClick={(event) => inspect(records.get(selected)!, event.currentTarget)}>Inspect selected</button>}
    </div>
    {graph.repetitions.length > 0 && <div className="architecture-repetitions" aria-label="Repeated groups">
      {graph.repetitions.map((r) => <label key={r.id}>{r.label} ({r.instances.length}) <select aria-label={`Expand instance of ${r.label}`} value="" onChange={(event) => {
        const id = event.target.value; select(id);
        const next = new Set(expanded); next.add(id);
        let parent = records.get(id)?.parent_id;
        while (parent) { next.add(parent); parent = records.get(parent)?.parent_id; }
        centerPending.current = id; replaceExpanded(next);
      }}><option value="" disabled>Choose instance…</option>{r.instances.map((i) => <option key={i.node_id} value={i.node_id}>{i.index} · {i.variant} · {records.get(i.node_id)!.label}</option>)}</select></label>)}
    </div>}
    <div className="architecture-coverage">{graph.coverage === 'partial' ? 'Partial architecture coverage' : 'Complete within declared scope'} · {graph.scope.replaceAll('_', ' ')}
      {selected && <> · Selected: {records.get(selected)?.label} ({variants.get(selected) ?? selected})</>}
    </div>
    {graph.diagnostics.length > 0 && <details className="architecture-diagnostics"><summary>{graph.diagnostics.length} architecture diagnostics</summary>
      {graph.diagnostics.map((d, i) => <p key={i}>{d.message}</p>)}</details>}
    {result.error && <div role="alert">{result.error} <button onClick={() => setRetry(retry + 1)}>Retry layout</button></div>}
    {!result.layout && !result.error && <p role="status">Laying out architecture…</p>}
    <div className="architecture-flow">
      <ReactFlow<CanvasNode> nodes={nodes} edges={edges} nodeTypes={nodeTypes} onlyRenderVisibleElements
        nodesDraggable={false} nodesConnectable={false} edgesReconnectable={false} deleteKeyCode={null}
        minZoom={0.00001} maxZoom={4} defaultViewport={view.viewport ?? { x: 0, y: 0, zoom: 1 }} panOnDrag zoomOnScroll
        onMoveEnd={(_, viewport) => { view.update({ viewport }); }} onNodeClick={(_, node) => select(node.id)}
        onNodeDoubleClick={(event, node) => inspect?.(node.data.record, event.currentTarget as HTMLElement)}
        onNodesChange={(changes) => { for (const change of changes) if (change.type === 'select' && change.selected) select(change.id); }}
        aria-label="Architecture canvas" />
    </div>
  </div>;
}
