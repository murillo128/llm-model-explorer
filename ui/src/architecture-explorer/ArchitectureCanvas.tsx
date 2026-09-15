import { memo, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Handle, Position, ReactFlow, ReactFlowProvider, useReactFlow } from '@xyflow/react';
import type { Node, NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './architecture.css';
import type { Graph, GraphNode, GraphView, Layout, PortPosition } from './graph';
import { requestLayout } from './layout';
import type { ProjectedNode, ProjectionOptions } from './projection';
import { connectionSet, endpointKey } from './projection';
import { deriveMlpGroups } from './derived-groups';
import { displayLabel, instanceOf, patternSummary } from './presentation';
import { Connection, ConnectionInspection } from './Connection';
import { ConnectionContext } from './connection-context';
import type { ConnectionEdge } from './Connection';
import type { EmphasisTarget } from './connection-context';

export interface ArchitectureSelection {
  modelId: string; sessionId: string; graphId: string; node: GraphNode; trigger: HTMLElement;
  filteredInputs?: string[];
}
interface CanvasProps {
  graph: Graph; modelId: string; sessionId: string; view: GraphView;
  onInspect?: ((selection: ArchitectureSelection) => void) | undefined;
}
type Data = { record: ProjectedNode; label: string; subtitle: string; ports: PortPosition[]; diagnostic: boolean;
  toggle: (id: string) => void; activate: (node: ProjectedNode, trigger: HTMLElement) => void;
  inspect: (node: ProjectedNode, trigger: HTMLElement) => void };
type CanvasNode = Node<Data, 'architecture'>;
const OperationNode = memo(function OperationNode({ data, selected }: NodeProps<CanvasNode>) {
  const interaction = useContext(ConnectionContext), node = data.record;
  return <div className="architecture-node" data-kind={node.kind} data-selected={selected} data-expanded={node.expanded}
    data-source-ids={JSON.stringify(node.sourceIds)} data-presentation={node.presentation ?? 'source'}>
    <div className="architecture-node-heading">
      <button className="nodrag nopan architecture-node-label" title={node.label} aria-label={`Select ${node.record?.label ?? node.label}`}
        aria-pressed={selected} onClick={(event) => { event.stopPropagation(); data.activate(node, event.currentTarget); }}
        onKeyDown={(event) => {
          if (event.key === 'ArrowRight' && node.kind === 'group' && !node.expanded || event.key === 'ArrowLeft' && node.kind === 'group' && node.expanded) {
            event.preventDefault(); event.stopPropagation(); data.toggle(node.id);
          }
        }}>{data.label}</button>
      {node.kind === 'group' && <button className="nodrag nopan architecture-expand" aria-label={`${node.expanded ? 'Collapse' : 'Expand'} ${node.label}`}
        aria-expanded={node.expanded} onClick={(event) => { event.stopPropagation(); data.toggle(node.id); }}>{node.expanded ? '−' : '+'}</button>}
      <button className="nodrag nopan architecture-info" aria-label={`Inspect ${node.label}`} onClick={(event) => { event.stopPropagation(); data.inspect(node, event.currentTarget); }}>ⓘ</button>
    </div>
    <div className="architecture-node-type" title={data.subtitle}>{data.subtitle}{data.diagnostic ? ' · diagnostic' : ''}</div>
    {data.ports.map((position) => {
      const port = node.ports.find((p) => p.id === position.portId)!;
      const target = { node_id: node.id, port_id: port.id };
      const active = interaction.ports.has(endpointKey(target));
      const hit = Math.min(20 / interaction.zoom, 23);
      return <div key={port.id}>
        <span className="architecture-port-label" data-emphasized={active} title={`${port.direction}: ${port.label}`}
          style={{ top: position.y - 7, ...(position.side === 'left' ? { left: position.x + 9 } : { right: 9 }) }}>{port.label}</span>
        <button className="architecture-port nodrag nopan" data-node-id={node.id} data-port-id={port.id}
          data-emphasized={active} aria-label={`${port.direction} port ${node.label}: ${port.label}`}
          style={{ left: position.x, top: position.y, width: hit, height: hit }}
          onPointerDown={(event) => event.stopPropagation()} onPointerEnter={() => interaction.hover({ port: target })}
          onPointerLeave={() => interaction.hover(null)} onFocus={() => interaction.focus({ port: target })} onBlur={() => interaction.focus(null)}
          onClick={(event) => event.stopPropagation()}>
          <span className="architecture-port-dot" />
        </button>
        {(['source', 'target'] as const).map((type) => <Handle key={type} type={type} id={`${type}:${port.id}`}
          position={position.side === 'left' ? Position.Left : Position.Right} isConnectable={false}
          className="architecture-handle" style={{ left: position.x, top: position.y }} />)}
      </div>;
    })}
  </div>;
});
const nodeTypes = { architecture: OperationNode }, edgeTypes = { connection: Connection };

export function ArchitectureCanvas(props: CanvasProps) { return <ReactFlowProvider><Canvas {...props} /></ReactFlowProvider>; }
function Canvas({ graph, modelId, sessionId, view, onInspect }: CanvasProps) {
  const flow = useReactFlow<CanvasNode>();
  const [options, setOptions] = useState<ProjectionOptions>(() => ({ expanded: view.expanded, repetitions: view.repetitions,
    dimensions: view.dimensions, exhaustive: view.exhaustive, showUnused: view.showUnused, showContext: view.showContext, deriveMlp: view.deriveMlp,
    ...(view.stateScope ? { stateScope: view.stateScope } : {}) }));
  const [selected, setSelected] = useState(view.selected), [dimensions, setDimensions] = useState(view.dimensions);
  const [focusId, setFocusId] = useState(view.focus), [pinned, setPinned] = useState(view.edge);
  const [temporary, setTemporary] = useState<EmphasisTarget | null>(null), [focused, setFocused] = useState<EmphasisTarget | null>(null);
  const [inspection, setInspection] = useState<{ edgeId?: string; nodeId?: string; trigger: HTMLElement } | null>(null);
  const [result, setResult] = useState<{ layout?: Layout; error?: string; options?: ProjectionOptions; invocation?: number }>({});
  const [retry, setRetry] = useState(0), [zoom, setZoom] = useState(view.viewport?.zoom ?? 1);
  const [panelWidth, setPanelWidth] = useState(1178);
  const panel = useRef<HTMLDivElement>(null), picker = useRef<HTMLSelectElement>(null);
  const appliedLayout = useRef<Layout | null>(null);
  const layoutCount = useRef(0), hoverFrame = useRef(0), focusFrame = useRef(0);
  const anchor = useRef<{ id: string; sourceId?: string | undefined; x: number; y: number } | null>(null);
  const centerPending = useRef<string | null>(null), fitPending = useRef(false), initialized = useRef(false);
  const records = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph]);
  const boxes = useMemo(() => new Map(result.layout?.boxes.map((b) => [b.id, b])), [result.layout]);
  const projected = useMemo(() => new Map(result.layout?.projection.nodes.map((n) => [n.id, n])), [result.layout]);
  const variants = useMemo(() => new Map(graph.repetitions.flatMap((r) => r.instances.map((i) => [i.node_id, `Instance ${i.index} · ${i.variant.replaceAll('_', ' ')}`] as const))), [graph]);
  const mlps = useMemo(() => deriveMlpGroups(graph), [graph]);
  const diagnosed = useMemo(() => new Set(graph.diagnostics.map((d) => d.node_id)), [graph.diagnostics]);
  const instance = useMemo(() => instanceOf(graph, selected), [graph, selected]);
  const windowSize = Math.max(2, Math.min(8, Math.floor((panelWidth - 140) / 240)));
  const select = useCallback((id: string) => { view.update({ selected: id }); setSelected(id); }, [view]);
  const focusContext = useCallback((id: string | null) => { view.update({ focus: id }); setFocusId(id); }, [view]);
  const change = useCallback((patch: Partial<ProjectionOptions>) => {
    setOptions((previous) => {
      const next = { ...previous, ...patch };
      view.update({ expanded: next.expanded, repetitions: next.repetitions ?? {}, exhaustive: next.exhaustive ?? false,
        showUnused: next.showUnused ?? false, showContext: next.showContext !== false, deriveMlp: next.deriveMlp !== false,
        stateScope: next.stateScope });
      return next;
    });
  }, [view]);
  const rememberAnchor = useCallback((id: string) => {
    const box = boxes.get(id);
    if (!box) return;
    const camera = flow.getViewport();
    anchor.current = { id, sourceId: projected.get(id)?.sourceIds[0], x: box.absoluteX * camera.zoom + camera.x, y: box.absoluteY * camera.zoom + camera.y };
  }, [boxes, flow, projected]);
  const withAncestors = useCallback((id: string, expanded: Set<string>) => {
    let node = records.get(id);
    while (node?.parent_id) { expanded.add(node.parent_id); node = records.get(node.parent_id); }
    return expanded;
  }, [records]);
  const chooseInstance = useCallback((id: string, open = true) => {
    const info = instanceOf(graph, id);
    if (!info) return;
    const previous = instanceOf(graph, selected);
    if (previous?.repetition.id === info.repetition.id) rememberAnchor(previous.instance.node_id);
    else {
      const range = [...projected.values()].find((n) => n.sourceIds.includes(info.instance.node_id));
      if (range) rememberAnchor(range.id);
    }
    const expanded = new Set(options.expanded);
    if (!options.exhaustive) for (const candidate of info.repetition.instances) {
      for (const expandedId of expanded) if (instanceOf(graph, expandedId)?.instance.node_id === candidate.node_id) expanded.delete(expandedId);
    }
    withAncestors(id, expanded); if (open) expanded.add(info.instance.node_id);
    const start = info.repetition.instances.findIndex((i) => i.node_id === info.instance.node_id);
    select(id); focusContext(info.instance.node_id);
    if (anchor.current) { anchor.current.id = info.instance.node_id; anchor.current.sourceId = info.instance.node_id; }
    change({ expanded: [...expanded], repetitions: { ...options.repetitions, [info.repetition.id]: { start, count: 1 } }, stateScope: undefined });
  }, [change, focusContext, graph, options, projected, rememberAnchor, select, selected, withAncestors]);
  const exploreStack = useCallback((id: string, start?: number) => {
    const repetition = graph.repetitions.find((r) => r.id === id)!;
    const current = options.repetitions?.[id];
    const range = [...projected.values()].find((n) => n.repetitionId === id);
    if (range) rememberAnchor(range.id);
    const expanded = withAncestors(repetition.instances[0]!.node_id, new Set(options.expanded));
    for (const item of repetition.instances) expanded.delete(item.node_id);
    focusContext(repetition.parent_id);
    change({ expanded: [...expanded], exhaustive: false, stateScope: undefined,
      repetitions: { ...options.repetitions, [id]: { start: Math.max(0, Math.min(repetition.instances.length - 1, start ?? current?.start ?? 0)), count: windowSize } } });
  }, [change, focusContext, graph.repetitions, options, projected, rememberAnchor, windowSize, withAncestors]);
  const toggle = useCallback((id: string) => {
    const node = projected.get(id);
    if (node?.repetitionId) { exploreStack(node.repetitionId, node.instances ? graph.repetitions.find((r) => r.id === node.repetitionId)!.instances.findIndex((i) => i.node_id === node.instances![0]!.node_id) : 0); return; }
    rememberAnchor(id);
    const expanded = new Set(options.expanded);
    if (expanded.has(id)) expanded.delete(id); else expanded.add(id);
    change({ expanded: [...expanded], exhaustive: false });
  }, [change, exploreStack, graph.repetitions, options.expanded, projected, rememberAnchor]);
  const reveal = useCallback((id: string) => {
    select(id);
    const info = instanceOf(graph, id);
    const expanded = withAncestors(id, new Set(options.expanded));
    const repetitions = { ...options.repetitions };
    if (info) repetitions[info.repetition.id] = { start: info.repetition.instances.findIndex((i) => i.node_id === info.instance.node_id), count: 1 };
    const mlp = mlps.find((g) => g.sourceIds.includes(id));
    if (mlp && options.deriveMlp !== false && !options.exhaustive) expanded.add(mlp.id);
    centerPending.current = id;
    change({ expanded: [...expanded], repetitions, stateScope: undefined });
  }, [change, graph, mlps, options, select, withAncestors]);
  const nativeInspect = useCallback((record: GraphNode, trigger: HTMLElement) => {
    select(record.id); setInspection(null);
    const filteredInputs = options.exhaustive || options.showUnused ? [] : result.layout?.projection.unusedInputs.filter((p) => p.node_id === record.id).map((p) => p.port_id) ?? [];
    onInspect?.({ modelId, sessionId, graphId: graph.graph_id, node: record, trigger, filteredInputs });
  }, [graph.graph_id, modelId, onInspect, options.exhaustive, options.showUnused, result.layout, select, sessionId]);
  const inspect = useCallback((node: ProjectedNode, trigger: HTMLElement) => {
    if (node.record) nativeInspect(records.get(node.record.id)!, trigger);
    else setInspection({ nodeId: node.id, trigger });
  }, [nativeInspect, records]);
  const activate = useCallback((node: ProjectedNode, trigger: HTMLElement) => {
    if (node.repetitionId) exploreStack(node.repetitionId);
    else if (node.presentation === 'mlp') { focusContext(node.id); toggle(node.id); }
    else inspect(node, trigger);
  }, [exploreStack, focusContext, inspect, toggle]);
  const overview = useCallback(() => {
    focusContext(null); fitPending.current = true;
    change({ expanded: graph.nodes.filter((n) => n.kind === 'group' && !n.parent_id).map((n) => n.id), repetitions: {}, exhaustive: false, stateScope: undefined });
  }, [change, focusContext, graph.nodes]);
  const focusLayer = useCallback(() => {
    if (!instance) return;
    chooseInstance(instance.instance.node_id);
    centerPending.current = instance.instance.node_id;
  }, [chooseInstance, instance]);
  const stateFocus = useCallback(() => {
    if (!instance) return;
    const id = instance.instance.node_id;
    const expanded = withAncestors(id, new Set(options.expanded)); expanded.add(id);
    for (const n of graph.nodes) if (n.kind === 'group' && instanceOf(graph, n.id)?.instance.node_id === id) expanded.add(n.id);
    focusContext(id); fitPending.current = true;
    change({ expanded: [...expanded], exhaustive: false, stateScope: id });
  }, [change, focusContext, graph, instance, options.expanded, withAncestors]);
  const fit = useCallback(() => {
    let focusNodes: CanvasNode[] | undefined;
    if (focusId && boxes.has(focusId)) focusNodes = flow.getNodes().filter((n) => n.id === focusId);
    void flow.fitView({ ...(focusNodes?.length ? { nodes: focusNodes } : {}), padding: 0.1, minZoom: 0.00001, maxZoom: 1 });
  }, [boxes, flow, focusId]);
  useEffect(() => {
    const element = panel.current!;
    const observer = new ResizeObserver(([entry]) => { if (entry) setPanelWidth(entry.contentRect.width); });
    observer.observe(element); return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const controller = new AbortController(); layoutCount.current++;
    void requestLayout(graph, options, controller.signal).then((layout) => {
      if (!controller.signal.aborted) {
        setResult({ layout, options, invocation: layoutCount.current });
        if (view.edge && !layout.projection.edges.some((e) => e.id === view.edge)) { view.update({ edge: null }); setPinned(null); setInspection(null); }
      }
    }, () => {
      if (!controller.signal.aborted) setResult({ options, invocation: layoutCount.current, error: 'Layout failed or exceeded 10 seconds. Retry or collapse groups.' });
    });
    return () => controller.abort();
  }, [graph, options, retry, view]);
  useEffect(() => {
    if (!result.layout || appliedLayout.current === result.layout) return;
    const frame = requestAnimationFrame(() => {
      const camera = flow.getViewport();
      const center = centerPending.current && boxes.get(centerPending.current);
      const substitute = anchor.current?.sourceId && [...projected.values()].find((n) => n.sourceIds.includes(anchor.current!.sourceId!));
      const at = anchor.current && (boxes.get(anchor.current.id) ?? (substitute ? boxes.get(substitute.id) : undefined));
      if (fitPending.current) fit();
      else if (center) void flow.setCenter(center.absoluteX + Math.min(center.width / 2, 360), center.absoluteY + Math.min(center.height / 2, 240), { zoom: Math.max(camera.zoom, 0.8) });
      else if (at && anchor.current) void flow.setViewport({ ...camera, x: anchor.current.x - at.absoluteX * camera.zoom, y: anchor.current.y - at.absoluteY * camera.zoom });
      else if (!initialized.current) {
        if (view.viewport) void flow.setViewport(view.viewport);
        else void flow.fitView({ padding: 0.06, minZoom: 0.65, maxZoom: 1 });
      }
      appliedLayout.current = result.layout!;
      initialized.current = true; anchor.current = null; centerPending.current = null; fitPending.current = false;
    });
    return () => cancelAnimationFrame(frame);
  }, [result.layout, boxes, flow, projected, view, fit]);
  useEffect(() => () => { cancelAnimationFrame(hoverFrame.current); cancelAnimationFrame(focusFrame.current); }, []);
  const hover = useCallback((target: EmphasisTarget | null) => {
    cancelAnimationFrame(hoverFrame.current);
    if (target) setTemporary(target); else hoverFrame.current = requestAnimationFrame(() => setTemporary(null));
  }, []);
  const keyboardFocus = useCallback((target: EmphasisTarget | null) => {
    cancelAnimationFrame(focusFrame.current);
    if (target) setFocused(target); else focusFrame.current = requestAnimationFrame(() => setFocused(null));
  }, []);
  const pin = useCallback((id: string, trigger: HTMLElement) => { view.update({ edge: id }); setPinned(id); setInspection({ edgeId: id, trigger }); }, [view]);
  const emphasis = useMemo(() => {
    const projection = result.layout?.projection;
    if (!projection) return new Set<string>();
    return new Set(connectionSet(projection, temporary ?? focused ?? { edgeId: pinned ?? '' }));
  }, [focused, pinned, result.layout, temporary]);
  const emphasizedPorts = useMemo(() => new Set(result.layout?.projection.edges.filter((e) => emphasis.has(e.id)).flatMap((e) => [endpointKey(e.source), endpointKey(e.target)])), [emphasis, result.layout]);
  const interaction = useMemo(() => ({ emphasized: emphasis, ports: emphasizedPorts, zoom, hover, focus: keyboardFocus, pin }), [emphasis, emphasizedPorts, hover, keyboardFocus, pin, zoom]);
  const nodes = useMemo<CanvasNode[]>(() => (result.layout?.boxes ?? []).map((box) => {
    const record = projected.get(box.id)!;
    return { id: box.id, type: 'architecture', position: { x: box.x, y: box.y },
      ...(box.parentId ? { parentId: box.parentId } : {}), width: box.width, height: box.height,
      style: { width: box.width, height: box.height, pointerEvents: record.expanded ? 'none' : 'auto' }, zIndex: 200, selected: selected === box.id,
      data: { record, label: displayLabel(record, graph), subtitle: record.summary?.replaceAll('linear attention', 'linear').replaceAll('full attention', 'full') ?? variants.get(record.id)?.replace(/^Instance \d+ · /, '') ?? record.record?.operation?.replaceAll('_', ' ') ?? record.kind,
        ports: result.layout!.ports.filter((p) => p.nodeId === box.id), diagnostic: diagnosed.has(box.id), toggle, activate, inspect } };
  }), [activate, diagnosed, graph, inspect, projected, result.layout, selected, toggle, variants]);
  const edges = useMemo<ConnectionEdge[]>(() => (result.layout?.projection.edges ?? []).map((edge) => ({
    id: edge.id, source: edge.source.node_id, target: edge.target.node_id, sourceHandle: `source:${edge.source.port_id}`,
    targetHandle: `target:${edge.target.port_id}`, type: 'connection', focusable: false, selectable: false,
    zIndex: emphasis.has(edge.id) ? 100 : 2,
    data: { connection: edge, route: result.layout!.routes.find((r) => r.id === edge.id)!, projection: result.layout!.projection, dimensions },
  })), [dimensions, emphasis, result.layout]);
  const activeInspectionEdge = result.layout?.projection.edges.find((e) => e.id === inspection?.edgeId);
  const activeInspectionNode = inspection?.nodeId ? projected.get(inspection.nodeId) : undefined;
  const sourceNodeIds = useMemo(() => {
    const represented = new Set(result.layout?.projection.nodes.flatMap((n) => n.sourceIds));
    return graph.nodes.filter((n) => represented.has(n.id)).map((n) => n.id);
  }, [graph.nodes, result.layout]);
  return <div ref={panel} className="architecture-explorer" aria-label="Architecture graph" data-graph-id={graph.graph_id}
    data-node-count={graph.nodes.length} data-edge-count={graph.edges.length} data-visible-nodes={nodes.length}
    data-visible-edges={edges.length} data-layout-ms={result.layout?.milliseconds} data-layout-count={result.invocation ?? 0} aria-busy={result.options !== options}
    data-source-node-ids={JSON.stringify(sourceNodeIds)} data-represented-edge-ids={JSON.stringify(result.layout?.edgeIds ?? [])}>
    <div className="architecture-toolbar" aria-label="Graph navigation">
      <button onClick={overview}>Overview</button>
      <button onClick={() => { focusContext(null); change({ expanded: graph.nodes.filter((n) => n.kind === 'group').map((n) => n.id), exhaustive: true, stateScope: undefined }); }}>Expand all</button>
      <button onClick={() => { focusContext(null); change({ expanded: [], repetitions: {}, exhaustive: false, stateScope: undefined }); }}>Collapse all</button>
      <button onClick={fit}>Fit graph</button>
      <button disabled={!selected} onClick={() => selected && reveal(selected)}>Center selected</button>
      <button aria-label="Zoom graph in" onClick={() => { void flow.zoomIn(); }}>+</button>
      <button aria-label="Zoom graph out" onClick={() => { void flow.zoomOut(); }}>−</button>
      <label><input type="checkbox" checked={dimensions} onChange={(event) => { if (selected) rememberAnchor(selected); view.update({ dimensions: event.target.checked }); setDimensions(event.target.checked); change({ dimensions: event.target.checked }); }} /> Show dimensions</label>
      <label>Component <select ref={picker} aria-label="Select graph component" value={selected ?? ''} onChange={(event) => reveal(event.target.value)}>
        <option value="" disabled>Select…</option>{graph.nodes.map((n) => <option key={n.id} value={n.id}>{n.label} · {variants.get(n.id) ?? n.id}</option>)}
      </select></label>
      {selected && records.get(selected)?.kind === 'group' && <button onClick={() => toggle(selected)}>Toggle selected group</button>}
      {selected && onInspect && <button onClick={(event) => nativeInspect(records.get(selected)!, event.currentTarget)}>Inspect selected</button>}
    </div>
    {graph.repetitions.length > 0 && <div className="architecture-repetitions" aria-label="Repeated groups">
      {graph.repetitions.map((r) => {
        const window = options.repetitions?.[r.id];
        return <div className="architecture-repetition" key={r.id}>
          <label>{r.label} ({r.instances.length}) <select aria-label={`Expand instance of ${r.label}`} value={instance?.repetition.id === r.id ? instance.instance.node_id : ''} onChange={(event) => chooseInstance(event.target.value)}>
            <option value="" disabled>Choose instance…</option>{r.instances.map((i) => <option key={i.node_id} value={i.node_id}>{i.index} · {i.variant} · {records.get(i.node_id)!.label}</option>)}
          </select></label>
          <button aria-label={`Explore stack ${r.label}`} onClick={() => exploreStack(r.id)}>Explore stack</button>
          {window && <>
            <button aria-label={`Previous window ${r.label}`} disabled={window.start === 0} onClick={() => exploreStack(r.id, Math.max(0, window.start - windowSize))}>←</button>
            <span>{r.instances[window.start]?.index}–{r.instances[Math.min(r.instances.length - 1, window.start + window.count - 1)]?.index}</span>
            <button aria-label={`Next window ${r.label}`} disabled={window.start + window.count >= r.instances.length} onClick={() => exploreStack(r.id, window.start + windowSize)}>→</button>
          </>}
          <span className="architecture-pattern" title={patternSummary(r.instances)}>{patternSummary(r.instances)}</span>
        </div>;
      })}
    </div>}
    <div className="architecture-focus-controls">
      {instance && <><span>Layer {instance.instance.index} · {instance.instance.variant.replaceAll('_', ' ')}</span>
        <button onClick={focusLayer}>{options.stateScope ? 'Back to layer' : 'Focus layer'}</button>
        {mlps.some((g) => g.parentId === instance.instance.node_id) && <button onClick={() => {
          const mlp = mlps.find((g) => g.parentId === instance.instance.node_id)!;
          const expanded = withAncestors(instance.instance.node_id, new Set(options.expanded)); expanded.add(instance.instance.node_id); expanded.add(mlp.id);
          focusContext(mlp.id); centerPending.current = mlp.id; change({ expanded: [...expanded], deriveMlp: true, exhaustive: false, stateScope: undefined });
        }}>Focus MLP</button>}
        <button onClick={stateFocus}>State dependencies</button></>}
      <label><input type="checkbox" checked={options.showUnused === true} onChange={(event) => change({ showUnused: event.target.checked })} /> Unused interfaces</label>
      <label><input type="checkbox" checked={options.showContext !== false} onChange={(event) => change({ showContext: event.target.checked })} /> Context</label>
      <label><input type="checkbox" checked={options.deriveMlp !== false} onChange={(event) => change({ deriveMlp: event.target.checked })} /> Group MLP</label>
      {pinned && <button onClick={() => { setPinned(null); view.update({ edge: null }); setInspection(null); }}>Clear connection selection</button>}
    </div>
    <div className="architecture-coverage">{graph.coverage === 'partial' ? 'Partial architecture coverage' : 'Complete within declared scope'} · {graph.scope.replaceAll('_', ' ')}
      {options.stateScope && ' · State dependencies only; other flows are filtered'}
      {!options.exhaustive && !options.showUnused && result.layout?.projection.filteredEdgeIds.length ? ' · Unconsumed interface branches filtered' : ''}
      {selected && <> · Selected: {records.get(selected)?.label} ({variants.get(selected) ?? selected})</>}
    </div>
    {graph.diagnostics.length > 0 && <details className="architecture-diagnostics"><summary>{graph.diagnostics.length} architecture diagnostics</summary>
      {graph.diagnostics.map((d, i) => <p key={i}>{d.message}</p>)}</details>}
    {result.error && <div role="alert">{result.error} <button onClick={() => setRetry(retry + 1)}>Retry layout</button></div>}
    {!result.layout && !result.error && <p role="status">Laying out architecture…</p>}
    <div className="architecture-flow">
      <ConnectionContext.Provider value={interaction}>
        <ReactFlow<CanvasNode, ConnectionEdge> nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} onlyRenderVisibleElements
          zIndexMode="manual" nodesDraggable={false} nodesConnectable={false} edgesReconnectable={false} deleteKeyCode={null}
          minZoom={0.00001} maxZoom={4} defaultViewport={view.viewport ?? { x: 0, y: 0, zoom: 1 }} panOnDrag zoomOnScroll
          onViewportChange={(viewport) => setZoom(viewport.zoom)} onMoveEnd={(_, viewport) => view.update({ viewport })}
          onNodeClick={(event, node) => activate(node.data.record, event.currentTarget as HTMLElement)}
          onNodesChange={(changes) => { for (const value of changes) if (value.type === 'select' && value.selected && records.has(value.id)) select(value.id); }}
          aria-label="Architecture canvas" />
      </ConnectionContext.Provider>
      {inspection && (activeInspectionEdge || activeInspectionNode) && <ConnectionInspection graph={graph} edge={activeInspectionEdge} node={activeInspectionNode}
        trigger={inspection.trigger} onClose={() => setInspection(null)} inspectNode={(id) => {
          if (picker.current) nativeInspect(records.get(id)!, picker.current);
        }} />}
    </div>
  </div>;
}
