import { memo, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Handle, Position, ReactFlow, ReactFlowProvider, useReactFlow } from '@xyflow/react';
import type { Node, NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './architecture.css';
import type { Graph, GraphNode, GraphView, Layout, PortPosition } from './graph';
import { requestLayout } from './layout';
import { useCanvasCallback } from './useCanvasCallback';
import { useLayoutCamera } from './useLayoutCamera';
import type { ProjectedNode, ProjectionOptions } from './projection';
import { connectionSet, endpointKey } from './projection';
import { deriveMlpGroups } from './derived-groups';
import { semanticRole } from './semantic-role';
import { displayLabel, instanceOf } from './presentation';
import { ArchitectureControls } from './ArchitectureControls';
import type { NavigationItem, ControlSelection } from './ArchitectureControls';
import { Connection, ConnectionInspection } from './Connection';
import { ConnectionContext } from './connection-context';
import { connectionHitResolver } from './connection-hit';
import { componentScope } from './scope';
import { cardDoubleClick, cardNavigation, cardSelection } from './card-actions';
import { backFromComponent, enterComponent, expandComponent, projectionOptions, returnToModel, snapshotView } from './scope-navigation';
import { bindTemplateLayout, commonNode, enterSharedStructure, remapNode, templateGraph } from './shared-structure';
import type { ConnectionEdge } from './Connection';
import type { EmphasisTarget } from './connection-context';

export interface ArchitectureSelection {
  modelId: string; sessionId: string; graphId: string; node: GraphNode; trigger: HTMLElement;
  filteredInputs?: string[];
  structureOnly?: { label: string; role: string };
  templateInstanceId?: string;
}
interface CanvasProps {
  graph: Graph; modelId: string; sessionId: string; view: GraphView;
  onInspect?: ((selection: ArchitectureSelection) => void) | undefined;
  onDismissInspection?: (() => void) | undefined;
}
type Data = { record: ProjectedNode; label: string; subtitle: string; ports: PortPosition[]; diagnostic: boolean;
  toggle: (id: string) => void; select: (id: string) => void; activate: (node: ProjectedNode, trigger: HTMLElement) => void;
  navigation: ReturnType<typeof cardNavigation>; navigate: (navigation: ReturnType<typeof cardNavigation>) => void;
  inspect: (node: ProjectedNode, trigger: HTMLElement) => void };
type CanvasNode = Node<Data, 'architecture'>;
const OperationNode = memo(function OperationNode({ data, selected }: NodeProps<CanvasNode>) {
  const interaction = useContext(ConnectionContext), node = data.record;
  const navigationName = `${data.navigation.action}: ${node.label} (${cardSelection(node)})`;
  return <div className="architecture-node nopan" data-kind={node.kind} data-selected={selected} data-expanded={node.expanded}
    data-source-ids={JSON.stringify(node.sourceIds)} data-presentation={node.presentation ?? 'source'}>
    <div className="architecture-node-heading" onKeyDown={(event) => event.stopPropagation()}>
      <button className="nodrag nopan architecture-node-label" title={node.label} aria-label={`Select ${node.record?.label ?? node.label}`}
        aria-pressed={selected} onClick={(event) => { event.stopPropagation(); data.select(cardSelection(node)); }}
        onDoubleClick={(event) => { event.stopPropagation(); data.activate(node, event.currentTarget); }}
        onKeyDown={(event) => {
          if (event.key === 'ArrowRight' && node.kind === 'group' && !node.expanded || event.key === 'ArrowLeft' && node.kind === 'group' && node.expanded) {
            event.preventDefault(); event.stopPropagation(); data.toggle(node.id);
          }
        }}>{data.label}</button>
      {node.kind === 'group' && <button className="nodrag nopan architecture-expand" aria-label={`${node.expanded ? 'Collapse' : 'Expand'} ${node.label}`}
        title={`${node.expanded ? 'Collapse' : 'Expand'} ${node.label}`}
        aria-expanded={node.expanded} onDoubleClick={(event) => event.stopPropagation()}
        onClick={(event) => { event.stopPropagation(); if (event.detail < 2) data.toggle(node.id); }}>{node.expanded ? '−' : '+'}</button>}
      <button className="nodrag nopan architecture-navigate" aria-label={navigationName}
        title={`${navigationName}${data.navigation.reason ? ` — ${data.navigation.reason}` : ''}`}
        aria-disabled={!data.navigation.target} onDoubleClick={(event) => event.stopPropagation()}
        onClick={(event) => { event.stopPropagation(); if (event.detail < 2) data.navigate(data.navigation); }}>
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
          <path d="M4 4l6 6m4 4 6 6M4 20l6-6m4-4 6-6" />
          <path d={data.navigation.action === 'Explore component'
            ? 'M4 9V4h5m6 0h5v5M4 15v5h5m6 0h5v-5'
            : 'M5 10h5V5m4 0v5h5M5 14h5v5m4 0v-5h5'} />
        </svg>
      </button>
      <button className="nodrag nopan architecture-info" aria-label={`Inspect ${node.label}`} title={`Inspect ${node.label}`} onDoubleClick={(event) => event.stopPropagation()}
        onClick={(event) => { event.stopPropagation(); if (event.detail < 2) data.inspect(node, event.currentTarget); }}>ⓘ</button>
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
          onClick={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()}>
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
function Canvas({ graph, modelId, sessionId, view, onInspect, onDismissInspection }: CanvasProps) {
  const flow = useReactFlow<CanvasNode>();
  const [options, setOptions] = useState<ProjectionOptions>(() => projectionOptions(view));
  const [selected, setSelected] = useState(view.selected), [dimensions, setDimensions] = useState(view.dimensions);
  const [focusId, setFocusId] = useState(view.focus), [pinned, setPinned] = useState(view.edge);
  const [activeStack, setActiveStack] = useState(view.activeStack);
  const [temporary, setTemporary] = useState<EmphasisTarget | null>(null), [focused, setFocused] = useState<EmphasisTarget | null>(null);
  const [inspection, setInspection] = useState<{ edgeId?: string; nodeId?: string; trigger: HTMLElement } | null>(null);
  const [layoutResult, setResult] = useState<{ layout?: Layout; error?: string; options?: ProjectionOptions; invocation?: number; input?: Graph }>({});
  const [shared, setShared] = useState(view.shared);
  const [notice, setNotice] = useState(view.notice);
  const template = graph.templates?.find((t) => t.id === shared?.templateId);
  const anchorInstance = template?.instances.find((i) => i.node_id === shared?.anchorId);
  const concreteInstance = template?.instances.find((i) => i.node_id === shared?.instanceId) ?? null;
  const sharedActive = Boolean(shared);
  const layoutInput = useMemo(() => {
    if (!sharedActive) return { graph };
    try {
      if (!anchorInstance) throw new Error('Shared structure correspondence changed. Return to the model view.');
      return { graph: templateGraph(graph, anchorInstance) };
    } catch (error) { return { error: error instanceof Error ? error.message : 'Shared structure is unavailable.' }; }
  }, [graph, anchorInstance, sharedActive]);
  const result = useMemo(() => {
    if (sharedActive && layoutResult.input !== layoutInput.graph) return {};
    if (!sharedActive || !layoutResult.layout || !template || !anchorInstance) return layoutResult;
    try { return { ...layoutResult, layout: bindTemplateLayout(layoutResult.layout, graph, template, anchorInstance, concreteInstance) }; }
    catch (error) { return { ...layoutResult, layout: undefined, error: error instanceof Error ? error.message : 'Shared structure is unavailable.' }; }
  }, [layoutResult, layoutInput, graph, template, anchorInstance, concreteInstance, sharedActive]);
  const [retry, setRetry] = useState(0), [zoom, setZoom] = useState(view.viewport?.zoom ?? 1);
  const [panelWidth, setPanelWidth] = useState(1178);
  const panel = useRef<HTMLDivElement>(null), picker = useRef<HTMLButtonElement>(null);
  const flowContainer = useRef<HTMLDivElement>(null);
  const savedViewport = useRef(view.viewport);
  const layoutCount = useRef(0), hoverFrame = useRef(0), focusFrame = useRef(0);
  const anchor = useRef<{ id: string; sourceId?: string | undefined; x: number; y: number } | null>(null);
  const centerPending = useRef<string | null>(null), fitPending = useRef(false), initialized = useRef(false);
  const restorePending = useRef<GraphView['viewport']>(undefined), scopeCameraPending = useRef(false);
  const records = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph]);
  const boxes = useMemo(() => new Map(result.layout?.boxes.map((b) => [b.id, b])), [result.layout]);
  const projected = useMemo(() => new Map(result.layout?.projection.nodes.map((n) => [n.id, n])), [result.layout]);
  const variants = useMemo(() => new Map(graph.repetitions.flatMap((r) => r.instances.map((i) => [i.node_id, `Instance ${i.index} · ${i.variant.replaceAll('_', ' ')}`] as const))), [graph]);
  const mlps = useMemo(() => [
    ...graph.nodes.flatMap((node) => node.kind === 'group' && node.parent_id && semanticRole(node) === 'mlp'
      ? [{ id: node.id, parentId: node.parent_id, label: node.label, sourceIds: node.children }] : []),
    ...deriveMlpGroups(graph),
  ], [graph]);
  const diagnosed = useMemo(() => new Set(graph.diagnostics.map((d) => d.node_id)), [graph.diagnostics]);
  const instance = useMemo(() => instanceOf(graph, mlps.find((g) => g.id === focusId)?.parentId ?? focusId), [focusId, graph, mlps]);
  const stack = graph.repetitions.find((r) => r.id === activeStack);
  const scope = useMemo(() => options.scope ? componentScope(graph, options.scope) : undefined, [graph, options.scope]);
  const windowSize = Math.max(2, Math.min(8, Math.floor((panelWidth - 140) / 240)));
  // Event proxies share only the committed render. Independently memoized
  // closures can chain older render contexts together and retain their layouts.
  const select = useCanvasCallback((id: string) => { view.update({ selected: id, edge: null }); setSelected(id); setPinned(null); setInspection(null); });
  const focusContext = useCanvasCallback((id: string | null, stackId?: string) => {
    // A stack overview focuses its parent; all other navigation derives the
    // repetition from the source instance, including a derived MLP's owner.
    const repetitionId = stackId ?? instanceOf(graph, mlps.find((g) => g.id === id)?.parentId ?? id)?.repetition.id ?? null;
    view.update({ focus: id, activeStack: repetitionId });
    setFocusId(id); setActiveStack(repetitionId);
  });
  const change = useCanvasCallback((patch: Partial<ProjectionOptions>) => {
    setOptions((previous) => {
      const next = { ...previous, ...patch };
      view.update({ expanded: next.expanded, repetitions: next.repetitions ?? {}, exhaustive: next.exhaustive ?? false,
        showUnused: next.showUnused ?? false, showContext: next.showContext !== false, deriveMlp: next.deriveMlp !== false,
        stateScope: next.stateScope, scope: next.scope });
      return next;
    });
  });
  const syncView = useCanvasCallback((initial: boolean = false) => {
    onDismissInspection?.(); setShared(view.shared);
    setOptions(projectionOptions(view)); setSelected(view.selected); setPinned(view.edge);
    setDimensions(view.dimensions); setFocusId(view.focus); setActiveStack(view.activeStack);
    setInspection(null); setTemporary(null); setFocused(null);
    cancelAnimationFrame(hoverFrame.current); cancelAnimationFrame(focusFrame.current);
    anchor.current = null; centerPending.current = null; fitPending.current = false;
    restorePending.current = initial ? undefined : view.viewport; scopeCameraPending.current = initial;
  });
  const leaveIsolation = () => {
    returnToModel(view); syncView(); return projectionOptions(view);
  };
  const isolate = useCanvasCallback((id: string) => {
    enterComponent(view, graph, id, flow.getViewport()); syncView(true);
    picker.current?.focus();
  });
  const back = useCanvasCallback(() => { backFromComponent(view); syncView(); picker.current?.focus(); });
  const openShared = useCanvasCallback((id: string, instanceId: string | null = null) => {
    const target = graph.templates?.find((t) => t.id === id);
    if (!target) { setNotice('Shared structure is no longer available. Choose an ordinary component.'); return; }
    enterSharedStructure(view, target, instanceId, flow.getViewport()); syncView(true);
    picker.current?.focus();
  });
  const chooseSharedInstance = useCanvasCallback((id: string | null) => {
    if (!shared || !template || !anchorInstance) return;
    const target = id ? template.instances.find((i) => i.node_id === id) : anchorInstance;
    if (!target) { setNotice('Instance correspondence changed; return to the model view.'); return; }
    onDismissInspection?.();
    const nextSelected = remapNode(selected, concreteInstance ?? anchorInstance, target);
    if (selected && !nextSelected) setNotice('The previous operation has no verified correspondence; its selection was cleared.');
    const next = { ...shared, instanceId: id };
    view.update({ shared: next, selected: nextSelected }); setShared(next); setSelected(nextSelected);
    setInspection(null); setTemporary(null); setFocused(null);
  });
  const rememberAnchor = useCanvasCallback((id: string) => {
    const box = boxes.get(id);
    if (!box) return;
    const camera = flow.getViewport();
    anchor.current = { id, sourceId: projected.get(id)?.sourceIds[0], x: box.absoluteX * camera.zoom + camera.x, y: box.absoluteY * camera.zoom + camera.y };
  });
  const withAncestors = (id: string, expanded: Set<string>) => {
    let node = records.get(id);
    while (node?.parent_id) { expanded.add(node.parent_id); node = records.get(node.parent_id); }
    return expanded;
  };
  const chooseInstance = useCanvasCallback((id: string, open: boolean = true) => {
    const info = instanceOf(graph, id);
    if (!info) return;
    const previous = instanceOf(graph, selected);
    if (previous?.repetition.id === info.repetition.id) rememberAnchor(previous.instance.node_id);
    else {
      const range = [...projected.values()].find((n) => n.sourceIds.includes(info.instance.node_id));
      if (range) rememberAnchor(range.id);
    }
    const base = options.scope ? leaveIsolation() : options;
    const expanded = new Set(base.expanded);
    if (!base.exhaustive) for (const candidate of info.repetition.instances) {
      for (const expandedId of expanded) if (instanceOf(graph, expandedId)?.instance.node_id === candidate.node_id) expanded.delete(expandedId);
    }
    withAncestors(id, expanded); if (open) expanded.add(info.instance.node_id);
    const start = info.repetition.instances.findIndex((i) => i.node_id === info.instance.node_id);
    select(id); focusContext(info.instance.node_id);
    if (anchor.current) { anchor.current.id = info.instance.node_id; anchor.current.sourceId = info.instance.node_id; }
    change({ ...base, expanded: [...expanded], repetitions: { ...base.repetitions, [info.repetition.id]: { start, count: 1 } }, stateScope: undefined, scope: undefined });
  });
  const exploreStack = useCanvasCallback((id: string, start?: number) => {
    const repetition = graph.repetitions.find((r) => r.id === id)!;
    const base = options.scope ? leaveIsolation() : options;
    const current = base.repetitions?.[id];
    const range = [...projected.values()].find((n) => n.repetitionId === id);
    if (range) rememberAnchor(range.id);
    const expanded = withAncestors(repetition.instances[0]!.node_id, new Set(base.expanded));
    for (const item of repetition.instances) expanded.delete(item.node_id);
    focusContext(repetition.parent_id, id);
    change({ ...base, expanded: [...expanded], exhaustive: false, stateScope: undefined, scope: undefined,
      repetitions: { ...base.repetitions, [id]: { start: Math.max(0, Math.min(repetition.instances.length - 1, start ?? current?.start ?? 0)), count: windowSize } } });
  });
  const toggle = useCanvasCallback((id: string) => {
    const node = projected.get(id);
    if (node?.repetitionId) { exploreStack(node.repetitionId, node.instances ? graph.repetitions.find((r) => r.id === node.repetitionId)!.instances.findIndex((i) => i.node_id === node.instances![0]!.node_id) : 0); return; }
    rememberAnchor(id);
    const expanded = new Set(options.expanded);
    if (expanded.has(id)) expanded.delete(id); else expanded.add(id);
    change({ expanded: [...expanded], exhaustive: false });
  });
  const reveal = useCanvasCallback((id: string) => {
    const base = shared || scope && !scope.members.has(id) && scope.id !== id ? leaveIsolation() : options;
    select(id);
    const derived = mlps.find((group) => group.id === id && !records.has(id));
    const sourceId = derived?.sourceIds[0] ?? id;
    const info = instanceOf(graph, sourceId);
    focusContext(base.scope ?? info?.instance.node_id ?? id);
    const expanded = withAncestors(sourceId, new Set(base.expanded));
    const repetitions = { ...base.repetitions };
    if (info) repetitions[info.repetition.id] = { start: info.repetition.instances.findIndex((i) => i.node_id === info.instance.node_id), count: 1 };
    const mlp = mlps.find((g) => g.sourceIds.includes(id));
    if (mlp && base.deriveMlp !== false && !base.exhaustive) expanded.add(mlp.id);
    centerPending.current = id;
    restorePending.current = undefined;
    change({ ...base, scope: base.scope, expanded: [...expanded], repetitions, stateScope: undefined,
      ...(derived ? { deriveMlp: true, exhaustive: false } : {}) });
  });
  const viewInModel = useCanvasCallback((target?: string) => {
    const sharedReturn = shared ? snapshotView(view, flow.getViewport()) : undefined;
    if (shared && !concreteInstance) return;
    const id = target ?? concreteInstance?.node_id ?? (selected && (records.has(selected) || mlps.some((group) => group.id === selected)) ? selected : options.scope);
    // reveal performs the same exact-instance action after restoring global state.
    if (!id) return;
    const base = leaveIsolation();
    if (sharedReturn) view.update({ history: [sharedReturn], globalView: snapshotView(view) });
    const derived = mlps.find((group) => group.id === id && !records.has(id));
    const source = derived?.sourceIds[0] ?? id;
    const expanded = withAncestors(source, new Set(base.expanded));
    const group = mlps.find((item) => item.sourceIds.includes(id));
    if (group && base.deriveMlp !== false) expanded.add(group.id);
    const info = instanceOf(graph, source);
    const repetitions = { ...base.repetitions };
    if (info) repetitions[info.repetition.id] = { start: info.repetition.instances.findIndex((item) => item.node_id === info.instance.node_id), count: 1 };
    select(id); focusContext(derived ? id : info?.instance.node_id ?? id);
    centerPending.current = id; restorePending.current = undefined;
    change({ ...base, expanded: [...expanded], repetitions, scope: undefined, stateScope: undefined,
      ...(derived ? { deriveMlp: true, exhaustive: false } : {}) });
  });
  const navigateCard = useCanvasCallback((navigation: ReturnType<typeof cardNavigation>) => {
    if (!navigation.target) return;
    if (navigation.action === 'View in model') viewInModel(navigation.target); else isolate(navigation.target);
  });
  const nativeInspect = useCanvasCallback((record: GraphNode, trigger: HTMLElement) => {
    select(record.id); setInspection(null);
    const filteredInputs = options.exhaustive || options.showUnused ? [] : result.layout?.projection.unusedInputs.filter((p) => p.node_id === record.id).map((p) => p.port_id) ?? [];
    if (shared && template && anchorInstance && !concreteInstance) {
      const role = anchorInstance.nodes.find((m) => m.node_id === record.id)?.role;
      if (!role) return;
      onInspect?.({ modelId, sessionId, graphId: graph.graph_id, node: commonNode(record, role), trigger, filteredInputs,
        structureOnly: { label: template.label, role } });
    } else onInspect?.({ modelId, sessionId, graphId: graph.graph_id, node: record, trigger, filteredInputs,
      ...(concreteInstance ? { templateInstanceId: concreteInstance.node_id } : {}) });
  });
  const inspect = useCanvasCallback((node: ProjectedNode, trigger: HTMLElement) => {
    if (node.record) nativeInspect(records.get(node.record.id)!, trigger);
    else setInspection({ nodeId: node.id, trigger });
  });
  const activate = useCanvasCallback((node: ProjectedNode, trigger: HTMLElement) => {
    if (cardDoubleClick(node) === 'expand') toggle(node.id);
    else inspect(node, trigger);
  });
  const overview = useCanvasCallback(() => {
    if (options.scope) leaveIsolation();
    focusContext(null); fitPending.current = true;
    restorePending.current = undefined;
    change({ expanded: graph.nodes.filter((n) => n.kind === 'group' && !n.parent_id).map((n) => n.id), repetitions: {}, exhaustive: false, stateScope: undefined, scope: undefined });
  });
  const focusLayer = useCanvasCallback(() => {
    if (!instance) return;
    chooseInstance(instance.instance.node_id);
    centerPending.current = instance.instance.node_id;
  });
  const stateFocus = useCanvasCallback(() => {
    if (!instance) return;
    const id = instance.instance.node_id;
    const expanded = withAncestors(id, new Set(options.expanded)); expanded.add(id);
    for (const n of graph.nodes) if (n.kind === 'group' && instanceOf(graph, n.id)?.instance.node_id === id) expanded.add(n.id);
    focusContext(id); fitPending.current = true;
    change({ expanded: [...expanded], exhaustive: false, stateScope: id });
  });
  useEffect(() => {
    const element = panel.current!;
    const observer = new ResizeObserver(([entry]) => { if (entry) setPanelWidth(entry.contentRect.width); });
    observer.observe(element); return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const controller = new AbortController(); layoutCount.current++;
    if (!layoutInput.graph) {
      // Failure is local to this optional view; ordinary model navigation stays available.
      void Promise.resolve().then(() => { if (!controller.signal.aborted) setResult({ error: layoutInput.error, options }); });
      return () => controller.abort();
    }
    void requestLayout(layoutInput.graph, options, controller.signal).then((layout) => {
      if (!controller.signal.aborted) {
        setResult({ layout, options, invocation: layoutCount.current, input: layoutInput.graph });
        if (view.edge && !layout.projection.edges.some((e) => e.id === view.edge)) { view.update({ edge: null }); setPinned(null); setInspection(null); }
      }
    }, () => {
      if (!controller.signal.aborted) setResult({ options, input: layoutInput.graph, invocation: layoutCount.current, error: 'Layout failed or exceeded 10 seconds. Retry or collapse groups.' });
    });
    return () => controller.abort();
  }, [layoutInput, options, retry, view]);
  useEffect(() => () => { cancelAnimationFrame(hoverFrame.current); cancelAnimationFrame(focusFrame.current); }, []);
  const hover = useCanvasCallback((target: EmphasisTarget | null) => {
    cancelAnimationFrame(hoverFrame.current);
    if (target) setTemporary((previous) => {
      // Pointer movement along the same represented segment changes no state.
      if ('edgeIds' in target && previous && 'edgeIds' in previous && target.edgeIds.length === previous.edgeIds.length &&
        target.edgeIds.every((id, index) => id === previous.edgeIds[index])) return previous;
      return target;
    }); else hoverFrame.current = requestAnimationFrame(() => setTemporary(null));
  });
  const keyboardFocus = useCanvasCallback((target: EmphasisTarget | null) => {
    cancelAnimationFrame(focusFrame.current);
    if (target) setFocused(target); else focusFrame.current = requestAnimationFrame(() => setFocused(null));
  });
  const pin = useCanvasCallback((id: string, trigger: HTMLElement) => { view.update({ edge: id }); setPinned(id); setInspection({ edgeId: id, trigger }); });
  const lineHit = useMemo(() => connectionHitResolver(result.layout?.projection.edges ?? [], result.layout?.routes ?? []), [result.layout]);
  const emphasis = useMemo(() => {
    const projection = result.layout?.projection;
    if (!projection) return new Set<string>();
    const target = temporary ?? focused ?? { edgeId: pinned ?? '' };
    return new Set('edgeIds' in target ? projection.edges.filter((edge) => target.edgeIds.includes(edge.id)).map((edge) => edge.id) : connectionSet(projection, target));
  }, [focused, pinned, result.layout, temporary]);
  const emphasizedPorts = useMemo(() => new Set(result.layout?.projection.edges.filter((e) => emphasis.has(e.id)).flatMap((e) => [endpointKey(e.source), endpointKey(e.target)])), [emphasis, result.layout]);
  const interaction = useMemo(() => ({ emphasized: emphasis, ports: emphasizedPorts, zoom, lineHit, hover, focus: keyboardFocus, pin }), [emphasis, emphasizedPorts, hover, keyboardFocus, lineHit, pin, zoom]);
  const nodes = useMemo<CanvasNode[]>(() => (result.layout?.boxes ?? []).map((box) => {
    const record = projected.get(box.id)!;
    return { id: box.id, type: 'architecture', position: { x: box.x, y: box.y },
      ...(box.parentId ? { parentId: box.parentId } : {}), width: box.width, height: box.height,
      style: { width: box.width, height: box.height, pointerEvents: record.expanded ? 'none' : 'auto' }, zIndex: 200,
      selected: selected === cardSelection(record) || Boolean(selected && !projected.has(selected) && record.sourceIds.includes(selected)),
      data: { record, label: displayLabel(record, graph), subtitle: record.summary?.replaceAll('linear attention', 'linear').replaceAll('full attention', 'full') ?? variants.get(record.id)?.replace(/^Instance \d+ · /, '') ?? record.record?.operation?.replaceAll('_', ' ') ?? record.kind,
        ports: result.layout!.ports.filter((p) => p.nodeId === box.id), diagnostic: diagnosed.has(box.id), toggle, select, activate, inspect,
        navigation: cardNavigation(record, concreteInstance?.node_id ?? options.scope, sharedActive && !concreteInstance), navigate: navigateCard } };
  }), [activate, diagnosed, graph, inspect, projected, result.layout, selected, toggle, variants, select, options.scope, sharedActive, concreteInstance, navigateCard]);
  const cameraState = useLayoutCamera(layoutResult.layout, options,
    result.options === options && result.input === layoutInput.graph && !result.error, nodes, flowContainer, flow, async (fitLayout) => {
      const camera = flow.getViewport();
      const center = centerPending.current && boxes.get(centerPending.current);
      const substitute = anchor.current?.sourceId && [...projected.values()].find((n) => n.sourceIds.includes(anchor.current!.sourceId!));
      const at = anchor.current && (boxes.get(anchor.current.id) ?? (substitute ? boxes.get(substitute.id) : undefined));
      if (restorePending.current) await flow.setViewport(restorePending.current);
      else if (scopeCameraPending.current) await fitLayout({ padding: 0.1, minZoom: 0.8, maxZoom: 1 });
      else if (fitPending.current) await fitLayout({ padding: 0.1, minZoom: 0.00001, maxZoom: 1,
        ...(!options.scope && focusId && boxes.has(focusId) ? { nodes: flow.getNodes().filter((node) => node.id === focusId) } : {}) });
      else if (center) await flow.setCenter(center.absoluteX + Math.min(center.width / 2, 360), center.absoluteY + Math.min(center.height / 2, 240), { zoom: Math.max(camera.zoom, 0.8) });
      else if (at && anchor.current) await flow.setViewport({ ...camera, x: anchor.current.x - at.absoluteX * camera.zoom, y: anchor.current.y - at.absoluteY * camera.zoom });
      else if (!initialized.current) {
        if (savedViewport.current) await flow.setViewport(savedViewport.current);
        else await fitLayout({ padding: 0.06, minZoom: 0.65, maxZoom: 1 });
      }
    }, () => {
      initialized.current = true; anchor.current = null; centerPending.current = null; fitPending.current = false;
      restorePending.current = undefined; scopeCameraPending.current = false;
    });
  const fit = useCanvasCallback(() => {
    cameraState.cancel();
    let focusNodes: CanvasNode[] | undefined;
    if (!options.scope && focusId && boxes.has(focusId)) focusNodes = flow.getNodes().filter((n) => n.id === focusId);
    void flow.fitView({ ...(focusNodes?.length ? { nodes: focusNodes } : {}), padding: 0.1, minZoom: 0.00001, maxZoom: 1 });
  });
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
  const focusMlp = () => {
    const mlp = mlps.find((g) => g.parentId === instance?.instance.node_id);
    if (!mlp || !instance) return;
    const expanded = withAncestors(instance.instance.node_id, new Set(options.expanded)); expanded.add(instance.instance.node_id); expanded.add(mlp.id);
    focusContext(mlp.id); centerPending.current = mlp.id; change({ expanded: [...expanded], deriveMlp: true, exhaustive: false, stateScope: undefined });
  };
  const breadcrumbs: NavigationItem[] = [];
  const breadcrumbFocus = options.scope ?? focusId;
  const focusedMlp = mlps.find((g) => g.id === breadcrumbFocus);
  let ancestor = records.get(focusedMlp?.parentId ?? breadcrumbFocus ?? '');
  while (ancestor) {
    if (ancestor.parent_id || ancestor.kind !== 'group') {
      const record = ancestor;
      const info = instanceOf(graph, record.id);
      breadcrumbs.unshift({ id: record.id, kind: 'node', label: info?.instance.node_id === record.id ? `Layer ${info.instance.index}` :
        displayLabel({ id: record.id, kind: record.kind, label: record.label, record, sourceIds: [record.id], ports: [], expanded: false }, graph) });
      if (info?.instance.node_id === record.id) breadcrumbs.unshift({ id: info.repetition.id, kind: 'stack', label: info.repetition.label });
    }
    ancestor = ancestor.parent_id ? records.get(ancestor.parent_id) : undefined;
  }
  if (stack && !breadcrumbs.some((item) => item.kind === 'stack')) breadcrumbs.push({ id: stack.id, kind: 'stack', label: stack.label });
  if (focusedMlp) breadcrumbs.push({ id: focusedMlp.id, kind: 'node', label: records.has(focusedMlp.id) ? focusedMlp.label : 'MLP (derived)' });
  const selectedRecord = selected ? records.get(selected) : undefined;
  const selectedEdge = result.layout?.projection.edges.find((e) => e.id === pinned);
  const eligibleTemplate = !shared && selectedRecord ? graph.templates?.find((t) => t.instances.some((i) => i.nodes.some((m) => m.node_id === selectedRecord.id))) : undefined;
  const eligibleInstance = eligibleTemplate?.instances.find((i) => i.nodes.some((m) => m.node_id === selectedRecord?.id));
  const centerSelected = () => {
    if (!selectedRecord) return;
    if (!shared) { reveal(selectedRecord.id); return; }
    const presentation = [...projected.values()].find((n) => n.record?.id === selectedRecord.id);
    const box = presentation && boxes.get(presentation.id);
    if (box) void flow.setCenter(box.absoluteX + box.width / 2, box.absoluteY + box.height / 2, { zoom: flow.getZoom() });
  };
  const controlSelection: ControlSelection | undefined = selectedEdge ? {
    edge: true,
    label: `${projected.get(selectedEdge.source.node_id)?.label ?? 'Source'} → ${projected.get(selectedEdge.target.node_id)?.label ?? 'Destination'}`,
    detail: `${selectedEdge.source.port_id} → ${selectedEdge.target.port_id} · ${selectedEdge.originalEdgeIds.join(', ')}`,
    inspect: (trigger) => setInspection({ edgeId: selectedEdge.id, trigger }),
    center: () => { void flow.fitView({ nodes: flow.getNodes().filter((n) => n.id === selectedEdge.source.node_id || n.id === selectedEdge.target.node_id), padding: 0.2, minZoom: 0.00001, maxZoom: 1 }); },
    clear: () => { setPinned(null); view.update({ edge: null }); setInspection(null); },
  } : selectedRecord ? {
    edge: false, nodeId: selectedRecord.id,
    label: shared && !concreteInstance ? [...projected.values()].find((n) => n.record?.id === selectedRecord.id)?.label ?? template?.label ?? 'Shared operation' :
      displayLabel({ id: selectedRecord.id, kind: selectedRecord.kind, label: selectedRecord.label, record: selectedRecord, sourceIds: [selectedRecord.id], ports: [], expanded: false }, graph),
    detail: shared && !concreteInstance ? 'Verified common operation; choose an instance for weights.' : `${selectedRecord.label} · ${selectedRecord.id}`,
    shared: eligibleTemplate && eligibleInstance ? () => openShared(eligibleTemplate.id, eligibleInstance.node_id) : undefined,
    inspect: onInspect ? (trigger) => nativeInspect(selectedRecord, trigger) : undefined,
    center: centerSelected,
    clear: () => { setSelected(null); view.update({ selected: null }); },
    explore: !shared && ['group', 'operation'].includes(selectedRecord.kind) && selectedRecord.id !== options.scope ? () => isolate(selectedRecord.id) : undefined,
  } : selected && mlps.some((group) => group.id === selected) ? {
    edge: false, nodeId: selected, label: 'MLP (derived)', detail: selected,
    center: () => reveal(selected), clear: () => { setSelected(null); view.update({ selected: null }); },
    explore: selected !== options.scope ? () => isolate(selected) : undefined,
  } : undefined;
  return <div ref={panel} className="architecture-explorer" aria-label="Architecture graph" data-graph-id={graph.graph_id}
    data-template-id={shared?.templateId ?? ''} data-template-instance-id={shared?.instanceId ?? ''}
    data-scope-id={concreteInstance?.node_id ?? options.scope ?? ''} data-node-count={graph.nodes.length} data-edge-count={graph.edges.length} data-visible-nodes={nodes.length}
    data-visible-edges={edges.length} data-layout-ms={result.layout?.milliseconds} data-layout-count={result.invocation ?? 0} aria-busy={result.options !== options || !result.error && !cameraState.ready}
    data-source-node-ids={JSON.stringify(sourceNodeIds)} data-represented-edge-ids={JSON.stringify(result.layout?.edgeIds ?? [])}>
    {notice && <p role="status">{notice}</p>}
    <ArchitectureControls shared={shared && template ? { template, instanceId: shared.instanceId, choose: chooseSharedInstance } : undefined}
      openShared={openShared} returnContext={!scope && view.history.length ? back : undefined} graph={graph} focus={focusId} stack={stack} options={options} picker={picker}
      isolation={scope ? { back, viewInModel: shared && !concreteInstance ? undefined : () => viewInModel(), expand: () => change(expandComponent(view, graph)),
        nodeIds: scope.members, excludedEdges: result.layout?.projection.scope?.excludedEdgeIds ?? [] } : undefined}
      instanceId={instance?.instance.node_id ?? (stack ? stack.instances[options.repetitions?.[stack.id]?.start ?? 0]?.node_id : undefined)}
      visibleInstances={stack?.instances.filter((i) => projected.has(i.node_id)).map((i) => i.node_id) ?? []}
      breadcrumbs={shared && template ? [{ id: template.id, kind: 'node', label: template.label }] : breadcrumbs} selection={controlSelection} reveal={reveal} navigate={(item) => {
        if (shared && item.id === template?.id) return;
        if (item.kind === 'stack') exploreStack(item.id);
        else if (!scope && mlps.some((g) => g.id === item.id)) focusMlp();
        else { const info = instanceOf(graph, item.id); if (info?.instance.node_id === item.id) chooseInstance(item.id); else reveal(item.id); }
      }} overview={overview} fit={fit} chooseInstance={chooseInstance} exploreStack={exploreStack} windowSize={windowSize}
      expandAll={() => { const base = scope ? leaveIsolation() : options; focusContext(null); change({ ...base, scope: undefined, expanded: graph.nodes.filter((n) => n.kind === 'group').map((n) => n.id), exhaustive: true, stateScope: undefined }); }}
      collapseAll={() => { const base = scope ? leaveIsolation() : options; focusContext(null); change({ ...base, scope: undefined, expanded: [], repetitions: {}, exhaustive: false, stateScope: undefined }); }}
      focusLayer={!scope && instance ? focusLayer : undefined} focusMlp={!scope && instance && mlps.some((g) => g.parentId === instance.instance.node_id) ? focusMlp : undefined}
      stateFocus={!scope && instance ? stateFocus : undefined} toggleSelected={selected && projected.get(selected)?.kind === 'group' ? () => toggle(selected) : undefined}
      preferences={(patch) => {
        if (patch.dimensions !== undefined) {
          if (selected) rememberAnchor(selected);
          view.update({ dimensions: patch.dimensions }); setDimensions(patch.dimensions);
        }
        change(patch);
      }} zoomIn={() => { cameraState.cancel(); void flow.zoomIn(); }} zoomOut={() => { cameraState.cancel(); void flow.zoomOut(); }}
      filtered={!options.exhaustive && !options.showUnused && !!result.layout?.projection.filteredEdgeIds.length} />
    {result.error && <div role="alert">{result.error} <button onClick={() => setRetry(retry + 1)}>Retry layout</button>{shared && <button onClick={back}>Return to ordinary view</button>}</div>}
    {cameraState.error && <div role="alert">{cameraState.error} <button onClick={() => setRetry(retry + 1)}>Retry layout</button></div>}
    <div className="architecture-flow" ref={flowContainer}>
      {!result.layout && !result.error && <p className="architecture-loading" role="status">Laying out architecture…</p>}
      <ConnectionContext.Provider value={interaction}>
        <ReactFlow<CanvasNode, ConnectionEdge> nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} onlyRenderVisibleElements
          zIndexMode="manual" nodesDraggable={false} nodesConnectable={false} edgesReconnectable={false} deleteKeyCode={null}
          minZoom={0.00001} maxZoom={4} defaultViewport={view.viewport ?? { x: 0, y: 0, zoom: 1 }} panOnDrag zoomOnScroll
          onMoveStart={(event) => { if (event) cameraState.cancel(); }}
          onViewportChange={(viewport) => setZoom(viewport.zoom)} onMoveEnd={(_, viewport) => view.update({ viewport })}
          onNodeClick={(event, node) => { event.stopPropagation(); select(cardSelection(node.data.record)); }}
          onNodeDoubleClick={(event, node) => { event.stopPropagation(); activate(node.data.record, event.currentTarget as HTMLElement); }}
          onNodesChange={(changes) => { for (const value of changes) if (value.type === 'select' && value.selected) { const node = projected.get(value.id); if (node) select(cardSelection(node)); } }}
          aria-label="Architecture canvas" />
      </ConnectionContext.Provider>
      {inspection && (activeInspectionEdge || activeInspectionNode) && <ConnectionInspection graph={graph} edge={activeInspectionEdge} node={activeInspectionNode}
        structure={shared && !concreteInstance && activeInspectionEdge ? { source: projected.get(activeInspectionEdge.source.node_id)?.label ?? '', target: projected.get(activeInspectionEdge.target.node_id)?.label ?? '' } : undefined}
        explore={activeInspectionNode?.presentation === 'mlp' && activeInspectionNode.id !== options.scope ? () => isolate(activeInspectionNode.id) : undefined}
        trigger={inspection.trigger} onClose={() => setInspection(null)} inspectNode={(id) => {
          if (picker.current) nativeInspect(records.get(id)!, picker.current);
        }} />}
    </div>
  </div>;
}
