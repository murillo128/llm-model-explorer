import { memo, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Handle, Position, ReactFlow, ReactFlowProvider, useReactFlow } from '@xyflow/react';
import type { ReactNode } from 'react';
import type { Node, NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './architecture.css';
import type { Graph, GraphNode, GraphView, PortPosition } from './graph';
import { useLayoutRequest } from './useLayoutRequest';
import type { LayoutResult } from './useLayoutRequest';
import { cardMetrics, cardSummary, ownParameters } from './card-summary';
import type { CardSummary as Summary } from './card-summary';
import { CardParameters, SummaryText } from './CardSummary';
import { formatShape } from './graph';
import { useCanvasCallback } from './useCanvasCallback';
import { useGraphView } from './useGraphView';
import { selectComponent, toggleComponent } from './component-actions';
import { useLayoutCamera } from './useLayoutCamera';
import type { ProjectedNode, ProjectionOptions } from './projection';
import { connectionSet, endpointKey } from './projection';
import { deriveMlpGroups } from './derived-groups';
import { semanticRole } from './semantic-role';
import { displayLabel, instanceOf } from './presentation';
import { ArchitectureBrowser } from './ArchitectureBrowser';
import { ArchitectureWorkspace } from './ArchitectureWorkspace';
import { browserExpansionId } from './browser-model';
import { CameraDock } from './CameraDock';
import { ArchitectureControls } from './ArchitectureControls';
import type { NavigationItem, ControlSelection } from './ArchitectureControls';
import { Connection, ConnectionInspection } from './Connection';
import { ConnectionContext } from './connection-context';
import { connectionHitResolver } from './connection-hit';
import { componentScope } from './scope';
import { cardDoubleClick, cardExpandable, cardNavigation, cardSelection } from './card-actions';
import { backFromComponent, enterComponent, expandComponent, projectionOptions, returnToModel, snapshotView } from './scope-navigation';
import { bindTemplateLayout, bindTemplatePortSelection, commonNode, enterSharedStructure, remapNode, templateGraph } from './shared-structure';
import type { ConnectionEdge } from './Connection';
import type { EmphasisTarget } from './connection-context';
import { interfaceIndex } from './interfaces';
import type { BoundarySelection } from './interfaces';
import { initialViewport, overviewExpansion, visibleBounds } from './overview';

export interface ArchitectureSelection {
  modelId: string; sessionId: string; graphId: string; node?: GraphNode; boundary?: BoundarySelection; trigger: HTMLElement;
  parameterId?: string;
  filteredInputs?: string[];
  structureOnly?: { label: string; role: string };
  templateInstanceId?: string;
}
interface CanvasProps {
  notices?: ReactNode;
  graph: Graph; modelId: string; sessionId: string; view: GraphView;
  onInspect?: ((selection: ArchitectureSelection) => void) | undefined;
  onDismissInspection?: (() => void) | undefined;
}
type Data = { record: ProjectedNode; label: string; subtitle: string; summary: Summary; metrics: ReturnType<typeof cardMetrics>; dimensions: boolean; ports: PortPosition[]; diagnostic: boolean;
  toggle: (id: string) => void; select: (id: string) => void; activate: (node: ProjectedNode) => void;
  navigation: ReturnType<typeof cardNavigation>; navigate: (navigation: ReturnType<typeof cardNavigation>) => void;
  inspect: (node: ProjectedNode, trigger: HTMLElement) => void;
  matrix: (node: ProjectedNode, id: string, trigger: HTMLElement) => void };
type CanvasNode = Node<Data, 'architecture'>;
const OperationNode = memo(function OperationNode({ data, selected }: NodeProps<CanvasNode>) {
  const interaction = useContext(ConnectionContext), node = data.record;
  const navigationName = `${data.navigation.action}: ${node.label} (${cardSelection(node)})`;
  return <div className="architecture-node nopan" data-kind={node.kind} data-selected={selected} data-expanded={node.expanded}
    data-source-ids={JSON.stringify(node.sourceIds)} data-presentation={node.presentation ?? 'source'}>
    <div className="architecture-node-heading" onKeyDown={(event) => event.stopPropagation()}>
      <button className="nodrag nopan architecture-node-label" title={node.label} aria-label={`Select ${node.record?.label ?? node.label}`}
        aria-pressed={selected} onClick={(event) => { event.stopPropagation(); data.select(cardSelection(node)); }}
        onDoubleClick={(event) => { event.stopPropagation(); data.activate(node); }}
        onKeyDown={(event) => {
          if (cardExpandable(node) && (event.key === 'ArrowRight' && !node.expanded || event.key === 'ArrowLeft' && node.expanded)) {
            event.preventDefault(); event.stopPropagation(); data.toggle(node.id);
          }
        }}>{data.label}</button>
      {cardExpandable(node) && <button className="nodrag nopan architecture-expand" aria-label={`${node.expanded ? 'Collapse' : 'Expand'} ${node.label}`}
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
      <button className="nodrag nopan architecture-info" aria-label={`Inspect ${node.label}`} title={`Inspect ${node.label}${data.diagnostic ? ' · diagnostic available' : ''}`} aria-description={data.diagnostic ? 'Diagnostic available' : undefined} onDoubleClick={(event) => event.stopPropagation()}
        onClick={(event) => { event.stopPropagation(); if (event.detail < 2) data.inspect(node, event.currentTarget); }}>{data.diagnostic ? 'ⓘ !' : 'ⓘ'}</button>
    </div>
    {(data.summary.formula || data.subtitle || data.diagnostic) && <div className="architecture-node-type">
      <SummaryText text={data.summary.formula ?? (data.subtitle || 'diagnostic')} />
    </div>}
    {node.record && <CardParameters node={node.record} summary={data.summary} dimensions={data.dimensions} top={data.metrics.metadataTop}
      inspect={(trigger) => data.inspect(node, trigger)} matrix={(id, trigger) => data.matrix(node, id, trigger)} />}
    {data.ports.map((position) => {
      const port = node.ports.find((p) => p.id === position.portId)!;
      const target = { node_id: node.id, port_id: port.id };
      const active = interaction.ports.has(endpointKey(target));
      const hit = Math.min(20 / interaction.zoom, 23);
      return <div key={port.id}>
        <button className="architecture-port nodrag nopan" data-node-id={node.id} data-port-id={port.id}
          data-absolute-x={position.absoluteX} data-absolute-y={position.absoluteY}
          data-emphasized={active} aria-label={`${port.direction} port ${node.label}: ${port.interfaceLabel ?? port.label}`}
          title={`${port.direction}: ${port.interfaceLabel ?? port.label}${data.dimensions ? ` ${formatShape(port.shape)}` : ''}`}
          aria-description={data.dimensions ? formatShape(port.shape) : undefined}
          style={{ left: position.x, top: position.y, width: hit, height: hit }}
          onPointerDown={(event) => event.stopPropagation()} onPointerEnter={() => interaction.hover({ port: target })}
          onPointerLeave={() => interaction.hover(null)} onFocus={() => interaction.focus({ port: target })} onBlur={() => interaction.focus(null)}
          onClick={(event) => { event.stopPropagation(); interaction.selectPort?.(target); }} onDoubleClick={(event) => event.stopPropagation()}>
          <span className="architecture-port-dot" />
          <span className="architecture-port-label" data-emphasized={active} data-raised={position.label.raised}
            data-layout-bounds={JSON.stringify(position.label)}
            title={`${port.direction}: ${port.interfaceLabel ?? port.label}${data.dimensions ? ` ${formatShape(port.shape)}` : ''}`}
            style={{ left: position.label.x - position.absoluteX + hit / 2, top: position.label.y - position.absoluteY + hit / 2,
              width: position.label.width, height: position.label.height }}>
            {port.interfaceLabel ?? port.label}{data.dimensions && <span className="architecture-port-shape">{formatShape(port.shape)}</span>}
          </span>
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
function Canvas({ graph, modelId, sessionId, view, onInspect, onDismissInspection, notices }: CanvasProps) {
  const flow = useReactFlow<CanvasNode>();
  const options = useGraphView(view);
  const { selected, selectionMode, focus: focusId, edge: pinned, activeStack, shared } = view;
  const [temporary, setTemporary] = useState<EmphasisTarget | null>(null), [focused, setFocused] = useState<EmphasisTarget | null>(null);
  const [inspection, setInspection] = useState<{ edgeId?: string; nodeId?: string; trigger: HTMLElement } | null>(null);
  const [layoutResult, setResult] = useState<LayoutResult>({});
  const [notice, setNotice] = useState(view.notice);
  const template = graph.templates?.find((t) => t.id === shared?.templateId);
  const anchorInstance = template?.instances.find((i) => i.node_id === shared?.anchorId);
  const concreteInstance = template?.instances.find((i) => i.node_id === shared?.instanceId) ?? null;
  const sharedActive = Boolean(shared);
  const commonSelection = sharedActive && !concreteInstance && selectionMode === 'structure';
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
  const panel = useRef<HTMLDivElement>(null), picker = useRef<HTMLButtonElement>(null), browserSearch = useRef<HTMLInputElement>(null);
  const flowContainer = useRef<HTMLDivElement>(null);
  const savedViewport = useRef(view.viewport);
  const hoverFrame = useRef(0), focusFrame = useRef(0);
  const anchor = useRef<{ id: string; sourceId?: string | undefined; x: number; y: number } | null>(null);
  const centerPending = useRef<string | null>(null), fitPending = useRef(false), initialized = useRef(false);
  const restorePending = useRef<GraphView['viewport']>(undefined), scopeCameraPending = useRef(false);
  const collapsePending = useRef<ProjectionOptions | undefined>(undefined);
  const interfaces = useMemo(() => interfaceIndex(graph), [graph]);
  const records = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph]);
  const parameters = useMemo(() => new Map(graph.parameters.map((p) => [p.id, p])), [graph]);
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
  const select = useCanvasCallback((id: string) => {
    if (id === interfaces.outer.id && interfaces.outer.kind === 'model') {
      view.update({ selected: null, edge: null, boundary: { kind: 'boundary', owner: interfaces.outer, endpoints: [] } }); setInspection(null); return;
    }
    selectComponent(view, id, view.shared && !view.shared.instanceId ? 'structure' : 'source'); setInspection(null); });
  const selectSource = useCanvasCallback((id: string) => { selectComponent(view, id); setInspection(null); });
  const selectBoundary = useCanvasCallback((boundary: BoundarySelection) => {
    view.update({ boundary, selected: boundary.owner.kind === 'source' ? boundary.owner.id : null, edge: null,
      selectionMode: boundary.templatePort && !concreteInstance ? 'structure' : 'source',
      browser: { ...view.browser, selectedFamily: null } }); setInspection(null);
  });
  const selectFamily = useCanvasCallback((id: string) => {
    view.update({ selected: null, boundary: undefined, edge: null, browser: { ...view.browser, selectedFamily: id } }); setInspection(null);
  });
  const focusContext = useCanvasCallback((id: string | null, stackId?: string) => {
    // A stack overview focuses its parent; all other navigation derives the
    // repetition from the source instance, including a derived MLP's owner.
    const repetitionId = stackId ?? instanceOf(graph, mlps.find((g) => g.id === id)?.parentId ?? id)?.repetition.id ?? null;
    view.update({ focus: id, activeStack: repetitionId });
  });
  const change = useCanvasCallback((patch: Partial<ProjectionOptions>) => {
    view.update(patch);
  });
  const syncView = useCanvasCallback((initial: boolean = false) => {
    if (view.browser.selectedFamily) view.update({ browser: { ...view.browser, selectedFamily: null } });
    onDismissInspection?.();
    setInspection(null); setTemporary(null); setFocused(null);
    cancelAnimationFrame(hoverFrame.current); cancelAnimationFrame(focusFrame.current);
    anchor.current = null; centerPending.current = null; fitPending.current = false;
    collapsePending.current = undefined;
    restorePending.current = initial ? undefined : view.viewport; scopeCameraPending.current = initial;
  });
  const leaveIsolation = (retain = false) => {
    const previous = retain ? [...view.history, snapshotView(view, flow.getViewport())].slice(-16) : [];
    returnToModel(view);
    if (retain) view.update({ history: previous, globalView: snapshotView(view) });
    syncView(); return projectionOptions(view);
  };
  const isolate = useCanvasCallback((id: string) => {
    enterComponent(view, graph, id, flow.getViewport()); syncView(true);
    picker.current?.focus();
  });
  const back = useCanvasCallback(() => { backFromComponent(view); syncView(); picker.current?.focus(); });
  const openShared = useCanvasCallback((id: string, instanceId: string | null = null) => {
    const target = graph.templates?.find((t) => t.id === id);
    if (!target) { setNotice('Shared structure is no longer available. Choose an ordinary component.'); return; }
    view.update({ browser: { ...view.browser, selectedFamily: null } });
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
    const previousInstance = concreteInstance ?? anchorInstance;
    const boundary = view.boundary;
    const rebound = boundary ? boundary.endpoints.flatMap((endpoint) => {
      const role = previousInstance.ports.find((p) => p.node_id === endpoint.node_id && p.port_id === endpoint.port_id)?.role;
      const port = target.ports.find((p) => p.role === role);
      return port ? [{ node_id: port.node_id, port_id: port.port_id }] : [];
    }) : [];
    const owner = boundary && remapNode(boundary.owner.id, previousInstance, target);
    const nextBoundary: BoundarySelection | undefined = boundary?.templatePort
      ? bindTemplatePortSelection(graph, template, anchorInstance, id ? target : null, boundary.templatePort)
      : boundary && owner && rebound.length ? { kind: 'boundary', owner: { kind: 'source', id: owner }, endpoints: rebound } : undefined;
    view.update({ shared: next, selected: nextBoundary ? nextBoundary.owner.kind === 'source' ? nextBoundary.owner.id : null : nextSelected,
      selectionMode: id ? 'source' : 'structure', boundary: nextBoundary });
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
    change({ ...base, modelCollapsed: false, expanded: [...expanded], repetitions: { ...base.repetitions, [info.repetition.id]: { start, count: 1 } }, stateScope: undefined, scope: undefined });
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
    change({ ...base, modelCollapsed: false, expanded: [...expanded], exhaustive: false, stateScope: undefined, scope: undefined,
      repetitions: { ...base.repetitions, [id]: { start: Math.max(0, Math.min(repetition.instances.length - 1, start ?? current?.start ?? 0)), count: windowSize } } });
  });
  const toggle = useCanvasCallback((id: string) => {
    const node = projected.get(id);
    if (node) toggleComponent(view, graph, node, windowSize);
  });
  const reveal = useCanvasCallback((id: string) => {
    const base = shared || scope && !scope.members.has(id) && scope.id !== id ? leaveIsolation(true) : options;
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
    change({ ...base, scope: base.scope, modelCollapsed: false, expanded: [...expanded], repetitions, stateScope: undefined,
      ...(derived ? { deriveMlp: true, exhaustive: false } : {}) });
  });
  const viewInModel = useCanvasCallback((target?: string) => {
    if (shared && !concreteInstance && !target) return;
    const id = target ?? concreteInstance?.node_id ?? (selected && (records.has(selected) || mlps.some((group) => group.id === selected)) ? selected : options.scope);
    // reveal performs the same exact-instance action after restoring global state.
    if (!id) return;
    const base = leaveIsolation(true);
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
    change({ ...base, modelCollapsed: false, expanded: [...expanded], repetitions, scope: undefined, stateScope: undefined,
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
    if (node.presentation === 'model') {
      onInspect?.({ modelId, sessionId, graphId: graph.graph_id, trigger, boundary: { kind: 'boundary', owner: interfaces.outer, endpoints: [] } }); return;
    }
    if (node.record) nativeInspect(records.get(node.record.id)!, trigger);
    else setInspection({ nodeId: node.id, trigger });
  });
  const matrix = useCanvasCallback((node: ProjectedNode, parameterId: string, trigger: HTMLElement) => {
    if (shared && !concreteInstance) return;
    const record = node.record && records.get(node.record.id);
    if (!record || !ownParameters(record, parameters).some((p) => p.id === parameterId && p.inspection.status === 'available')) return;
    setInspection(null);
    onInspect?.({ modelId, sessionId, graphId: graph.graph_id, node: record, trigger, parameterId,
      ...(concreteInstance ? { templateInstanceId: concreteInstance.node_id } : {}) });
  });
  const activate = useCanvasCallback((node: ProjectedNode) => {
    if (cardDoubleClick(node)) toggle(node.id);
  });
  const overview = useCanvasCallback(() => {
    if (options.scope) leaveIsolation();
    focusContext(null); fitPending.current = true;
    restorePending.current = undefined;
    change({ modelCollapsed: false, expanded: overviewExpansion(graph), repetitions: {}, exhaustive: false, stateScope: undefined, scope: undefined });
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
  const clearInspection = useCanvasCallback(() => setInspection(null));
  useLayoutRequest(layoutInput, options, retry, view, rememberAnchor, setResult, clearInspection, flowContainer);
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
  const pin = useCanvasCallback((id: string, trigger: HTMLElement) => { view.update({ edge: id, boundary: undefined, browser: { ...view.browser, selectedFamily: null } }); setInspection({ edgeId: id, trigger }); });
  const selectPort = useCanvasCallback((endpoint: { node_id: string; port_id: string }) => {
    const node = projected.get(endpoint.node_id), port = node?.ports.find((p) => p.id === endpoint.port_id);
    if (!node || !port) return;
    if (port.templatePort && template && anchorInstance) {
      const boundary = bindTemplatePortSelection(graph, template, anchorInstance, concreteInstance, port.templatePort);
      if (boundary) selectBoundary(boundary);
      return;
    }
    selectBoundary({ kind: 'boundary', owner: { kind: node.presentation === 'model' ? 'model' : node.record ? 'source' : 'presentation', id: node.record?.id ?? node.id },
      endpoints: port.endpoints });
  });
  const boundaryPorts = useMemo(() => view.boundary ? result.layout?.projection.nodes.flatMap((n) => n.ports.filter((p) =>
    view.boundary!.templatePort ? p.templatePort?.templateId === view.boundary!.templatePort.templateId &&
      p.templatePort.nodeRole === view.boundary!.templatePort.nodeRole && p.templatePort.portRole === view.boundary!.templatePort.portRole
      : p.endpoints.some((e) => view.boundary!.endpoints.some((s) => endpointKey(s) === endpointKey(e)))).map((p) => ({ node_id: n.id, port_id: p.id }))) ?? [] : [], [view.boundary, result.layout]);
  const lineHit = useMemo(() => connectionHitResolver(result.layout?.projection.edges ?? [], result.layout?.routes ?? []), [result.layout]);
  const emphasis = useMemo(() => {
    const projection = result.layout?.projection;
    if (!projection) return new Set<string>();
    if (!temporary && !focused && boundaryPorts.length) return new Set(boundaryPorts.flatMap((port) => connectionSet(projection, { port })));
    const target = temporary ?? focused ?? { edgeId: pinned ?? '' };
    return new Set('edgeIds' in target ? target.edgeIds.flatMap((edgeId) => connectionSet(projection, { edgeId })) : connectionSet(projection, target));
  }, [focused, pinned, result.layout, temporary, boundaryPorts]);
  const emphasizedPorts = useMemo(() => {
    const ports = new Set(result.layout?.projection.edges.filter((e) => emphasis.has(e.id)).flatMap((e) => [endpointKey(e.source), endpointKey(e.target)]));
    const target = temporary ?? focused;
    if (target && 'port' in target) ports.add(endpointKey(target.port));
    if (!target) boundaryPorts.forEach((port) => ports.add(endpointKey(port)));
    return ports;
  }, [emphasis, result.layout, temporary, focused, boundaryPorts]);
  const interaction = useMemo(() => ({ emphasized: emphasis, ports: emphasizedPorts, zoom, lineHit, hover, focus: keyboardFocus, pin, selectPort }), [emphasis, emphasizedPorts, hover, keyboardFocus, lineHit, pin, selectPort, zoom]);
  // Keep shape annotations and row geometry on the same completed layout while
  // a preference change is awaiting the worker.
  const cardDimensions = Boolean(result.options?.dimensions);
  const nodes = useMemo<CanvasNode[]>(() => (result.layout?.boxes ?? []).map((box) => {
    const record = projected.get(box.id)!;
    const summary = cardSummary(record.record, parameters);
    const raised = new Set(result.layout!.ports.filter((port) => port.nodeId === box.id && port.label.raised).map((port) => port.portId));
    const subtitle = record.summary?.replaceAll('linear attention', 'linear').replaceAll('full attention', 'full') ?? variants.get(record.id)?.replace(/^Instance \d+ · /, '') ?? '';
    return { id: box.id, type: 'architecture', position: { x: box.x, y: box.y },
      ...(box.parentId ? { parentId: box.parentId } : {}), width: box.width, height: box.height,
      style: { width: box.width, height: box.height, pointerEvents: record.expanded ? 'none' : 'auto' }, zIndex: 200,
      selected: selected === cardSelection(record) && (!sharedActive || concreteInstance !== null || selectionMode === 'structure'),
      data: { record, label: displayLabel(record, graph), subtitle,
        summary, metrics: cardMetrics(record, summary, cardDimensions, Boolean(subtitle || diagnosed.has(box.id)), raised), dimensions: cardDimensions, matrix,
        ports: result.layout!.ports.filter((p) => p.nodeId === box.id), diagnostic: diagnosed.has(box.id), toggle, select, activate, inspect,
        navigation: cardNavigation(record, concreteInstance?.node_id ?? options.scope, sharedActive && !concreteInstance), navigate: navigateCard } };
  }), [activate, diagnosed, graph, inspect, projected, result.layout, selected, toggle, variants, select, options.scope, sharedActive, concreteInstance, navigateCard, parameters, cardDimensions, matrix, selectionMode]);
  const cameraState = useLayoutCamera(layoutResult.layout, options,
    result.options === options && result.input === layoutInput.graph && !result.error, nodes, flowContainer, flow, async (fitLayout, size) => {
      const camera = flow.getViewport();
      const center = centerPending.current && boxes.get(centerPending.current);
      const substitute = anchor.current?.sourceId && [...projected.values()].find((n) => n.sourceIds.includes(anchor.current!.sourceId!));
      const at = anchor.current && (boxes.get(anchor.current.id) ?? (substitute ? boxes.get(substitute.id) : undefined));
      if (collapsePending.current === options) await fitLayout({ padding: 0.1, minZoom: 0.00001, maxZoom: 1 });
      else if (restorePending.current) await flow.setViewport(restorePending.current);
      else if (scopeCameraPending.current) await fitLayout({ padding: 0.1, minZoom: 0.8, maxZoom: 1 });
      else if (fitPending.current) await fitLayout({ padding: 0.1, minZoom: 0.00001, maxZoom: 1,
        ...(!options.scope && focusId && boxes.has(focusId) ? { nodes: flow.getNodes().filter((node) => node.id === focusId) } : {}) });
      else if (center) await flow.setCenter(center.absoluteX + Math.min(center.width / 2, 360), center.absoluteY + Math.min(center.height / 2, 240), { zoom: Math.max(camera.zoom, 0.8) });
      else if (at && anchor.current) await flow.setViewport({ ...camera, x: anchor.current.x - at.absoluteX * camera.zoom, y: anchor.current.y - at.absoluteY * camera.zoom });
      else if (!initialized.current) {
        if (savedViewport.current) await flow.setViewport(savedViewport.current);
        else await flow.setViewport(initialViewport(visibleBounds(result.layout!), size.width, size.height));
      }
    }, () => {
      initialized.current = true; anchor.current = null; centerPending.current = null; fitPending.current = false;
      restorePending.current = undefined; scopeCameraPending.current = false;
      collapsePending.current = undefined;
    });
  const cancelCamera = useCanvasCallback(() => {
    view.update({ initialOverview: false }); cameraState.cancel(); initialized.current = true;
    anchor.current = null; centerPending.current = null; fitPending.current = false;
    restorePending.current = undefined; scopeCameraPending.current = false; collapsePending.current = undefined;
  });
  const fit = useCanvasCallback(() => {
    cancelCamera();
    let focusNodes: CanvasNode[] | undefined;
    if (!options.scope && focusId && boxes.has(focusId)) focusNodes = flow.getNodes().filter((n) => n.id === focusId);
    void flow.fitView({ ...(focusNodes?.length ? { nodes: focusNodes } : {}), padding: 0.1, minZoom: 0.00001, maxZoom: 1 });
  });
  const edges = useMemo<ConnectionEdge[]>(() => (result.layout?.projection.edges ?? []).map((edge) => ({
    id: edge.id, source: edge.source.node_id, target: edge.target.node_id, sourceHandle: `source:${edge.source.port_id}`,
    targetHandle: `target:${edge.target.port_id}`, type: 'connection', focusable: false, selectable: false,
    zIndex: emphasis.has(edge.id) ? 100 : 2,
    data: { connection: edge, route: result.layout!.routes.find((r) => r.id === edge.id)!, projection: result.layout!.projection, dimensions: cardDimensions },
  })), [cardDimensions, emphasis, result.layout]);
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
  const selectedCard = selected ? [...projected.values()].find((node) => cardSelection(node) === selected) : undefined;
  const selectedEdge = result.layout?.projection.edges.find((e) => e.id === pinned);
  const eligibleTemplate = !shared && selectedRecord ? graph.templates?.find((t) => t.instances.some((i) => i.nodes.some((m) => m.node_id === selectedRecord.id))) : undefined;
  const eligibleInstance = eligibleTemplate?.instances.find((i) => i.nodes.some((m) => m.node_id === selectedRecord?.id));
  const centerInLayout = (id: string) => {
    // The requested reveal/center owns the next layout's camera, even while
    // the current scope is still awaiting its initial fit.
    scopeCameraPending.current = false; fitPending.current = false;
    reveal(id);
  };
  const centerSelected = useCanvasCallback(() => {
    if (!selectedRecord) return;
    if (!shared || !commonSelection && !insideBrowserScope(selectedRecord.id)) { centerInLayout(selectedRecord.id); return; }
    const presentation = [...projected.values()].find((n) => n.record?.id === selectedRecord.id);
    const box = presentation && boxes.get(presentation.id);
    if (box) {
      cancelCamera();
      void flow.setCenter(box.absoluteX + box.width / 2, box.absoluteY + box.height / 2, { zoom: flow.getZoom() });
    } else {
      const mapped = browserExpansionId(graph, view, selectedRecord.id);
      centerPending.current = mapped; restorePending.current = undefined;
      scopeCameraPending.current = false; fitPending.current = false;
      change({ expanded: [...withAncestors(mapped, new Set(view.expanded))] });
    }
  });
  function insideBrowserScope(id: string) {
    if (!scope) return true;
    if (shared) return concreteInstance?.nodes.some((node) => node.node_id === id) ?? false;
    return scope.members.has(id) || scope.id === id;
  }
  const toggleBrowser = useCanvasCallback((id: string) => {
    const record = records.get(id);
    if (!record || record.kind !== 'group' || !record.children.length) return;
    const outside = !insideBrowserScope(id);
    if (outside) { const selection = view.selectionMode === 'source' ? view.selected : null; leaveIsolation(true); selectComponent(view, selection); }
    const mapped = browserExpansionId(graph, view, id), source = records.get(mapped)!;
    const base = projectionOptions(view);
    const expanded = withAncestors(mapped, new Set(base.expanded));
    // Exact expanded instances are already visible alongside repetition windows.
    // Preserve those windows and let the shared toggle change only this component.
    view.update({ expanded: [...expanded], modelCollapsed: false, stateScope: undefined });
    toggleComponent(view, graph, { id: mapped, kind: source.kind, label: source.label, record: source,
      sourceIds: [mapped], ports: [], expanded: base.exhaustive || base.expanded.includes(mapped) }, windowSize);
    if (outside) { centerPending.current = id; restorePending.current = undefined; }
  });
  const viewSelectionInModel = useCanvasCallback(() => { if (selected) viewInModel(selected); });
  const viewScopeInModel = useCanvasCallback(() => viewInModel());
  const expandCurrentComponent = useCanvasCallback(() => change(expandComponent(view, graph)));
  const selectedFamily = graph.templates?.find((family) => family.id === view.browser.selectedFamily);
  // Conditional controls can remain in a detached React tree after navigation.
  // Give every control a proxy so that tree cannot retain a Canvas layout.
  const exploreFamily = useCanvasCallback(() => { if (selectedFamily) openShared(selectedFamily.id); });
  const clearFamily = useCanvasCallback(() => view.update({ browser: { ...view.browser, selectedFamily: null } }));
  const inspectSelected = useCanvasCallback((trigger: HTMLElement) => {
    if (view.boundary) {
      // Model-wide search targets source interfaces even while the canvas shows
      // a neutral template. Only an actual template port has common metadata.
      const common = view.boundary.templatePort && shared && !concreteInstance && template && anchorInstance;
      const node = common ? [...projected.values()].find((n) => n.record?.id === view.boundary!.owner.id)?.record : undefined;
      const role = common ? anchorInstance.nodes.find((m) => m.node_id === view.boundary!.owner.id)?.role : undefined;
      onInspect?.({ modelId, sessionId, graphId: graph.graph_id, trigger, boundary: view.boundary,
        ...(node ? { node } : {}), ...(common ? { structureOnly: { label: template.label, role: view.boundary.templatePort?.portRole ?? role ?? 'interface' } } : {}) });
    }
    else if (selectedEdge) setInspection({ edgeId: selectedEdge.id, trigger });
    else if (selectedRecord) nativeInspect(selectedRecord, trigger);
  });
  const centerEdge = useCanvasCallback(() => {
    if (!selectedEdge) return;
    cancelCamera();
    void flow.fitView({ nodes: flow.getNodes().filter((n) => n.id === selectedEdge.source.node_id || n.id === selectedEdge.target.node_id), padding: 0.2, minZoom: 0.00001, maxZoom: 1 });
  });
  const centerDerived = useCanvasCallback(() => { if (selected) centerInLayout(selected); });
  const clearSelection = useCanvasCallback(() => {
    if (selectedEdge) { view.update({ edge: null }); setInspection(null); }
    else selectComponent(view, null);
  });
  const exploreSelected = useCanvasCallback(() => { if (selected) isolate(selected); });
  const shareSelected = useCanvasCallback(() => { if (eligibleTemplate && eligibleInstance) openShared(eligibleTemplate.id, eligibleInstance.node_id); });
  const navigate = useCanvasCallback((item: NavigationItem) => {
    if (shared && item.id === template?.id) return;
    if (item.kind === 'stack') exploreStack(item.id);
    else if (!scope && mlps.some((g) => g.id === item.id)) focusMlp();
    else { const info = instanceOf(graph, item.id); if (info?.instance.node_id === item.id) chooseInstance(item.id); else reveal(item.id); }
  });
  const expandAll = useCanvasCallback(() => { const base = scope ? leaveIsolation() : options; focusContext(null); change({ ...base, scope: undefined, modelCollapsed: false, expanded: graph.nodes.filter((n) => n.kind === 'group').map((n) => n.id), exhaustive: true, stateScope: undefined }); });
  const collapseAll = useCanvasCallback(() => {
    // A common template role is not a concrete source selection. Bound ports
    // retain their real endpoints, but shed the shared-only correspondence tag.
    const boundary = view.selectionMode === 'source' ? view.boundary : undefined;
    const selection = { selected: view.selectionMode === 'source' ? view.selected : null, selectionMode: 'source' as const,
      boundary: boundary ? { kind: boundary.kind, owner: boundary.owner, endpoints: boundary.endpoints } : undefined };
    const base = scope ? leaveIsolation() : options;
    focusContext(null);
    anchor.current = null; view.expansionAnchor = undefined; centerPending.current = null;
    restorePending.current = undefined; scopeCameraPending.current = false; fitPending.current = false;
    view.update({ ...base, ...selection, scope: undefined, expanded: [], modelCollapsed: true, repetitions: {}, exhaustive: false, stateScope: undefined });
    collapsePending.current = view.getProjectionOptions();
  });
  const toggleSelected = useCanvasCallback(() => { if (selectedCard) toggle(selectedCard.id); });
  const preferences = useCanvasCallback((patch: Partial<ProjectionOptions>) => {
    if (patch.dimensions !== undefined) {
      if (selected) rememberAnchor(selected);
      view.update({ dimensions: patch.dimensions });
    }
    change(patch);
  });
  const zoomIn = useCanvasCallback(() => { cancelCamera(); void flow.zoomIn(); });
  const zoomOut = useCanvasCallback(() => { cancelCamera(); void flow.zoomOut(); });
  const centerBoundary = useCanvasCallback(() => {
    if (!view.boundary) return;
    const boundary = view.boundary;
    if (boundary.owner.kind !== 'model') centerInLayout(boundaryPorts[0]?.node_id ?? boundary.owner.id);
    else { if (options.scope) leaveIsolation(true); fit(); }
    selectBoundary(boundary);
  });
  const controlSelection: ControlSelection | undefined = view.boundary ? {
    edge: false, label: view.boundary.endpoints.length || view.boundary.templatePort ? 'Interface ports' : 'Model',
    detail: view.boundary.templatePort ? `Shared interface · ${view.boundary.templatePort.portRole}` :
      view.boundary.endpoints.map((p) => `${records.get(p.node_id)?.label ?? p.node_id} · ${p.port_id}`).join(', '),
    inspect: inspectSelected, center: centerBoundary, clear: clearSelection,
  } : selectedEdge ? {
    edge: true,
    label: `${projected.get(selectedEdge.source.node_id)?.label ?? 'Source'} → ${projected.get(selectedEdge.target.node_id)?.label ?? 'Destination'}`,
    detail: `${selectedEdge.source.port_id} → ${selectedEdge.target.port_id} · ${selectedEdge.originalEdgeIds.join(', ')}`,
    inspect: inspectSelected, center: centerEdge, clear: clearSelection,
  } : selectedRecord ? {
    edge: false, nodeId: selectedRecord.id,
    label: commonSelection ? [...projected.values()].find((n) => n.record?.id === selectedRecord.id)?.label ?? template?.label ?? 'Shared operation' :
      displayLabel({ id: selectedRecord.id, kind: selectedRecord.kind, label: selectedRecord.label, record: selectedRecord, sourceIds: [selectedRecord.id], ports: [], expanded: false }, graph),
    detail: commonSelection ? 'Verified common operation; choose an instance for weights.' : `${selectedRecord.label} · ${selectedRecord.id}`,
    shared: eligibleTemplate && eligibleInstance ? shareSelected : undefined,
    inspect: onInspect && (!shared || commonSelection || insideBrowserScope(selectedRecord.id)) ? inspectSelected : undefined,
    center: centerSelected,
    viewInModel: options.scope && !commonSelection ? viewSelectionInModel : undefined,
    clear: clearSelection,
    explore: !shared && ['group', 'operation'].includes(selectedRecord.kind) && selectedRecord.id !== options.scope ? exploreSelected : undefined,
  } : selected && mlps.some((group) => group.id === selected) ? {
    edge: false, nodeId: selected, label: 'MLP (derived)', detail: selected,
    center: centerDerived, clear: clearSelection,
    explore: selected !== options.scope ? exploreSelected : undefined,
  } : undefined;
  return <ArchitectureWorkspace browser={<ArchitectureBrowser graph={graph} view={view} searchRef={browserSearch}
    select={selectSource} selectBoundary={selectBoundary} selectFamily={selectFamily} toggle={toggleBrowser} exploreStack={exploreStack} />}>
    <div ref={panel} tabIndex={-1} className="architecture-explorer explorer-card" aria-label="Architecture graph" data-graph-id={graph.graph_id}
    data-template-id={shared?.templateId ?? ''} data-template-instance-id={shared?.instanceId ?? ''}
    data-scope-id={concreteInstance?.node_id ?? options.scope ?? ''} data-node-count={graph.nodes.length} data-edge-count={graph.edges.length} data-visible-nodes={nodes.length}
    data-visible-edges={edges.length} data-layout-ms={result.layout?.milliseconds} data-layout-count={result.invocation ?? 0} aria-busy={result.options !== options || !result.error && !cameraState.ready}
    data-source-node-ids={JSON.stringify(sourceNodeIds)} data-represented-edge-ids={JSON.stringify(result.layout?.edgeIds ?? [])}>
    <ArchitectureControls shared={shared && template ? { template, instanceId: shared.instanceId, choose: chooseSharedInstance } : undefined}
      family={selectedFamily ? { label: selectedFamily.label, explore: exploreFamily, clear: clearFamily } : undefined}
      returnContext={!scope && view.history.length ? back : undefined} graph={graph} focus={focusId} stack={stack} options={options} picker={picker}
      isolation={scope ? { back, viewInModel: shared && !concreteInstance ? undefined : viewScopeInModel, expand: expandCurrentComponent } : undefined}
      instanceId={instance?.instance.node_id ?? (stack ? stack.instances[options.repetitions?.[stack.id]?.start ?? 0]?.node_id : undefined)}
      visibleInstances={stack?.instances.filter((i) => projected.has(i.node_id)).map((i) => i.node_id) ?? []}
      breadcrumbs={shared && template ? [{ id: template.id, kind: 'node', label: template.label }] : breadcrumbs}
      selection={controlSelection} navigate={navigate} overview={overview}
      chooseInstance={chooseInstance} exploreStack={exploreStack} windowSize={windowSize} expandAll={expandAll} collapseAll={collapseAll}
      focusLayer={!scope && instance ? focusLayer : undefined} focusMlp={!scope && instance && mlps.some((g) => g.parentId === instance.instance.node_id) ? focusMlp : undefined}
      stateFocus={!scope && instance ? stateFocus : undefined} toggleSelected={selectedCard && cardExpandable(selectedCard) ? toggleSelected : undefined}
      preferences={preferences} derivedMlpAvailable={mlps.some((group) => !records.has(group.id))}
      filtered={!options.exhaustive && !options.showUnused && !!result.layout?.projection.filteredEdgeIds.length} />
    {notices}
    {notice && <p role="status">{notice}</p>}
    {result.error && <div role="alert">{result.error} <button onClick={() => setRetry(retry + 1)}>Retry layout</button>{shared && <button onClick={back}>Return to ordinary view</button>}</div>}
    {cameraState.error && <div role="alert">{cameraState.error} <button onClick={() => setRetry(retry + 1)}>Retry layout</button></div>}
    <div className="architecture-flow" ref={flowContainer}>
      {!result.layout && !result.error && <p className="architecture-loading" role="status">Laying out architecture…</p>}
      <ConnectionContext.Provider value={interaction}>
        <ReactFlow<CanvasNode, ConnectionEdge> nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} onlyRenderVisibleElements
          zIndexMode="manual" nodesDraggable={false} nodesConnectable={false} edgesReconnectable={false} deleteKeyCode={null}
          minZoom={0.00001} maxZoom={4} defaultViewport={view.viewport ?? { x: 0, y: 0, zoom: 1 }} panOnDrag zoomOnScroll
          onMoveStart={(event) => { if (event) cancelCamera(); }}
          onViewportChange={(viewport) => setZoom(viewport.zoom)} onMoveEnd={(_, viewport) => view.update({ viewport })}
          onNodeClick={(event, node) => { event.stopPropagation(); select(cardSelection(node.data.record)); }}
          onNodeDoubleClick={(event, node) => { event.stopPropagation(); activate(node.data.record); }}
          onNodesChange={(changes) => { for (const value of changes) if (value.type === 'select' && value.selected) { const node = projected.get(value.id); if (node) select(cardSelection(node)); } }}
          aria-label="Architecture canvas" />
      </ConnectionContext.Provider>
      <CameraDock zoomIn={zoomIn} zoomOut={zoomOut} fit={fit} />
      {inspection && (activeInspectionEdge || activeInspectionNode) && <ConnectionInspection graph={graph} edge={activeInspectionEdge} node={activeInspectionNode}
        structure={shared && !concreteInstance && activeInspectionEdge ? { source: projected.get(activeInspectionEdge.source.node_id)?.label ?? '', target: projected.get(activeInspectionEdge.target.node_id)?.label ?? '' } : undefined}
        explore={activeInspectionNode?.presentation === 'mlp' && activeInspectionNode.id !== options.scope ? () => isolate(activeInspectionNode.id) : undefined}
        trigger={inspection.trigger} onClose={() => setInspection(null)} inspectNode={(id) => {
          if (picker.current) nativeInspect(records.get(id)!, picker.current);
        }} />}
    </div>
  </div></ArchitectureWorkspace>;
}
