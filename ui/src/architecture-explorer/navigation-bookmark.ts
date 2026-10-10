import type { components } from '../api/generated/types';
import type { Graph, GraphNode, GraphSnapshot, GraphView } from './graph';
import { GraphView as View } from './graph';
import { deriveMlpGroups } from './derived-groups';
import { depthRanges, indexedNodeId, indexedStacks } from './indexed-repetition';
import { bodyId, bodyRangeId } from './repeated-body';
import { componentScope } from './scope';
import { projectionOptions, snapshotView } from './scope-navigation';
import { projectGraph } from './projection';
import { interfaceIndex } from './interfaces';
import type { BoundarySelection } from './interfaces';
import { overviewExpansion } from './overview';
import { ownParameters } from './card-summary';
import { commonNode, templateGraph } from './shared-structure';
import type { ArchitectureSelection } from './ArchitectureCanvas';

type Source = { key: string; kind: GraphNode['kind']; operation?: string; parents: string[] };
type Locator = { type: 'model' } | { type: 'source'; source: Source } | { type: 'mlp'; owner: Source } |
  { type: 'repeat'; repetition: string; first?: number; last?: number } |
  { type: 'indexed'; repetition: string; family: string; role: string; first: number } |
  { type: 'body'; repetition: string; first: number; slots: string[]; slot?: number; role?: string };
type Endpoint = { node: Source; port: string; direction: 'input' | 'output' };
type Edge = { source: Endpoint; target: Endpoint; kind: Graph['edges'][number]['kind'] };
type Boundary = { owner: Locator | 'model'; kind: BoundarySelection['owner']['kind']; endpoints: Endpoint[];
  template?: { key: string; nodeRole: string; portRole: string } };
export interface CameraAnchor { target: string; zoom: number; x: number; y: number }
interface PortableView {
  selected: Locator[]; scope: Locator[]; focus: Locator[]; expanded: Locator[];
  flags: Pick<GraphSnapshot, 'dimensions' | 'exhaustive' | 'showUnused' | 'showContext' | 'deriveMlp' | 'modelCollapsed' | 'selectionMode'>;
  repetitions: { key: string; first: number; count: number }[]; activeStack?: string | undefined; stateScope?: Source | undefined;
  shared?: { family: string; anchor: Source; instance: Source | null } | null;
  edge?: Edge; paths?: Edge[][]; boundary?: Boundary | undefined;
  camera?: { target: Locator; zoom: number; x: number; y: number };
}
export interface ArchitectureBookmark {
  namespace?: string;
  current: PortableView;
  history: PortableView[];
  global?: PortableView;
  browser: Omit<GraphView['browser'], 'families' | 'selectedFamily'> & { families: string[]; selectedFamily: string | null };
  inspection?: { unavailable?: true; node?: Source; boundary?: Boundary | undefined; parameter?: string; tensor?: string | undefined;
    structure?: { family: string; role: string }; instance?: Source | undefined };
}
const only = <T,>(items: T[]): T | undefined => items.length === 1 ? items[0] : undefined;

/** Ephemeral lookup. Only bounded typed locators leave this call; no graph is retained. */
class Navigation {
  readonly nodes: Map<string, GraphNode>;
  readonly locators = new Map<string, Locator>();
  constructor(readonly graph: Graph, windows: GraphSnapshot['repetitions'] = {}) {
    this.nodes = new Map(graph.nodes.map((n) => [n.id, n]));
    this.locators.set(interfaceIndex(graph).outer.id, { type: 'model' });
    for (const node of graph.nodes) {
      const source = this.source(node.id);
      if (source) this.locators.set(node.id, { type: 'source', source });
    }
    for (const mlp of deriveMlpGroups(graph)) {
      const owner = this.source(mlp.parentId);
      if (owner) this.locators.set(mlp.id, { type: 'mlp', owner });
    }
    for (const rep of graph.repetitions) {
      if (!rep.navigation_key) continue;
      const repetition = rep.navigation_key;
      this.locators.set(`repeat:${rep.id}`, { type: 'repeat', repetition });
      const addRange = (id: string, instances: typeof rep.instances) => {
        if (instances.length) this.locators.set(id, { type: 'repeat', repetition,
          first: instances[0]!.index, last: instances.at(-1)!.index });
      };
      addRange(`repeat:${rep.id}:0:${rep.instances.length - 1}`, rep.instances);
      for (const range of depthRanges(graph, rep)) {
        addRange(`repeat:${range.id}`, range.instances);
        // Projection uses window positions in its local repetition slice.
        addRange(`repeat:${range.id}:0:${range.instances.length - 1}`, range.instances);
      }
      const window = windows[rep.id];
      if (window) addRange(`repeat:${rep.id}:${window.start}:${Math.min(rep.instances.length, window.start + window.count) - 1}`,
        rep.instances.slice(window.start, window.start + window.count));
      for (const body of rep.bodies ?? []) {
        const first = rep.instances[body.start]?.index;
        const slots = body.slots.map((id) => graph.templates?.find((t) => t.id === id)?.navigation_key);
        if (first === undefined || slots.some((key) => !key)) continue;
        const base = { type: 'body' as const, repetition, first, slots: slots as string[] };
        this.locators.set(bodyId(rep, body), base);
        for (const range of body.ranges) {
          const scope = bodyRangeId(rep, body, range.start);
          this.locators.set(scope, { ...base, slot: range.start });
          const family = graph.templates?.find((t) => t.id === body.slots[range.start]);
          const anchor = family?.instances.find((i) => i.node_id === rep.instances[body.start + range.start]?.node_id);
          for (const node of anchor?.nodes ?? []) this.locators.set(indexedNodeId(scope, node.node_id), { ...base, slot: range.start, role: node.role });
        }
      }
    }
    for (const [id, stack] of indexedStacks(graph)) {
      const rep = graph.repetitions.find((r) => r.instances.some((i) => i.node_id === stack.instances[0]?.node_id));
      if (!rep?.navigation_key || !stack.template.navigation_key) continue;
      const first = rep.instances.find((i) => i.node_id === stack.instances[0]?.node_id)!.index;
      for (const node of stack.instances[0]!.nodes) this.locators.set(indexedNodeId(id, node.node_id),
        { type: 'indexed', repetition: rep.navigation_key, family: stack.template.navigation_key, role: node.role, first });
    }
  }
  source(id: string | undefined): Source | undefined {
    const node = id && this.nodes.get(id);
    if (!node || !node.navigation_key) return;
    const parents: string[] = [];
    let parent = node.parent_id;
    while (parent) {
      const record = this.nodes.get(parent);
      if (!record?.navigation_key) return;
      parents.push(record.navigation_key); parent = record.parent_id;
    }
    return { key: node.navigation_key, kind: node.kind, ...(node.operation ? { operation: node.operation } : {}), parents };
  }
  resolveSource(source: Source | undefined) {
    if (!source) return;
    const node = only(this.graph.nodes.filter((n) => n.navigation_key === source.key));
    return node && node.kind === source.kind && node.operation === source.operation &&
      JSON.stringify(this.source(node.id)?.parents) === JSON.stringify(source.parents) ? node.id : undefined;
  }
  resolve(locator: Locator | undefined): string | undefined {
    if (!locator) return;
    if (locator.type === 'model') return interfaceIndex(this.graph).outer.id;
    if (locator.type === 'source') return this.resolveSource(locator.source);
    if (locator.type === 'mlp') {
      const owner = this.resolveSource(locator.owner);
      return only(deriveMlpGroups(this.graph).filter((m) => m.parentId === owner))?.id;
    }
    const exact = [...this.locators].filter(([, value]) => JSON.stringify(value) === JSON.stringify(locator));
    // A repeat range can have two equivalent spellings; prefer the standard projection ID.
    if (locator.type === 'repeat') {
      const rep = only(this.graph.repetitions.filter((r) => r.navigation_key === locator.repetition));
      if (!rep) return;
      if (locator.first === undefined) return `repeat:${rep.id}`;
      const first = rep.instances.findIndex((i) => i.index === locator.first);
      if (first < 0) return;
      const last = rep.instances.findLastIndex((i) => i.index <= locator.last!);
      return last >= first ? `repeat:${rep.id}:${first}:${last}` : undefined;
    }
    return only(exact)?.[0];
  }
  chain(id: string | null | undefined): Locator[] {
    if (!id) return [];
    const result: Locator[] = [];
    const locator = this.locators.get(id);
    if (locator) result.push(locator);
    let source = this.nodes.get(id) ?? (locator?.type === 'mlp' ? this.nodes.get(this.resolveSource(locator.owner)!) : undefined);
    if (locator?.type === 'mlp' && source && this.locators.has(source.id)) result.push(this.locators.get(source.id)!);
    if (!source && locator && 'repetition' in locator) {
      source = this.nodes.get(this.graph.repetitions.find((r) => r.navigation_key === locator.repetition)?.parent_id ?? '');
      if (source && this.locators.has(source.id)) result.push(this.locators.get(source.id)!);
    }
    while (source?.parent_id) {
      source = this.nodes.get(source.parent_id);
      if (source && this.locators.has(source.id)) result.push(this.locators.get(source.id)!);
    }
    return result;
  }
  endpoint(nodeId: string, portId: string): Endpoint | undefined {
    const node = this.source(nodeId), port = this.nodes.get(nodeId)?.ports.find((p) => p.id === portId);
    return node && port?.navigation_key ? { node, port: port.navigation_key, direction: port.direction } : undefined;
  }
  resolveEndpoint(endpoint: Endpoint) {
    const node = this.resolveSource(endpoint.node), port = node && only(this.nodes.get(node)!.ports.filter((p) =>
      p.navigation_key === endpoint.port && p.direction === endpoint.direction));
    return node && port ? { node_id: node, port_id: port.id } : undefined;
  }
  boundary(boundary: BoundarySelection | undefined): Boundary | undefined {
    if (!boundary) return;
    const owner = boundary.owner.kind === 'model' ? 'model' : this.locators.get(boundary.owner.id);
    const endpoints = boundary.endpoints.map((e) => this.endpoint(e.node_id, e.port_id));
    const template = boundary.templatePort && this.graph.templates?.find((t) => t.id === boundary.templatePort!.templateId);
    if (!owner || endpoints.some((e) => !e) || boundary.templatePort && !template?.navigation_key) return;
    return { owner, kind: boundary.owner.kind, endpoints: endpoints as Endpoint[], ...(template?.navigation_key ? {
      template: { key: template.navigation_key, nodeRole: boundary.templatePort!.nodeRole, portRole: boundary.templatePort!.portRole } } : {}) };
  }
  resolveBoundary(boundary: Boundary | undefined): BoundarySelection | undefined {
    if (!boundary) return;
    const owner = boundary.owner === 'model' ? interfaceIndex(this.graph).outer.id : this.resolve(boundary.owner);
    const endpoints = boundary.endpoints.map((e) => this.resolveEndpoint(e));
    const template = boundary.template && only(this.graph.templates?.filter((t) => t.navigation_key === boundary.template!.key) ?? []);
    if (!owner || endpoints.some((e) => !e) || boundary.template && (!template || !template.instances.every((i) =>
      i.nodes.some((n) => n.role === boundary.template!.nodeRole) && i.ports.some((p) => p.role === boundary.template!.portRole)))) return;
    return { kind: 'boundary', owner: { kind: boundary.kind, id: owner }, endpoints: endpoints as BoundarySelection['endpoints'],
      ...(template ? { templatePort: { kind: 'template-port', templateId: template.id, nodeRole: boundary.template!.nodeRole, portRole: boundary.template!.portRole } } : {}) };
  }
}

function presentation(graph: Graph, view: GraphSnapshot) {
  const template = graph.templates?.find((t) => t.id === view.shared?.templateId);
  const anchor = template?.instances.find((i) => i.node_id === view.shared?.anchorId);
  return projectGraph(anchor ? templateGraph(graph, anchor) : graph, projectionOptions(view));
}

function captureView(graph: Graph, view: GraphSnapshot, anchor?: CameraAnchor): PortableView {
  anchor ??= view.cameraAnchor ?? view.restoreCamera;
  const nav = new Navigation(graph, view.repetitions);
  const sourceEdge = graph.edges.find((e) => e.id === view.edge);
  const connection = view.edge && presentation(graph, view).edges.find((e) => e.id === view.edge);
  const paths = connection ? connection.paths.map((path) => path.map((e) => {
    const source = nav.endpoint(e.source.node_id, e.source.port_id), target = nav.endpoint(e.target.node_id, e.target.port_id);
    return source && target ? { source, target, kind: e.kind } : undefined;
  })) : undefined;
  const from = sourceEdge && nav.endpoint(sourceEdge.source.node_id, sourceEdge.source.port_id);
  const to = sourceEdge && nav.endpoint(sourceEdge.target.node_id, sourceEdge.target.port_id);
  const family = graph.templates?.find((t) => t.id === view.shared?.templateId);
  const sharedAnchor = nav.source(view.shared?.anchorId), instance = nav.source(view.shared?.instanceId ?? undefined);
  const cameraTarget = anchor?.target ?? view.selected ?? view.scope ?? interfaceIndex(graph).outer.id;
  const target = cameraTarget && nav.locators.get(cameraTarget);
  const zoom = anchor?.zoom ?? view.viewport?.zoom;
  return {
    selected: nav.chain(view.selected), scope: nav.chain(view.scope), focus: nav.chain(view.focus),
    expanded: view.expanded.flatMap((id) => nav.locators.get(id) ?? []),
    flags: { dimensions: view.dimensions, exhaustive: view.exhaustive, showUnused: view.showUnused, showContext: view.showContext,
      deriveMlp: view.deriveMlp, modelCollapsed: view.modelCollapsed, selectionMode: view.selectionMode },
    repetitions: Object.entries(view.repetitions).flatMap(([id, window]) => {
      const rep = graph.repetitions.find((r) => r.id === id), first = rep?.instances[window.start]?.index;
      return rep?.navigation_key && first !== undefined ? [{ key: rep.navigation_key, first, count: window.count }] : [];
    }),
    ...(view.activeStack ? { activeStack: graph.repetitions.find((r) => r.id === view.activeStack)?.navigation_key } : {}),
    ...(view.stateScope ? { stateScope: nav.source(view.stateScope) } : {}),
    ...(family?.navigation_key && sharedAnchor && (!view.shared!.instanceId || instance) ? {
      shared: { family: family.navigation_key, anchor: sharedAnchor, instance: instance ?? null } } : view.shared ? { shared: null } : {}),
    ...(paths?.length && paths.every((path) => path.every(Boolean)) ? { paths: paths as Edge[][] } : {}),
    ...(from && to ? { edge: { source: from, target: to, kind: sourceEdge!.kind } } : {}),
    ...(nav.boundary(view.boundary) ? { boundary: nav.boundary(view.boundary) } : {}),
    ...(target && zoom ? { camera: { target, zoom, x: anchor?.x ?? 0.5, y: anchor?.y ?? 0.5 } } : {}),
  };
}

export function captureBookmark(graph: Graph, view: GraphView, inspected?: ArchitectureSelection | null, inventory?: components['schemas']['TensorInventory']): ArchitectureBookmark {
  const nav = new Navigation(graph), template = graph.templates?.find((t) => t.id === view.shared?.templateId);
  const node = inspected?.node && nav.source(inspected.node.id);
  const parameter = graph.parameters.find((p) => p.id === inspected?.parameterId);
  const tensorId = parameter?.inspection.status === 'available' ? parameter.inspection.tensor_id : undefined;
  const tensorName = tensorId ? inventory?.tensors.find((t) => t.id === tensorId)?.name ?? parameter?.name : undefined;
  const structure = inspected?.structureOnly && template?.navigation_key ? { family: template.navigation_key, role: inspected.structureOnly.role } : undefined;
  return { ...(graph.navigation_namespace ? { namespace: graph.navigation_namespace } : {}),
    current: captureView(graph, view, view.cameraAnchor), history: view.history.slice(-16).map((v) => captureView(graph, v)),
    ...(view.globalView ? { global: captureView(graph, view.globalView) } : {}),
    browser: { ...view.browser, families: view.browser.families.flatMap((id) => graph.templates?.find((t) => t.id === id)?.navigation_key ?? []),
      selectedFamily: graph.templates?.find((t) => t.id === view.browser.selectedFamily)?.navigation_key ?? null },
    ...(inspected ? { inspection: {
      ...(inspected.structureOnly && !structure || !node && !nav.boundary(inspected.boundary) && !structure ? { unavailable: true as const } : {}),
      ...(node ? { node } : {}), ...(nav.boundary(inspected.boundary) ? { boundary: nav.boundary(inspected.boundary) } : {}),
      ...(parameter ? { parameter: parameter.name, tensor: tensorName } : {}),
      ...(structure ? { structure } : {}), ...(nav.source(inspected.templateInstanceId) ? { instance: nav.source(inspected.templateInstanceId) } : {}),
    } } : {}) };
}

function restoreView(graph: Graph, saved: PortableView): { view: GraphView; fallback: boolean } {
  const view = new View(overviewExpansion(graph)), nav = new Navigation(graph);
  Object.assign(view, saved.flags);
  let fallback = false;
  const nearest = (chain: Locator[]) => {
    for (const [position, locator] of chain.entries()) {
      const id = nav.resolve(locator);
      if (id) { fallback ||= position > 0; return id; }
    }
    fallback ||= chain.length > 0; return undefined;
  };
  view.scope = nearest(saved.scope);
  if (view.scope) { try { componentScope(graph, view.scope); } catch { view.scope = undefined; fallback = true; } }
  view.selected = nearest(saved.selected) ?? null; view.focus = nearest(saved.focus) ?? null;
  view.expanded = saved.expanded.flatMap((locator) => nav.resolve(locator) ?? []);
  if (view.scope && !view.expanded.includes(view.scope)) view.expanded.push(view.scope);
  view.stateScope = nav.resolveSource(saved.stateScope);
  view.activeStack = only(graph.repetitions.filter((r) => r.navigation_key === saved.activeStack))?.id ?? null;
  for (const window of saved.repetitions) {
    const rep = only(graph.repetitions.filter((r) => r.navigation_key === window.key));
    if (!rep?.instances.length) continue;
    const found = rep.instances.findIndex((i) => i.index >= window.first);
    const start = found < 0 ? rep.instances.length - 1 : found;
    view.repetitions[rep.id] = { start, count: Math.min(Math.max(1, window.count), rep.instances.length - start) };
  }
  if (saved.shared === null) {
    view.scope = undefined; view.selected = null; view.focus = null; fallback = true;
  } else if (saved.shared) {
    const template = only(graph.templates?.filter((t) => t.navigation_key === saved.shared!.family) ?? []);
    const anchor = nav.resolveSource(saved.shared.anchor), instance = nav.resolveSource(saved.shared.instance ?? undefined);
    if (template && anchor && template.instances.some((i) => i.node_id === anchor) &&
      (!saved.shared.instance || instance && template.instances.some((i) => i.node_id === instance))) {
      view.shared = { templateId: template.id, anchorId: anchor, instanceId: instance ?? null };
      view.scope = anchor; view.selectionMode = instance ? 'source' : 'structure';
    } else {
      // Never promote a neutral structure or a removed concrete binding to layer zero.
      view.scope = undefined; view.selected = null; view.focus = null; fallback = true;
    }
  }
  view.boundary = nav.resolveBoundary(saved.boundary);
  if (saved.edge) {
    const source = nav.resolveEndpoint(saved.edge.source), target = nav.resolveEndpoint(saved.edge.target);
    view.edge = only(graph.edges.filter((e) => e.kind === saved.edge!.kind &&
      e.source.node_id === source?.node_id && e.source.port_id === source?.port_id &&
      e.target.node_id === target?.node_id && e.target.port_id === target?.port_id))?.id ?? null;
  }
  const projection = presentation(graph, view);
  if (saved.paths) {
    const edgeIds = saved.paths.map((path) => path.map((edge) => {
      const source = nav.resolveEndpoint(edge.source), target = nav.resolveEndpoint(edge.target);
      return only(graph.edges.filter((e) => e.kind === edge.kind && e.source.node_id === source?.node_id && e.source.port_id === source?.port_id &&
        e.target.node_id === target?.node_id && e.target.port_id === target?.port_id))?.id;
    }));
    view.edge = edgeIds.every((path) => path.every(Boolean)) ? only(projection.edges.filter((edge) =>
      JSON.stringify(edge.paths.map((path) => path.map((e) => e.id))) === JSON.stringify(edgeIds)))?.id ?? null : null;
  }
  if (view.selected && !graph.nodes.some((n) => n.id === view.selected) && !projection.nodes.some((n) => n.id === view.selected)) {
    view.selected = nearest(saved.selected.slice(1)) ?? null; fallback = true;
  }
  if (!view.selected && !view.scope && fallback) {
    view.expanded = overviewExpansion(graph); view.modelCollapsed = false; view.initialOverview = true;
  } else {
    view.initialOverview = false;
    const target = saved.camera && nav.resolve(saved.camera.target);
    const cameraTarget = fallback ? view.selected ?? view.scope : target ?? view.selected ?? view.scope;
    if (cameraTarget) view.restoreCamera = { target: cameraTarget, zoom: saved.camera?.zoom ?? 1,
      x: fallback ? 0.5 : saved.camera?.x ?? 0.5, y: fallback ? 0.5 : saved.camera?.y ?? 0.5 };
  }
  return { view, fallback };
}

export function restoreBookmark(graph: Graph, saved: ArchitectureBookmark): GraphView {
  if (!saved.namespace || saved.namespace !== graph.navigation_namespace) {
    const view = new View(overviewExpansion(graph)); view.initialOverview = true;
    view.notice = 'Architecture changed; returned to the model overview.'; return view;
  }
  const { view, fallback } = restoreView(graph, saved.current);
  view.history = saved.history.slice(-16).map((item) => restoreView(graph, item)).filter((r) => !r.fallback).map((r) => snapshotView(r.view));
  if (saved.global) { const result = restoreView(graph, saved.global); if (!result.fallback) view.globalView = snapshotView(result.view); }
  const family = (key: string | null) => key ? only(graph.templates?.filter((t) => t.navigation_key === key) ?? [])?.id : undefined;
  view.browser = { ...saved.browser, families: saved.browser.families.flatMap((key) => family(key) ?? []), selectedFamily: family(saved.browser.selectedFamily) ?? null };
  if (fallback) view.notice = 'Architecture changed; showing the nearest available component or model overview.';
  if (saved.current.boundary && !view.boundary || (saved.current.edge || saved.current.paths) && !view.edge) view.notice ??= 'The previous connection is no longer available.';
  return view;
}

export function restoreInspection(graph: Graph, saved: ArchitectureBookmark, inventory: components['schemas']['TensorInventory'],
  sessionId: string, modelId: string, trigger: HTMLElement): ArchitectureSelection | undefined {
  if (!saved.inspection || saved.inspection.unavailable || !saved.namespace || saved.namespace !== graph.navigation_namespace) return;
  const nav = new Navigation(graph), spec = saved.inspection;
  const id = nav.resolveSource(spec.node), node = id ? nav.nodes.get(id) : undefined;
  const boundary = nav.resolveBoundary(spec.boundary);
  if (spec.boundary && !boundary || spec.node && !node) return;
  const instance = nav.resolveSource(spec.instance);
  if (spec.instance && !instance) return;
  if (spec.structure) {
    const template = only(graph.templates?.filter((t) => t.navigation_key === spec.structure!.family) ?? []);
    const member = template?.instances[0]?.nodes.find((n) => n.role === spec.structure!.role);
    const record = member && nav.nodes.get(member.node_id);
    return record && template ? { sessionId, modelId, graphId: graph.graph_id, trigger,
      node: commonNode(record, spec.structure.role), structureOnly: { label: template.label, role: spec.structure.role } } : undefined;
  }
  const parameter = spec.parameter && node ? only(ownParameters(node, new Map(graph.parameters.map((p) => [p.id, p]))).filter((p) => p.name === spec.parameter)) : undefined;
  if (spec.parameter && !parameter) return;
  if (spec.tensor) {
    if (!parameter || parameter.inspection.status !== 'available') return;
    const tensorId = parameter.inspection.tensor_id;
    const tensor = inventory.tensors.find((t) => t.id === tensorId);
    if (!tensor || tensor.name !== spec.tensor || ![1, 2].includes(tensor.rank) ||
      parameter.logical_shape?.length !== tensor.rank || parameter.logical_shape.some((d, i) => d.kind !== 'constant' || d.value !== tensor.shape[i])) return;
  }
  return { sessionId, modelId, graphId: graph.graph_id, trigger, ...(node ? { node } : {}), ...(boundary ? { boundary } : {}),
    ...(parameter ? { parameterId: parameter.id } : {}), ...(instance ? { templateInstanceId: instance } : {}) };
}
