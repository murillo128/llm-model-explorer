import type { components } from '../api/generated/types';
import type { Projection, ProjectionOptions } from './projection';
import { projectGraph } from './projection';
import { componentScope } from './scope';
import { projectionOptions } from './scope-navigation';
import { sharedContext, type SharedStructure } from './shared-structure';
import { interfaceIndex, interfaceSelection } from './interfaces';
import type { BoundarySelection } from './interfaces';

export type Graph = components['schemas']['ArchitectureGraph'];
export type GraphNode = components['schemas']['ArchitectureNode'];
export type Shape = components['schemas']['ArchitectureShape'];
export interface Box { id: string; parentId?: string; x: number; y: number; width: number; height: number; absoluteX: number; absoluteY: number }
export interface Point { x: number; y: number }
export interface PortPosition extends Point { nodeId: string; portId: string; absoluteX: number; absoluteY: number; side: 'left' | 'right' }
export interface Route { id: string; sections: Point[][]; junctions: Point[]; labels?: { x: number; y: number; width: number; height: number; lines: string[] }[] }
export interface Layout { boxes: Box[]; ports: PortPosition[]; routes: Route[]; projection: Projection; edgeIds: string[]; width: number; height: number; milliseconds: number }
export type GraphSnapshot = Pick<GraphView, 'selected' | 'selectionMode' | 'dimensions' | 'edge' | 'focus' | 'activeStack' | 'repetitions' |
  'exhaustive' | 'showUnused' | 'showContext' | 'deriveMlp' | 'stateScope' | 'expanded' | 'viewport' | 'scope' | 'shared' | 'boundary' | 'modelCollapsed'>;
export class GraphView {
  private revision = 0;
  private readonly listeners = new Set<() => void>();
  private projection: ProjectionOptions | undefined;
  /** Transient layout intent, never retained in a scope snapshot. */
  expansionAnchor: string | undefined;
  takeExpansionAnchor() { const id = this.expansionAnchor; this.expansionAnchor = undefined; return id; }
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  getRevision = () => this.revision;
  getProjectionOptions() {
    const next = projectionOptions(this), previous = this.projection;
    // Selection, inspection and camera notifications must not invalidate layout.
    if (!previous || Object.keys(previous).length !== Object.keys(next).length ||
      (Object.keys(next) as (keyof ProjectionOptions)[]).some((key) => next[key] !== previous[key])) this.projection = next;
    return this.projection!;
  }
  // Navigation presentation only; lifetime follows this backend/model/graph view.
  browser = { query: '', treeScroll: 0, searchScroll: 0, families: [] as string[], selectedFamily: null as string | null };
  shared: SharedStructure | undefined;
  notice: string | undefined;
  scope: string | undefined;
  history: GraphSnapshot[] = [];
  globalView: GraphSnapshot | undefined;
  selected: string | null = null;
  boundary: BoundarySelection | undefined;
  modelCollapsed = false;
  selectionMode: 'source' | 'structure' = 'source';
  dimensions = false;
  edge: string | null = null;
  focus: string | null = null;
  activeStack: string | null = null;
  repetitions: NonNullable<ProjectionOptions['repetitions']> = {};
  exhaustive = false;
  showUnused = false;
  showContext = true;
  deriveMlp = true;
  stateScope: string | undefined;
  constructor(public expanded: string[] = []) {}
  update(patch: Partial<Omit<GraphView, 'update'>>) {
    if (!Object.entries(patch).some(([key, value]) => Reflect.get(this, key) !== value)) return;
    Object.assign(this, patch); this.revision++;
    this.listeners.forEach((listener) => listener());
  }
  viewport: { x: number; y: number; zoom: number } | undefined;
}
/** Owned by the mounted backend shell, retained across explorer/session switches. No graph copies. */
export class GraphViews {
  private readonly views = new Map<string, GraphView>();
  get(model: string, graph: Graph): GraphView {
    const key = JSON.stringify([model, graph.graph_id]);
    let view = this.views.get(key);
    if (!view) {
      // Old graph identities for this model cannot contain valid navigation
      // references to the replacement graph. Bound retained models as well.
      let clearedShared = false;
      for (const existing of this.views.keys()) if ((JSON.parse(existing) as string[])[0] === model) {
        clearedShared ||= Boolean(this.views.get(existing)?.shared); this.views.delete(existing);
      }
      view = new GraphView(graph.nodes.filter((n) => n.kind === 'group' && !n.parent_id).map((n) => n.id));
      if (clearedShared) view.notice = "Graph changed; the shared structure selection was cleared.";
      this.views.set(key, view);
    }
    this.views.delete(key); this.views.set(key, view);
    while (this.views.size > 8) this.views.delete(this.views.keys().next().value!);
    const ids = new Set(graph.nodes.map((n) => n.id));
    const interfaces = interfaceIndex(graph);
    const declaration = view.selected && interfaces.declarations.get(view.selected)?.[0];
    if (declaration) { view.boundary = interfaceSelection(declaration); view.selected = declaration.owner.kind === 'source' ? declaration.owner.id : null; view.edge = null; }
    if (view.selected && interfaces.tools.has(view.selected)) {
      view.selected = null; view.edge = null; view.notice = 'The selected tool capability is no longer a model component.';
    }
    if (view.boundary && !view.boundary.endpoints.every((p) => graph.nodes.some((n) => n.id === p.node_id && n.ports.some((port) => port.id === p.port_id)))) view.boundary = undefined;
    if (view.boundary?.templatePort) {
      const target = view.boundary.templatePort, template = graph.templates?.find((t) => t.id === target.templateId);
      if (view.shared?.templateId !== target.templateId || !template?.instances.every((instance) =>
        instance.nodes.some((n) => n.role === target.nodeRole && ids.has(n.node_id)) && instance.ports.some((p) => p.role === target.portRole &&
          graph.nodes.some((n) => n.id === p.node_id && n.ports.some((port) => port.id === p.port_id))))) {
        view.boundary = undefined; view.notice = 'Shared interface correspondence changed; the port selection was cleared.';
      }
    }
    view.browser.families = view.browser.families.filter((id) => graph.templates?.some((t) => t.id === id));
    if (!graph.templates?.some((t) => t.id === view.browser.selectedFamily)) view.browser.selectedFamily = null;
    const expanded = view.expanded.filter((id) => interfaces.eligible(id) && (ids.has(id) || id.startsWith('mlp:') && ids.has(id.slice(4))));
    if (expanded.length !== view.expanded.length) view.expanded = expanded;
    const repetitions = Object.entries(view.repetitions).filter(([id]) => graph.repetitions.some((r) => r.id === id));
    if (repetitions.length !== Object.keys(view.repetitions).length) view.repetitions = Object.fromEntries(repetitions);
    if (view.activeStack && !graph.repetitions.some((r) => r.id === view.activeStack)) view.activeStack = null;
    if (view.focus && (!interfaces.eligible(view.focus) || !ids.has(view.focus) && !(view.focus.startsWith('mlp:') && ids.has(view.focus.slice(4))))) view.focus = null;
    if (view.stateScope && !ids.has(view.stateScope)) view.stateScope = undefined;
    if (view.shared) {
      try { sharedContext(graph, view.shared); }
      catch {
        const reset = new GraphView(graph.nodes.filter((n) => n.kind === 'group' && !n.parent_id).map((n) => n.id));
        reset.notice = 'Shared structure correspondence changed; returned to the model view.';
        this.views.set(key, reset); return reset;
      }
    }
    if (view.scope) {
      try { componentScope(graph, view.scope); }
      catch {
        const reset = new GraphView(graph.nodes.filter((n) => n.kind === 'group' && !n.parent_id).map((n) => n.id));
        this.views.set(key, reset); return reset;
      }
    }
    // A card selection may name a local repetition/context presentation. Retain
    // it across remounts only while that exact presentation still exists.
    if (view.selected && !ids.has(view.selected) && !(view.selected.startsWith('mlp:') && ids.has(view.selected.slice(4))) &&
      !projectGraph(graph, projectionOptions(view)).nodes.some((node) => node.id === view.selected)) view.selected = null;
    return view;
  }
}
export function formatShape(shape: Shape): string {
  if (shape === null) return '? (unknown rank)';
  if (!shape.length) return 'scalar';
  return '[' + shape.map((d) => d.kind === 'constant' ? String(d.value) : d.kind === 'symbol' ? d.name :
    d.kind === 'expression' ? `(${d.text})` : `? (${d.reason})`).join(' × ') + ']';
}
