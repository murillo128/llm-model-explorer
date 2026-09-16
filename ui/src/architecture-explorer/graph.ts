import type { components } from '../api/generated/types';
import type { Projection, ProjectionOptions } from './projection';
import { projectGraph } from './projection';
import { componentScope } from './scope';
import { projectionOptions } from './scope-navigation';
import { sharedContext, type SharedStructure } from './shared-structure';

export type Graph = components['schemas']['ArchitectureGraph'];
export type GraphNode = components['schemas']['ArchitectureNode'];
export type Shape = components['schemas']['ArchitectureShape'];
export interface Box { id: string; parentId?: string; x: number; y: number; width: number; height: number; absoluteX: number; absoluteY: number }
export interface Point { x: number; y: number }
export interface PortPosition extends Point { nodeId: string; portId: string; absoluteX: number; absoluteY: number; side: 'left' | 'right' }
export interface Route { id: string; sections: Point[][]; junctions: Point[]; labels?: { x: number; y: number; width: number; height: number; lines: string[] }[] }
export interface Layout { boxes: Box[]; ports: PortPosition[]; routes: Route[]; projection: Projection; edgeIds: string[]; width: number; height: number; milliseconds: number }
export type GraphSnapshot = Pick<GraphView, 'selected' | 'dimensions' | 'edge' | 'focus' | 'activeStack' | 'repetitions' |
  'exhaustive' | 'showUnused' | 'showContext' | 'deriveMlp' | 'stateScope' | 'expanded' | 'viewport' | 'scope' | 'shared'>;
export class GraphView {
  shared: SharedStructure | undefined;
  notice: string | undefined;
  scope: string | undefined;
  history: GraphSnapshot[] = [];
  globalView: GraphSnapshot | undefined;
  selected: string | null = null;
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
  update(patch: Partial<Omit<GraphView, 'update'>>) { Object.assign(this, patch); }
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
    view.expanded = view.expanded.filter((id) => ids.has(id) || id.startsWith('mlp:') && ids.has(id.slice(4)));
    view.repetitions = Object.fromEntries(Object.entries(view.repetitions).filter(([id]) => graph.repetitions.some((r) => r.id === id)));
    if (view.activeStack && !graph.repetitions.some((r) => r.id === view.activeStack)) view.activeStack = null;
    if (view.focus && !ids.has(view.focus) && !(view.focus.startsWith('mlp:') && ids.has(view.focus.slice(4)))) view.focus = null;
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
