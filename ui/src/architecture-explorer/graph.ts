import type { components } from '../api/generated/types';
import type { Projection, ProjectionOptions } from './projection';

export type Graph = components['schemas']['ArchitectureGraph'];
export type GraphNode = components['schemas']['ArchitectureNode'];
export type Shape = components['schemas']['ArchitectureShape'];
export interface Box { id: string; parentId?: string; x: number; y: number; width: number; height: number; absoluteX: number; absoluteY: number }
export interface Point { x: number; y: number }
export interface PortPosition extends Point { nodeId: string; portId: string; absoluteX: number; absoluteY: number; side: 'left' | 'right' }
export interface Route { id: string; sections: Point[][]; junctions: Point[]; labels?: { x: number; y: number; width: number; height: number; lines: string[] }[] }
export interface Layout { boxes: Box[]; ports: PortPosition[]; routes: Route[]; projection: Projection; edgeIds: string[]; width: number; height: number; milliseconds: number }
export class GraphView {
  selected: string | null = null;
  dimensions = false;
  edge: string | null = null;
  focus: string | null = null;
  repetitions: NonNullable<ProjectionOptions['repetitions']> = {};
  exhaustive = false;
  showUnused = false;
  showContext = true;
  deriveMlp = true;
  stateScope: string | undefined;
  constructor(public expanded: string[] = []) {}
  update(patch: Partial<Omit<GraphView, 'update'>>) { Object.assign(this, patch); }
  viewport?: { x: number; y: number; zoom: number };
}
/** Owned by the mounted backend shell, retained across explorer/session switches. No graph copies. */
export class GraphViews {
  private readonly views = new Map<string, GraphView>();
  get(model: string, graph: Graph): GraphView {
    const key = JSON.stringify([model, graph.graph_id]);
    let view = this.views.get(key);
    if (!view) {
      view = new GraphView(graph.nodes.filter((n) => n.kind === 'group' && !n.parent_id).map((n) => n.id));
      this.views.set(key, view);
    }
    const ids = new Set(graph.nodes.map((n) => n.id));
    view.expanded = view.expanded.filter((id) => ids.has(id) || id.startsWith('mlp:') && ids.has(id.slice(4)));
    view.repetitions = Object.fromEntries(Object.entries(view.repetitions).filter(([id]) => graph.repetitions.some((r) => r.id === id)));
    if (view.stateScope && !ids.has(view.stateScope)) view.stateScope = undefined;
    if (view.selected && !ids.has(view.selected)) view.selected = null;
    return view;
  }
}
export function formatShape(shape: Shape): string {
  if (shape === null) return '? (unknown rank)';
  if (!shape.length) return 'scalar';
  return '[' + shape.map((d) => d.kind === 'constant' ? String(d.value) : d.kind === 'symbol' ? d.name :
    d.kind === 'expression' ? `(${d.text})` : `? (${d.reason})`).join(' × ') + ']';
}
