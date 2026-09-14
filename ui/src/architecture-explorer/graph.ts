import type { components } from '../api/generated/types';

export type Graph = components['schemas']['ArchitectureGraph'];
export type GraphNode = components['schemas']['ArchitectureNode'];
export type Shape = components['schemas']['ArchitectureShape'];
export interface Box { id: string; parentId?: string; x: number; y: number; width: number; height: number; absoluteX: number; absoluteY: number }
export interface Layout { boxes: Box[]; edgeIds: string[]; width: number; height: number; milliseconds: number }
export class GraphView {
  selected: string | null = null;
  dimensions = false;
  constructor(public expanded: string[] = []) {}
  update(patch: Partial<Pick<GraphView, 'expanded' | 'selected' | 'dimensions' | 'viewport'>>) { Object.assign(this, patch); }
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
    view.expanded = view.expanded.filter((id) => ids.has(id));
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
/** Ordered containment packing: linear work, no semantic edge inference or recursion.
 * Each group retains its explicit ports. Hidden internal edges reappear unchanged on expansion.
 * Connections crossing a collapsed boundary already terminate at that group's contract ports.
 */
export function layoutGraph(graph: Graph, expandedIds: string[]): Layout {
  const started = performance.now();
  const expanded = new Set(expandedIds);
  const records = new Map(graph.nodes.map((n) => [n.id, n]));
  const roots = graph.nodes.filter((n) => !n.parent_id);
  const ordered: GraphNode[] = [];
  const stack = [...roots].reverse();
  while (stack.length) {
    const node = stack.pop()!;
    ordered.push(node);
    if (node.kind === 'group' && expanded.has(node.id)) {
      for (let i = node.children.length - 1; i >= 0; i--) stack.push(records.get(node.children[i]!)!);
    }
  }
  const boxes = new Map<string, Box>();
  for (let i = ordered.length - 1; i >= 0; i--) {
    const node = ordered[i]!;
    const children = node.kind === 'group' && expanded.has(node.id) ? node.children.map((id) => boxes.get(id)!) : [];
    // Ports remain distinct even on collapsed groups with many crossings.
    let height = Math.max(100, node.ports.length * 20 + 60), width = 300;
    if (children.length) {
      height = Math.max(110, node.ports.length * 20 + 70);
      for (const child of children) { child.x = 44; child.y = height; height += child.height + 60; width = Math.max(width, child.width + 88); }
    }
    boxes.set(node.id, { id: node.id, ...(node.parent_id ? { parentId: node.parent_id } : {}), x: 0, y: 0, width, height, absoluteX: 0, absoluteY: 0 });
  }
  let width = 0, height = 0;
  for (const root of roots) { const box = boxes.get(root.id)!; box.x = width; width += box.width + 100; height = Math.max(height, box.height); }
  for (const node of ordered) {
    const box = boxes.get(node.id)!;
    const parent = box.parentId ? boxes.get(box.parentId) : undefined;
    box.absoluteX = (parent?.absoluteX ?? 0) + box.x;
    box.absoluteY = (parent?.absoluteY ?? 0) + box.y;
  }
  return { boxes: ordered.map((n) => boxes.get(n.id)!), edgeIds: graph.edges.filter((e) => boxes.has(e.source.node_id) && boxes.has(e.target.node_id)).map((e) => e.id),
    width: Math.max(0, width - 100), height, milliseconds: performance.now() - started };
}
