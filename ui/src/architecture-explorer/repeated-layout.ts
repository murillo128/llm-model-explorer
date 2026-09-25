import type { ELK as ElkEngine, ElkExtendedEdge, ElkNode } from 'elkjs/lib/elk-api';
import type { Graph } from './graph';
import type { Projection } from './projection';

export interface LayoutStub { edgeId: string; side: 'source' | 'target' }

/** Lay out large, explicitly repeated interiors independently. A signature is
 * built from actual projected topology and geometry, never labels or ID names.
 * Equivalent interiors share ELK work, but every output record is rebound to
 * its own concrete node, port and edge identities. */
export async function layoutRepeatedInteriors(graph: Graph, projection: Projection,
  elkNodes: Map<string, ElkNode>, root: ElkNode, engine: ElkEngine, signal?: AbortSignal):
  Promise<{ restore: (laidOut: ElkNode) => void; stubs: Map<string, LayoutStub> }> {
  const visible = new Map(projection.nodes.map((node) => [node.id, node]));
  const proposed = new Set(graph.repetitions.filter((repeat) => repeat.instances.length >= 64 &&
    visible.get(repeat.parent_id)?.expanded && repeat.instances.every((item) =>
      visible.get(item.node_id)?.parentId === repeat.parent_id)).map((repeat) => repeat.parent_id));
  if (!proposed.size) return { restore: () => {}, stubs: new Map<string, LayoutStub>() };

  // Do not split overlapping scopes. The innermost explicit repeated interiors
  // carry the large fan-out, while their outer stack remains one global flow.
  for (const id of [...proposed]) {
    for (let parent = visible.get(id)?.parentId; parent; parent = visible.get(parent)?.parentId) {
      if (proposed.has(parent)) proposed.delete(parent);
    }
  }
  const owner = (id: string) => {
    for (let current: string | undefined = id; current; current = visible.get(current)?.parentId) {
      if (proposed.has(current)) return current;
    }
    return undefined;
  };
  const edgesByScope = new Map([...proposed].map((id) => [id, [] as ElkExtendedEdge[]]));
  const stubs = new Map<string, LayoutStub>();
  root.edges = (root.edges ?? []).filter((edge, index) => {
    const projected = projection.edges[index]!;
    const source = owner(projected.source.node_id), target = owner(projected.target.node_id);
    if (source && source === target) { edgesByScope.get(source)!.push(edge); return false; }
    for (const [scope, side] of [[source, 'source'], [target, 'target']] as const) {
      if (!scope || projected[side].node_id === scope) continue;
      const candidate = elkNodes.get(scope)!;
      const portId = `proxy-${edge.id}-${side}`;
      const outgoing = side === 'source';
      candidate.ports!.push({ id: portId, x: outgoing ? candidate.width! : 0, y: 64, width: 0, height: 0,
        layoutOptions: { 'elk.port.side': outgoing ? 'EAST' : 'WEST' } });
      const stubId = `stub-${edge.id}-${side}`;
      edgesByScope.get(scope)!.push({ id: stubId,
        sources: [outgoing ? edge.sources[0]! : portId], targets: [outgoing ? portId : edge.targets[0]!] });
      stubs.set(stubId, { edgeId: projected.id, side });
      if (outgoing) edge.sources = [portId]; else edge.targets = [portId];
    }
    return true;
  });

  const snapshots = new Map<string, ElkNode>();
  const routed = new Map<string, ElkExtendedEdge[]>();
  const cache = new Map<string, { input: ElkNode; output: ElkNode }>();
  for (const id of proposed) {
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    const input = elkNodes.get(id)!;
    const wrapper: ElkNode = { id: `repeated-root-${input.id}`, children: [input], edges: edgesByScope.get(id)!,
      layoutOptions: { 'elk.algorithm': 'layered', 'elk.direction': 'RIGHT', 'elk.edgeRouting': 'ORTHOGONAL',
        'elk.hierarchyHandling': 'INCLUDE_CHILDREN', 'elk.randomSeed': '1' } };
    const signature = structuralSignature(wrapper);
    const previous = cache.get(signature);
    const inputSnapshot = previous ? undefined : structuredClone(wrapper);
    const output = previous ? rebind(previous.input, previous.output, wrapper) : await engine.layout(wrapper);
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    if (!previous) cache.set(signature, { input: inputSnapshot!, output: structuredClone(output) });
    const local = output.children![0]!;
    snapshots.set(id, structuredClone(local));
    const x = local.x ?? 0, y = local.y ?? 0;
    const edges = (output.edges ?? []).map((edge) => {
      const shifted = structuredClone(edge);
      const atWrapper = !edge.container || edge.container === wrapper.id;
      const dx = atWrapper ? x : 0, dy = atWrapper ? y : 0;
      if (atWrapper) shifted.container = input.id;
      for (const section of shifted.sections ?? []) for (const point of [section.startPoint, ...(section.bendPoints ?? []), section.endPoint]) {
        point.x -= dx; point.y -= dy;
      }
      for (const point of shifted.junctionPoints ?? []) { point.x -= dx; point.y -= dy; }
      for (const label of shifted.labels ?? []) { label.x = (label.x ?? 0) - dx; label.y = (label.y ?? 0) - dy; }
      return shifted;
    });
    routed.set(id, edges);
    // ELK sees a complete, fixed-size compound boundary as one leaf. Its
    // independent interior is restored after the global route is computed.
    input.width = local.width!; input.height = local.height!;
    input.ports = local.ports ?? [];
    input.children = []; input.edges = [];
    input.layoutOptions = { ...input.layoutOptions, 'elk.portConstraints': 'FIXED_POS' };
  }
  return { stubs, restore: (laidOut: ElkNode) => {
    const byElkId = new Map([...snapshots].map(([id, saved]) => [elkNodes.get(id)!.id, { id, saved }]));
    const restore = (node: ElkNode) => {
      const state = byElkId.get(node.id);
      if (state) { node.children = state.saved.children ?? []; node.edges = [...(state.saved.edges ?? []), ...(routed.get(state.id) ?? [])]; }
      for (const child of node.children ?? []) restore(child);
    };
    restore(laidOut);
  } };
}

function walk(root: ElkNode) {
  const nodes: ElkNode[] = [];
  const visit = (node: ElkNode) => { nodes.push(node); for (const child of node.children ?? []) visit(child); };
  visit(root); return nodes;
}

function structuralSignature(root: ElkNode) {
  const nodes = walk(root);
  const ports = new Map(nodes.flatMap((node, i) => (node.ports ?? []).map((port, p) => [port.id, `${i}:${p}`] as const)));
  const role = (id: string) => {
    const indexed = ports.get(id);
    if (!indexed) throw new Error('A repeated interior edge has no local port. Collapse groups and retry.');
    return indexed;
  };
  return JSON.stringify({
    nodes: nodes.map((node) => ({ width: node.width, height: node.height, options: node.layoutOptions,
      ports: (node.ports ?? []).map((port) => ({ x: port.x, y: port.y, width: port.width, height: port.height,
        options: port.layoutOptions, labels: port.labels?.map((label) => [label.width, label.height, label.text]) })),
      children: node.children?.length ?? 0 })),
    edges: (root.edges ?? []).map((edge) => ({ sources: edge.sources.map(role), targets: edge.targets.map(role),
      // ELK receives the measured label box. Its text does not change routing;
      // instance-specific shape strings must not defeat repeated geometry reuse.
      labels: edge.labels?.map((label) => [label.width, label.height, label.layoutOptions]) })),
  });
}

function rebind(from: ElkNode, output: ElkNode, to: ElkNode): ElkNode {
  const oldNodes = walk(from), newNodes = walk(to);
  const ids = new Map<string, string>();
  oldNodes.forEach((node, index) => {
    const target = newNodes[index]!;
    ids.set(node.id, target.id);
    (node.ports ?? []).forEach((port, p) => ids.set(port.id, target.ports![p]!.id));
  });
  (from.edges ?? []).forEach((edge, index) => {
    const target = to.edges![index]!;
    ids.set(edge.id, target.id);
    (edge.labels ?? []).forEach((label, p) => { if (label.id && target.labels?.[p]?.id) ids.set(label.id, target.labels[p]!.id!); });
  });
  const clone = (value: unknown): unknown => {
    if (typeof value === 'string') return ids.get(value) ?? value;
    if (Array.isArray(value)) return value.map(clone);
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, clone(item)]));
    return value;
  };
  const rebound = clone(output) as ElkNode;
  for (const [index, edge] of (rebound.edges ?? []).entries()) {
    for (const [labelIndex, label] of (edge.labels ?? []).entries()) {
      const text = to.edges?.[index]?.labels?.[labelIndex]?.text;
      if (text !== undefined) label.text = text;
    }
  }
  return rebound;
}
