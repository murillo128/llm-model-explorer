import type { components } from '../api/generated/types';
import type { Graph, GraphNode } from './graph';
import { deriveMlpGroups } from './derived-groups';
import { componentScope } from './scope';

export type Endpoint = components['schemas']['ArchitectureEdge']['source'];
type SourceEdge = Graph['edges'][number];
type SourcePort = GraphNode['ports'][number];
export interface ProjectedPort {
  id: string; label: string; direction: 'input' | 'output'; shape: SourcePort['shape'];
  /** Exact source endpoints represented by this presentation alias. */
  endpoints: Endpoint[];
}
export interface ProjectedNode {
  id: string; parentId?: string; kind: GraphNode['kind']; label: string;
  sourceIds: string[]; record?: GraphNode; ports: ProjectedPort[]; expanded: boolean;
  presentation?: 'repetition' | 'range' | 'mlp' | 'external'; repetitionId?: string;
  instances?: Graph['repetitions'][number]['instances']; summary?: string;
}
export interface ProjectedEdge {
  id: string; source: Endpoint; target: Endpoint; kind: SourceEdge['kind'];
  /** Each path is ordered from the original source to original destination.
   * Shared forwarding prefixes may occur in genuine fan-out paths. */
  paths: SourceEdge[][]; originalEdgeIds: string[];
}
export interface ProjectionOptions {
  expanded: string[];
  scope?: string | undefined;
  repetitions?: Record<string, { start: number; count: number }>;
  exhaustive?: boolean;
  deriveMlp?: boolean;
  showUnused?: boolean;
  showContext?: boolean;
  stateScope?: string | undefined;
  /** Layout-only label visibility; it never changes source projection semantics. */
  dimensions?: boolean;
}
export interface Projection {
  nodes: ProjectedNode[]; edges: ProjectedEdge[];
  /** Every omitted source edge has an explicit reason, for inspection/tests. */
  hiddenEdgeIds: string[]; filteredEdgeIds: string[];
  unusedInputs: Endpoint[];
  scope?: { id: string; nodeIds: string[]; excludedNodeIds: string[]; excludedEdgeIds: string[] };
}
export const endpointKey = (e: Endpoint): string => JSON.stringify([e.node_id, e.port_id]);
const endpoint = (node_id: string, port_id: string): Endpoint => ({ node_id, port_id });

export function variantSummary(instances: Graph['repetitions'][number]['instances']): string {
  const counts = new Map<string, number>();
  for (const i of instances) counts.set(i.variant, (counts.get(i.variant) ?? 0) + 1);
  return [...counts].map(([variant, count]) => `${count} ${variant.replaceAll('_', ' ')}`).join(' · ');
}

/** Derive view records only. The validated graph and all original record objects
 * remain untouched. Only group interfaces are transparent; computations stop
 * traversal, even when two operation ports have the same shape or label. */
export function projectGraph(graph: Graph, options: ProjectionOptions): Projection {
  const records = new Map(graph.nodes.map((n) => [n.id, n]));
  const scope = options.scope ? componentScope(graph, options.scope) : undefined;
  // Stop at the first external endpoint. In particular, an external bypass or
  // an exit/re-entry must never be composed into an invented internal path.
  const scopedEdges = scope ? graph.edges.filter((edge) =>
    scope.members.has(edge.source.node_id) || scope.members.has(edge.target.node_id)) : graph.edges;
  const expanded = new Set(options.expanded);
  const exhaustive = options.exhaustive === true;
  const outgoing = new Map<string, SourceEdge[]>(), incoming = new Map<string, SourceEdge[]>();
  for (const e of graph.edges) {
    const s = endpointKey(e.source), t = endpointKey(e.target);
    outgoing.set(s, [...(outgoing.get(s) ?? []), e]); incoming.set(t, [...(incoming.get(t) ?? []), e]);
  }
  const instanceRepetition = new Map(graph.repetitions.flatMap((r) => r.instances.map((i) => [i.node_id, r] as const)));
  const derived = options.deriveMlp === false || exhaustive ? [] : deriveMlpGroups(graph)
    .filter((group) => !scope || group.sourceIds.every((id) => scope.members.has(id)));
  const mlpMembers = new Map(derived.flatMap((g) => g.sourceIds.map((id) => [id, g] as const)));
  const nodes: ProjectedNode[] = [];
  const owners = new Map<string, ProjectedNode>();
  const visible = new Map<string, ProjectedNode>();
  const added = new Set<string>();
  function assign(id: string, owner: ProjectedNode) {
    const queue = [id];
    while (queue.length) {
      const n = records.get(queue.pop()!)!; owners.set(n.id, owner);
      if (n.kind === 'group') queue.push(...n.children);
    }
  }
  function add(n: ProjectedNode) { nodes.push(n); visible.set(n.id, n); }
  function visit(id: string, parentId?: string) {
    const record = records.get(id)!;
    const repetition = instanceRepetition.get(id);
    if (repetition && !exhaustive && id !== scope?.id) {
      const window = options.repetitions?.[repetition.id];
      const selected = repetition.instances.map((i) => expanded.has(i.node_id));
      const start = Math.max(0, Math.min(repetition.instances.length - 1, Math.trunc(window?.start ?? 0)));
      const count = Math.max(1, Math.trunc(window?.count ?? 1));
      const open = repetition.instances.map((_, i) => selected[i] || Boolean(window && i >= start && i < start + count));
      const at = repetition.instances.findIndex((i) => i.node_id === id);
      if (!open[at]) {
        let first = at, last = at;
        while (first > 0 && !open[first - 1]) first--;
        while (last + 1 < open.length && !open[last + 1]) last++;
        // A range only combines contiguous siblings. Interleaved unrelated
        // children must stay explicit and cannot be swallowed by repetition.
        const parent = records.get(repetition.parent_id)!;
        if (parent.kind === 'group') {
          const childIndex = parent.children.indexOf(id);
          while (first < at && parent.children[childIndex - (at - first)] !== repetition.instances[first]!.node_id) first++;
          while (last > at && parent.children[childIndex + (last - at)] !== repetition.instances[last]!.node_id) last--;
        }
        const rangeId = `repeat:${repetition.id}:${first}:${last}`;
        if (added.has(rangeId)) return;
        added.add(rangeId);
        const instances = repetition.instances.slice(first, last + 1);
        const whole = instances.length === repetition.instances.length;
        const n: ProjectedNode = { id: rangeId, ...(parentId ? { parentId } : {}), kind: 'group',
          label: whole ? `${repetition.label} ×${instances.length}` : `Instances ${instances[0]!.index}–${instances.at(-1)!.index}`,
          sourceIds: instances.map((i) => i.node_id), ports: [], expanded: false,
          presentation: whole ? 'repetition' : 'range', repetitionId: repetition.id, instances, summary: variantSummary(instances) };
        add(n); for (const i of instances) assign(i.node_id, n); return;
      }
    }
    const mlp = mlpMembers.get(id);
    if (mlp && !added.has(mlp.id)) {
      added.add(mlp.id);
      const n: ProjectedNode = { id: mlp.id, ...(parentId ? { parentId } : {}), kind: 'group', label: mlp.label,
        sourceIds: mlp.sourceIds, ports: [], expanded: expanded.has(mlp.id), presentation: 'mlp', summary: 'Derived · 5 operations' };
      add(n);
      if (n.expanded) for (const member of mlp.sourceIds) visit(member, n.id);
      else for (const member of mlp.sourceIds) assign(member, n);
      return;
    }
    if (mlp && parentId !== mlp.id) return;
    const n: ProjectedNode = { id, ...(parentId ? { parentId } : {}), kind: record.kind, label: record.label,
      sourceIds: [id], record, ports: [], expanded: record.kind === 'group' && record.children.length > 0 && (exhaustive || expanded.has(id)) };
    add(n); owners.set(id, n);
    if (record.kind === 'group') {
      if (n.expanded) for (const child of record.children) visit(child, id);
      else for (const child of record.children) assign(child, n);
    }
  }
  if (scope) for (const id of scope.roots) visit(id);
  else for (const n of graph.nodes) if (!n.parent_id) visit(n.id);

  const sourcePort = (e: Endpoint) => records.get(e.node_id)!.ports.find((p) => p.id === e.port_id)!;
  const transparent = (e: Endpoint) => {
    const owner = owners.get(e.node_id)!;
    return owner?.id === e.node_id && owner.expanded && owner.kind === 'group';
  };
  // Consumption is actual reachability through group boundary ports only. It
  // does not depend on a variant name, module label or tensor dimensions.
  const consumedMemo = new Map<string, boolean>();
  function consumed(e: Endpoint, seen = new Set<string>()): boolean {
    const key = endpointKey(e), cached = consumedMemo.get(key);
    if (cached !== undefined) return cached;
    if (seen.has(key)) return false;
    const record = records.get(e.node_id)!;
    if (record.kind !== 'group') return true;
    const nextSeen = new Set(seen).add(key);
    const result = (outgoing.get(key) ?? []).some((edge) => consumed(edge.target, nextSeen));
    consumedMemo.set(key, result); return result;
  }
  const unusedInputs = graph.nodes.filter((n) => n.kind === 'group').flatMap((n) =>
    n.ports.filter((p) => p.direction === 'input' && !consumed(endpoint(n.id, p.id))).map((p) => endpoint(n.id, p.id)));
  const unused = new Set(unusedInputs.map(endpointKey));
  const hidden = new Set<string>(), filtered = new Set<string>();
  const edges: ProjectedEdge[] = [];
  const edgeGroups = new Map<string, ProjectedEdge>();
  function external(original: Endpoint, direction: 'input' | 'output'): ProjectedNode {
    const id = `external:${direction}:${endpointKey(original)}`;
    let alias = visible.get(id);
    if (!alias) {
      const record = records.get(original.node_id)!;
      alias = { id, kind: direction === 'output' ? 'input' : 'output',
        label: direction === 'output' ? 'Component input' : 'Component output',
        sourceIds: [record.id], ports: [], expanded: false, presentation: 'external',
        summary: `${direction === 'output' ? 'From' : 'To'} ${record.label}` };
      add(alias);
    }
    return alias;
  }
  function addPort(owner: ProjectedNode, original: Endpoint, direction: 'input' | 'output', signal: string): Endpoint {
    const port = sourcePort(original);
    // A collapsed aggregate can share an input ONLY for the same exact source
    // signal. Its endpoint list retains all separately declared consumers.
    const id = owner.id === original.node_id ? original.port_id : `${direction}:${signal}`;
    let projected = owner.ports.find((p) => p.id === id);
    if (!projected) {
      projected = { id, label: port.label, direction: owner.expanded ? port.direction : direction, shape: port.shape, endpoints: [] };
      owner.ports.push(projected);
    }
    if (!projected.endpoints.some((p) => endpointKey(p) === endpointKey(original))) projected.endpoints.push(original);
    return endpoint(owner.id, id);
  }
  function emit(path: SourceEdge[]) {
    const source = path[0]!.source, target = path.at(-1)!.target;
    const sourceMember = owners.get(source.node_id), targetMember = owners.get(target.node_id);
    if (sourceMember && sourceMember === targetMember && (!sourceMember.expanded || source.node_id !== target.node_id)) {
      for (const e of path) hidden.add(e.id); return;
    }
    if (!exhaustive && options.showUnused !== true && unused.has(endpointKey(target))) {
      for (const e of path) filtered.add(e.id); return;
    }
    const sourceOwner = sourceMember ?? external(source, 'output');
    const targetOwner = targetMember ?? external(target, 'input');
    const signal = endpointKey(source);
    const s = addPort(sourceOwner, source, 'output', signal);
    const t = addPort(targetOwner, target, 'input', signal);
    const kind = path.some((e) => e.kind === 'state') ? 'state' : path[0]!.kind;
    const key = JSON.stringify([s, t, kind]);
    let edge = edgeGroups.get(key);
    if (!edge) {
      edge = { id: `connection:${path.map((e) => e.id).join(':')}`, source: s, target: t, kind, paths: [], originalEdgeIds: [] };
      edgeGroups.set(key, edge); edges.push(edge);
    }
    edge.paths.push(path);
    for (const e of path) if (!edge.originalEdgeIds.includes(e.id)) edge.originalEdgeIds.push(e.id);
  }
  let visits = 0;
  const maxVisits = Math.max(100_000, graph.edges.length * 128);
  for (const edge of scopedEdges) {
    const sOwner = owners.get(edge.source.node_id)!, tOwner = owners.get(edge.target.node_id)!;
    if (sOwner && sOwner === tOwner && !sOwner.expanded) { hidden.add(edge.id); continue; }
    if (transparent(edge.source) && incoming.has(endpointKey(edge.source))) continue;
    const queue: { path: SourceEdge[]; seen: Set<string> }[] = [{ path: [edge], seen: new Set([edge.id]) }];
    while (queue.length) {
      if (++visits > maxVisits) throw new Error('Boundary projection exceeded its recoverable work limit. Collapse groups and retry.');
      const { path, seen } = queue.pop()!;
      const target = path.at(-1)!.target;
      const next = transparent(target) ? outgoing.get(endpointKey(target)) : undefined;
      if (!next?.length) { emit(path); continue; }
      for (const e of next) {
        if (seen.has(e.id)) throw new Error('Cyclic boundary forwarding cannot be projected. Collapse this group.');
        queue.push({ path: [...path, e], seen: new Set(seen).add(e.id) });
      }
    }
  }
  // Declared ports remain available for exact inspection, including unused
  // interfaces. Expanded forwarding ports live in source records, not duplicate
  // renderable connections. Keep ordinary operation ports even when unconnected.
  for (const n of nodes) if (n.record && !n.expanded) {
    for (const p of n.record.ports) {
      if (!exhaustive && !options.showUnused && unused.has(endpointKey(endpoint(n.id, p.id)))) continue;
      if (!n.ports.some((v) => v.id === p.id)) n.ports.push({ ...p, endpoints: [endpoint(n.id, p.id)] });
    }
  }
  let shownNodes = nodes, shownEdges = edges;
  if (!exhaustive && options.showContext === false) {
    const context = new Set(nodes.filter((n) => n.kind === 'context' ||
      n.presentation === 'external' && n.sourceIds.some((id) => records.get(id)!.kind === 'context')).map((n) => n.id));
    shownNodes = shownNodes.filter((n) => !context.has(n.id));
    shownEdges = shownEdges.filter((e) => {
      if (!context.has(e.source.node_id) && !context.has(e.target.node_id)) return true;
      e.originalEdgeIds.forEach((id) => filtered.add(id)); return false;
    });
  }
  if (!exhaustive && options.stateScope) {
    const belongs = (id: string) => {
      let node = records.get(id);
      while (node) { if (node.id === options.stateScope) return true; node = node.parent_id ? records.get(node.parent_id) : undefined; }
      return false;
    };
    shownEdges = shownEdges.filter((e) => {
      if (e.kind === 'state' && e.paths.some((p) => p.some((s) => belongs(s.source.node_id) || belongs(s.target.node_id)))) return true;
      e.originalEdgeIds.forEach((id) => filtered.add(id)); return false;
    });
    const keep = new Set(shownEdges.flatMap((e) => [e.source.node_id, e.target.node_id]));
    for (const id of [...keep]) { let n = visible.get(id); while (n?.parentId) { keep.add(n.parentId); n = visible.get(n.parentId); } }
    shownNodes = shownNodes.filter((n) => keep.has(n.id));
  }
  const represented = new Set(shownEdges.flatMap((e) => e.originalEdgeIds));
  if (scope) {
    const endpoints = new Set(shownEdges.flatMap((edge) => [edge.source.node_id, edge.target.node_id]));
    shownNodes = shownNodes.filter((node) => node.presentation !== 'external' || endpoints.has(node.id));
  }
  // This partition makes omissions auditable without discarding source records.
  for (const e of graph.edges) if (!represented.has(e.id) && !filtered.has(e.id)) hidden.add(e.id);
  for (const id of represented) { hidden.delete(id); filtered.delete(id); }
  for (const id of filtered) hidden.delete(id);
  return { nodes: shownNodes, edges: shownEdges, hiddenEdgeIds: [...hidden], filteredEdgeIds: [...filtered], unusedInputs,
    ...(scope ? { scope: { id: scope.id, nodeIds: [...scope.members],
      excludedNodeIds: graph.nodes.filter((node) => !scope.members.has(node.id)).map((node) => node.id),
      excludedEdgeIds: graph.edges.filter((edge) => !scope.members.has(edge.source.node_id) && !scope.members.has(edge.target.node_id)).map((edge) => edge.id),
    } } : {}) };
}

/** Exact visible endpoints, including genuine fan-out; never traverse through
 * a computational operation during hover/focus. */
export function connectionSet(projection: Projection, target: { edgeId: string } | { port: Endpoint }): string[] {
  if ('edgeId' in target) return projection.edges.filter((e) => e.id === target.edgeId).map((e) => e.id);
  const key = endpointKey(target.port);
  return projection.edges.filter((e) => endpointKey(e.source) === key || endpointKey(e.target) === key).map((e) => e.id);
}
