import type { Graph, GraphNode } from './graph';
import type { Endpoint } from './projection';

export type BoundaryOwner = { kind: 'source' | 'model' | 'presentation'; id: string };
export interface ModelInterface {
  node: GraphNode; endpoint: Endpoint; owner: BoundaryOwner;
  direction: 'input' | 'output'; label: string;
  /** Existing owner endpoints reached only through explicit group forwarding. */
  ownerPorts: Endpoint[];
}
export interface BoundarySelection {
  kind: 'boundary'; owner: BoundaryOwner; endpoints: Endpoint[];
  templatePort?: TemplatePortTarget;
}
/** Presentation identity within verified correspondence, independent of any
 * concrete instance's endpoints. It remains usable in structure-only mode. */
export interface TemplatePortTarget {
  kind: 'template-port'; templateId: string; nodeRole: string; portRole: string;
}
const key = (p: Endpoint) => JSON.stringify([p.node_id, p.port_id]);
const cache = new WeakMap<Graph, ReturnType<typeof buildInterfaceIndex>>();

/** Classification never reads labels, evaluates expressions, or traverses an operation. */
function buildInterfaceIndex(graph: Graph) {
  const records = new Map(graph.nodes.map((n) => [n.id, n]));
  const incoming = new Map<string, Graph['edges']>(), outgoing = new Map<string, Graph['edges']>();
  const incident = new Map<string, Graph['edges']>();
  for (const edge of graph.edges) {
    for (const id of new Set([edge.source.node_id, edge.target.node_id])) {
      const list = incident.get(id) ?? []; list.push(edge); incident.set(id, list);
    }
    const ins = incoming.get(key(edge.target)) ?? []; ins.push(edge); incoming.set(key(edge.target), ins);
    const outs = outgoing.get(key(edge.source)) ?? []; outs.push(edge); outgoing.set(key(edge.source), outs);
  }
  const tools = new Set<string>(), notices = new Map<string, string>();
  const declarations = new Map<string, ModelInterface[]>();
  for (const node of graph.nodes) {
    const edges = incident.get(node.id) ?? [];
    const tokenizer = node.references.some((r) => r.kind === 'tokenizer');
    if (node.kind === 'context' && tokenizer && !node.ports.length && !edges.length &&
      !node.parameter_ids.length && node.references.every((r) => r.kind === 'tokenizer') && !node.formula && !node.operation) {
      tools.add(node.id); continue;
    }
    if (node.kind !== 'input' && node.kind !== 'output') continue;
    const inputs = node.ports.filter((p) => p.direction === 'input');
    const outputs = node.ports.filter((p) => p.direction === 'output');
    const passive = !node.parameter_ids.length && !node.references.length && !node.formula && edges.every((e) => e.kind === 'data');
    const valid = passive && (node.kind === 'input'
      ? !inputs.length && outputs.length > 0 && edges.every((e) => e.source.node_id === node.id)
      : inputs.length === 1 && outputs.length <= 1 && edges.length <= 1 && edges.every((e) => e.target.node_id === node.id));
    if (!valid) { notices.set(node.id, 'Interface mapping is ambiguous; the original component and connections remain visible.'); continue; }
    const ports = node.kind === 'input' ? outputs : inputs;
    declarations.set(node.id, ports.map((port) => ({ node, endpoint: { node_id: node.id, port_id: port.id },
      owner: { kind: 'source', id: node.parent_id ?? '' }, direction: node.kind === 'input' ? 'input' : 'output',
      label: ports.length > 1 ? `${node.label} · ${port.label}` : node.label, ownerPorts: [] })));
  }
  const roots = graph.nodes.filter((n) => !n.parent_id && !tools.has(n.id) && !declarations.has(n.id));
  let modelId = 'presentation:model';
  while (records.has(modelId)) modelId += ':model';
  const outer: BoundaryOwner = roots.length === 1 && roots[0]!.kind === 'group'
    ? { kind: 'source', id: roots[0]!.id } : { kind: 'model', id: modelId };
  const signals = new Map<string, ModelInterface[]>();
  for (const interfaces of declarations.values()) for (const item of interfaces) {
    if (!item.owner.id) item.owner = outer;
    const pending = [item.endpoint], seen = new Set<string>();
    while (pending.length) {
      const endpoint = pending.pop()!, id = key(endpoint);
      if (seen.has(id)) continue;
      seen.add(id);
      const matches = signals.get(id) ?? []; matches.push(item); signals.set(id, matches);
      const port = records.get(endpoint.node_id)!.ports.find((p) => p.id === endpoint.port_id)!;
      if (endpoint.node_id === item.owner.id && port.direction === item.direction) item.ownerPorts.push(endpoint);
      const edges = (item.direction === 'input' ? outgoing : incoming).get(id) ?? [];
      for (const edge of edges) {
        if (edge.kind !== 'data') continue;
        const next = item.direction === 'input' ? edge.target : edge.source;
        if (records.get(next.node_id)?.kind === 'group') pending.push(next);
      }
    }
  }
  // Verify incoming identities without crossing a computational operation. A
  // group port fed by a declaration and a computation is not one supplied signal.
  const competing = new Set<string>();
  for (const [id, items] of signals) {
    const endpoint = JSON.parse(id) as [string, string];
    if (records.get(endpoint[0])?.kind !== 'group') continue;
    for (const item of items.filter((v) => v.direction === 'input')) {
      const pending = [id], seen = new Set<string>();
      while (pending.length) {
        const current = pending.pop()!;
        if (seen.has(current)) continue;
        seen.add(current);
        for (const edge of incoming.get(current) ?? []) {
          if (records.get(edge.source.node_id)?.kind === 'group' && edge.kind === 'data') pending.push(key(edge.source));
          else if (edge.kind !== 'data' || key(edge.source) !== key(item.endpoint)) competing.add(item.node.id);
        }
      }
    }
  }
  for (const items of declarations.values()) for (const item of items) {
    if (item.direction !== 'output') continue;
    for (const port of item.ownerPorts) {
      const pending = [key(port)], seen = new Set<string>(), sources = new Set<string>();
      while (pending.length) {
        const current = pending.pop()!;
        if (seen.has(current)) continue;
        seen.add(current);
        for (const edge of incoming.get(current) ?? []) {
          if (edge.kind !== 'data') competing.add(item.node.id);
          if (records.get(edge.source.node_id)?.kind === 'group') pending.push(key(edge.source));
          else sources.add(key(edge.source));
        }
      }
      if (sources.size > 1) competing.add(item.node.id);
    }
  }
  // Never reuse a boundary that combines independent declarations.
  for (const [id, items] of declarations) if (competing.has(id) || items.some((item) => item.ownerPorts.some((p) => (signals.get(key(p))?.length ?? 0) > 1))) {
    declarations.delete(id); notices.set(id, 'Distinct interfaces share a boundary; the original declarations remain visible.');
  }
  for (const [id, items] of signals) signals.set(id, items.filter((item) => declarations.has(item.node.id)));
  const interfaces = [...declarations.values()].flat();
  const eligible = (id: string) => !declarations.has(id) && !tools.has(id);
  const finalRoots = graph.nodes.filter((n) => !n.parent_id && eligible(n.id));
  const finalOuter: BoundaryOwner = finalRoots.length === 1 && finalRoots[0]!.kind === 'group'
    ? { kind: 'source', id: finalRoots[0]!.id } : { kind: 'model', id: modelId };
  for (const item of interfaces) if (!item.node.parent_id) {
    item.owner = finalOuter; item.ownerPorts = item.ownerPorts.filter((p) => p.node_id === finalOuter.id);
  }
  return { outer: finalOuter, roots: finalRoots, declarations, interfaces, tools, notices, signals, eligible };
}

export function interfaceIndex(graph: Graph) {
  let index = cache.get(graph);
  if (!index) { index = buildInterfaceIndex(graph); cache.set(graph, index); }
  return index;
}

export function interfaceSelection(item: ModelInterface): BoundarySelection {
  return { kind: 'boundary', owner: item.owner, endpoints: [item.endpoint] };
}

/** Resolve concrete ports against the full graph, including declarations outside
 * an isolated/template subgraph. Follow only the index's exact group forwarding. */
export function resolveInterfaceEndpoints(graph: Graph, endpoints: Endpoint[]): Endpoint[] {
  const index = interfaceIndex(graph), resolved = new Map(endpoints.map((p) => [key(p), p]));
  for (const endpoint of endpoints) for (const item of index.signals.get(key(endpoint)) ?? []) {
    resolved.set(key(item.endpoint), item.endpoint);
  }
  return [...resolved.values()];
}
