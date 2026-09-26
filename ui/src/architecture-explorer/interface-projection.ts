import type { Graph } from './graph';
import { interfaceIndex, interfaceNotices } from './interfaces';
import type { ModelInterface } from './interfaces';
import type { Endpoint, ProjectedEdge, ProjectedNode, ProjectedPort, Projection, ProjectionOptions } from './projection';

const key = (p: Endpoint) => JSON.stringify([p.node_id, p.port_id]);

/** Change presentation endpoints only. Source edges and their terminal declarations
 * remain the evidence for every route, including forwarding absorbed by a port. */
export function projectInterfaces(graph: Graph, base: Projection, options: ProjectionOptions): Projection {
  const index = interfaceIndex(graph);
  const records = new Map(graph.nodes.map((n) => [n.id, n]));
  const baseNodes = new Map(base.nodes.map((n) => [n.id, n]));
  const removed = new Set([...index.declarations.keys(), ...index.tools]);
  const sourcePort = (p: Endpoint) => records.get(p.node_id)!.ports.find((v) => v.id === p.port_id)!;
  const passiveExternal = (node: ProjectedNode) => node.presentation === 'external' && node.ports.length > 0 &&
    node.ports.every((p) => p.endpoints.every((e) => index.signals.get(key(e))?.filter((item) =>
      item.direction === (node.kind === 'input' ? 'input' : 'output')).length === 1));
  const nodes = base.nodes.filter((n) => !removed.has(n.id) && !passiveExternal(n));
  const visible = new Map(nodes.map((n) => [n.id, n]));
  const sourceOwners = new Map(nodes.flatMap((n) => n.sourceIds.map((id) => [id, n] as const)));
  const nearest = (id: string): ProjectedNode | undefined => {
    for (let node = records.get(id); node; node = node.parent_id ? records.get(node.parent_id) : undefined) {
      const owner = visible.get(node.id) ?? sourceOwners.get(node.id);
      if (owner) return owner;
    }
    return undefined;
  };
  if (!options.scope && index.outer.kind === 'model') {
    const wrapper: ProjectedNode = { id: index.outer.id, kind: 'group', label: 'Model', presentation: 'model',
      sourceIds: index.roots.map((n) => n.id), ports: [], expanded: !options.modelCollapsed || Boolean(options.exhaustive),
      componentCount: index.roots.length, summary: `Presentation boundary · ${graph.scope} · ${graph.coverage}` };
    for (const node of nodes) if (!node.parentId) node.parentId = wrapper.id;
    nodes.unshift(wrapper); visible.set(wrapper.id, wrapper);
  }
  for (const node of nodes) if (node.record?.kind === 'group') {
    node.componentCount = node.record.children.filter(index.eligible).length;
    if (!node.componentCount) node.expanded = false;
  }
  const portAt = (node: ProjectedNode, original: Endpoint): ProjectedPort => {
    let port = node.ports.find((p) => p.id === original.port_id);
    if (!port) { port = { ...sourcePort(original), endpoints: [original] }; node.ports.push(port); }
    for (const item of index.signals.get(key(original)) ?? []) {
      if (!port.endpoints.some((p) => key(p) === key(item.endpoint))) port.endpoints.push(item.endpoint);
      port.interfaces = [...new Set([...(port.interfaces ?? []), item.node.id])];
    }
    return port;
  };
  const addInterface = (item: ModelInterface, owner: ProjectedNode, original?: Endpoint): Endpoint => {
    const exact = original ?? item.ownerPorts.find((p) => p.node_id === owner.id);
    const id = exact?.port_id ?? `interface:${key(item.endpoint)}`;
    let port = owner.ports.find((p) => p.id === id);
    if (!port) {
      port = { id, label: exact ? sourcePort(exact).label : item.label, direction: item.direction,
        shape: sourcePort(exact ?? item.endpoint).shape, endpoints: exact ? [exact] : [] };
      owner.ports.push(port);
    }
    if (!port.endpoints.some((p) => key(p) === key(item.endpoint))) port.endpoints.push(item.endpoint);
    port.interfaces = [...new Set([...(port.interfaces ?? []), item.node.id])];
    if (exact && !port.label.includes(item.label)) port.interfaceLabel = item.label;
    return { node_id: owner.id, port_id: id };
  };
  const ownerOf = (item: ModelInterface) => options.scope ? nearest(item.owner.id) ?? nearest(options.scope)
    : visible.get(item.owner.id) ?? nearest(item.owner.id);
  // Isolation exposes the source component's complete interface, independently
  // of external declarations or whether a port has a routed connection.
  const isolated = options.scope ? visible.get(options.scope) : undefined;
  if (isolated?.record) for (const port of isolated.record.ports) portAt(isolated, { node_id: isolated.id, port_id: port.id });
  const expandedBoundary = (endpoint: Endpoint) => visible.get(endpoint.node_id)?.expanded &&
    (endpoint.node_id === options.scope || Boolean(index.signals.get(key(endpoint))?.length));
  // Complete declared interfaces remain available even without visible edges.
  for (const item of index.interfaces) {
    if (options.scope && !base.scope?.nodeIds.includes(item.node.id)) continue;
    const owner = ownerOf(item);
    if (owner) {
      const matches = item.ownerPorts.filter((p) => p.node_id === owner.id);
      if (matches.length) matches.forEach((p) => addInterface(item, owner, p)); else addInterface(item, owner);
    }
  }
  const absorbed: ProjectedEdge['paths'] = [];
  const edges: ProjectedEdge[] = [];
  const seenPaths = new Set<string>();
  const mapped = (original: Endpoint, fallback: Endpoint, path: Graph['edges'], other: Endpoint): Endpoint => {
    const item = index.signals.get(key(original))?.[0];
    const fallbackNode = baseNodes.get(fallback.node_id);
    const external = fallbackNode && passiveExternal(fallbackNode);
    if (item && (index.declarations.has(original.node_id) || external)) {
      const owner = ownerOf(item);
      if (owner) {
        const boundary = path.flatMap((e) => [e.source, e.target]).find((p) => p.node_id === owner.id &&
          (records.get(owner.id)?.kind !== 'group' || sourcePort(p).direction === item.direction));
        const exact = boundary ?? (other.node_id === owner.id ? other : undefined);
        return addInterface(item, owner, exact);
      }
    }
    const direct = visible.get(original.node_id);
    if (direct && expandedBoundary(original)) {
      portAt(direct, original); return original;
    }
    return fallback;
  };
  for (const edge of base.edges) for (const path of edge.paths) {
    const cuts = [0];
    for (let i = 1; i < path.length; i++) {
      const ep = path[i]!.source;
      if (expandedBoundary(ep)) cuts.push(i);
    }
    cuts.push(path.length);
    for (let at = 1; at < cuts.length; at++) {
      const first = cuts[at - 1]!, last = cuts[at]!;
      const segment = path.slice(first, last), start = segment[0]!.source, end = segment.at(-1)!.target;
      const source = mapped(start, first ? start : edge.source, path, end);
      const target = mapped(end, last < path.length ? end : edge.target, path, start);
      const signature = JSON.stringify(segment.map((e) => e.id));
      if (seenPaths.has(signature)) continue;
      seenPaths.add(signature);
      if (key(source) === key(target)) { absorbed.push(segment); continue; }
      if (!visible.has(source.node_id) || !visible.has(target.node_id)) continue;
      edges.push({ ...edge, id: cuts.length === 2 ? edge.id : `interface-connection:${signature}`,
        source, target, paths: [segment], originalEdgeIds: segment.map((e) => e.id) });
    }
  }
  const grouped = new Map<string, ProjectedEdge>();
  for (const edge of edges) {
    const id = JSON.stringify([edge.source, edge.target, edge.kind]);
    const previous = grouped.get(id);
    if (previous) { previous.paths.push(...edge.paths); previous.originalEdgeIds = [...new Set([...previous.originalEdgeIds, ...edge.originalEdgeIds])]; }
    else grouped.set(id, edge);
  }
  let shownNodes = nodes, shownEdges = [...grouped.values()];
  const wrapper = visible.get(index.outer.id);
  if (!options.scope && wrapper?.presentation === 'model' && !wrapper.expanded) {
    for (const item of index.interfaces) addInterface(item, wrapper);
    shownNodes = [wrapper]; shownEdges = [];
  }
  // Discard obsolete operation-port aliases that belonged to removed cards,
  // while retaining every explicit source port and interface.
  const used = new Set(shownEdges.flatMap((e) => [key(e.source), key(e.target)]));
  for (const node of shownNodes) node.ports = node.ports.filter((p) => p.interfaces?.length ||
    node.record?.ports.some((v) => v.id === p.id) || used.has(key({ node_id: node.id, port_id: p.id })));
  const represented = new Set([...shownEdges.flatMap((e) => e.originalEdgeIds), ...absorbed.flat().map((e) => e.id)]);
  const filtered = base.filteredEdgeIds.filter((id) => !represented.has(id));
  const hidden = graph.edges.filter((e) => !represented.has(e.id) && !filtered.includes(e.id)).map((e) => e.id);
  return { ...base, nodes: shownNodes, edges: shownEdges, hiddenEdgeIds: hidden, filteredEdgeIds: filtered,
    boundaryPaths: absorbed, notices: interfaceNotices(graph) };
}
