/** Test-only oracles. No runtime import from projection, grouping or layout. */
import assert from 'node:assert/strict';
import type { Graph } from '../src/architecture-explorer/graph';
import type { Endpoint, Projection } from '../src/architecture-explorer/projection';

const key = (endpoint: Endpoint) => JSON.stringify([endpoint.node_id, endpoint.port_id]);
const sorted = <T>(items: T[]) => items.sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
const kinds = (values: string[]) => values.filter((value, i) => i === 0 || value !== values[i - 1]);
type Wire = { source: Endpoint; target: Endpoint; kinds: string[] };

/** Explicit reviewed correspondence, never inferred from labels, shapes or reachability.
 * Omitted entries retain their source ID. Tensor inventory IDs are never remapped.
 * Use endpoint keys JSON.stringify([node_id, port_id]) for port-role overrides.
 */
export interface SemanticNames {
  nodes?: Record<string, string>;
  ports?: Record<string, string>;
  parameters?: Record<string, string>;
  repetitions?: Record<string, string>;
}

function contractedWires(graph: Graph): Wire[] {
  let wires: Wire[] = graph.edges.map((edge) => ({ source: edge.source, target: edge.target, kinds: [edge.kind] }));
  // Eliminate boundary endpoints by relational composition, preserving a multiset.
  // A computational input and output are NEVER joined, even for identity/reshape.
  for (const node of graph.nodes) if (node.kind === 'group') for (const port of node.ports) {
    const boundary = key({ node_id: node.id, port_id: port.id });
    const incoming = wires.filter((wire) => key(wire.target) === boundary);
    const outgoing = wires.filter((wire) => key(wire.source) === boundary);
    if (!incoming.length || !outgoing.length) continue; // Preserve open interfaces.
    assert(!incoming.some((wire) => key(wire.source) === boundary), 'Cyclic group forwarding');
    wires = wires.filter((wire) => key(wire.source) !== boundary && key(wire.target) !== boundary);
    for (const before of incoming) for (const after of outgoing) wires.push({
      source: before.source, target: after.target, kinds: kinds([...before.kinds, ...after.kinds]),
    });
  }
  return wires;
}

/** Operation-level contract across producer revisions. Group count, graph/edge
 * hashes, containment wrappers and provenance revision strings are not semantics.
 * Non-group ports, endpoint roles, state/context kinds, multiplicity, repetition
 * ownership, operation attributes/formulas and logical bindings remain exact.
 * API validation is a separate prerequisite; this is not a schema validator.
 */
export function semanticSnapshot(graph: Graph, names: SemanticNames = {}) {
  const nodeName = (id: string) => names.nodes?.[id] ?? id;
  const parameterName = (id: string) => names.parameters?.[id] ?? id;
  const repetitionName = (id: string) => names.repetitions?.[id] ?? id;
  const endpoint = (value: Endpoint) => [nodeName(value.node_id), names.ports?.[key(value)] ?? value.port_id];
  for (const ids of [graph.nodes.map((n) => nodeName(n.id)), graph.parameters.map((p) => parameterName(p.id)),
    graph.repetitions.map((r) => repetitionName(r.id))]) assert.equal(new Set(ids).size, ids.length, 'Correspondence must be injective');
  const references = (refs: Graph['nodes'][number]['references']) => sorted(refs.map((ref) =>
    ref.kind === 'parameter' ? { ...ref, parameter_id: parameterName(ref.parameter_id) } : ref));
  const records = new Map(graph.nodes.map((node) => [node.id, node]));
  const membership = new Map(graph.repetitions.flatMap((rep) => rep.instances.map((instance) =>
    [instance.node_id, { repetition: repetitionName(rep.id), index: instance.index, variant: instance.variant }] as const)));
  const ownership = (id: string) => {
    const owners = [];
    for (let node = records.get(id); node; node = node.parent_id ? records.get(node.parent_id) : undefined) {
      if (membership.has(node.id)) owners.unshift(membership.get(node.id)!);
    }
    return owners;
  };
  const portRecord = (nodeId: string, port: Graph['nodes'][number]['ports'][number]) => ({
    role: endpoint({ node_id: nodeId, port_id: port.id })[1], direction: port.direction, label: port.label, shape: port.shape,
  });
  for (const node of graph.nodes) {
    const roles = node.ports.map((port) => endpoint({ node_id: node.id, port_id: port.id })[1]);
    assert.equal(new Set(roles).size, roles.length, 'Port correspondence must be injective');
  }
  const wires = contractedWires(graph);
  const terminals = new Map<string, Endpoint>();
  for (const wire of wires) for (const end of [wire.source, wire.target]) {
    if (records.get(end.node_id)!.kind === 'group') terminals.set(key(end), end);
  }
  return {
    scope: graph.scope, coverage: graph.coverage, symbols: sorted([...graph.symbols]),
    diagnostics: sorted(graph.diagnostics.map((diagnostic) => ({ ...diagnostic,
      ...(diagnostic.node_id ? { node_id: nodeName(diagnostic.node_id) } : {}),
      ...(diagnostic.parameter_id ? { parameter_id: parameterName(diagnostic.parameter_id) } : {}),
    }))),
    nodes: sorted(graph.nodes.filter((node) => node.kind !== 'group').map((node) => ({
      id: nodeName(node.id), kind: node.kind, operation: node.operation, formula: node.formula,
      ports: sorted(node.ports.map((port) => portRecord(node.id, port))), ownership: ownership(node.id),
      attributes: sorted(node.attributes.map(({ name, value }) => ({ name, value }))),
      parameters: node.parameter_ids.map(parameterName), references: references(node.references),
    }))),
    groupBindings: sorted(graph.nodes.filter((node) => node.kind === 'group' && node.parameter_ids.length).map((node) => ({
      id: nodeName(node.id), parameters: node.parameter_ids.map(parameterName), references: references(node.references),
    }))),
    interfaces: sorted([...terminals.values()].map((end) => ({ endpoint: endpoint(end),
      port: portRecord(end.node_id, records.get(end.node_id)!.ports.find((port) => port.id === end.port_id)!),
    }))),
    wires: sorted(wires.map((wire) => ({ source: endpoint(wire.source), target: endpoint(wire.target), kinds: wire.kinds }))),
    repetitions: sorted(graph.repetitions.map((rep) => ({ id: repetitionName(rep.id),
      instances: rep.instances.map((instance) => ({ ...instance, node_id: nodeName(instance.node_id) })),
    }))),
    parameters: sorted(graph.parameters.map((record) => {
      const { provenance, ...parameter } = record; void provenance;
      return { ...parameter,
      id: parameterName(parameter.id), ...(parameter.binding === 'alias' ? { alias_of: parameterName(parameter.alias_of) } : {}),
    }; })),
  };
}

export function assertSemanticEquivalent(before: Graph, after: Graph, beforeNames?: SemanticNames, afterNames?: SemanticNames) {
  assert.deepEqual(semanticSnapshot(after, afterNames), semanticSnapshot(before, beforeNames), 'Operation-level semantics changed');
}

/** Common unchanged-source oracle for compact, focused, filtered and exhaustive views. */
export function assertTraceability(graph: Graph, projected: Projection, exhaustive = false) {
  const sourceEdges = new Map(graph.edges.map((edge) => [edge.id, edge]));
  const sourceNodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const nodes = new Map(projected.nodes.map((node) => [node.id, node]));
  assert.equal(nodes.size, projected.nodes.length);
  const paths = new Set<string>();
  for (const node of projected.nodes) {
    assert(node.sourceIds.length > 0);
    for (const id of node.sourceIds) assert(sourceNodes.has(id));
    if (node.record) assert.equal(node.record, sourceNodes.get(node.id), 'Source record identity changed');
    for (const port of node.ports) for (const original of port.endpoints) {
      assert(sourceNodes.get(original.node_id)?.ports.some((item) => item.id === original.port_id));
      if (original === port.endpoints[0]) assert.deepEqual(port.shape, sourceNodes.get(original.node_id)!.ports.find((item) => item.id === original.port_id)!.shape);
    }
  }
  for (const edge of projected.edges) {
    const source = nodes.get(edge.source.node_id)?.ports.find((port) => port.id === edge.source.port_id);
    const target = nodes.get(edge.target.node_id)?.ports.find((port) => port.id === edge.target.port_id);
    const conventionalDirection = source?.direction === 'output' && target?.direction === 'input';
    const groupBoundaryForwarding = edge.paths.some((path) => {
      const first = path[0]!, last = path.at(-1)!;
      const firstSource = sourceNodes.get(first.source.node_id)!;
      const firstTarget = sourceNodes.get(first.target.node_id)!;
      const lastSource = sourceNodes.get(last.source.node_id)!;
      const lastTarget = sourceNodes.get(last.target.node_id)!;
      const firstSourcePort = firstSource.ports.find((port) => port.id === first.source.port_id)!;
      const firstTargetPort = firstTarget.ports.find((port) => port.id === first.target.port_id)!;
      const lastSourcePort = lastSource.ports.find((port) => port.id === last.source.port_id)!;
      const lastTargetPort = lastTarget.ports.find((port) => port.id === last.target.port_id)!;
      return (firstSource.kind === 'group' && firstTarget.parent_id === firstSource.id &&
        firstSourcePort.direction === 'input' && firstTargetPort.direction === 'input') ||
        (lastTarget.kind === 'group' && lastSource.parent_id === lastTarget.id &&
          lastSourcePort.direction === 'output' && lastTargetPort.direction === 'output');
    });
    assert(conventionalDirection || groupBoundaryForwarding, 'Invalid projected edge directions');
    const originals = new Set<string>();
    assert(edge.paths.length > 0);
    for (const path of edge.paths) {
      assert(path.length > 0);
      assert(source!.endpoints.some((end) => key(end) === key(path[0]!.source)));
      assert(target!.endpoints.some((end) => key(end) === key(path.at(-1)!.target)));
      assert.equal(edge.kind, path.some((item) => item.kind === 'state') ? 'state' : path[0]!.kind);
      const pathKey = JSON.stringify(path.map((item) => item.id));
      assert(!paths.has(pathKey), 'Signal path duplicated'); paths.add(pathKey);
      path.forEach((item, i) => {
        assert.equal(item, sourceEdges.get(item.id)); originals.add(item.id);
        if (i) {
          assert.deepEqual(item.source, path[i - 1]!.target);
          assert.equal(sourceNodes.get(item.source.node_id)!.kind, 'group', 'Cannot contract an operation');
        }
      });
    }
    assert.equal(new Set(edge.originalEdgeIds).size, edge.originalEdgeIds.length);
    assert.deepEqual(new Set(edge.originalEdgeIds), originals);
  }
  const represented = new Set(projected.edges.flatMap((edge) => edge.originalEdgeIds));
  const hidden = new Set(projected.hiddenEdgeIds), filtered = new Set(projected.filteredEdgeIds);
  assert(![...represented].some((id) => hidden.has(id) || filtered.has(id)));
  assert(![...hidden].some((id) => filtered.has(id)));
  assert.deepEqual(new Set([...represented, ...hidden, ...filtered]), new Set(sourceEdges.keys()));
  if (exhaustive) {
    assert.deepEqual(new Set(nodes.keys()), new Set(sourceNodes.keys()));
    assert.deepEqual(projected.hiddenEdgeIds, []); assert.deepEqual(projected.filteredEdgeIds, []);
    const wires = projected.edges.flatMap((edge) => edge.paths.map((path) => ({
      source: path[0]!.source, target: path.at(-1)!.target, kinds: kinds(path.map((item) => item.kind)),
    })));
    assert.deepEqual(sorted(wires), sorted(contractedWires(graph)), 'Exhaustive directed signal multiset changed');
  }
}
