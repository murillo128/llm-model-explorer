import { describe, expect, it } from 'vitest';
import { assertSemanticEquivalent, assertTraceability, semanticSnapshot, type SemanticNames } from '../../tests/architecture-invariants';
import { makeProjectionFixture, type ProjectionFixtureOptions } from '../../tests/architecture-projection-fixture';
import { contractResponse } from '../../tests/architecture-fixtures';
import { validateArchitecture } from '../api/architecture-validation';
import type { Graph } from './graph';
import { projectGraph } from './projection';

const ep = (node_id: string, port_id: string) => ({ node_id, port_id });
const fixture = () => makeProjectionFixture({ variants: ['full_attention', 'linear_attention', 'full_attention'], hiddenSize: 29 });
function validate(graph: Graph) {
  validateArchitecture({ status: 'available', model_id: 'oracle', diagnostics: [], graph }, {
    modelId: 'oracle', tokenizerAvailable: true, inventory: { tensors: [], coverage: 'partial', diagnostics: [] },
  });
}

/** Authored producer change: wrap Q projection, with exactly two forwarding
 * segments. This does not call the production derived-group/projection helpers. */
function wrapQuery(graph: Graph) {
  const attention = graph.nodes.find((node) => node.id === 'layer-0.attention')!;
  const query = graph.nodes.find((node) => node.id === 'layer-0.attention.Q')!;
  if (attention.kind !== 'group') throw new Error('Expected authored attention');
  attention.children[attention.children.indexOf(query.id)] = 'query-group';
  query.parent_id = 'query-group';
  graph.nodes.push({ id: 'query-group', kind: 'group', label: 'Q component', parent_id: attention.id,
    children: [query.id], ports: structuredClone(query.ports), parameter_ids: [], references: [], attributes: [], provenance: [] });
  graph.edges.find((edge) => edge.target.node_id === query.id)!.target = ep('query-group', 'x');
  graph.edges.find((edge) => edge.source.node_id === query.id)!.source = ep('query-group', 'out');
  graph.edges.push(
    { id: 'query-in', source: ep('query-group', 'x'), target: ep(query.id, 'x'), kind: 'data', provenance: [] },
    { id: 'query-out', source: ep(query.id, 'out'), target: ep('query-group', 'out'), kind: 'data', provenance: [] },
  );
}

function renameProducerIds(graph: Graph): SemanticNames {
  const names: Required<SemanticNames> = { nodes: {}, parameters: {}, repetitions: {}, ports: {} };
  const nodeIds = new Map(graph.nodes.map((node, i) => [node.id, `revision-node-${i}`]));
  const parameterIds = new Map(graph.parameters.map((parameter, i) => [parameter.id, `revision-parameter-${i}`]));
  const endpoint = (end: ReturnType<typeof ep>) => ep(nodeIds.get(end.node_id)!, `port-${end.port_id}`);
  graph.graph_id = 'new-producer-identity';
  for (const node of graph.nodes) {
    const old = node.id; node.id = nodeIds.get(old)!; names.nodes[node.id] = old;
    if (node.parent_id) node.parent_id = nodeIds.get(node.parent_id)!;
    if (node.kind === 'group') node.children = node.children.map((id) => nodeIds.get(id)!);
    for (const port of node.ports) {
      const oldPort = port.id; port.id = `port-${oldPort}`;
      names.ports[JSON.stringify([node.id, port.id])] = oldPort;
    }
    node.parameter_ids = node.parameter_ids.map((id) => parameterIds.get(id)!);
    for (const ref of node.references) if (ref.kind === 'parameter') ref.parameter_id = parameterIds.get(ref.parameter_id)!;
    node.provenance = [{ kind: 'description', source: 'new-producer', revision: '2' }];
  }
  graph.edges.forEach((edge, i) => { edge.id = `revision-edge-${i}`; edge.source = endpoint(edge.source); edge.target = endpoint(edge.target); });
  for (const parameter of graph.parameters) {
    const old = parameter.id; parameter.id = parameterIds.get(old)!; names.parameters[parameter.id] = old;
    if (parameter.binding === 'alias') parameter.alias_of = parameterIds.get(parameter.alias_of)!;
  }
  graph.repetitions.forEach((rep, i) => {
    const old = rep.id; rep.id = `revision-repetition-${i}`; names.repetitions[rep.id] = old;
    rep.parent_id = nodeIds.get(rep.parent_id)!;
    rep.instances.forEach((instance) => { instance.node_id = nodeIds.get(instance.node_id)!; });
  });
  for (const diagnostic of graph.diagnostics) {
    if (diagnostic.node_id) diagnostic.node_id = nodeIds.get(diagnostic.node_id)!;
    if (diagnostic.parameter_id) diagnostic.parameter_id = parameterIds.get(diagnostic.parameter_id)!;
  }
  return names;
}

describe('independent operation-level safety oracle', () => {
  it('accepts only transparent new boundaries with explicit correspondence across producer IDs', () => {
    const before = fixture(), after = structuredClone(before);
    wrapQuery(after); validate(before); validate(after);
    const names = renameProducerIds(after); validate(after);
    expect(after.graph_id).not.toBe(before.graph_id);
    expect(after.nodes.length).toBe(before.nodes.length + 1);
    expect(after.edges.length).toBe(before.edges.length + 2);
    assertSemanticEquivalent(before, after, undefined, names);
    // Reverse order is a representation change, not a different computation.
    after.nodes.reverse(); after.edges.reverse(); after.parameters.reverse();
    assertSemanticEquivalent(before, after, undefined, names);
  });

  it('has a hand-written endpoint oracle that stops at every computational operation', () => {
    const wires = semanticSnapshot(fixture()).wires;
    const incoming = (node: string) => wires.filter((wire) => wire.target[0] === node);
    expect(incoming('layer-0.attention.Q')).toEqual([
      { source: ['layer-0.input-norm', 'out'], target: ['layer-0.attention.Q', 'x'], kinds: ['data'] },
    ]);
    expect(incoming('layer-0.residual-1')).toEqual(expect.arrayContaining([
      { source: ['embedding', 'out'], target: ['layer-0.residual-1', 'residual'], kinds: ['data'] },
      { source: ['layer-0.attention.output', 'out'], target: ['layer-0.residual-1', 'update'], kinds: ['data'] },
    ]));
    expect(incoming('layer-0.attention.core').filter((wire) => wire.kinds.includes('state'))).toEqual([
      { source: ['layer-0.prior-K', 'out'], target: ['layer-0.attention.core', 'prior_K'], kinds: ['state'] },
      { source: ['layer-0.prior-V', 'out'], target: ['layer-0.attention.core', 'prior_V'], kinds: ['state'] },
    ]);
    expect(wires.some((wire) => wire.source[0] === 'embedding' && wire.target[0] === 'layer-0.attention.Q')).toBe(false);
  });

  const mutations: [string, (graph: Graph) => void][] = [
    ['dropped residual', (graph) => { graph.edges = graph.edges.filter((edge) => !(edge.target.node_id === 'layer-0.residual-1' && edge.target.port_id === 'residual')); }],
    ['swapped equal-shaped Q/K ports', (graph) => { for (const edge of graph.edges) if (edge.target.node_id === 'layer-0.attention.core') {
      if (edge.target.port_id === 'Q') edge.target.port_id = 'K'; else if (edge.target.port_id === 'K') edge.target.port_id = 'Q';
    } }],
    ['incorrect instance weight', (graph) => { const last = graph.nodes.find((node) => node.id === 'layer-2.attention.Q')!;
      last.parameter_ids = [...graph.nodes.find((node) => node.id === 'layer-0.attention.Q')!.parameter_ids]; }],
    ['merged prior K/V states', (graph) => { graph.edges.find((edge) => edge.source.node_id === 'layer-0.prior-V')!.source.node_id = 'layer-0.prior-K'; }],
    ['merged next K/V states', (graph) => { graph.edges.find((edge) => edge.target.node_id === 'layer-0.next-V')!.target.node_id = 'layer-0.next-K'; }],
    ['cross-instance state ownership', (graph) => { graph.edges.find((edge) => edge.source.node_id === 'layer-2.prior-K')!.source.node_id = 'layer-0.prior-K'; }],
    ['duplicate signal', (graph) => { graph.edges.push({ ...structuredClone(graph.edges[0]!), id: 'duplicate-signal' }); }],
    ['state reclassified as data', (graph) => { graph.edges.find((edge) => edge.kind === 'state')!.kind = 'data'; }],
    ['context reclassified as data', (graph) => { graph.edges[0]!.kind = 'context'; }],
    ['operation changed', (graph) => { graph.nodes.find((node) => node.id === 'layer-0.input-norm')!.operation = 'identity'; }],
    ['operation attributes changed', (graph) => { graph.nodes.find((node) => node.id === 'layer-0.attention.core')!.attributes.push({ name: 'causal', value: false, provenance: [] }); }],
    ['port shape changed', (graph) => { graph.nodes.find((node) => node.id === 'layer-0.attention.Q')!.ports[0]!.shape = null; }],
    ['variant order changed', (graph) => { graph.repetitions[0]!.instances[0]!.variant = 'linear_attention'; }],
    ['instance index changed', (graph) => { graph.repetitions[0]!.instances[0]!.index = 99; }],
    ['instance sequence reversed', (graph) => { graph.repetitions[0]!.instances.reverse(); }],
  ];
  it.each(mutations)('rejects controlled negative: %s', (_name, mutate) => {
    const before = fixture(), after = structuredClone(before); mutate(after);
    expect(() => assertSemanticEquivalent(before, after)).toThrow('Operation-level semantics changed');
  });

  it('rejects contracting a real identity operation as a group', () => {
    const before = fixture(), after = structuredClone(before);
    const query = after.nodes.find((node) => node.id === 'layer-0.attention.Q')!;
    const attention = after.nodes.find((node) => node.id === query.parent_id)!;
    if (attention.kind !== 'group') throw new Error('Expected group');
    attention.children.push('identity');
    after.nodes.push({ ...structuredClone(query), id: 'identity', operation: 'identity', parameter_ids: [], references: [] });
    after.edges.find((edge) => edge.target.node_id === query.id)!.target.node_id = 'identity';
    after.edges.push({ id: 'identity-output', source: ep('identity', 'out'), target: ep(query.id, 'x'), kind: 'data', provenance: [] });
    validate(after);
    expect(() => assertSemanticEquivalent(before, after)).toThrow('Operation-level semantics changed');
  });

  it.each(['alias', 'tensor identity', 'logical shape', 'storage geometry', 'fused region'])('preserves %s independently of graph IDs', (change) => {
    const before = structuredClone(contractResponse.graph), after = structuredClone(before);
    const alias = after.parameters.find((parameter) => parameter.id === 'alias')!;
    if (change === 'alias' && alias.binding === 'alias') alias.alias_of = 'quantized';
    if (change === 'tensor identity') alias.inspection = { status: 'available', tensor_id: 'wrong-inventory-tensor' };
    if (change === 'logical shape') alias.logical_shape = [{ kind: 'constant', value: 6 }];
    if (change === 'storage geometry') alias.storage[0]!.shape = [3, 2];
    if (change === 'fused region') { const fused = after.parameters.find((parameter) => parameter.binding === 'fused_region')!;
      if (fused.binding === 'fused_region') fused.region.description = 'Different rows'; }
    expect(() => assertSemanticEquivalent(before, after)).toThrow('Operation-level semantics changed');
    const renamed = structuredClone(before), names = renameProducerIds(renamed);
    assertSemanticEquivalent(before, renamed, undefined, names);
  });

  it('does not conceal model-local partial diagnostics or allow many-to-one correspondence', () => {
    const before = makeProjectionFixture({ count: 1, repetitions: false, partial: true }), after = structuredClone(before);
    after.diagnostics = [];
    expect(() => assertSemanticEquivalent(before, after)).toThrow('Operation-level semantics changed');
    expect(() => semanticSnapshot(before, { nodes: { 'layer-0.prior-conv': 'state', 'layer-0.prior-delta': 'state' } })).toThrow('injective');
  });
});

describe('reusable exhaustive and focused source assertions', () => {
  it.each([
    { variants: ['full_attention', 'full_attention'], hiddenSize: 8 },
    { variants: ['linear_attention', 'full_attention', 'full_attention', 'linear_attention'], hiddenSize: 29 },
    { count: 3, secondStack: 2, hiddenSize: 17 },
    { count: 1, repetitions: false, partial: true },
  ] satisfies ProjectionFixtureOptions[])('retains exact source and multiset paths: %j', (options) => {
    const graph = makeProjectionFixture(options), before = structuredClone(graph);
    for (const repetition of graph.repetitions) for (const instance of [repetition.instances[0]!, repetition.instances.at(-1)!]) {
      assertTraceability(graph, projectGraph(graph, { expanded: ['model', repetition.parent_id, instance.node_id] }));
    }
    assertTraceability(graph, projectGraph(graph, { expanded: [], exhaustive: true }), true);
    expect(graph).toEqual(before);
  });

  it('catches loss of a duplicate signal even when every source-edge ID is still accounted for', () => {
    const graph = fixture(); graph.edges.push({ ...structuredClone(graph.edges[0]!), id: 'parallel-path' });
    graph.edges.push({ ...structuredClone(graph.edges.find((edge) => edge.target.node_id === 'embedding')!), id: 'parallel-exit' });
    const projected = projectGraph(graph, { expanded: [], exhaustive: true });
    assertTraceability(graph, projected, true);
    const edge = projected.edges.find((item) => item.paths.length > 1)!;
    expect(edge.paths).toHaveLength(4); edge.paths.pop();
    expect(new Set(edge.paths.flatMap((path) => path.map((item) => item.id)))).toEqual(new Set(edge.originalEdgeIds));
    expect(() => assertTraceability(graph, projected, true)).toThrow();
  });
});
