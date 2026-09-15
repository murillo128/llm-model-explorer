import { describe, expect, it } from 'vitest';
import { assertTraceability } from '../../tests/architecture-invariants';
import { makeProjectionFixture, type ProjectionFixtureOptions } from '../../tests/architecture-projection-fixture';
import { validateArchitecture } from '../api/architecture-validation';
import { deriveMlpGroups } from './derived-groups';
import { connectionSet, endpointKey, projectGraph, type Endpoint, type Projection } from './projection';

const ep = (node_id: string, port_id: string): Endpoint => ({ node_id, port_id });
const ids = (projection: Projection) => projection.nodes.map((n) => n.id);
const fixture = (options: ProjectionFixtureOptions = {}) => makeProjectionFixture({ count: 4, ...options });
const selected = (index: number, extra: string[] = []) => ({ expanded: ['model', `layer-${index}`, ...extra] });
function freeze(value: unknown) {
  if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); }
}
describe('source-preserving visible projection', () => {
  it.each([
    {}, { variants: ['full_attention', 'linear_attention', 'full_attention'] },
    { count: 5, secondStack: 2, hiddenSize: 29 }, { count: 2, repetitions: false, partial: true },
  ] satisfies ProjectionFixtureOptions[])('validates authored fixtures against the actual API contract: %j', (options) => {
    const graph = makeProjectionFixture(options);
    expect(() => validateArchitecture({ model_id: 'authored-model', status: 'available', diagnostics: [], graph }, {
      modelId: 'authored-model', tokenizerAvailable: !options.secondStack,
      inventory: { tensors: [], coverage: 'partial', diagnostics: [] },
    })).not.toThrow();
    expect(new Set(graph.nodes.map((n) => n.id)).size).toBe(graph.nodes.length);
    expect(new Set(graph.parameters.map((p) => p.id)).size).toBe(graph.parameters.length);
  });

  it('starts with one truthful 24-instance compact stack and excludes all descendants', () => {
    const graph = makeProjectionFixture(), projected = projectGraph(graph, { expanded: ['model'] });
    const stack = projected.nodes.find((n) => n.presentation === 'repetition')!;
    expect(stack.label).toBe('Decoder layers ×24'); expect(stack.instances).toEqual(graph.repetitions[0]!.instances);
    expect(stack.summary).toBe('18 linear attention · 6 full attention');
    expect(stack.sourceIds).toEqual(Array.from({ length: 24 }, (_, i) => `layer-${i}`));
    expect(projected.nodes.some((n) => n.id.startsWith('layer-'))).toBe(false);
    expect(ids(projected)).toEqual(expect.arrayContaining(['tokenizer', 'model', 'embedding', 'final-norm', 'head']));
    expect(projected.nodes.find((n) => n.id === 'tokenizer')!.ports).toEqual([]);
    expect(projected.edges.some((e) => e.source.node_id === 'tokenizer' || e.target.node_id === 'tokenizer')).toBe(false);
    expect(projected.nodes.length).toBeLessThan(15); assertTraceability(graph, projected);
  });

  it('round-trips compact, selected, window and exhaustive states without mutating source records', () => {
    const graph = fixture(), before = JSON.stringify(graph); freeze(graph);
    const compact = projectGraph(graph, { expanded: ['model'] });
    for (const options of [selected(0), selected(3, ['layer-3.attention']),
      { expanded: ['model'], repetitions: { 'decoder-layers': { start: 1, count: 2 } } },
      { expanded: [], exhaustive: true }]) assertTraceability(graph, projectGraph(graph, options));
    expect(JSON.stringify(graph)).toBe(before);
    expect(projectGraph(graph, { expanded: ['model'] })).toEqual(compact);
  });

  it('selects actual first/last instances with different weights, auxiliaries and owned state topology', () => {
    const graph = fixture();
    const first = projectGraph(graph, selected(0)), last = projectGraph(graph, selected(3));
    expect(first.nodes.filter((n) => n.presentation === 'range').flatMap((n) => n.sourceIds)).toEqual(['layer-1', 'layer-2', 'layer-3']);
    expect(last.nodes.filter((n) => n.presentation === 'range').flatMap((n) => n.sourceIds)).toEqual(['layer-0', 'layer-1', 'layer-2']);
    expect(first.nodes.find((n) => n.id === 'layer-0.input-norm')!.record!.parameter_ids).not.toEqual(
      last.nodes.find((n) => n.id === 'layer-3.input-norm')!.record!.parameter_ids);
    expect(first.nodes.filter((n) => n.kind === 'state').map((n) => n.id)).toEqual(['layer-0.prior-conv', 'layer-0.next-conv', 'layer-0.prior-delta', 'layer-0.next-delta']);
    expect(last.nodes.filter((n) => n.kind === 'state').map((n) => n.id)).toEqual(['layer-3.prior-K', 'layer-3.next-K', 'layer-3.prior-V', 'layer-3.next-V']);
    expect(first.unusedInputs).toContainEqual(ep('layer-0', 'positions')); expect(first.unusedInputs).toContainEqual(ep('layer-0', 'mask'));
    expect(last.unusedInputs).toContainEqual(ep('layer-3', 'current_mask'));
    expect(first.unusedInputs).not.toContainEqual(ep('layer-0', 'current_mask'));
    expect(last.unusedInputs).not.toContainEqual(ep('layer-3', 'positions'));
    for (const [projection, instance] of [[first, 0], [last, 3]] as const) {
      expect(projection.nodes.filter((n) => n.record?.parent_id?.startsWith('layer-')).every((n) => n.record!.parent_id!.startsWith(`layer-${instance}`))).toBe(true);
    }
  });

  it('uses configurable contiguous windows without rewriting non-periodic variant order', () => {
    const graph = fixture({ count: 7, variants: ['full_attention', 'linear_attention', 'full_attention', 'full_attention', 'linear_attention', 'linear_attention', 'full_attention'] });
    const projection = projectGraph(graph, { expanded: ['model'], repetitions: { 'decoder-layers': { start: 2, count: 3 } } });
    expect(projection.nodes.filter((n) => n.presentation === 'range').map((n) => n.instances!.map((i) => i.index))).toEqual([[0, 1], [5, 6]]);
    expect(projection.nodes.filter((n) => /^layer-\d$/.test(n.id)).map((n) => n.id)).toEqual(['layer-2', 'layer-3', 'layer-4']);
    expect(projection.nodes.find((n) => n.instances?.[0]?.index === 0)!.instances!.map((i) => i.variant)).toEqual(['full_attention', 'linear_attention']);
    for (const start of [-20, 6, 99]) {
      const boundary = projectGraph(graph, { expanded: ['model'], repetitions: { 'decoder-layers': { start, count: 2 } } });
      expect(boundary.nodes.flatMap((n) => n.instances?.map((i) => i.node_id) ?? (/^layer-\d$/.test(n.id) ? [n.id] : []))).toEqual(graph.repetitions[0]!.instances.map((i) => i.node_id));
      assertTraceability(graph, boundary);
    }
  });

  it('controls two independent repetitions without a tokenizer or implicit cross-stack state', () => {
    const graph = fixture({ count: 3, secondStack: 2 });
    const projection = projectGraph(graph, { expanded: ['model', 'encoder', 'predictor', 'encoder.layer-1'],
      repetitions: { 'predictor-layers': { start: 1, count: 1 } } });
    expect(graph.repetitions.map((r) => r.id)).toEqual(['encoder-layers', 'predictor-layers']);
    expect(graph.nodes.some((n) => n.references.some((r) => r.kind === 'tokenizer'))).toBe(false);
    expect(ids(projection)).toEqual(expect.arrayContaining(['encoder.layer-1', 'predictor.layer-1', 'representations']));
    expect(ids(projection)).not.toContain('tokenizer'); expect(ids(projection)).not.toContain('head');
    expect(graph.edges.filter((e) => e.kind === 'state').every((e) => e.source.node_id.split('.')[0] === e.target.node_id.split('.')[0])).toBe(true);
    assertTraceability(graph, projection);
  });

  it('handles graphs without repetitions and explicit partial/unknown regions', () => {
    const graph = fixture({ count: 2, repetitions: false, partial: true });
    const projection = projectGraph(graph, { expanded: ['model'] });
    expect(projection.nodes.some((n) => n.presentation === 'repetition' || n.presentation === 'range')).toBe(false);
    expect(ids(projection)).toEqual(expect.arrayContaining(['layer-0', 'layer-1', 'unknown-component']));
    expect(projection.nodes.find((n) => n.id === 'unknown-component')!.record!.description).toContain('unknown');
    expect(graph.coverage).toBe('partial'); expect(graph.diagnostics[0]!.node_id).toBe('unknown-component');
    assertTraceability(graph, projection);
  });

  it('keeps an interleaved unrelated sibling outside compact repetition ranges', () => {
    const graph = fixture({ count: 3 });
    const parent = graph.nodes.find((n) => n.id === 'model')!;
    if (parent.kind !== 'group') throw new Error('Expected model group');
    parent.children.splice(parent.children.indexOf('layer-0') + 1, 0, 'independent-context');
    graph.nodes.push({ id: 'independent-context', parent_id: parent.id, kind: 'context', label: 'Independent context',
      ports: [], parameter_ids: [], references: [], attributes: [], provenance: [] });
    const projection = projectGraph(graph, { expanded: ['model'] });
    expect(ids(projection)).toContain('independent-context');
    expect(projection.nodes.filter((n) => n.presentation === 'range').map((n) => n.sourceIds)).toEqual([
      ['layer-0'], ['layer-1', 'layer-2'],
    ]);
    expect(projection.nodes.flatMap((n) => n.sourceIds).filter((id) => id === 'independent-context')).toHaveLength(1);
    assertTraceability(graph, projection);
  });

  it('exhaustively exposes every source record and original edge, overriding detail filters', () => {
    const graph = makeProjectionFixture();
    const projection = projectGraph(graph, { expanded: [], exhaustive: true, stateScope: 'layer-0', showContext: false });
    expect(new Set(ids(projection))).toEqual(new Set(graph.nodes.map((n) => n.id)));
    expect(new Set(projection.edges.flatMap((e) => e.originalEdgeIds))).toEqual(new Set(graph.edges.map((e) => e.id)));
    expect(projection.nodes.some((n) => n.presentation)).toBe(false);
    expect(projection.hiddenEdgeIds).toEqual([]); expect(projection.filteredEdgeIds).toEqual([]);
    assertTraceability(graph, projection, true);
  });

  it('preserves bypass dependencies across compressed before/after ranges without a false middle endpoint', () => {
    const graph = fixture({ count: 5 });
    const first = graph.nodes.find((n) => n.id === 'layer-0')!, last = graph.nodes.find((n) => n.id === 'layer-4')!;
    const shape = first.ports[0]!.shape;
    first.ports.push({ id: 'skip_out', direction: 'output', label: 'skip', shape });
    last.ports.push({ id: 'skip_in', direction: 'input', label: 'skip', shape });
    graph.edges.push({ id: 'bypass-internal-source', source: ep('layer-0.input-norm', 'out'), target: ep(first.id, 'skip_out'), kind: 'data', provenance: [] },
      { id: 'bypass-external', source: ep(first.id, 'skip_out'), target: ep(last.id, 'skip_in'), kind: 'data', provenance: [] },
      { id: 'bypass-internal-target', source: ep(last.id, 'skip_in'), target: ep('layer-4.residual-1', 'residual'), kind: 'data', provenance: [] });
    const projection = projectGraph(graph, selected(2));
    const bypass = projection.edges.find((e) => e.originalEdgeIds.includes('bypass-external'))!;
    expect(projection.nodes.find((n) => n.id === bypass.source.node_id)!.sourceIds).toEqual(['layer-0', 'layer-1']);
    expect(projection.nodes.find((n) => n.id === bypass.target.node_id)!.sourceIds).toEqual(['layer-3', 'layer-4']);
    expect(bypass.paths).toEqual([[graph.edges.find((e) => e.id === 'bypass-external')!]]);
    expect([bypass.source.node_id, bypass.target.node_id]).not.toContain('layer-2'); assertTraceability(graph, projection);
  });

  it('composes nested boundary forwarding once and retains exact port signal identities', () => {
    const graph = fixture();
    const projection = projectGraph(graph, selected(3, ['layer-3.attention']));
    const positions = projection.edges.filter((e) => e.source.node_id === 'positions' && e.target.node_id.startsWith('layer-3.attention.rope-'));
    expect(positions.map((e) => e.target.node_id).sort()).toEqual(['layer-3.attention.rope-K', 'layer-3.attention.rope-Q']);
    for (const edge of positions) {
      expect(edge.paths).toHaveLength(1); expect(edge.paths[0]).toHaveLength(4);
      expect(edge.paths[0]!.map((e) => e.target.node_id)).toEqual(['model', 'layer-3', 'layer-3.attention', edge.target.node_id]);
    }
    const keys = projection.edges.map((e) => JSON.stringify([e.source, e.target, e.kind]));
    expect(new Set(keys).size).toBe(keys.length);
    expect(projection.nodes.find((n) => n.id === 'layer-3.attention.core')!.ports.filter((p) => ['Q', 'K', 'V'].includes(p.id))).toHaveLength(3);
    assertTraceability(graph, projection);
  });

  it('targets exact input/output ports and individual fan-out branches without traversing operations', () => {
    const projection = projectGraph(fixture(), selected(3, ['layer-3.attention']));
    const branches = projection.edges.filter((e) => e.source.node_id === 'layer-3.input-norm');
    expect(branches.map((e) => e.target.node_id).sort()).toEqual(['layer-3.attention.K', 'layer-3.attention.Q', 'layer-3.attention.V']);
    expect(connectionSet(projection, { port: ep('layer-3.input-norm', 'out') }).sort()).toEqual(branches.map((e) => e.id).sort());
    for (const branch of branches) {
      expect(connectionSet(projection, { edgeId: branch.id })).toEqual([branch.id]);
      expect(connectionSet(projection, { port: branch.target })).toEqual([branch.id]);
    }
    const core = 'layer-3.attention.core';
    for (const port of ['Q', 'K', 'V', 'prior_K', 'prior_V']) {
      const edges = connectionSet(projection, { port: ep(core, port) });
      expect(edges).toHaveLength(1);
      expect(projection.edges.find((e) => e.id === edges[0])!.target).toEqual(ep(core, port));
    }
    const inputs = connectionSet(projection, { port: ep(core, 'Q') });
    expect(inputs.some((id) => connectionSet(projection, { port: ep(core, 'out') }).includes(id))).toBe(false);
    expect(connectionSet(projection, { port: ep(core, 'missing') })).toEqual([]);
  });

  it('keeps the two residual additions and separate prior/next K/V routes independently addressable', () => {
    const graph = fixture(), projection = projectGraph(graph, selected(3, ['layer-3.attention']));
    for (const index of [1, 2]) {
      const node = `layer-3.residual-${index}`;
      const residual = connectionSet(projection, { port: ep(node, 'residual') });
      const update = connectionSet(projection, { port: ep(node, 'update') });
      expect(residual).toHaveLength(1); expect(update).toHaveLength(1); expect(residual).not.toEqual(update);
    }
    const states = projectGraph(graph, { ...selected(3, ['layer-3.attention']), stateScope: 'layer-3' });
    expect(states.edges).toHaveLength(4); expect(states.edges.every((e) => e.kind === 'state')).toBe(true);
    expect(states.edges.map((e) => [e.source.node_id, e.target.node_id])).toEqual(expect.arrayContaining([
      ['layer-3.prior-K', 'layer-3.attention.core'], ['layer-3.attention.core', 'layer-3.next-K'],
      ['layer-3.prior-V', 'layer-3.attention.core'], ['layer-3.attention.core', 'layer-3.next-V'],
    ]));
    expect(states.filteredEdgeIds.length).toBeGreaterThan(0); assertTraceability(graph, states);
  });

  it('computes auxiliary consumption from source edges and exposes suppressed declared interfaces', () => {
    const graph = fixture();
    // Deliberately misleading variant metadata must not change source consumption.
    graph.repetitions[0]!.instances[0]!.variant = 'full_attention';
    const compact = projectGraph(graph, selected(0)), detail = projectGraph(graph, { ...selected(0), showUnused: true });
    expect(compact.unusedInputs).toContainEqual(ep('layer-0', 'positions'));
    expect(compact.unusedInputs).not.toContainEqual(ep('layer-0', 'current_mask'));
    expect(compact.filteredEdgeIds.length).toBeGreaterThan(detail.filteredEdgeIds.length);
    expect(detail.edges.some((e) => endpointKey(e.target) === endpointKey(ep('layer-0', 'positions')))).toBe(true);
    expect(graph.nodes.find((n) => n.id === 'layer-0')!.ports.map((p) => p.id)).toEqual(['x', 'positions', 'mask', 'current_mask', 'out']);
    assertTraceability(graph, compact); assertTraceability(graph, detail);
  });
});

describe('justified derived MLP presentation', () => {
  it('groups exactly five supported operations, exposes real selected references and reverses without changing edges', () => {
    const graph = fixture();
    const groups = deriveMlpGroups(graph); expect(groups).toHaveLength(4);
    const mlp = groups.find((g) => g.parentId === 'layer-3')!;
    expect(mlp.sourceIds).toEqual(['layer-3.gate', 'layer-3.up', 'layer-3.silu', 'layer-3.multiply', 'layer-3.down']);
    const compact = projectGraph(graph, selected(3)), expanded = projectGraph(graph, selected(3, [mlp.id]));
    expect(compact.nodes.find((n) => n.id === mlp.id)!.presentation).toBe('mlp');
    expect(compact.nodes.some((n) => mlp.sourceIds.includes(n.id))).toBe(false);
    expect(expanded.nodes.filter((n) => n.parentId === mlp.id).map((n) => n.id)).toEqual(mlp.sourceIds);
    const gate = expanded.nodes.find((n) => n.id === 'layer-3.gate')!.record!;
    expect(graph.parameters.find((p) => p.id === gate.parameter_ids[0])!.name).toBe('model.layers.3.mlp.gate_proj.weight');
    const original = projectGraph(graph, { ...selected(3), deriveMlp: false });
    expect(original.nodes.some((n) => n.presentation === 'mlp')).toBe(false);
    expect(original.nodes.filter((n) => mlp.sourceIds.includes(n.id)).every((n) => n.parentId === 'layer-3')).toBe(true);
    for (const projection of [compact, expanded, original]) assertTraceability(graph, projection);
  });

  it.each(['label-only', 'different-signal', 'same-multiply-port', 'extra-gate-consumer', 'different-module'])(
    'leaves unsupported structure explicit: %s', (change) => {
      const graph = fixture({ count: 1 });
      if (change === 'label-only') graph.parameters.forEach((p, i) => { p.name = `weight.${i}`; });
      if (change === 'different-signal') graph.edges.find((e) => e.target.node_id === 'layer-0.up')!.source = ep('layer-0.residual-1', 'out');
      if (change === 'same-multiply-port') graph.edges.find((e) => e.source.node_id === 'layer-0.up')!.target.port_id = 'gate';
      if (change === 'extra-gate-consumer') graph.edges.push({ id: 'extra-consumer', source: ep('layer-0.gate', 'out'), target: ep('layer-0.residual-2', 'update'), kind: 'data', provenance: [] });
      if (change === 'different-module') graph.parameters.find((p) => p.name.endsWith('.mlp.up_proj.weight'))!.name = 'other.layers.0.mlp.up_proj.weight';
      expect(deriveMlpGroups(graph)).toEqual([]);
      const projection = projectGraph(graph, selected(0));
      expect(ids(projection)).toEqual(expect.arrayContaining(['layer-0.gate', 'layer-0.up', 'layer-0.silu', 'layer-0.multiply', 'layer-0.down']));
    });
});
