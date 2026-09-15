import { describe, expect, it } from 'vitest';
import { assertTraceability } from '../../tests/architecture-invariants';
import { makeProjectionFixture } from '../../tests/architecture-projection-fixture';
import { makeExplicitFixture } from '../../tests/architecture-explicit-fixture';
import type { Graph } from './graph';
import { connectionSet, projectGraph } from './projection';
import { componentScope } from './scope';
import { validateArchitecture } from '../api/architecture-validation';

const ep = (node_id: string, port_id: string) => ({ node_id, port_id });
const sourcePaths = (graph: Graph, scope: string) => projectGraph(graph, { scope, expanded: [scope], showUnused: true });
const validate = (graph: Graph) => validateArchitecture({ status: 'available', model_id: 'fixture', diagnostics: [], graph },
  { modelId: 'fixture', inventory: { tensors: [], coverage: 'complete', diagnostics: [] }, tokenizerAvailable: false });

describe('isolated component projection', () => {
  it.each(['layer-3', 'layer-3.attention', 'layer-3.attention.core', 'mlp:layer-3.gate'])(
    'keeps exact source objects and accounts for excluded records in %s', (scope) => {
      const graph = makeProjectionFixture({ count: 4 });
      const before = JSON.stringify(graph);
      const global = projectGraph(graph, { expanded: ['model', 'layer-3'] });
      const projection = sourcePaths(graph, scope);
      assertTraceability(graph, projection);
      const members = new Set(projection.scope!.nodeIds), excluded = new Set(projection.scope!.excludedNodeIds);
      expect([...members].some((id) => excluded.has(id))).toBe(false);
      expect(new Set([...members, ...excluded])).toEqual(new Set(graph.nodes.map((node) => node.id)));
      expect(projection.nodes.filter((node) => node.presentation !== 'external').flatMap((node) => node.sourceIds)
        .every((id) => members.has(id))).toBe(true);
      expect(projection.nodes.find((node) => node.id === scope)?.parentId).toBeUndefined();
      const represented = new Set(projection.edges.flatMap((edge) => edge.originalEdgeIds));
      for (const edge of graph.edges) {
        const crossing = members.has(edge.source.node_id) !== members.has(edge.target.node_id);
        if (crossing) expect(represented.has(edge.id), edge.id).toBe(true);
        const outside = !members.has(edge.source.node_id) && !members.has(edge.target.node_id);
        expect(projection.scope!.excludedEdgeIds.includes(edge.id)).toBe(outside);
        if (outside) expect(represented.has(edge.id)).toBe(false);
      }
      expect(JSON.stringify(graph)).toBe(before);
      expect(projectGraph(graph, { expanded: ['model', 'layer-3'], scope: undefined })).toEqual(global);
    });

  it('retains true external fan-out, mask/positions and independent K/V states', () => {
    const graph = makeProjectionFixture({ count: 4 }), projection = sourcePaths(graph, 'layer-3.attention');
    const branches = projection.edges.filter((edge) => edge.paths.some((path) => path[0]!.source.node_id === 'layer-3.input-norm'));
    expect(branches.map((edge) => edge.paths[0]!.at(-1)!.target.node_id).sort()).toEqual(
      ['layer-3.attention.K', 'layer-3.attention.Q', 'layer-3.attention.V']);
    expect(new Set(branches.map((edge) => edge.source.node_id)).size).toBe(1);
    expect(connectionSet(projection, { port: branches[0]!.source }).sort()).toEqual(branches.map((edge) => edge.id).sort());
    for (const edge of branches) expect(connectionSet(projection, { port: edge.target })).toEqual([edge.id]);
    const paths = projection.edges.flatMap((edge) => edge.paths.map((path) => [path[0]!.source, path.at(-1)!.target]));
    expect(paths).toEqual(expect.arrayContaining([
      [ep('layer-3.prior-K', 'out'), ep('layer-3.attention.core', 'prior_K')],
      [ep('layer-3.prior-V', 'out'), ep('layer-3.attention.core', 'prior_V')],
      [ep('layer-3.attention.core', 'next_K'), ep('layer-3.next-K', 'x')],
      [ep('layer-3.attention.core', 'next_V'), ep('layer-3.next-V', 'x')],
      [ep('layer-3', 'positions'), ep('layer-3.attention.rope-Q', 'positions')],
      [ep('layer-3', 'mask'), ep('layer-3.attention.core', 'mask')],
    ]));
    expect(projection.scope!.excludedNodeIds).toContain('layer-3.residual-1');
    const residual = graph.edges.find((edge) => edge.target.node_id === 'layer-3.residual-1' && edge.target.port_id === 'residual')!;
    expect(projection.scope!.excludedEdgeIds).toContain(residual.id);
    assertTraceability(graph, projection);
  });

  it('keeps equal-shaped leaf inputs distinct and never pulls in a containing derived group', () => {
    const graph = makeProjectionFixture({ count: 4 });
    const core = sourcePaths(graph, 'layer-3.attention.core');
    const inputs = core.edges.filter((edge) => ['Q', 'K', 'V'].includes(edge.target.port_id));
    expect(inputs).toHaveLength(3); expect(new Set(inputs.map((edge) => edge.source.node_id)).size).toBe(3);
    const leaf = sourcePaths(graph, 'layer-3.gate');
    expect(leaf.nodes.filter((node) => node.presentation !== 'external').map((node) => node.id)).toEqual(['layer-3.gate']);
    expect(leaf.nodes.some((node) => node.presentation === 'mlp')).toBe(false);
    assertTraceability(graph, core); assertTraceability(graph, leaf);
  });

  it.each(['encoder.layer-1.attention', 'predictor.layer-1.mlp'])(
    'uses exact authored components in independent stacks: %s', (scope) => {
      const graph = makeExplicitFixture({ count: 3, secondStack: 2 });
      const projection = sourcePaths(graph, scope);
      expect(projection.nodes.some((node) => node.presentation === 'mlp' || node.instances)).toBe(false);
      const members = projection.nodes.filter((node) => node.record?.kind === 'operation');
      expect(members.length).toBeGreaterThan(0);
      expect(members.every((node) => node.record!.parent_id === scope)).toBe(true);
      expect(members.flatMap((node) => node.record!.parameter_ids).every((id) => id.includes(scope.split('.')[0]!))).toBe(true);
      assertTraceability(graph, projection);
    });

  it('preserves reversible filters and valid partial/no-repetition scopes', () => {
    const graph = makeProjectionFixture({ count: 1, repetitions: false, partial: true });
    const options = { scope: 'layer-0', expanded: ['layer-0', 'layer-0.attention'] };
    const filtered = projectGraph(graph, options), complete = projectGraph(graph, { ...options, showUnused: true });
    expect(filtered.filteredEdgeIds.length).toBeGreaterThan(complete.filteredEdgeIds.length);
    expect(complete.unusedInputs).toContainEqual(ep('layer-0', 'positions'));
    expect(graph.nodes.find((node) => node.id === 'layer-0')!.ports.some((port) => port.id === 'positions')).toBe(true);
    const state = projectGraph(graph, { ...options, stateScope: 'layer-0' });
    expect(state.edges).toHaveLength(4); expect(state.edges.every((edge) => edge.kind === 'state')).toBe(true);
    for (const projection of [filtered, complete, state, sourcePaths(graph, 'unknown-component')]) assertTraceability(graph, projection);
  });

  it('does not splice an external re-entry or bypass into an internal path', () => {
    const ports = [{ id: 'in', label: 'in', direction: 'input' as const, shape: [] },
      { id: 'out', label: 'out', direction: 'output' as const, shape: [] }];
    const base = { ports, parameter_ids: [], references: [], attributes: [], provenance: [] };
    const graph: Graph = { graph_id: 'reentrant', scope: 'language_model', coverage: 'partial', symbols: [], parameters: [], repetitions: [], diagnostics: [],
      nodes: [{ ...base, id: 'component', kind: 'group', label: 'Component', children: ['a', 'b'] },
        { ...base, id: 'outside', kind: 'group', label: 'External boundary', children: ['external-op'] },
        { ...base, id: 'external-op', kind: 'operation', label: 'External operation', parent_id: 'outside' },
        { ...base, id: 'external-sink', kind: 'output', label: 'External sink' },
        ...['a', 'b'].map((id) => ({ ...base, id, kind: 'operation' as const, label: id, parent_id: 'component' }))],
      edges: [
        { id: 'exit-internal', source: ep('a', 'out'), target: ep('component', 'out'), kind: 'state', provenance: [] },
        { id: 'exit', source: ep('component', 'out'), target: ep('outside', 'in'), kind: 'state', provenance: [] },
        { id: 'external-in', source: ep('outside', 'in'), target: ep('external-op', 'in'), kind: 'state', provenance: [] },
        { id: 'external-out', source: ep('external-op', 'out'), target: ep('outside', 'out'), kind: 'state', provenance: [] },
        { id: 'reenter', source: ep('outside', 'out'), target: ep('component', 'in'), kind: 'state', provenance: [] },
        { id: 'reenter-internal', source: ep('component', 'in'), target: ep('b', 'in'), kind: 'state', provenance: [] },
        { id: 'bypass', source: ep('outside', 'out'), target: ep('external-sink', 'in'), kind: 'data', provenance: [] },
      ] };
    validate(graph);
    const projection = sourcePaths(graph, 'component');
    expect(projection.edges.flatMap((edge) => edge.paths.map((path) => path.map((item) => item.id)))).toEqual([['exit-internal', 'exit'], ['reenter', 'reenter-internal']]);
    expect(projection.scope!.excludedEdgeIds).toEqual(['external-in', 'external-out', 'bypass']);
    expect(projection.edges.some((edge) => edge.source.node_id === 'a' && edge.target.node_id === 'b')).toBe(false);
    expect(projection.nodes.filter((node) => node.presentation === 'external')).toHaveLength(2);
    assertTraceability(graph, projection);
  });

  it('applies the reversible context filter to the original external node kind', () => {
    const base = { parameter_ids: [], references: [], attributes: [], provenance: [] };
    const input = { id: 'in', label: 'in', direction: 'input' as const, shape: [] };
    const output = { id: 'out', label: 'out', direction: 'output' as const, shape: [] };
    const graph: Graph = { graph_id: 'context-filter', scope: 'language_model', coverage: 'partial', symbols: [], parameters: [], repetitions: [], diagnostics: [],
      nodes: [{ ...base, id: 'context', kind: 'context', label: 'Context', ports: [output] },
        { ...base, id: 'component', kind: 'group', label: 'Component', ports: [input], children: ['op'] },
        { ...base, id: 'op', kind: 'operation', label: 'Operation', ports: [input], parent_id: 'component' }],
      edges: [{ id: 'enter', source: ep('context', 'out'), target: ep('component', 'in'), kind: 'context', provenance: [] },
        { id: 'use', source: ep('component', 'in'), target: ep('op', 'in'), kind: 'context', provenance: [] }] };
    validate(graph);
    for (const scope of [undefined, 'component']) {
      const options = { scope, expanded: ['component'], showUnused: true };
      const shown = projectGraph(graph, { ...options, showContext: true });
      const hidden = projectGraph(graph, { ...options, showContext: false });
      expect(shown.edges).toHaveLength(1); expect(hidden.edges).toHaveLength(0);
      expect(hidden.filteredEdgeIds).toEqual(['enter', 'use']);
      expect(hidden.nodes.some((node) => node.presentation === 'external')).toBe(false);
      expect(projectGraph(graph, { ...options, showContext: true })).toEqual(shown);
      assertTraceability(graph, shown); assertTraceability(graph, hidden);
    }
  });

  it('rejects obsolete scopes rather than substituting a guessed component', () => {
    const graph = makeProjectionFixture({ count: 1 });
    expect(() => componentScope(graph, 'missing')).toThrow('no longer available');
    expect(() => componentScope(graph, 'mlp:missing')).toThrow('no longer available');
  });
});
