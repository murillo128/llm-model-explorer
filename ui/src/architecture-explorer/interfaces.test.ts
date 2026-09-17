import { describe, expect, it } from 'vitest';
import { interfaceFixture } from '../../tests/architecture-interface-fixture';
import { assertTraceability, assertInterfaceCoverage, semanticSnapshot } from '../../tests/architecture-invariants';
import { interfaceIndex } from './interfaces';
import { connectionSet, projectGraph } from './projection';
import { browserIndex } from './browser-model';
import { GraphViews } from './graph';
import { layoutGraph } from './auto-layout';

describe('source declarations become exact container interfaces', () => {
  it.each([false, true])('preserves every expanded isolated boundary, including disconnected ports (many=%s)', (many) => {
    const graph = interfaceFixture('hybrid', many), before = semanticSnapshot(graph);
    const projection = projectGraph(graph, { scope: 'language', expanded: ['language'], showUnused: true });
    const owner = projection.nodes.find((n) => n.id === 'language')!;
    const declared = graph.nodes.find((n) => n.id === 'language')!.ports;
    expect(owner.ports.map((p) => p.id)).toEqual(declared.map((p) => p.id));
    for (const port of declared) expect(owner.ports.find((p) => p.id === port.id)).toMatchObject({
      ...port, endpoints: expect.arrayContaining([{ node_id: 'language', port_id: port.id }]),
    });
    const output = { node_id: 'language', port_id: 'out' };
    const routes = projection.edges.filter((e) => connectionSet(projection, { port: output }).includes(e.id));
    expect(routes).toHaveLength(2);
    expect(routes.flatMap((e) => e.paths.flat()).map((e) => e.id).sort()).toEqual(
      graph.edges.filter((e) => e.source.node_id === 'language' && e.source.port_id === 'out' ||
        e.target.node_id === 'language' && e.target.port_id === 'out').map((e) => e.id).sort());
    if (many) expect(connectionSet(projection, { port: { node_id: 'language', port_id: 'auxiliary_17' } })).toEqual([]);
    assertTraceability(graph, projection);
    expect(semanticSnapshot(graph)).toEqual(before);
  });

  it.each(['dense', 'hybrid', 'visual'] as const)('preserves %s computations and every source record', (kind) => {
    const graph = interfaceFixture(kind), before = semanticSnapshot(graph), index = interfaceIndex(graph);
    const expected = kind === 'visual' ? ['video', 'context_indices', 'target_indices', 'representations'] : ['Token IDs', 'positions', 'mask', 'current_mask', 'logits'];
    expect([...index.declarations.keys()]).toEqual(expected);
    expect(index.notices.size).toBe(0);
    expect(index.outer.kind).toBe(kind === 'dense' ? 'source' : 'model');
    for (const options of [{ expanded: [], exhaustive: true }, { expanded: [] }, { expanded: [], modelCollapsed: true }, { expanded: [], showContext: false }]) {
      const projection = projectGraph(graph, options);
      expect(projection.nodes.some((n) => expected.includes(n.id) || n.id === 'tokenizer')).toBe(false);
      for (const id of expected) expect(projection.nodes.some((n) => n.ports.some((p) => p.interfaces?.includes(id)))).toBe(true);
      assertTraceability(graph, projection, Boolean(options.exhaustive));
    }
    const entries = browserIndex(graph);
    expect(entries.some((e) => expected.includes(e.node.id) || e.node.id === 'tokenizer')).toBe(false);
    expect(entries.some((e) => e.node.id === (kind === 'visual' ? 'predictor' : 'LM head'))).toBe(true);
    expect(semanticSnapshot(graph)).toEqual(before);
  });

  it('reuses exact existing boundary ports without losing declared output descriptors', () => {
    const graph = interfaceFixture('hybrid');
    // Remove the external head/output so the sole computational root owns the graph interface.
    graph.nodes = graph.nodes.filter((n) => n.id !== 'LM head' && n.id !== 'logits');
    graph.edges = graph.edges.filter((e) => e.source.node_id !== 'LM head' && e.target.node_id !== 'LM head' && e.target.node_id !== 'logits');
    const projection = projectGraph(graph, { expanded: [], exhaustive: true });
    const owner = projection.nodes.find((n) => n.id === 'language')!;
    expect(owner.ports.filter((p) => p.interfaces?.includes('positions'))).toHaveLength(1);
    expect(owner.ports.find((p) => p.id === 'positions')?.endpoints).toContainEqual({ node_id: 'positions', port_id: 'out' });
    expect(projection.boundaryPaths).toHaveLength(4);
    assertTraceability(graph, projection, true);
  });

  it('retains real operations with interface-like names and ambiguous declarations', () => {
    const graph = interfaceFixture();
    const node = graph.nodes.find((n) => n.id === 'positions')!;
    node.kind = 'operation'; node.operation = 'generate_positions';
    const mask = graph.nodes.find((n) => n.id === 'mask')!;
    mask.formula = 'construct_mask(x)';
    const index = interfaceIndex(graph);
    expect(index.declarations.has(node.id)).toBe(false);
    expect(index.declarations.has(mask.id)).toBe(false);
    expect(index.notices.has(mask.id)).toBe(true);
    const projection = projectGraph(graph, { expanded: [], exhaustive: true });
    expect(projection.nodes.map((n) => n.id)).toContain('positions');
    expect(projection.nodes.map((n) => n.id)).toContain('mask');
    assertTraceability(graph, projection, true);
  });

  it('keeps connected tokenizer context and distinct state records', () => {
    const graph = interfaceFixture();
    const source = graph.nodes.find((n) => n.id === 'positions')!;
    source.kind = 'state';
    graph.edges.find((e) => e.source.node_id === source.id)!.kind = 'state';
    const tokenizer = graph.nodes.find((n) => n.id === 'tokenizer')!;
    tokenizer.ports = [...source.ports];
    expect(interfaceIndex(graph).tools.size).toBe(0);
    const projection = projectGraph(graph, { expanded: [], exhaustive: true });
    expect(projection.nodes.map((n) => n.id)).toEqual(expect.arrayContaining(['positions', 'tokenizer', 'LM head']));
    assertTraceability(graph, projection, true);
  });

  it('restores a former declaration selection to an owner/port without changing the camera', () => {
    const graph = interfaceFixture(), views = new GraphViews(), view = views.get('m', graph);
    view.selected = 'logits'; view.focus = 'tokenizer'; view.expanded = ['model', 'logits'];
    view.viewport = { x: 73, y: -42, zoom: 0.9 };
    expect(views.get('m', graph)).toBe(view);
    expect(view.selected).toBe('model');
    expect(view.boundary?.endpoints).toEqual([{ node_id: 'logits', port_id: 'x' }]);
    expect(view.viewport).toEqual({ x: 73, y: -42, zoom: 0.9 });
    expect(view.focus).toBeNull(); expect(view.expanded).toEqual(['model']);
  });

  it('keeps nested input anchors and exact hover routes, stopping at computations', () => {
    const graph = interfaceFixture('hybrid'), projection = projectGraph(graph, { expanded: [], exhaustive: true });
    const port = projection.nodes.find((n) => n.presentation === 'model')!.ports.find((p) => p.interfaces?.includes('positions'))!;
    const ids = connectionSet(projection, { port: { node_id: 'presentation:model', port_id: port.id } });
    const edges = projection.edges.filter((e) => ids.includes(e.id));
    expect(edges.flatMap((e) => e.originalEdgeIds)).toEqual(['wire-2', 'wire-3']);
    expect(edges.at(-1)?.target).toEqual({ node_id: 'Embedding', port_id: 'positions' });
    const intermediate = projection.nodes.find((n) => n.id === 'language')!.ports.find((p) => p.id === 'positions')!;
    expect(intermediate.interfaces).toEqual(['positions']);
    expect(intermediate.endpoints).toContainEqual({ node_id: 'positions', port_id: 'out' });
    expect(connectionSet(projection, { edgeId: edges[1]!.id })).toEqual(expect.arrayContaining(ids));
  });

  it('retains a declaration when its forwarded boundary has a competing computational source', () => {
    const graph = interfaceFixture('hybrid');
    graph.edges.push({ ...graph.edges[0]!, id: 'competing', source: { node_id: 'LM head', port_id: 'out' },
      target: { node_id: 'language', port_id: 'positions' } });
    const index = interfaceIndex(graph);
    expect(index.declarations.has('positions')).toBe(false);
    expect(index.notices.has('positions')).toBe(true);
    const projection = projectGraph(graph, { expanded: [], exhaustive: true });
    expect(projection.nodes.some((n) => n.id === 'positions')).toBe(true);
    assertTraceability(graph, projection, true);
  });

  it('keeps disconnected unknown-shape endpoints distinct, including multiple ports on one declaration', () => {
    const graph = interfaceFixture(), declaration = graph.nodes.find((n) => n.id === 'positions')!;
    declaration.ports[0]!.shape = null;
    declaration.ports.push({ ...declaration.ports[0]!, id: 'second', label: 'Second supplied signal' });
    const projection = projectGraph(graph, { expanded: [], exhaustive: true });
    const ports = projection.nodes.find((n) => n.id === 'model')!.ports.filter((p) => p.interfaces?.includes('positions'));
    expect(ports).toHaveLength(2);
    expect(ports.map((p) => p.shape)).toEqual([null, null]);
    expect(new Set(ports.map((p) => p.id)).size).toBe(2);
    expect(connectionSet(projection, { port: { node_id: 'model', port_id: ports[1]!.id } })).toEqual([]);
    expect(graph.nodes.find((n) => n.id === 'logits')!.ports).toHaveLength(2);
    expect(projection.nodes.find((n) => n.id === 'model')!.ports.filter((p) => p.interfaces?.includes('logits'))).toHaveLength(1);
    assertTraceability(graph, projection, true);
  });

  it('does not reuse an output boundary fed by distinct computational signals', () => {
    const graph = interfaceFixture(), model = graph.nodes.find((n) => n.id === 'model')!;
    model.ports.push({ id: 'out', label: 'out', direction: 'output', shape: null });
    graph.edges.find((e) => e.target.node_id === 'logits')!.target = { node_id: 'model', port_id: 'out' };
    graph.edges.push({ ...graph.edges[0]!, id: 'second-producer', source: { node_id: 'Embedding', port_id: 'out' }, target: { node_id: 'model', port_id: 'out' } },
      { ...graph.edges[0]!, id: 'terminal-output', source: { node_id: 'model', port_id: 'out' }, target: { node_id: 'logits', port_id: 'x' } });
    expect(interfaceIndex(graph).declarations.has('logits')).toBe(false);
    expect(interfaceIndex(graph).notices.has('logits')).toBe(true);
    assertTraceability(graph, projectGraph(graph, { expanded: [], exhaustive: true }), true);
  });

  it('retains every port with generated geometry for many interfaces', async () => {
    const graph = interfaceFixture('hybrid', true);
    for (const dimensions of [false, true]) {
      const layout = await layoutGraph(graph, { expanded: [], exhaustive: true, dimensions });
      assertInterfaceCoverage(graph, layout.projection);
      const model = layout.projection.nodes.find((n) => n.presentation === 'model')!;
      expect(model.ports).toHaveLength(23);
      const box = layout.boxes.find((b) => b.id === model.id)!;
      for (const port of layout.ports.filter((p) => p.nodeId === model.id)) expect(port.x).toBe(port.side === 'left' ? 0 : box.width);
      const positions = layout.ports.filter((p) => p.nodeId === model.id && p.side === 'left').map((p) => p.y).sort((a,b) => a-b);
      for (let i=1;i<positions.length;i++) expect(positions[i]! - positions[i-1]!).toBeGreaterThanOrEqual(20);
      const ordered = layout.ports.filter((p) => p.nodeId === model.id && p.side === 'left').sort((a,b) => a.y-b.y).map((p) => p.portId);
      expect(ordered).toEqual(model.ports.filter((p) => p.direction === 'input').map((p) => p.id));
      for (const edge of layout.projection.edges) {
        const route = layout.routes.find((r) => r.id === edge.id)!;
        for (const endpoint of [edge.source, edge.target]) {
          const port = layout.ports.find((p) => p.nodeId === endpoint.node_id && p.portId === endpoint.port_id)!;
          expect(route.sections.flatMap((s) => [s[0], s.at(-1)])).toContainEqual({ x: port.absoluteX, y: port.absoluteY });
        }
        for (const section of route.sections) for (let i = 1; i < section.length; i++) {
          expect(section[i]!.x === section[i-1]!.x || section[i]!.y === section[i-1]!.y).toBe(true);
        }
      }
    }
  });
});
