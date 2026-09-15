import { describe, expect, it } from 'vitest';
import { connectionHitResolver } from './connection-hit';
import type { Route } from './graph';
import type { ProjectedEdge } from './projection';

const edge = (id: string, node = 'norm', port = 'out'): ProjectedEdge => ({ id,
  source: { node_id: node, port_id: port }, target: { node_id: id, port_id: 'x' },
  kind: 'data', paths: [], originalEdgeIds: [],
});
const route = (id: string, ...sections: number[][][]): Route => ({ id, junctions: [],
  sections: sections.map((points) => points.map(([x, y]) => ({ x: x!, y: y! }))),
});

describe('generated connection hits', () => {
  const edges = ['Q', 'K', 'V', 'side'].map((id) => edge(id));
  const routes = [
    route('Q', [[0, 0], [10, 0], [10, -20], [40, -20]]),
    // Subdivision and reversed section order cannot change the shared signal.
    route('K', [[0, 0], [5, 0]], [[40, 0], [5, 0]]),
    route('V', [[0, 0], [10, 0], [10, 20], [40, 20]]),
    route('side', [[0, 0], [0, 40], [40, 40]]),
  ];
  it('resolves the trunk to its represented branches regardless of SVG owner', () => {
    const hit = connectionHitResolver(edges, routes);
    for (const id of ['Q', 'K', 'V']) {
      expect(hit(id, { x: 7, y: 0 })).toEqual(['Q', 'K', 'V']);
      // CSS line hit targets extend beyond the exact centerline, also at zoom.
      expect(hit(id, { x: 7, y: 2 })).toEqual(['Q', 'K', 'V']);
    }
  });
  it('keeps exclusive branches individually targetable after the split', () => {
    const hit = connectionHitResolver(edges, routes);
    expect(hit('Q', { x: 20, y: -20 })).toEqual(['Q']);
    expect(hit('K', { x: 20, y: 0 })).toEqual(['K']);
    expect(hit('V', { x: 20, y: 20 })).toEqual(['V']);
    expect(hit('side', { x: 20, y: 40 })).toEqual(['side']);
  });
  it('highlights only branches represented in a partially shared segment', () => {
    const hit = connectionHitResolver(edges.slice(0, 3), [
      route('Q', [[0, 0], [10, 0], [10, 20]]),
      route('K', [[0, 0], [20, 0], [20, 20]]),
      route('V', [[0, 0], [30, 0], [30, 20]]),
    ]);
    expect(hit('V', { x: 5, y: 0 })).toEqual(['Q', 'K', 'V']);
    expect(hit('V', { x: 15, y: 0 })).toEqual(['K', 'V']);
    expect(hit('V', { x: 25, y: 0 })).toEqual(['V']);
  });
  it('does not join different source ports/nodes, nearby lines or point crossings', () => {
    const hit = connectionHitResolver([
      edge('owner'), edge('other-port', 'norm', 'another'), edge('other-node', 'other'),
      edge('near'), edge('crossing'), edge('touching'),
    ], [
      route('owner', [[0, 0], [40, 0]]),
      route('other-port', [[0, 0], [40, 0]]), route('other-node', [[0, 0], [40, 0]]),
      route('near', [[0, 1], [40, 1]]), route('crossing', [[20, -10], [20, 10]]),
      route('touching', [[40, 0], [50, 0]]),
    ]);
    expect(hit('owner', { x: 20, y: 0 })).toEqual(['owner']);
    expect(hit('owner', { x: 40, y: 0 })).toEqual(['owner']);
  });
  it('uses exact projected aliases and handles vertical, zero-length and missing routes', () => {
    const hit = connectionHitResolver([edge('a', 'range', 'output:source'), edge('b', 'range', 'output:source'),
      edge('different', 'range', 'output:other')], [
      route('a', [[0, 0], [0, 0], [0, 40]]), route('b', [[0, 10], [0, 30]]),
      route('different', [[0, 0], [0, 40]]),
    ]);
    expect(hit('a', { x: 0, y: 20 })).toEqual(['a', 'b']);
    expect(hit('a', { x: 0, y: 5 })).toEqual(['a']);
    expect(hit('missing', { x: 0, y: 20 })).toEqual(['missing']);
  });
});
