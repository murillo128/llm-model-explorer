import { describe, expect, it } from 'vitest';
import { arrowTransform, canonicalPoints, canonicalSections, roundedSection, routeDisplays, targetTangent } from './route-display';
import { connectionHitResolver } from './connection-hit';
import type { ProjectedEdge } from './projection';
import type { Layout } from './graph';

const points = [{ x: 0, y: 0 }, { x: 60, y: 0 }, { x: 60, y: 60 }, { x: 120, y: 60 }];
describe('connection display geometry', () => {
  it('removes duplicate/collinear vertices, but preserves reversals and genuine junctions', () => {
    const spine = [{ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 10, y: 0 }, { x: 20, y: 0 }, { x: 15, y: 0 }];
    expect(canonicalPoints(spine)).toEqual([{ x: 0, y: 0 }, { x: 20, y: 0 }, { x: 15, y: 0 }]);
    expect(canonicalPoints(spine, [{ x: 10, y: 0 }])).toEqual(spine.slice(1));
  });
  it('rounds orthogonal corners with an eight-unit fillet shared by its hit samples', () => {
    const display = roundedSection(points, [], [], points[0], points.at(-1));
    expect(display.path).toBe('M0,0 L52,0 Q60,0 60,8 L60,52 Q60,60 68,60 L120,60');
    expect(display.samples).toContainEqual({ x: 58, y: 2 });
    expect(display.samples[0]).toEqual({ x: 0, y: 0 });
    expect(display.samples.at(-1)).toEqual({ x: 120, y: 60 });
  });
  it('clamps locally for short segments, terminal approaches, padded obstacles and branch points', () => {
    expect(roundedSection([{ x: 0, y: 0 }, { x: 6, y: 0 }, { x: 6, y: 4 }], [], []).path)
      .toBe('M0,0 L4,0 Q6,0 6,2 L6,4');
    const protectedTurn = roundedSection(points, [{ x: 60, y: 0 }], [], points[0], points.at(-1));
    expect(protectedTurn.path).toContain('L60,0 L60,52 Q');
    expect(roundedSection(points, [{ x: 56, y: 0 }], []).path).toContain('L56,0 Q60,0 60,4');
    const obstacle = [{ x: 50, y: 5, width: 2, height: 2 }];
    const constrained = roundedSection(points, [], obstacle);
    expect(constrained.path).toContain('L56,0 Q60,0 60,4');
    expect(constrained.path).toContain('Q60,60 68,60');
    const terminal = [{ x: 0, y: 0 }, { x: 60, y: 0 }, { x: 60, y: 60 }, { x: 92, y: 60 }];
    expect(roundedSection(terminal, [], [], terminal[0], terminal.at(-1)).path).toContain('L60,58 Q60,60 62,60');
  });
  it('resolves the real target and final nonzero tangent even with reversed or shuffled sections', () => {
    const route = { id: 'route', junctions: [], sections: [
      [{ x: 100, y: 40 }, { x: 100, y: 40 }, { x: 70, y: 40 }], [{ x: 0, y: 0 }, { x: 70, y: 0 }, { x: 70, y: 40 }],
    ] };
    expect(targetTangent(route, { x: 100, y: 40 })).toEqual({ x: 1, y: 0 });
    expect(targetTangent({ ...route, sections: [[{ x: 0, y: 20 }, { x: 0, y: 0 }]] }, { x: 0, y: 0 })).toEqual({ x: 0, y: -1 });
    expect(() => targetTangent(route, { x: 100, y: 0 })).toThrow('target terminal');
    expect(arrowTransform({ x: 100, y: 40 }, { x: 1, y: 0 }, 1)).toBe('translate(91.5 40) rotate(0) scale(1)');
    expect(arrowTransform({ x: 100, y: 40 }, { x: 1, y: 0 }, 0.1)).toBe(arrowTransform({ x: 100, y: 40 }, { x: 1, y: 0 }, 0.8));
    expect(canonicalSections([[{ x: 40, y: 0 }, { x: 64, y: 0 }], [{ x: 0, y: 0 }, { x: 40, y: 0 }]], []))
      .toEqual([[{ x: 0, y: 0 }, { x: 64, y: 0 }]]);
  });
  it('hits rounded exclusive bends without joining a nearby sibling, and preserves exact common trunks', () => {
    const edges = ['one', 'two'].map((id) => ({ id, source: { node_id: 'source', port_id: 'out' }, target: { node_id: id, port_id: 'in' }, kind: 'data', paths: [], originalEdgeIds: [id] }) as ProjectedEdge);
    const routes = [
      { id: 'one', sections: [points], junctions: [{ x: 30, y: 0 }] },
      { id: 'two', sections: [[{ x: 0, y: 0 }, { x: 30, y: 0 }, { x: 30, y: 90 }, { x: 120, y: 90 }]], junctions: [{ x: 30, y: 0 }] },
    ];
    const displays = new Map(routes.map((r) => [r.id, { sections: r.sections.map((s) => roundedSection(s, r.junctions, [])), target: r.sections[0]!.at(-1)!, tangent: { x: 1, y: 0 } }]));
    const hit = connectionHitResolver(edges, routes, displays);
    expect(hit('one', { x: 58, y: 2 })).toEqual(['one']);
    expect(hit('one', { x: 20, y: 0 })).toEqual(['one', 'two']);
    expect(hit('two', { x: 32, y: 84 })).toEqual(['two']);
  });
  it('protects a sibling branch inside a segment across a spatial bucket boundary', async () => {
    const edges = ['one', 'two'].map((id) => ({ id, source: { node_id: 'source', port_id: 'out' },
      target: { node_id: id, port_id: 'in' }, kind: 'data', paths: [], originalEdgeIds: [id] }) as ProjectedEdge);
    const layout: Layout = {
      boxes: [], projection: { nodes: [], edges, hiddenEdgeIds: [], filteredEdgeIds: [], unusedInputs: [] },
      edgeIds: ['one', 'two'], width: 120, height: 120, milliseconds: 0,
      ports: ([['source', 'out', -28, 0], ['one', 'in', 92, 60], ['two', 'in', 92, -60]] as const).map(([nodeId, portId, x, y]) => ({
        nodeId, portId, x, y, absoluteX: x, absoluteY: y, side: 'left',
        label: { x, y: 500, width: 20, height: 16, clearance: 2, raised: false },
      })),
      routes: [
        { id: 'one', sections: [[{ x: -28, y: 0 }, { x: 32, y: 0 }, { x: 32, y: 60 }, { x: 92, y: 60 }]], junctions: [] },
        { id: 'two', sections: [[{ x: -28, y: 0 }, { x: 28, y: 0 }, { x: 28, y: -60 }, { x: 92, y: -60 }]],
          junctions: [{ x: 28, y: 0 }, { x: 28, y: 0 }] },
      ],
    };
    const displays = await routeDisplays(layout);
    expect(displays.get('one')!.sections[0]!.path).toBe('M-28,0 L28,0 Q32,0 32,4 L32,52 Q32,60 40,60 L92,60');
    expect(displays.get('two')!.sections[0]!.path).toContain('L28,0 L28,-52 Q');
    const hit = connectionHitResolver(edges, layout.routes, displays);
    expect(hit('one', { x: 20, y: 0 })).toEqual(['one', 'two']);
    expect(hit('one', { x: 31, y: 1 })).toEqual(['one']);
  });
});
