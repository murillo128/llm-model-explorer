import { describe, expect, it } from 'vitest';
import { arrowTransform, canonicalPoints, canonicalSections, roundedSection, targetTangent } from './route-display';
import { connectionHitResolver } from './connection-hit';
import type { ProjectedEdge } from './projection';

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
});
