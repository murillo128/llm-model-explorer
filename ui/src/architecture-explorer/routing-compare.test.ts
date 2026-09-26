import { it } from 'vitest';
import { layoutGraph } from './auto-layout';
import { rotaryContextFixture } from '../../tests/architecture-routing-fixture';
import { interfaceFixture } from '../../tests/architecture-interface-fixture';
import type { Graph } from './graph';
import type { ProjectionOptions } from './projection';

it('reports compact routing geometry', async () => {
  const cases: [string, Graph, ProjectionOptions][] = [
    ['rotary-context', rotaryContextFixture(), { expanded: ['component'], showContext: true }],
    ['dense-many', interfaceFixture('dense', true), { expanded: ['model'], showUnused: true }],
    ['hybrid-many', interfaceFixture('hybrid', true), { expanded: ['language'], showUnused: true }],
  ];
  for (const [name, graph, options] of cases) {
    const layout = await layoutGraph(graph, options);
    const bends = layout.routes.reduce((sum, route) => sum + route.sections.reduce((n, section) => n + Math.max(0, section.length - 2), 0), 0);
    let labelIntersections = 0;
    for (const port of layout.ports) {
      const label = port.label, padding = label.clearance;
      const left = label.x - padding, right = label.x + label.width + padding;
      const top = label.y - padding, bottom = label.y + label.height + padding;
      for (const route of layout.routes) for (const section of route.sections) for (let i = 1; i < section.length; i++) {
        const a = section[i - 1]!, b = section[i]!;
        if (a.x === b.x ? a.x > left && a.x < right && Math.max(a.y, b.y) > top && Math.min(a.y, b.y) < bottom :
          a.y === b.y && a.y > top && a.y < bottom && Math.max(a.x, b.x) > left && Math.min(a.x, b.x) < right) labelIntersections++;
      }
    }
    console.log('routing comparison', JSON.stringify({ name, nodes: layout.boxes.length, edges: layout.routes.length,
      bends, labelIntersections, width: Math.round(layout.width), height: Math.round(layout.height) }));
  }
});
