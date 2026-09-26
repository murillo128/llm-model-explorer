import { expect } from '@playwright/test';
import type { Locator, Page } from '@playwright/test';

/** Find a trunk or exclusive branch from generated SVG geometry, independently
 * of the production hit resolver. The pointer still uses native browser hits. */
export async function fanoutPoint(page: Page, edges: Locator[], expected: Locator[]) {
  const ids = await Promise.all(edges.map(async (edge) => (await edge.getAttribute('data-edge-id'))!));
  const expectedIds = await Promise.all(expected.map(async (edge) => (await edge.getAttribute('data-edge-id'))!));
  const result = await page.evaluate(({ ids, expectedIds }) => {
    type Point = { x: number; y: number };
    const routes = ids.map((id) => {
      const element = [...document.querySelectorAll('.architecture-connection')].find((edge) => edge.getAttribute('data-edge-id') === id)!;
      const segments = [...element.querySelectorAll<SVGPathElement>('.architecture-edge-hit')].flatMap((path) => {
        const matrix = path.getScreenCTM()!;
        const values = path.getAttribute('d')!.match(/[-+]?(?:\d*\.?\d+)(?:e[-+]?\d+)?/gi)!.map(Number);
        const points: DOMPoint[] = [];
        for (let i = 0; i < values.length; i += 2) points.push(new DOMPoint(values[i], values[i + 1]).matrixTransform(matrix));
        return points.slice(1).map((point, i) => [points[i]!, point] as const);
      });
      return { id, segments };
    });
    function distance(point: Point, [a, b]: readonly [Point, Point]) {
      const dx = b.x - a.x, dy = b.y - a.y;
      const t = Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / (dx * dx + dy * dy || 1)));
      return Math.hypot(point.x - a.x - t * dx, point.y - a.y - t * dy);
    }
    for (const { segments } of routes) for (const [a, b] of segments) {
      // A long SVG segment may contain both the trunk and a short exclusive tail.
      // Sample each interval between real route junctions; fixed fractions of the
      // entire segment can all land on the trunk despite a reachable branch.
      const dx = b.x - a.x, dy = b.y - a.y, lengthSquared = dx * dx + dy * dy;
      if (!lengthSquared) continue;
      const cuts = new Set([0, 1]);
      for (const route of routes) for (const segment of route.segments) for (const p of segment) {
        const t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSquared;
        if (t > 0 && t < 1 && distance(p, [a, b]) < 0.01) cuts.add(t);
      }
      const ordered = [...cuts].sort((a, b) => a - b);
      const samples = ordered.slice(1).flatMap((end, i) => [0.5, 0.25, 0.75].map((t) => ordered[i]! + (end - ordered[i]!) * t));
      for (const fraction of samples) {
        const point = { x: a.x + dx * fraction, y: a.y + dy * fraction };
        if (point.x < 1 || point.x > innerWidth - 1 || point.y < 1 || point.y > innerHeight - 1) continue;
        const distances = routes.map((route) => ({ id: route.id, distance: Math.min(...route.segments.map((segment) => distance(point, segment))) }));
        const represented = distances.filter((route) => route.distance < 0.01).map((route) => route.id).sort();
        if (JSON.stringify(represented) !== JSON.stringify([...expectedIds].sort())) continue;
        // Exclusive samples stay clear of other branches' 12 CSS pixel hit corridors.
        if (distances.some((route) => !expectedIds.includes(route.id) && route.distance < 7)) continue;
        const hitId = document.elementFromPoint(point.x, point.y)?.closest('.architecture-connection')?.getAttribute('data-edge-id');
        if (hitId && expectedIds.includes(hitId)) return { ...point, hitId };
      }
    }
    return null;
  }, { ids, expectedIds });
  expect(result, 'Generated fan-out must expose a native-pointer target for the requested trunk/branch').not.toBeNull();
  return result!;
}
