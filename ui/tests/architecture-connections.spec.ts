import { fanoutPoint } from './architecture-pointer';
import { openShared, chooseInstance, findComponent, graphAction, graphPreference } from './architecture-controls';
import { expect, test } from '@playwright/test';
import type { Locator, Page, TestInfo } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import { makeProjectionFixture } from './architecture-projection-fixture';
import type { Graph } from '../src/architecture-explorer/graph';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;
const graph = (page: Page) => page.getByLabel('Architecture graph', { exact: true });
const connection = (page: Page, source: string, from: string, target: string, to: string) => page.locator(
  `.architecture-connection[data-source-node=${JSON.stringify(source)}][data-source-port=${JSON.stringify(from)}][data-target-node=${JSON.stringify(target)}][data-target-port=${JSON.stringify(to)}]`);
const port = (page: Page, node: string, id: string) => page.locator(
  `.architecture-port[data-node-id=${JSON.stringify(node)}][data-port-id=${JSON.stringify(id)}]`);
async function capture(page: Page, info: TestInfo, name: string) {
  const path = info.outputPath(`${name}.png`);
  await page.screenshot({ path });
  await info.attach(name, { path, contentType: 'image/png' });
}
async function ready(page: Page) {
  await expect(graph(page)).toHaveAttribute('aria-busy', 'false');
  await expect(graph(page)).toHaveAttribute('data-layout-count', /^[1-9]\d*$/);
  await expect(page.getByText('Laying out architecture…', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.locator('.architecture-node').first()).toBeAttached();
}
async function open(page: Page, fixture = 'connections') {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(harness);
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption(fixture);
  await expect(graph(page)).toHaveAttribute('data-graph-id', /^authored-projection-/);
  await ready(page);
}
async function layer(page: Page, index: number) {
  await chooseInstance(page, 'Decoder layers', `layer-${index}`);
  await graphAction(page, 'Focus layer');
  await ready(page);
}
async function fullAttention(page: Page) {
  await open(page);
  await layer(page, 3);
  await findComponent(page, 'layer-3.attention');
  await graphAction(page, 'Toggle selected group');
  await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  await expect(port(page, 'layer-3.attention.core', 'K')).toBeVisible();
}
async function stableState(page: Page, clearHover = true) {
  // aria-busy covers the current layout and its committed camera action.
  await ready(page);
  if (clearHover) await page.mouse.move(0, 0);
  // Flush temporary pointer emphasis; frame count does not establish camera readiness.
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  return page.evaluate(() => ({
    layoutCount: document.querySelector('[aria-label="Architecture graph"]')?.getAttribute('data-layout-count'),
    camera: document.querySelector('.react-flow__viewport')?.getAttribute('style'),
    nodes: [...document.querySelectorAll<HTMLElement>('.react-flow__node')].map((node) => ({
      id: node.dataset.id, position: node.style.transform, width: node.style.width, height: node.style.height,
    })),
    ports: [...document.querySelectorAll<HTMLElement>('.architecture-port')].map((port) => ({
      node: port.dataset.nodeId, port: port.dataset.portId, left: port.style.left, top: port.style.top,
    })),
    routes: [...document.querySelectorAll('.architecture-connection')].map((edge) => ({
      id: edge.getAttribute('data-edge-id')!, paths: [...edge.querySelectorAll('.architecture-edge-line')].map((path) => path.getAttribute('d')),
    })).sort((a, b) => a.id.localeCompare(b.id)),
  }));
}
async function unchanged(page: Page, before: Awaited<ReturnType<typeof stableState>>) {
  expect(await stableState(page, false)).toEqual(before);
}
async function emphasized(page: Page, expected: Locator[]) {
  const ids = await Promise.all(expected.map(async (edge) => {
    await expect(edge).toHaveCount(1);
    await expect(edge.locator('.architecture-edge-line[marker-end]')).toHaveCount(1);
    return (await edge.getAttribute('data-edge-id'))!;
  }));
  await expect.poll(() => page.locator('.architecture-connection[data-emphasized="true"]').evaluateAll((edges) =>
    edges.map((edge) => edge.getAttribute('data-edge-id')!).sort())).toEqual(ids.sort());
  const endpointIds = new Set<string>();
  for (const edge of expected) {
    for (const end of ['source', 'target']) endpointIds.add(JSON.stringify([
      await edge.getAttribute(`data-${end}-node`), await edge.getAttribute(`data-${end}-port`),
    ]));
  }
  await expect.poll(() => page.locator('.architecture-port[data-emphasized="true"]').evaluateAll((ports) =>
    ports.map((port) => JSON.stringify([port.getAttribute('data-node-id'), port.getAttribute('data-port-id')])).sort()))
    .toEqual([...endpointIds].sort());
}
async function hoverDot(point: Locator) {
  await expect(point).toBeVisible();
  await point.locator('.architecture-port-dot').hover();
}
/** Choose an actual interior path point where native hit testing identifies this
 * route. No node dragging or manually supplied layout coordinates are involved. */
async function hoverLine(page: Page, edge: Locator) {
  const result = await edge.evaluate((element) => {
    const expected = element.getAttribute('data-edge-id');
    const hits: string[] = [];
    for (const path of element.querySelectorAll<SVGPathElement>('.architecture-edge-hit')) {
      const matrix = path.getScreenCTM(); if (!matrix) continue;
      const length = path.getTotalLength();
      for (const fraction of [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]) {
        const p = path.getPointAtLength(length * fraction).matrixTransform(matrix);
        if (p.x < 1 || p.x > innerWidth - 1 || p.y < 1 || p.y > innerHeight - 1) continue;
        const hit = document.elementFromPoint(p.x, p.y);
        const target = hit?.closest('.architecture-connection');
        if (target?.getAttribute('data-edge-id') === expected) return { location: { x: p.x, y: p.y }, hits };
        hits.push(`${Math.round(p.x)},${Math.round(p.y)}: ${hit?.tagName}.${hit?.getAttribute('class')} node=${hit?.closest('.react-flow__node')?.getAttribute('data-id')} edge=${target?.getAttribute('data-edge-id')}`);
      }
    }
    return { location: null, hits };
  });
  expect(result.location, `Each connection needs its own pointer-targetable line interior: ${result.hits.join('; ')}`).not.toBeNull();
  await page.mouse.move(result.location!.x, result.location!.y);
  return result.location!;
}
function originalIds(source: Graph, segments: [string, string, string, string][]) {
  return segments.map(([s, from, t, to]) => {
    const edge = source.edges.find((edge) => edge.source.node_id === s && edge.source.port_id === from && edge.target.node_id === t && edge.target.port_id === to);
    expect(edge, `Missing independently authored segment ${s}.${from} → ${t}.${to}`).toBeTruthy();
    return edge!.id;
  });
}

test('exact source and destination dots, fan-out branches and line middles highlight complete forwarded connections', async ({ page }, info) => {
  await fullAttention(page);
  const source = makeProjectionFixture({ count: 4 });
  const fanout = ['Q', 'K', 'V'].map((name) => connection(page, 'layer-3.input-norm', 'out', `layer-3.attention.${name}`, 'x'));
  for (const [index, name] of ['Q', 'K', 'V'].entries()) expect(JSON.parse((await fanout[index]!.getAttribute('data-original-edge-ids'))!)).toEqual(originalIds(source, [
    ['layer-3.input-norm', 'out', 'layer-3.attention', 'x'], ['layer-3.attention', 'x', `layer-3.attention.${name}`, 'x'],
  ]));
  const before = await stableState(page);
  await hoverDot(port(page, 'layer-3.input-norm', 'out'));
  await emphasized(page, fanout);
  await unchanged(page, before);
  await capture(page, info, 'source-dot-fanout');
  // Card sizes can move shared-trunk breakpoints. Sample a genuinely exclusive
  // branch from SVG geometry, rather than treating the owning path as exclusive.
  const keyBranch = await fanoutPoint(page, fanout, [fanout[1]!]);
  await page.mouse.move(keyBranch.x, keyBranch.y); await emphasized(page, [fanout[1]!]);
  await unchanged(page, before);
  await capture(page, info, 'line-hover-complete-forwarding');
  await hoverDot(port(page, 'layer-3.attention.K', 'x')); await emphasized(page, [fanout[1]!]);
  // A neighboring source branch remains individually targetable.
  const valueBranch = await fanoutPoint(page, fanout, [fanout[2]!]);
  await page.mouse.move(valueBranch.x, valueBranch.y); await emphasized(page, [fanout[2]!]);
  await page.mouse.move(0, 0); await emphasized(page, []);
  await unchanged(page, before);
});

test('shared fan-out trunk identifies every branch through native trunk, branch and port transitions', async ({ page }, info) => {
  await fullAttention(page);
  const fanout = ['Q', 'K', 'V'].map((name) => connection(page, 'layer-3.input-norm', 'out', `layer-3.attention.${name}`, 'x'));
  const before = await stableState(page);
  const trunk = await fanoutPoint(page, fanout, fanout);
  await page.mouse.move(trunk.x, trunk.y);
  await emphasized(page, fanout);
  await unchanged(page, before);
  await capture(page, info, 'shared-trunk-fanout');
  // Follow the native trunk owner's route to an exclusive branch. This needs
  // pointer-move resolution even when the enclosing SVG connection is unchanged.
  const owner = fanout[(await Promise.all(fanout.map((edge) => edge.getAttribute('data-edge-id')))).indexOf(trunk.hitId)]!;
  const branch = await fanoutPoint(page, fanout, [owner]);
  await page.mouse.move(branch.x, branch.y);
  await emphasized(page, [owner]);
  await unchanged(page, before);
  await capture(page, info, 'exclusive-branch-after-trunk');
  await hoverDot(port(page, 'layer-3.input-norm', 'out')); await emphasized(page, fanout);
  await hoverDot(port(page, 'layer-3.attention.K', 'x')); await emphasized(page, [fanout[1]!]);
  await unchanged(page, before);
  // Pin a single branch; temporary trunk and endpoint emphasis restores it on exit.
  const pinned = fanout[2]!;
  await page.mouse.move(0, 0); await pinned.focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog', { name: 'Connection inspection', exact: true })).toBeVisible();
  await page.keyboard.press('Escape'); await expect(pinned).toBeFocused();
  await page.getByRole('button', { name: 'Fit view', exact: true }).focus();
  await page.mouse.move(trunk.x, trunk.y); await emphasized(page, fanout);
  await page.mouse.move(branch.x, branch.y); await emphasized(page, [owner]);
  await hoverDot(port(page, 'layer-3.input-norm', 'out')); await emphasized(page, fanout);
  await page.mouse.move(0, 0); await emphasized(page, [pinned]);
  await unchanged(page, before);
  await graphAction(page, 'Zoom graph in');
  const zoomed = await stableState(page);
  const zoomedTrunk = await fanoutPoint(page, fanout, fanout);
  await page.mouse.move(zoomedTrunk.x, zoomedTrunk.y); await emphasized(page, fanout);
  await capture(page, info, 'zoomed-shared-trunk-fanout');
  const zoomedBranch = await fanoutPoint(page, fanout, [fanout[1]!]);
  await page.mouse.move(zoomedBranch.x, zoomedBranch.y); await emphasized(page, [fanout[1]!]);
  await hoverDot(port(page, 'layer-3.input-norm', 'out')); await emphasized(page, fanout);
  await page.mouse.move(0, 0); await emphasized(page, [pinned]);
  await unchanged(page, zoomed);
});

test('authored MLP boundary forwarding preserves shared trunks and exact port hover', async ({ page }) => {
  await open(page, 'components');
  await layer(page, 3);
  await findComponent(page, 'layer-3.mlp');
  await graphAction(page, 'Toggle selected group');
  await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  const fanout = ['gate', 'up'].map((name) => connection(page, 'layer-3.post-norm', 'out', `layer-3.${name}`, 'x'));
  const before = await stableState(page);
  await hoverDot(port(page, 'layer-3.post-norm', 'out')); await emphasized(page, fanout);
  const trunk = await fanoutPoint(page, fanout, fanout);
  await page.mouse.move(trunk.x, trunk.y); await emphasized(page, fanout);
  const branch = await fanoutPoint(page, fanout, [fanout[1]!]);
  await page.mouse.move(branch.x, branch.y); await emphasized(page, [fanout[1]!]);
  await hoverDot(port(page, 'layer-3.gate', 'x')); await emphasized(page, [fanout[0]!]);
  await unchanged(page, before);
});

test('isolated Attention boundaries retain exact fan-out, trunk, branch, endpoint and pin interactions', async ({ page }, info) => {
  await fullAttention(page);
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await expect(graph(page)).toHaveAttribute('data-scope-id', 'layer-3.attention');
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  const branches = ['Q', 'K', 'V'].map((name) => page.locator(
    `.architecture-connection[data-target-node="layer-3.attention.${name}"][data-target-port="x"]`));
  const sourceId = (await branches[0]!.getAttribute('data-source-node'))!;
  const sourcePort = (await branches[0]!.getAttribute('data-source-port'))!;
  expect(sourceId.startsWith('external:')).toBe(true);
  const source = makeProjectionFixture({ count: 4 });
  for (const [i, name] of ['Q', 'K', 'V'].entries()) expect(JSON.parse((await branches[i]!.getAttribute('data-original-edge-ids'))!)).toEqual(originalIds(source, [
    ['layer-3.input-norm', 'out', 'layer-3.attention', 'x'], ['layer-3.attention', 'x', `layer-3.attention.${name}`, 'x'],
  ]));
  const before = await stableState(page);
  await hoverDot(port(page, sourceId, sourcePort)); await emphasized(page, branches); await unchanged(page, before);
  await capture(page, info, 'isolated-boundary-port');
  const trunk = await fanoutPoint(page, branches, branches);
  await page.mouse.move(trunk.x, trunk.y); await emphasized(page, branches); await unchanged(page, before);
  await capture(page, info, 'isolated-shared-trunk');
  const branch = await fanoutPoint(page, branches, [branches[1]!]);
  await page.mouse.move(branch.x, branch.y); await emphasized(page, [branches[1]!]);
  await hoverDot(port(page, 'layer-3.attention.V', 'x')); await emphasized(page, [branches[2]!]);
  await page.mouse.move(0, 0); await branches[0]!.focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog', { name: 'Connection inspection', exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Fit view', exact: true }).focus();
  await page.mouse.move(trunk.x, trunk.y); await emphasized(page, branches);
  await page.mouse.move(0, 0); await emphasized(page, [branches[0]!]); await unchanged(page, before);
});

test('same-shaped inputs and separate K/V state routes keep exact identity under mouse and keyboard emphasis', async ({ page }) => {
  await fullAttention(page);
  const source = makeProjectionFixture({ count: 4 });
  const k = connection(page, 'layer-3.attention.rope-K', 'out', 'layer-3.attention.core', 'K');
  const v = connection(page, 'layer-3.attention.V', 'out', 'layer-3.attention.core', 'V');
  const priorK = connection(page, 'layer-3.prior-K', 'out', 'layer-3.attention.core', 'prior_K');
  const priorV = connection(page, 'layer-3.prior-V', 'out', 'layer-3.attention.core', 'prior_V');
  const nextK = connection(page, 'layer-3.attention.core', 'next_K', 'layer-3.next-K', 'x');
  const nextV = connection(page, 'layer-3.attention.core', 'next_V', 'layer-3.next-V', 'x');
  const before = await stableState(page);
  await hoverDot(port(page, 'layer-3.attention.core', 'K')); await emphasized(page, [k]);
  await hoverDot(port(page, 'layer-3.attention.core', 'V')); await emphasized(page, [v]);
  for (const [name, incoming, outgoing] of [['K', priorK, nextK], ['V', priorV, nextV]] as const) {
    expect(JSON.parse((await incoming.getAttribute('data-original-edge-ids'))!)).toEqual(originalIds(source, [
      [`layer-3.prior-${name}`, 'out', 'layer-3.attention', `prior_${name}`],
      ['layer-3.attention', `prior_${name}`, 'layer-3.attention.core', `prior_${name}`],
    ]));
    await hoverLine(page, incoming); await emphasized(page, [incoming]);
    await hoverDot(port(page, 'layer-3.attention.core', `next_${name}`)); await emphasized(page, [outgoing]);
    await page.mouse.move(0, 0); await port(page, 'layer-3.attention.core', `prior_${name}`).focus();
    await emphasized(page, [incoming]);
    await page.getByRole('button', { name: 'Fit view', exact: true }).focus();
  }
  await emphasized(page, []); await unchanged(page, before);
});

test('residual and MLP inputs stop at operations; pin survives temporary hover, zoom and inspection close', async ({ page }, info) => {
  await open(page); await layer(page, 3);
  await graphAction(page, 'Focus MLP');
  await ready(page); await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  const gate = connection(page, 'layer-3.silu', 'out', 'layer-3.multiply', 'gate');
  const up = connection(page, 'layer-3.up', 'out', 'layer-3.multiply', 'up');
  const down = connection(page, 'layer-3.multiply', 'out', 'layer-3.down', 'x');
  const before = await stableState(page);
  await hoverDot(port(page, 'layer-3.multiply', 'gate')); await emphasized(page, [gate]);
  await expect(down).toHaveAttribute('data-emphasized', 'false');
  await page.mouse.move(0, 0); await gate.focus(); await emphasized(page, [gate]);
  await page.keyboard.press('Enter');
  const inspection = page.getByRole('dialog', { name: 'Connection inspection', exact: true });
  await expect(inspection).toBeVisible();
  await expect(inspection).toContainText('layer-3.silu');
  await expect(inspection).toContainText('layer-3.multiply');
  await expect(page.getByRole('button', { name: 'Close connection inspection', exact: true })).toBeFocused();
  for (const key of ['Shift+Tab', 'Shift+Tab', 'Tab', 'Tab', 'Tab']) {
    await page.keyboard.press(key);
    expect(await inspection.evaluate((dialog) => dialog.contains(document.activeElement))).toBe(true);
  }
  await page.keyboard.press('Escape'); await expect(inspection).toHaveCount(0); await expect(gate).toBeFocused();
  await page.getByRole('button', { name: 'Fit view', exact: true }).focus();
  await hoverLine(page, up); await emphasized(page, [up]);
  await page.mouse.move(0, 0); await emphasized(page, [gate]);
  await unchanged(page, before);
  await graphAction(page, 'Zoom graph in');
  const zoomed = await stableState(page);
  await hoverLine(page, up); await emphasized(page, [up]);
  await capture(page, info, 'zoomed-line-hover');
  await unchanged(page, zoomed);
  const linePoint = await hoverLine(page, up);
  await page.mouse.click(linePoint.x, linePoint.y);
  await expect(inspection).toBeVisible();
  await expect(inspection).toContainText('layer-3.up');
  await page.getByRole('button', { name: 'Close connection inspection', exact: true }).click();
  await expect(inspection).toHaveCount(0); await expect(up).toBeFocused();
  await page.mouse.move(0, 0); await emphasized(page, [up]);
  await unchanged(page, zoomed);
  await page.getByRole('button', { name: 'Clear connection selection', exact: true }).click();
  await page.mouse.move(0, 0); await emphasized(page, []);
  await graphAction(page, /^(Back to layer|Focus layer)$/);
  await ready(page); await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  const residual = connection(page, 'layer-3.residual-1', 'out', 'layer-3.residual-2', 'residual');
  const norm = connection(page, 'layer-3.residual-1', 'out', 'layer-3.post-norm', 'x');
  await hoverDot(port(page, 'layer-3.residual-2', 'residual')); await emphasized(page, [residual]);
  await hoverDot(port(page, 'layer-3.residual-1', 'out')); await emphasized(page, [residual, norm]);
});

test('24-instance compact navigation, first/last identity, MLP/state focus and exhaustive round trip', async ({ page }, info) => {
  await open(page, 'hybrid'); await page.setViewportSize({ width: 1178, height: 900 });
  const source = makeProjectionFixture();
  await expect(page.locator('.architecture-node[data-presentation="repetition"]')).toContainText('24');
  await expect(page.locator('.architecture-node[data-presentation="repetition"]')).toContainText('18 linear');
  await expect(page.locator('.architecture-node[data-presentation="repetition"]')).toContainText('6 full');
  expect(Number(await graph(page).getAttribute('data-visible-nodes'))).toBeLessThan(20);
  // Keep the compact model cards inside both viewports for this DOM-coordinate
  // comparison; resizing a narrower pane may legitimately change viewport culling.
  await page.getByRole('button', { name: 'Collapse browser', exact: true }).click();
  const beforeResize = await stableState(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  const afterResize = await stableState(page);
  // A host resize may update the camera; generated node/port/route coordinates
  // and the completed layout count must remain unchanged.
  expect({ ...afterResize, camera: beforeResize.camera }).toEqual(beforeResize);
  await page.setViewportSize({ width: 1178, height: 900 });
  const restoredSize = await stableState(page);
  expect({ ...restoredSize, camera: beforeResize.camera }).toEqual(beforeResize);
  await page.getByRole('button', { name: 'Expand browser', exact: true }).click();
  await page.getByRole('button', { name: /Explore stack/ }).first().click(); await ready(page);
  await expect(page.getByRole('button', { name: /Previous window/ }).first()).toBeDisabled();
  await page.getByRole('button', { name: /Next window/ }).first().click(); await ready(page);
  await expect(page.getByRole('button', { name: /Previous window/ }).first()).toBeEnabled();
  await page.getByRole('button', { name: /Previous window/ }).first().click(); await ready(page);
  await layer(page, 0);
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-0');
  await expect(port(page, 'layer-0.attention', 'current_mask')).toBeAttached();
  await expect(port(page, 'layer-0.attention', 'positions')).toHaveCount(0);
  await expect(port(page, 'layer-0.attention', 'mask')).toHaveCount(0);
  await graphPreference(page, 'Unused interfaces', true); await ready(page);
  await expect(port(page, 'layer-0.attention', 'positions')).toBeAttached();
  await expect(page.locator('.architecture-connection[data-target-node="layer-0.attention"][data-target-port="positions"]')).toHaveCount(0);
  await graphPreference(page, 'Unused interfaces', false); await ready(page);
  await graphAction(page, 'State dependencies'); await ready(page);
  await expect(page.locator('.architecture-connection[data-kind="data"]')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Select Prior conv', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Select Prior delta', exact: true })).toBeVisible();
  await graphAction(page, 'Back to layer'); await ready(page);
  await layer(page, 3);
  await expect(port(page, 'layer-3.attention', 'positions')).toBeAttached();
  await expect(port(page, 'layer-3.attention', 'mask')).toBeAttached();
  await expect(port(page, 'layer-3.attention', 'current_mask')).toHaveCount(0);
  await graphAction(page, 'State dependencies'); await ready(page);
  await expect(page.getByRole('button', { name: 'Select Prior K', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Select Prior V', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Select Prior conv', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Model overview', exact: true }).click(); await ready(page);
  await layer(page, 23);
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-23');
  await expect(page.getByRole('combobox', { name: /Expand instance of/ }).locator('option:checked')).toContainText(/instance 23/i);
  await graphPreference(page, 'Show dimensions', true); await ready(page);
  await graphAction(page, 'Show all operations'); await ready(page);
  await expect(graph(page)).toHaveAttribute('data-visible-nodes', String(source.nodes.length));
  expect(JSON.parse((await graph(page).getAttribute('data-source-node-ids'))!)).toEqual(source.nodes.map((node) => node.id));
  expect(JSON.parse((await graph(page).getAttribute('data-represented-edge-ids'))!)).toEqual(source.edges.map((edge) => edge.id));
  const coordinates = info.outputPath('authored-exhaustive-coordinates.json');
  await writeFile(coordinates, JSON.stringify(await stableState(page)));
  await info.attach('authored-exhaustive-coordinates', { path: coordinates, contentType: 'application/json' });
  await page.getByRole('button', { name: 'Model overview', exact: true }).click(); await ready(page);
  expect(Number(await graph(page).getAttribute('data-visible-nodes'))).toBeLessThan(20);
  expect(await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.scrollHeight])).toEqual([1178, 900]);
});

interface WorkerObservation {
  live: number;
  created: number;
  mode: 'normal' | 'fail-next' | 'hold-next';
  held: number;
  release: (() => void)[];
}
test('obsolete model layout replies cannot replace the latest graph; a failed worker retries locally', async ({ page }) => {
  await page.addInitScript(() => {
    const state: WorkerObservation = { live: 0, created: 0, mode: 'fail-next', held: 0, release: [] };
    Object.assign(window, { architectureLayoutWorkers: state });
    const Original = window.Worker;
    window.Worker = class extends Original {
      private alive = true;
      private delayed = false;
      private handler: ((this: Worker, event: MessageEvent) => unknown) | null = null;
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options); state.live++; state.created++;
      }
      override set onmessage(callback: ((this: Worker, event: MessageEvent) => unknown) | null) {
        this.handler = callback;
        super.onmessage = (event) => {
          if (this.delayed) {
            state.held++;
            state.release.push(() => { state.held--; callback?.call(this, event); });
          } else callback?.call(this, event);
        };
      }
      override get onmessage() { return this.handler; }
      override postMessage(message: unknown, transfer: Transferable[] | StructuredSerializeOptions = []) {
        if (state.mode === 'fail-next') {
          state.mode = 'normal'; queueMicrotask(() => this.onerror?.call(this, new ErrorEvent('error', { message: 'Authored worker failure' })));
          return;
        }
        if (state.mode === 'hold-next') { state.mode = 'normal'; this.delayed = true; }
        if (Array.isArray(transfer)) super.postMessage(message, transfer);
        else super.postMessage(message, transfer);
      }
      override terminate() { if (this.alive) { this.alive = false; state.live--; } super.terminate(); }
    };
  });
  const state = () => page.evaluate(() => {
    const value = (window as typeof window & { architectureLayoutWorkers: WorkerObservation }).architectureLayoutWorkers;
    return { live: value.live, created: value.created, held: value.held };
  });
  await page.goto(harness);
  await expect(page.getByRole('alert')).toContainText(/Layout|layout/);
  await expect.poll(async () => (await state()).live).toBe(0);
  await page.getByRole('button', { name: 'Retry layout', exact: true }).click(); await ready(page);
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.evaluate(() => { (window as typeof window & { architectureLayoutWorkers: WorkerObservation }).architectureLayoutWorkers.mode = 'hold-next'; });
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('connections');
  await expect.poll(async () => (await state()).held).toBe(1);
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('visual-stacks'); await ready(page);
  await expect(graph(page)).toHaveAttribute('data-graph-id', makeProjectionFixture({ count: 3, secondStack: 2 }).graph_id);
  const before = await stableState(page);
  await page.evaluate(() => { const state = (window as typeof window & { architectureLayoutWorkers: WorkerObservation }).architectureLayoutWorkers; state.release.splice(0).forEach((release) => release()); });
  await expect.poll(async () => (await state()).held).toBe(0);
  await unchanged(page, before);
  await expect(page.getByRole('button', { name: 'Select Tokenizer capability', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click();
  await expect.poll(async () => (await state()).live).toBe(0);
});

test('shared structure retains geometry, exact QKV hit sets and semantic selection across nonconsecutive instances', async ({ page }, info) => {
  await open(page, 'templates');
  await openShared(page, 'shared-full-attention'); await ready(page);
  await expect(graph(page)).toHaveAttribute('data-template-instance-id', '');
  await expect(page.getByText('No instance selected; weights require a choice.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'View in model', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  await page.locator('[data-id="layer-0.attention.Q"]').getByRole('button', { name: 'Inspect Q', exact: true }).click();
  await expect(page.locator('output')).toHaveText('Structure only: Q');
  const before = await stableState(page);
  const picker = page.getByLabel('Shared structure instance', { exact: true });
  await expect(picker.locator('option').nth(2)).toHaveText('Decoder layers / Layer 2 · family instance 2 of 2');
  await picker.selectOption('layer-2.attention'); await ready(page);
  await expect(page.locator('output')).toHaveText('');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-2.attention.Q');
  await expect(graph(page)).toHaveAttribute('data-source-node-ids', /layer-2.attention.Q/);
  await unchanged(page, before);
  const fanout = ['Q', 'K', 'V'].map((name) => connection(page, 'layer-0.attention', 'x', `layer-0.attention.${name}`, 'x'));
  for (const edge of fanout) await expect(edge).toHaveCount(1);
  const trunk = await fanoutPoint(page, fanout, fanout);
  await page.mouse.move(trunk.x, trunk.y); await emphasized(page, fanout);
  const branch = await fanoutPoint(page, fanout, [fanout[1]!]);
  await page.mouse.move(branch.x, branch.y); await emphasized(page, [fanout[1]!]);
  await hoverDot(port(page, 'layer-0.attention.V', 'x')); await emphasized(page, [fanout[2]!]);
  await unchanged(page, before);
  await fanout[1]!.focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog', { name: 'Connection inspection', exact: true })).toContainText('layer-2.attention.K');
  const ids = await fanout[1]!.getAttribute('data-original-edge-ids');
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Previous shared instance', exact: true }).click();
  await expect(graph(page)).toHaveAttribute('data-template-instance-id', 'layer-0.attention');
  expect(await fanout[1]!.getAttribute('data-original-edge-ids')).not.toBe(ids);
  await emphasized(page, [fanout[1]!]); await unchanged(page, before);
  await page.getByRole('button', { name: 'Inspect connection', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Connection inspection', exact: true })).toContainText('layer-0.attention.K');
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Clear connection selection', exact: true }).click();
  await page.getByRole('button', { name: 'Next shared instance', exact: true }).click();
  await page.getByRole('button', { name: 'View in model', exact: true }).click(); await ready(page);
  await expect(graph(page)).toHaveAttribute('data-template-id', '');
  // The minimal toolbar's selected-item action reveals the exact selected source.
  // The root card's separate View in model action remains covered by card navigation.
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-2.attention.Q');
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  await expect(graph(page)).toHaveAttribute('data-template-instance-id', 'layer-2.attention');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-2.attention.Q');
  const restored = await stableState(page);
  expect({ ...restored, layoutCount: before.layoutCount }).toEqual(before);
  await capture(page, info, 'shared-attention-instance-2');
  for (const width of [1178, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await picker.selectOption('layer-0.attention');
    await picker.selectOption('layer-2.attention');
    expect(await graph(page).getAttribute('data-layout-count')).toBe(restored.layoutCount);
    await capture(page, info, `shared-controls-${width}`);
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole('button', { name: 'Model overview', exact: true }).click(); await ready(page);
  await findComponent(page, 'layer-2.attention.K'); await ready(page);
  await page.getByRole('button', { name: 'Explore structure', exact: true }).click(); await ready(page);
  await expect(picker).toHaveValue('layer-2.attention');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-2.attention.K');
});

test('optional template metadata leaves ordinary overview, exhaustive projection, routing and isolation unchanged', async ({ page }) => {
  const states = [];
  for (const fixture of ['templates-absent', 'templates']) {
    await open(page, fixture);
    await expect(page.locator('[data-family-id]')).toHaveCount(fixture === 'templates' ? 1 : 0);
    const overview = await stableState(page);
    await graphAction(page, 'Show all operations'); await ready(page);
    const exhaustive = await stableState(page);
    await findComponent(page, 'layer-2.attention'); await ready(page);
    await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
    await page.getByRole('button', { name: 'Fit view', exact: true }).click();
    const isolated = await stableState(page);
    states.push({ overview, exhaustive, isolated });
  }
  expect(states[1]).toEqual(states[0]);
});
