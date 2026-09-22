import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type { Layout } from '../src/architecture-explorer/graph';
import { cameraProbe, releaseLayouts } from './architecture-camera-probe';
import { findComponent, graphAction, openShared, selectComponent } from './architecture-controls';
import { minimumOverviewScale, overviewScale, visibleBounds } from '../src/architecture-explorer/overview';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;
declare global { interface Window { overviewLayouts: Layout[]; paintedNodes: string[][] } }
const panel = (page: Page) => page.getByLabel('Architecture graph', { exact: true });
async function ready(page: Page) {
  await expect(panel(page)).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByRole('alert')).toHaveCount(0);
}
const sample = (page: Page) => page.evaluate(() => ({ ...window.cameraProbe.sample(), layouts: window.overviewLayouts,
  completed: window.cameraProbe.events.filter((e) => e.event.endsWith('completed')),
  requested: window.cameraProbe.events.filter((e) => e.event.endsWith('requested')), painted: window.paintedNodes }));
test.beforeEach(async ({ page }) => {
  await cameraProbe(page);
  await page.addInitScript(() => {
    window.overviewLayouts = []; window.paintedNodes = [];
    const Original = window.Worker;
    window.Worker = class extends Original {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener('message', (event: MessageEvent<{ layout?: Layout }>) => {
          if (event.data.layout) window.overviewLayouts.push(event.data.layout);
        });
      }
    };
    const install = window.cameraProbe.install;
    window.cameraProbe.install = (store, flow) => {
      install(store, flow);
      if (!window.paintedNodes.length) store.subscribe((s, previous) => {
        if (s.nodes !== previous.nodes) window.paintedNodes.push(s.nodes.map((n) => n.id));
      });
    };
  });
});

for (const fixture of ['overview-compact', 'overview-synthetic', 'overview-wide', 'overview-fanout', 'overview-training', 'interface-visual', 'interface-many', 'mixed-stacks']) {
  test(`${fixture}: one readable level or one bounded collapsed fallback, top aligned`, async ({ page }, info) => {
    await page.goto(`${harness}?fixture=${fixture}`); await ready(page);
    const state = await sample(page), candidate = state.layouts[0]!, final = state.layouts.at(-1)!;
    const fits = overviewScale(visibleBounds(candidate), state.actual[0]!, state.actual[1]!)! >= minimumOverviewScale;
    expect(state.layouts).toHaveLength(fits ? 1 : 2);
    expect(candidate.projection.nodes.filter((n) => n.expanded)).toHaveLength(1);
    const outer = candidate.projection.nodes.find((n) => n.expanded)!;
    expect(candidate.projection.nodes.every((n) => n.id === outer.id || n.parentId === outer.id)).toBe(true);
    if (!fits) {
      expect(final.boxes.map((b) => b.id)).toEqual([outer.id]);
      expect(state.painted.every((ids) => ids.length <= 1)).toBe(true);
    }
    expect(state.camera[2]).toBeGreaterThanOrEqual(minimumOverviewScale);
    expect(state.camera[2]).toBeLessThanOrEqual(1);
    const bounds = visibleBounds(final);
    expect(bounds.y * state.camera[2]! + state.camera[1]!).toBeCloseTo(16);
    if (bounds.width * state.camera[2]! <= state.actual[0]! - 32)
      expect((bounds.x + bounds.width / 2) * state.camera[2]! + state.camera[0]!).toBeCloseTo(state.actual[0]! / 2);
    expect(state.requested.map((e) => e.event)).toEqual(['setViewport requested']);
    expect(state.completed.map((e) => e.event)).toEqual(['setViewport completed']);
    await info.attach('overview-metrics', { body: JSON.stringify({ fixture, viewport: state.actual, candidate: visibleBounds(candidate),
      final: bounds, candidateScale: overviewScale(visibleBounds(candidate), state.actual[0]!, state.actual[1]!), camera: state.camera,
      layouts: state.layouts.length, cameraRequests: state.requested.length }, null, 2), contentType: 'application/json' });
    await info.attach('overview', { body: await page.screenshot(), contentType: 'image/png' });
  });
}

test('Collapse all exits isolation, preserves hidden selection and fits final bounds exactly once', async ({ page }, info) => {
  await page.goto(`${harness}?fixture=templates`); await ready(page);
  await findComponent(page, 'layer-2.attention'); await ready(page);
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await selectComponent(page, 'layer-2.attention.Q');
  await page.evaluate(() => { window.cameraProbe.events = []; });
  await graphAction(page, 'Collapse model'); await ready(page);
  const state = await sample(page);
  expect(state.nodes).toBe(1);
  expect(state.requested.map((e) => e.event)).toEqual(['setViewport requested']);
    expect(state.completed.map((e) => e.event)).toEqual(['setViewport completed']);
  expect(state.requested[0]!.nodes).toBe(1);
  expect(state.camera[2]).toBeLessThanOrEqual(1);
  await expect(panel(page)).toHaveAttribute('data-scope-id', '');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-2.attention.Q');
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  await expect.poll(async () => (await sample(page)).camera).toEqual(state.camera);
  await info.attach('collapse-final', { body: JSON.stringify(state.requested), contentType: 'application/json' });
  const retained = await sample(page);
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click(); await ready(page);
  expect((await sample(page)).camera).toEqual(retained.camera);
  expect((await sample(page)).nodes).toBe(1);
});

test('user camera movement supersedes pending initialization and a pending global collapse', async ({ page }) => {
  await page.goto(`${harness}?fixture=overview-compact`); await ready(page);
  for (const action of ['initialize', 'collapse']) {
    await page.evaluate(() => { window.cameraProbe.holdLayout = true; window.cameraProbe.events = []; });
    if (action === 'initialize') await page.getByLabel('Fixture', { exact: true }).selectOption('overview-wide');
    else await graphAction(page, 'Collapse all');
    await expect.poll(() => page.evaluate(() => window.cameraProbe.layouts.length)).toBeGreaterThan(0);
    await graphAction(page, 'Zoom graph in');
    const user = await sample(page);
    await releaseLayouts(page); await ready(page);
    expect((await sample(page)).camera).toEqual(user.camera);
    // Initial user movement also suppresses the uncommitted readability collapse.
    if (action === 'initialize') expect((await sample(page)).nodes).toBeGreaterThan(1);
  }
});

test('a new component choice or model supersedes obsolete collapse work', async ({ page }) => {
  await page.goto(`${harness}?fixture=templates`); await ready(page);
  await findComponent(page, 'layer-2.attention'); await ready(page);
  await page.evaluate(() => { window.cameraProbe.holdLayout = true; window.cameraProbe.events = []; });
  await graphAction(page, 'Collapse all');
  await expect.poll(() => page.evaluate(() => window.cameraProbe.layouts.length)).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await releaseLayouts(page); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'layer-2.attention');
  expect((await sample(page)).requested).toHaveLength(1);
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  await page.evaluate(() => { window.cameraProbe.holdLayout = true; });
  await graphAction(page, 'Collapse all');
  await expect.poll(() => page.evaluate(() => window.cameraProbe.layouts.length)).toBeGreaterThan(0);
  await page.getByLabel('Fixture', { exact: true }).selectOption('overview-compact');
  await releaseLayouts(page); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-graph-id', 'overview-compact');
  expect((await sample(page)).nodes).toBe(2);
});

test('zero-to-nonzero viewport waits without rendering or classifying the candidate', async ({ page }) => {
  await page.route('**/tests/architecture.html*', async (route) => {
    const response = await route.fetch();
    await route.fulfill({ response, body: (await response.text()).replace('</head>', '<style id="zero">.architecture-flow { display:none !important }</style></head>') });
  });
  await page.goto(`${harness}?fixture=overview-compact`);
  await expect.poll(() => page.evaluate(() => window.overviewLayouts.length)).toBe(1);
  expect((await sample(page)).nodes).toBe(0);
  expect((await sample(page)).requested).toHaveLength(0);
  await page.locator('#zero').evaluate((e) => e.remove()); await ready(page);
  expect((await sample(page)).layouts).toHaveLength(1);
  expect((await sample(page)).nodes).toBe(2);
});

test('selection during delayed camera completion does not request a second initialization', async ({ page }) => {
  await page.addInitScript(() => { window.cameraProbe.holdCompletion = true; });
  await page.goto(`${harness}?fixture=overview-compact`);
  await expect.poll(() => page.evaluate(() => window.cameraProbe.completions.length)).toBe(1);
  await selectComponent(page, 'Operation');
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  expect((await sample(page)).requested).toHaveLength(1);
  await page.evaluate(() => { window.cameraProbe.holdCompletion = false; window.cameraProbe.completions.splice(0).forEach((done) => done()); });
  await ready(page);
  expect((await sample(page)).requested).toHaveLength(1);
});

test('initial worker failure retries the bounded candidate policy without losing source records', async ({ page }) => {
  await page.addInitScript(() => {
    let first = true;
    const Original = window.Worker;
    window.Worker = class extends Original {
      postMessage(message: unknown) {
        if (first) {
          first = false;
          queueMicrotask(() => this.onerror?.call(this, new ErrorEvent('error', { message: 'Injected failure' })));
        } else super.postMessage(message);
      }
    };
  });
  await page.goto(`${harness}?fixture=overview-wide`);
  await expect(page.getByRole('alert')).toContainText('Layout failed');
  await page.getByRole('button', { name: 'Retry layout', exact: true }).click(); await ready(page);
  const state = await sample(page);
  expect(state.nodes).toBe(1); expect(state.layouts).toHaveLength(2);
  await graphAction(page, 'Show all operations'); await ready(page);
  expect((await sample(page)).nodes).toBe(49);
});

test('rapid individual expansion supersedes pending collapse framing and keeps the user zoom', async ({ page }) => {
  await page.goto(`${harness}?fixture=overview-compact`); await ready(page);
  await graphAction(page, 'Zoom graph in');
  const before = await sample(page);
  await page.evaluate(() => { window.cameraProbe.holdLayout = true; window.cameraProbe.events = []; });
  await graphAction(page, 'Collapse all');
  await expect.poll(() => page.evaluate(() => window.cameraProbe.layouts.length)).toBeGreaterThan(0);
  await page.locator('[data-node-id="Model"] .architecture-browser-disclosure').click();
  await releaseLayouts(page); await ready(page);
  const after = await sample(page);
  expect(after.nodes).toBe(2);
  expect(after.camera[2]).toBe(before.camera[2]);
  expect(after.requested.filter((e) => e.event === 'fitView requested')).toHaveLength(0);
});

test('global collapse never turns a structure-only role into a concrete anchor instance', async ({ page }) => {
  await page.goto(`${harness}?fixture=templates`); await ready(page);
  await openShared(page, 'shared-full-attention'); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-template-instance-id', '');
  await graphAction(page, 'Collapse model'); await ready(page);
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveCount(0);
  expect((await sample(page)).nodes).toBe(1);
});

test('global collapse retains a concrete nonzero shared port across explorer reopening', async ({ page }) => {
  await page.goto(`${harness}?fixture=templates`); await ready(page);
  await openShared(page, 'shared-full-attention'); await ready(page);
  await page.getByLabel('Shared structure instance', { exact: true }).selectOption('layer-2.attention'); await ready(page);
  const port = page.locator('.architecture-port[data-node-id="layer-0.attention"][data-port-id="x"]');
  await port.focus(); await port.press('Enter');
  await graphAction(page, 'Collapse model'); await ready(page);
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click(); await ready(page);
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  const inspected = await page.locator('output').textContent();
  expect(inspected).toContain('layer-2.attention');
  expect(inspected).not.toContain('templatePort');
  expect(inspected).not.toContain('layer-0.attention');
});

test('Show all after global collapse keeps the Model open when one child contracts', async ({ page }) => {
  await page.goto(`${harness}?fixture=templates`); await ready(page);
  await graphAction(page, 'Collapse all'); await ready(page);
  await graphAction(page, 'Show all operations'); await ready(page);
  const before = await sample(page);
  await selectComponent(page, 'layer-2.attention');
  await graphAction(page, 'Toggle selected group'); await ready(page);
  const after = await sample(page), nodes = after.layouts.at(-1)!.projection.nodes;
  expect(after.nodes).toBeLessThan(before.nodes);
  expect(nodes.find((n) => n.presentation === 'model')?.expanded).toBe(true);
  expect(nodes.some((n) => n.id === 'layer-0.attention.Q')).toBe(true);
  expect(nodes.some((n) => n.id === 'layer-2.attention.Q')).toBe(false);
});
