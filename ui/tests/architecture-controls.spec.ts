import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type { Graph, Layout } from '../src/architecture-explorer/graph';
import type { ProjectionOptions } from '../src/architecture-explorer/projection';
import { assertTraceability } from './architecture-invariants';
import { chooseInstance, findComponent, graphAction, viewOptions } from './architecture-controls';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;
interface ControlProbe { requests: { graph: Graph; options: ProjectionOptions }[]; layouts: Layout[] }
declare global { interface Window { architectureControlProbe: ControlProbe } }
async function ready(page: Page) {
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('aria-busy', 'false');
  await expect(page.locator('.architecture-node').first()).toBeAttached();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}
async function snapshot(page: Page) {
  await ready(page);
  return page.evaluate(() => ({
    requests: window.architectureControlProbe.requests,
    layouts: window.architectureControlProbe.layouts,
    count: document.querySelector('[data-layout-count]')?.getAttribute('data-layout-count'),
    camera: document.querySelector('.react-flow__viewport')?.getAttribute('style'),
  }));
}
function assertTransportProjection(graph: Graph, layout: Layout, exhaustive = false) {
  // postMessage and Playwright serialize requests and replies separately. Prove
  // exact source record contents before restoring references for the shared
  // in-process identity oracle; no projection fields or expectations change.
  const projection = structuredClone(layout.projection);
  for (const node of projection.nodes) if (node.record) {
    const source = graph.nodes.find((n) => n.id === node.record!.id)!;
    expect(node.record).toEqual(source); node.record = source;
  }
  for (const edge of projection.edges) edge.paths = edge.paths.map((path) => path.map((item) => {
    const source = graph.edges.find((e) => e.id === item.id)!;
    expect(item).toEqual(source); return source;
  }));
  assertTraceability(graph, projection, exhaustive);
}
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.architectureControlProbe = { requests: [], layouts: [] };
    const Original = window.Worker;
    window.Worker = class extends Original {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener('message', (event: MessageEvent<{ layout?: Layout }>) => {
          if (event.data.layout) window.architectureControlProbe.layouts.push(event.data.layout);
        });
      }
      override postMessage(message: unknown, transfer: Transferable[] | StructuredSerializeOptions = []) {
        if (message && typeof message === 'object' && 'graph' in message) window.architectureControlProbe.requests.push(structuredClone(message) as ControlProbe['requests'][number]);
        if (Array.isArray(transfer)) super.postMessage(message, transfer); else super.postMessage(message, transfer);
      }
    };
  });
  await page.goto(harness); await ready(page);
});

test('search and overflow preserve the canvas, projection, generated routes, camera and layout count', async ({ page }) => {
  const canvas = await page.locator('.react-flow').elementHandle();
  const before = await snapshot(page);
  assertTransportProjection(before.requests.at(-1)!.graph, before.layouts.at(-1)!);
  await page.getByRole('searchbox', { name: 'Search components', exact: true }).focus();
  const search = page.getByRole('searchbox', { name: 'Search components', exact: true });
  await expect(search).toBeFocused();
  await search.fill('linear');
  await page.keyboard.press('ArrowDown'); await page.keyboard.press('ArrowUp');
  await search.press('Home'); await search.type('absent ');
  await expect(page.getByRole('group', { name: 'Model search results', exact: true })).toContainText('No matching components');
  expect(await snapshot(page)).toEqual(before);
  await search.press('Escape');
  await expect(page.getByRole('searchbox', { name: 'Search components', exact: true })).toBeFocused();
  const options = await viewOptions(page);
  await expect(options.getByRole('button', { name: 'Show all operations' })).toBeFocused();
  await expect(options.getByLabel('Show dimensions')).not.toBeChecked();
  for (const name of ['Collapse all', 'Center selection', 'Zoom graph in', 'Zoom graph out']) await expect(options.getByRole('button', { name, exact: true })).toBeVisible();
  for (const name of ['Unused interfaces', 'Context', 'Group MLP']) await expect(options.getByLabel(name, { exact: true })).toBeVisible();
  await options.locator('summary').click();
  expect(await snapshot(page)).toEqual(before);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'View options', exact: true })).toBeFocused();
  expect(await canvas!.evaluate((element) => element === document.querySelector('.react-flow'))).toBe(true);
  expect(await snapshot(page)).toEqual(before);
});

test('keyboard search selects before explicit Center reveals collapsed repeated names with parent context and exact source bindings', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('mixed-stacks'); await ready(page);
  await expect(page.getByRole('combobox', { name: /Expand instance of/ })).toHaveCount(0);
  await page.getByRole('searchbox', { name: 'Search components', exact: true }).focus();
  const search = page.getByRole('searchbox', { name: 'Search components', exact: true });
  await search.fill('Q projection');
  const options = page.getByRole('group', { name: 'Model search results', exact: true }).locator('[data-node-id]');
  await expect(options).toHaveCount(5); // Encoder full instances 0/2/3; predictor 0/1.
  const contexts = await options.allTextContents();
  expect(new Set(contexts).size).toBe(5);
  expect(contexts.some((text) => text.includes('Encoder layers') && text.includes('Layer 3'))).toBe(true);
  expect(contexts.some((text) => text.includes('Predictor layers') && text.includes('Layer 1'))).toBe(true);
  await search.fill('Q projection encoder layer 3');
  await expect(options).toHaveCount(1);
  const id = await options.first().getAttribute('data-node-id');
  const compact = await snapshot(page);
  await search.press('Enter'); await ready(page);
  expect(await snapshot(page)).toEqual(compact);
  await page.getByRole('button', { name: 'Center selected', exact: true }).click(); await ready(page);
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', id!);
  await expect(page.getByRole('combobox', { name: /Expand instance of Encoder/ })).toHaveValue('encoder.layer-3');
  const before = await snapshot(page);
  const graph = before.requests.at(-1)!.graph;
  const node = graph.nodes.find((n) => n.id === id)!;
  expect(graph.parameters.find((p) => node.parameter_ids.includes(p.id))?.name).toBe('model.encoder.layers.3.self_attn.q_proj.weight');
  assertTransportProjection(graph, before.layouts.at(-1)!);
  const changedRecord = structuredClone(before.layouts.at(-1)!);
  changedRecord.projection.nodes.find((n) => n.record)!.record!.label = 'wrong source record';
  expect(() => assertTransportProjection(graph, changedRecord)).toThrow();
  const changedEdge = structuredClone(before.layouts.at(-1)!);
  changedEdge.projection.edges[0]!.paths[0]![0]!.target.port_id = 'wrong source port';
  expect(() => assertTransportProjection(graph, changedEdge)).toThrow();

  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  await expect(page.locator('output')).toContainText(id!);
  expect(await snapshot(page)).toEqual(before);
});

test('browser and canvas disclosure preserve the same multi-instance window and neighboring cards', async ({ page }) => {
  const results = [];
  for (const surface of ['canvas', 'browser'] as const) {
    await page.goto(`${harness}?fixture=components`); await ready(page);
    await page.getByRole('button', { name: 'Explore stack Decoder layers', exact: true }).click();
    const before = await snapshot(page), options = before.requests.at(-1)!.options;
    const count = page.viewportSize()!.width <= 760 ? 2 : 4;
    expect(options.repetitions).toEqual({ 'decoder-layers': { start: 0, count } });
    const disclosure = surface === 'canvas' ? page.locator('[data-id="layer-0"] .architecture-expand') :
      page.getByRole('tree', { name: 'Model components', exact: true }).locator('[data-node-id="layer-0"] .architecture-browser-disclosure');
    await disclosure.click();
    const opened = await snapshot(page), projection = opened.layouts.at(-1)!.projection;
    expect(opened.requests.at(-1)!.options.repetitions).toEqual(options.repetitions);
    expect(projection.nodes.find((node) => node.id === 'layer-0')?.expanded).toBe(true);
    for (let index = 1; index < count; index++) {
      expect(projection.nodes.find((node) => node.id === `layer-${index}`)?.expanded).toBe(false);
    }
    assertTransportProjection(opened.requests.at(-1)!.graph, opened.layouts.at(-1)!);
    await disclosure.click();
    const closed = await snapshot(page);
    expect(closed.requests.at(-1)!.options.repetitions).toEqual(options.repetitions);
    expect(closed.layouts.at(-1)!.projection).toEqual(before.layouts.at(-1)!.projection);
    results.push({ options: opened.requests.at(-1)!.options, projection });
  }
  expect(results[1]).toEqual(results[0]);
});

for (const window of ['compact', 'multi-instance'] as const) test(`browser disclosure reveals only the hidden exact instance and preserves the ${window} window`, async ({ page }) => {
  await page.goto(`${harness}?fixture=components-large`); await ready(page);
  if (window === 'multi-instance') await page.getByRole('button', { name: 'Explore stack Decoder layers', exact: true }).click();
  const before = await snapshot(page), options = before.requests.at(-1)!.options;
  const instanceIds = (layout: Layout) => layout.projection.nodes.filter((node) => /^layer-\d+$/.test(node.id)).map((node) => node.id);
  const visible = instanceIds(before.layouts.at(-1)!);
  expect(visible).not.toContain('layer-10');
  await page.getByRole('searchbox', { name: 'Search components', exact: true }).fill('layer-10');
  const disclosure = page.getByRole('group', { name: 'Model search results', exact: true })
    .locator('[data-node-id="layer-10"] .architecture-browser-disclosure');
  await disclosure.click();
  const opened = await snapshot(page);
  expect(opened.requests.at(-1)!.options.repetitions).toEqual(options.repetitions);
  expect(instanceIds(opened.layouts.at(-1)!)).toEqual([...visible, 'layer-10']);
  expect(opened.layouts.at(-1)!.projection.nodes.find((node) => node.id === 'layer-10')?.expanded).toBe(true);
  assertTransportProjection(opened.requests.at(-1)!.graph, opened.layouts.at(-1)!);
  await disclosure.click();
  const closed = await snapshot(page);
  expect(closed.requests.at(-1)!.options.repetitions).toEqual(options.repetitions);
  expect(closed.layouts.at(-1)!.projection).toEqual(before.layouts.at(-1)!.projection);
});

test('breadcrumbs, first/last and mixed variants preserve two independent stack windows and return context', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('mixed-stacks'); await ready(page);
  await page.getByRole('button', { name: 'Explore stack Encoder layers', exact: true }).click(); await ready(page);
  const picker = page.getByRole('combobox', { name: 'Expand instance of Encoder layers', exact: true });
  expect(await picker.locator('option').allTextContents()).toEqual(['Choose instance…', 'Instance 0 · full attention', 'Instance 1 · linear attention', 'Instance 2 · full attention', 'Instance 3 · full attention', 'Instance 4 · linear attention']);
  await page.getByRole('button', { name: 'Open instance', exact: true }).click(); await ready(page);
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'encoder.layer-0');
  await expect(page.getByRole('button', { name: 'Previous instance of Encoder layers', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Next instance of Encoder layers', exact: true }).click(); await ready(page);
  await expect(picker).toHaveValue('encoder.layer-1');
  await expect(picker.locator('option:checked')).toHaveText('Instance 1 · linear attention');
  await picker.selectOption('encoder.layer-4'); await ready(page);
  await expect(page.getByRole('button', { name: 'Next instance of Encoder layers', exact: true })).toBeDisabled();
  await expect(page.locator('.architecture-visible-range')).toHaveText('Visible 4');
  const encoderOptions = (await snapshot(page)).requests.at(-1)!.options.repetitions!['encoder-layers']!;
  // Search can enter a second stack without resetting the encoder's expansion/window.
  await findComponent(page, 'predictor.layer-1.attention.Q'); await ready(page);
  await expect(page.getByRole('combobox', { name: /Expand instance of/ })).toHaveCount(1);
  await expect(page.getByRole('combobox', { name: 'Expand instance of Predictor layers' })).toHaveValue('predictor.layer-1');
  const selected = await snapshot(page);
  expect(selected.requests.at(-1)!.options.repetitions!['encoder-layers']).toEqual(encoderOptions);
  const crumb = page.getByRole('navigation', { name: 'Architecture focus' });
  await expect(crumb).toContainText('Predictor layers'); await expect(crumb).toContainText('Layer 1');
  await crumb.getByRole('button', { name: 'Predictor layers', exact: true }).click(); await ready(page);
  await expect(crumb.getByRole('button', { name: 'Layer 1', exact: true })).toHaveCount(0);
  await expect(page.getByRole('combobox', { name: 'Expand instance of Predictor layers' })).toHaveValue('predictor.layer-1');
  const retained = await snapshot(page);
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click(); await ready(page);
  expect((await snapshot(page)).requests.at(-1)!.options).toEqual(retained.requests.at(-1)!.options);
  await expect(crumb).toContainText('Predictor layers');
  await crumb.getByRole('button', { name: 'Model overview', exact: true }).click(); await ready(page);
  await expect(page.getByRole('combobox', { name: /Expand instance of/ })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Explore stack Encoder layers', exact: true })).toBeVisible();
});

for (const activation of ['pointer', 'keyboard'] as const) test(`cross-stack canvas MLP ${activation} navigation keeps focus and instance controls consistent`, async ({ page }) => {
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('mixed-stacks'); await ready(page);
  await findComponent(page, 'encoder.layer-3.gate'); await ready(page);
  await findComponent(page, 'predictor.layer-1.attention.Q'); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  const before = await snapshot(page);
  const windows = before.requests.at(-1)!.options.repetitions;
  expect(windows).toEqual({ 'encoder-layers': { start: 3, count: 1 }, 'predictor-layers': { start: 1, count: 1 } });
  const mlp = page.locator('[data-id="mlp:encoder.layer-3.gate"] .architecture-node-label');
  // Reach the other expanded stack through camera controls, without changing focus.
  const panel = (await page.locator('.architecture-flow').boundingBox())!;
  const inView = () => mlp.evaluateAll((elements, panel) => {
    // React Flow can cull the target until zoom exposes it; do not wait for
    // attachment before issuing the camera action that makes it reachable.
    const box = elements[0]?.getBoundingClientRect();
    return box !== undefined && box.width > 0 && box.height > 0 && box.x >= panel.x && box.y >= panel.y &&
      box.right <= panel.x + panel.width && box.bottom <= panel.y + panel.height;
  }, panel);
  for (let i = 0; i < 16 && !await inView(); i++) { await graphAction(page, 'Zoom graph out'); await ready(page); }
  expect(await inView()).toBe(true);
  const navigation = page.locator('[data-id="mlp:encoder.layer-3.gate"] .architecture-navigate');
  // The bounded narrow canvas can expose a distant stack at subpixel scale.
  // Focal camera zoom makes its native control actionable without changing focus.
  for (let i = 0; i < 16 && (await navigation.boundingBox())!.width < 12; i++) {
    const box = (await navigation.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -400);
    await expect.poll(async () => (await navigation.boundingBox())!.width).toBeGreaterThan(box.width);
  }
  if (activation === 'pointer') await navigation.click();
  else { await navigation.focus(); await page.keyboard.press('Enter'); }
  await ready(page);
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('data-scope-id', 'mlp:encoder.layer-3.gate');
  await page.getByRole('button', { name: 'View in model', exact: true }).click();
  await ready(page);
  const crumb = page.getByRole('navigation', { name: 'Architecture focus' });
  await expect(crumb).toContainText('Encoder layers');
  await expect(crumb).toContainText('Layer 3');
  await expect(crumb).toContainText('MLP (derived)');
  const picker = page.getByRole('combobox', { name: 'Expand instance of Encoder layers', exact: true });
  await expect(picker).toHaveValue('encoder.layer-3');
  await expect(picker.locator('option:checked')).toHaveText('Instance 3 · full attention');
  await expect(page.getByRole('combobox', { name: 'Expand instance of Predictor layers', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Previous instance of Encoder layers', exact: true })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Next instance of Encoder layers', exact: true })).toBeEnabled();
  await expect(page.locator('.architecture-visible-range')).toHaveText('Visible 3');
  // Explicit card navigation replaces the previously selected other-stack node.
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'mlp:encoder.layer-3.gate');
  const after = await snapshot(page);
  expect(after.requests.at(-1)!.graph).toEqual(before.requests.at(-1)!.graph);
  expect(after.requests.at(-1)!.options.repetitions).toEqual(windows);
  assertTransportProjection(after.requests.at(-1)!.graph, after.layouts.at(-1)!);
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click(); await ready(page);
  await expect(crumb).toContainText('MLP (derived)');
  await expect(picker).toHaveValue('encoder.layer-3');
  expect((await snapshot(page)).requests.at(-1)!.options.repetitions).toEqual(windows);
  await page.getByRole('button', { name: 'Open instance', exact: true }).click(); await ready(page);
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'encoder.layer-3');
  await page.getByRole('button', { name: 'Previous instance of Encoder layers', exact: true }).click(); await ready(page);
  await expect(picker).toHaveValue('encoder.layer-2');
  await page.getByRole('button', { name: 'Next instance of Encoder layers', exact: true }).click(); await ready(page);
  await expect(picker).toHaveValue('encoder.layer-3');
  expect((await snapshot(page)).requests.at(-1)!.options.repetitions).toEqual(windows);
});

test('edge pin, inspection and clear leave focus navigation and layout unchanged', async ({ page }) => {
  await chooseInstance(page, 'Layers', 'layer1');
  await graphAction(page, 'Show all operations');
  await findComponent(page, 'linear1'); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  const before = await snapshot(page);
  const breadcrumb = await page.getByRole('navigation', { name: 'Architecture focus' }).textContent();
  await page.locator('.architecture-connection').first().focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'Close connection inspection' })).toBeFocused();
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Inspect connection', exact: true }).click();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Inspect connection', exact: true })).toBeFocused();
  await page.getByRole('button', { name: 'Clear connection selection', exact: true }).click();
  expect(await page.getByRole('navigation', { name: 'Architecture focus' }).textContent()).toBe(breadcrumb);
  expect(await snapshot(page)).toEqual(before);
  await page.locator('.architecture-connection').first().focus(); await page.keyboard.press('Enter');
  await page.keyboard.press('Escape');
  await findComponent(page, 'linear0'); await ready(page);
  await expect(page.getByRole('button', { name: 'Inspect selected', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Clear connection selection', exact: true })).toHaveCount(0);
});

test('all operations and camera fit remain distinct; popovers stay in the panel at constrained widths', async ({ page }) => {
  // Use the safety suite's connected computation fixture for exhaustive topology
  // assertions; the minimal schema fixture contains open group input interfaces.
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('connections'); await ready(page);
  const before = await snapshot(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  expect((await snapshot(page)).layouts).toEqual(before.layouts);
  await graphAction(page, 'Show all operations'); await ready(page);
  const after = await snapshot(page);
  assertTransportProjection(after.requests.at(-1)!.graph, after.layouts.at(-1)!, true);
  for (const label of ['View options']) {
    await page.getByRole('button', { name: label, exact: true }).click();
    const panel = (await page.getByLabel('Architecture graph', { exact: true }).boundingBox())!;
    const popover = (await page.getByRole('dialog', { name: label, exact: true }).boundingBox())!;
    expect(popover.x).toBeGreaterThanOrEqual(panel.x); expect(popover.y).toBeGreaterThanOrEqual(panel.y);
    expect(popover.x + popover.width).toBeLessThanOrEqual(panel.x + panel.width);
    expect(popover.y + popover.height).toBeLessThanOrEqual(panel.y + panel.height);
    await page.keyboard.press('Escape');
  }
  expect(await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.scrollHeight])).toEqual([page.viewportSize()!.width, page.viewportSize()!.height]);
});


test('partial graphs without repetition retain component navigation and on-demand diagnostics', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('partial'); await ready(page);
  await expect(page.getByText('Partial coverage', { exact: true })).toBeVisible();
  await expect(page.getByRole('combobox', { name: /Expand instance of/ })).toHaveCount(0);
  await findComponent(page, 'unknown-component'); await ready(page);
  await expect(page.getByRole('navigation', { name: 'Architecture focus' })).toContainText('Unknown component');
  const before = await snapshot(page);
  const options = await viewOptions(page); await options.locator('summary').click();
  await expect(options).toContainText('Fixture intentionally includes an unresolved component.');
  expect(await snapshot(page)).toEqual(before);
  await page.keyboard.press('Escape');
});
