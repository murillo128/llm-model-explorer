import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type { Graph, Layout } from '../src/architecture-explorer/graph';
import type { ProjectionOptions } from '../src/architecture-explorer/projection';
import { findComponent, graphAction, graphPreference } from './architecture-controls';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;
interface IsolationProbe {
  requests: { graph: Graph; options: ProjectionOptions }[]; layouts: Layout[];
  reject: boolean; hold: boolean; release?: () => void;
}
declare global { interface Window { isolationProbe: IsolationProbe } }
const panel = (page: Page) => page.getByLabel('Architecture graph', { exact: true });
async function ready(page: Page) {
  await expect(panel(page)).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}
async function state(page: Page) {
  await ready(page);
  return page.evaluate(() => ({ options: window.isolationProbe.requests.at(-1)!.options,
    camera: document.querySelector('.react-flow__viewport')?.getAttribute('style'),
    selection: document.querySelector('[aria-label="Graph selection"]')?.getAttribute('data-node-id'),
    layout: window.isolationProbe.layouts.at(-1)!, count: window.isolationProbe.requests.length }));
}
async function isolate(page: Page, id: string) {
  await findComponent(page, id); await ready(page);
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await expect(panel(page)).toHaveAttribute('data-scope-id', id); await ready(page);
}
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.isolationProbe = { requests: [], layouts: [], reject: false, hold: false };
    const Original = window.Worker;
    window.Worker = class extends Original {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener('message', (event: MessageEvent<{ layout?: Layout }>) => {
          if (event.data.layout) window.isolationProbe.layouts.push(event.data.layout);
        });
      }
      override postMessage(message: unknown, transfer: Transferable[] | StructuredSerializeOptions = []) {
        if (message && typeof message === 'object' && 'graph' in message) {
          const probe = window.isolationProbe;
          probe.requests.push(structuredClone(message) as IsolationProbe['requests'][number]);
          const callback = this.onmessage;
          if (probe.reject) {
            probe.reject = false;
            setTimeout(() => callback?.call(this, new MessageEvent('message', { data: { error: 'Injected layout failure' } })), 10);
            return;
          }
          if (probe.hold) {
            probe.hold = false;
            // Preserve a queued old callback, even after worker cancellation and replacement.
            const layout = probe.layouts.at(-1)!;
            probe.release = () => callback?.call(this, new MessageEvent('message', { data: { layout } }));
            return;
          }
        }
        if (Array.isArray(transfer)) super.postMessage(message, transfer); else super.postMessage(message, transfer);
      }
    };
  });
  await page.goto(harness);
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('components'); await ready(page);
});

test('nested isolation and Back restore exact camera, selected instance, expansion and filters', async ({ page }) => {
  await findComponent(page, 'layer-3'); await ready(page);
  await graphAction(page, 'Toggle selected group'); await ready(page);
  await graphPreference(page, 'Unused interfaces', true); await ready(page);
  await graphAction(page, 'Zoom graph in'); await ready(page);
  const before = await state(page), canvas = await page.locator('.react-flow').elementHandle();
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'layer-3');
  await expect(page.getByText('Isolated component', { exact: true })).toBeVisible();
  await findComponent(page, 'layer-3.attention'); await ready(page);
  const layer = await state(page);
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await expect(page.getByRole('navigation', { name: 'Architecture focus' })).toContainText('Decoder layers');
  await expect(page.getByRole('navigation', { name: 'Architecture focus' })).toContainText('Layer 3');
  await expect(page.getByRole('navigation', { name: 'Architecture focus' })).toContainText('Full attention');
  await graphAction(page, 'Zoom graph out');
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  const restoredLayer = await state(page);
  expect(restoredLayer.options).toEqual(layer.options); expect(restoredLayer.camera).toBe(layer.camera);
  expect(restoredLayer.selection).toBe(layer.selection);
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  const restored = await state(page);
  expect(restored.options).toEqual(before.options); expect(restored.camera).toBe(before.camera); expect(restored.selection).toBe(before.selection);
  expect(restored.layout.boxes).toEqual(before.layout.boxes); expect(restored.layout.routes).toEqual(before.layout.routes);
  expect(await canvas!.evaluate((element) => element === document.querySelector('.react-flow'))).toBe(true);
});

test('component expansion, model-wide exhaustive access and View in model retain exact scope semantics', async ({ page }) => {
  await isolate(page, 'layer-3');
  await page.getByRole('button', { name: 'Expand component', exact: true }).click(); await ready(page);
  const expanded = await state(page);
  expect(expanded.options.exhaustive).toBe(false);
  expect(expanded.layout.projection.nodes.filter((node) => node.record?.kind === 'operation').every((node) => node.id.startsWith('layer-3.'))).toBe(true);
  expect(expanded.layout.projection.nodes.some((node) => node.id === 'model' || node.id === 'layer-2')).toBe(false);
  await findComponent(page, 'layer-3.attention.Q'); await ready(page);
  await page.getByRole('button', { name: 'View in model', exact: true }).click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', '');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-3.attention.Q');
  await expect(page.locator('[data-id="layer-3.attention.Q"]')).toBeVisible();
  await isolate(page, 'layer-3.attention');
  await graphAction(page, 'Show all operations in model'); await ready(page);
  const exhaustive = await state(page);
  expect(exhaustive.options.scope).toBeUndefined(); expect(exhaustive.options.exhaustive).toBe(true);
  expect(exhaustive.layout.projection.nodes.map((node) => node.id).sort()).toEqual(
    (await page.evaluate(() => window.isolationProbe.requests.at(-1)!.graph.nodes.map((node) => node.id))).sort());
  expect(exhaustive.layout.projection.hiddenEdgeIds).toEqual([]); expect(exhaustive.layout.projection.filteredEdgeIds).toEqual([]);
});

test('the existing derived MLP exposes explicit isolation without changing its source operations', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('connections'); await ready(page);
  await findComponent(page, 'layer-3.gate'); await ready(page);
  await graphAction(page, 'Focus MLP'); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  await page.locator('[data-id="mlp:layer-3.gate"]').getByRole('button', { name: 'Inspect MLP', exact: true }).click();
  await page.getByRole('dialog', { name: 'Group inspection', exact: true }).getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'mlp:layer-3.gate');
  const current = await state(page);
  expect(current.layout.projection.nodes.filter((node) => node.record?.kind === 'operation').map((node) => node.id)).toEqual([
    'layer-3.gate', 'layer-3.up', 'layer-3.silu', 'layer-3.multiply', 'layer-3.down']);
  await page.getByRole('button', { name: 'View in model', exact: true }).click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', '');
  await expect(page.locator('[data-id="mlp:layer-3.gate"]')).toBeVisible();
});

test('unrelated model growth leaves a small isolated component readable and its bounds unchanged', async ({ page }) => {
  await isolate(page, 'layer-3.mlp');
  const before = await state(page);
  const label = page.locator('[data-id="layer-3.gate"] .architecture-node-label');
  await expect(label).toBeVisible();
  const labelBox = (await label.boundingBox())!;
  expect(labelBox.height).toBeGreaterThanOrEqual(12);
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('components-large'); await ready(page);
  await isolate(page, 'layer-3.mlp');
  const after = await state(page);
  expect([after.layout.width, after.layout.height]).toEqual([before.layout.width, before.layout.height]);
  expect(after.layout.boxes).toEqual(before.layout.boxes); expect(after.layout.ports).toEqual(before.layout.ports);
  expect((await label.boundingBox())!.height).toBeCloseTo(labelBox.height, 2);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  expect((await state(page)).count).toBe(after.count);
});

test('search across scopes preserves independent stack context; explorer and model replacement fence views', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('mixed-stacks'); await ready(page);
  await isolate(page, 'encoder.layer-3.attention');
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'encoder.layer-3.attention');
  await findComponent(page, 'predictor.layer-1.attention.Q'); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', '');
  await expect(page.getByRole('combobox', { name: 'Expand instance of Predictor layers', exact: true })).toHaveValue('predictor.layer-1');
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  expect((await state(page)).layout.projection.nodes.filter((node) => node.record).map((node) => node.id)).toEqual(['predictor.layer-1.attention.Q']);
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('partial'); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', '');
  await isolate(page, 'unknown-component');
  await expect(page.getByText('Partial coverage', { exact: true })).toBeVisible();
});

test('layout failure allows retry and Back; queued late results cannot replace a newer scope or model', async ({ page }) => {
  await findComponent(page, 'layer-3.attention'); await ready(page);
  const before = await state(page);
  await page.evaluate(() => { window.isolationProbe.reject = true; });
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Layout failed');
  await page.getByRole('button', { name: 'Retry layout', exact: true }).click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'layer-3.attention');
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  expect((await state(page)).camera).toBe(before.camera);
  await page.evaluate(() => { window.isolationProbe.hold = true; });
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await expect(panel(page)).toHaveAttribute('aria-busy', 'true');
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  const current = await state(page);
  await page.evaluate(() => window.isolationProbe.release?.()); await ready(page);
  expect(await state(page)).toEqual(current);
  await page.evaluate(() => { window.isolationProbe.hold = true; });
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('partial'); await ready(page);
  const replacement = await state(page);
  await page.evaluate(() => window.isolationProbe.release?.()); await ready(page);
  expect(await state(page)).toEqual(replacement);
});
