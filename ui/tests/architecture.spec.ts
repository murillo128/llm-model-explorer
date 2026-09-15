import { findComponent, graphAction, graphPreference, viewOptions } from './architecture-controls';
import { expect, test } from '@playwright/test';
import { contractResponse, referenceFixture, references } from './architecture-fixtures';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;
test('nested expansion, instance identity, ports, dimensions, keyboard and camera restoration', async ({ page }, info) => {
  await page.goto(harness);
  const graph = page.getByLabel('Architecture graph', { exact: true });
  await expect(graph).toHaveAttribute('data-visible-nodes', '3');
  await page.getByRole('button', { name: /Explore stack/ }).click();
  await page.getByRole('combobox', { name: /Expand instance of/ }).selectOption('layer1');
  await expect(graph).toHaveAttribute('data-visible-nodes', '5');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer1');
  await expect(page.getByRole('button', { name: 'Select linear1', exact: true })).toBeVisible();
  const label = page.getByRole('button', { name: 'Select layer1', exact: true });
  await label.focus();
  const before = await label.boundingBox();
  await page.keyboard.press('ArrowLeft'); await expect(graph).toHaveAttribute('data-visible-nodes', '4');
  await expect.poll(async () => {
    const after = (await label.boundingBox())!;
    return Math.max(Math.abs(after.x - before!.x), Math.abs(after.y - before!.y));
  }).toBeLessThan(2);
  await page.keyboard.press('ArrowRight'); await expect(graph).toHaveAttribute('data-visible-nodes', '5');
  await graphAction(page, 'Show all operations');
  await expect(graph).toHaveAttribute('data-visible-nodes', '6');
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  expect(JSON.parse((await graph.getAttribute('data-represented-edge-ids'))!)).toEqual(contractResponse.graph.edges.map((edge) => edge.id));
  // The leave + between forwarding segments form one complete visible route.
  await expect(page.locator('.architecture-connection[data-source-node="linear0"][data-source-port="out"][data-target-node="layer1"][data-target-port="in"]')).toHaveCount(1);
  const shapeText = page.locator('.architecture-edge-label').filter({ hasText: /\[2\]/ });
  await expect(shapeText).toHaveCount(0);
  await graphPreference(page, 'Show dimensions', true); await expect(shapeText.first()).toBeVisible();
  await info.attach('expanded-port-graph', { body: await page.screenshot(), contentType: 'image/png' });
  await findComponent(page, 'linear1');
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).focus(); await page.keyboard.press('Enter');
  await expect(page.locator('output')).toContainText('linear1');
  await graphAction(page, 'Zoom graph in');
  const camera = page.locator('.react-flow__viewport');
  const transform = await camera.getAttribute('style');
  expect(await page.getByLabel('Untransformed prompt').evaluate((e) => getComputedStyle(e).transform)).toBe('none');
  await page.getByRole('button', { name: 'Toggle explorer' }).click();
  await expect(graph).toHaveCount(0);
  await page.getByRole('button', { name: 'Toggle explorer' }).click();
  await expect(graph).toHaveAttribute('data-visible-nodes', '6');
  await expect(camera).toHaveAttribute('style', transform!);
  await viewOptions(page);
  await expect(page.getByLabel('Show dimensions')).toBeChecked();
  await page.keyboard.press('Escape');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'linear1');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true);
});

test('safe labels and expressions remain inert; repeated unmount terminates workers', async ({ page }) => {
  await page.addInitScript(() => {
    const stats = { created: 0, live: 0 };
    Object.assign(window, { architectureWorkers: stats });
    const Original = window.Worker;
    window.Worker = class extends Original {
      private live = true;
      constructor(url: string | URL, options?: WorkerOptions) { super(url, options); stats.created++; stats.live++; }
      terminate() { if (this.live) { stats.live--; this.live = false; } super.terminate(); }
    };
  });
  const dialogs: string[] = [];
  page.on('dialog', (dialog) => { dialogs.push(dialog.message()); void dialog.dismiss(); });
  await page.goto(`${harness}?inert`);
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('data-visible-nodes', '3');
  await graphAction(page, 'Show all operations');
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Select <img src=x onerror=alert(1)>', exact: true })).toBeVisible();
  await graphPreference(page, 'Show dimensions', true);
  await expect(page.locator('.architecture-edge-label').filter({ hasText: 'window.alert(1)' })).toBeVisible();
  expect(dialogs).toEqual([]); await expect(page.locator('img')).toHaveCount(0);
  for (let i = 0; i < 3; i++) {
    await graphAction(page, 'Show all operations');
    await page.getByRole('button', { name: 'Toggle explorer' }).click();
    await expect.poll(() => page.evaluate(() => (window as typeof window & { architectureWorkers: { live: number } }).architectureWorkers.live)).toBe(0);
    await page.getByRole('button', { name: 'Toggle explorer' }).click();
    await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('data-visible-nodes', '6');
  }
});

for (const reference of references) test(`full-size ${reference.name}: all instances, bounded document, layout and memory evidence`, async ({ page }, info) => {
  await page.goto(harness);
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption(reference.name);
  const graph = page.getByLabel('Architecture graph', { exact: true });
  await expect(graph).toHaveAttribute('data-graph-id', `fixture-${reference.name}`);
  const started = performance.now();
  await graphAction(page, 'Show all operations');
  const fixture = referenceFixture(reference.name);
  await expect(graph).toHaveAttribute('data-visible-nodes', String(fixture.graph.nodes.length));
  const elapsed = performance.now() - started;
  await findComponent(page, `layer-${reference.stacks.length - 1}-${reference.stacks.at(-1)! - 1}-op-31`);
  await expect(page.getByRole('button', { name: 'Select full_attention operation 31', exact: true }).last()).toBeVisible();
  await expect(graph).toHaveAttribute('data-node-count', String(fixture.graph.nodes.length));
  const client = await page.context().newCDPSession(page);
  const heap = await client.send('Runtime.getHeapUsage');
  const metrics = { fixture: reference.name, revision: reference.revision, viewport: page.viewportSize(), nodes: fixture.graph.nodes.length,
    edges: fixture.graph.edges.length, renderedNodes: await page.locator('.react-flow__node').count(), layoutMs: Number(await graph.getAttribute('data-layout-ms')), interactionMs: elapsed, heap };
  await info.attach('architecture-layout-memory', { body: JSON.stringify(metrics, null, 2), contentType: 'application/json' });
  console.log(JSON.stringify(metrics));
  expect(metrics.renderedNodes).toBeLessThan(metrics.nodes);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true);
  await graphAction(page, 'Collapse all');
  await expect(graph).toHaveAttribute('data-visible-nodes', '3');
  // Selection stays concrete even while its ancestors hide it; Center selected reveals it again.
  await page.getByRole('button', { name: 'Center selected' }).click();
  await expect(page.getByRole('button', { name: 'Select full_attention operation 31', exact: true }).last()).toBeVisible();
});
