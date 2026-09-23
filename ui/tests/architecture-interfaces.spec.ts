import { expect, test } from '@playwright/test';
import type { Locator } from '@playwright/test';
import { viewOptions, graphAction, graphPreference, findComponent, openShared } from './architecture-controls';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;
const panel = '[aria-label="Architecture graph"]';
const ready = async (page: import('@playwright/test').Page) => {
  await expect(page.locator(panel)).toHaveAttribute('aria-busy', 'false');
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
};
const camera = (page: import('@playwright/test').Page) => page.locator('.react-flow__viewport').getAttribute('style');

for (const kind of ['dense', 'hybrid', 'visual'] as const) test(`${kind}: declarations stay as ports across browser, dimensions, collapse and restore`, async ({ page }, info) => {
  const calls: string[] = []; page.on('request', (r) => { if (/\/sessions\//.test(r.url())) calls.push(r.url()); });
  await page.goto(`${harness}?fixture=interface-${kind}`); await ready(page);
  const removed = kind === 'visual' ? ['video', 'context_indices', 'target_indices', 'representations'] : ['Token IDs', 'positions', 'mask', 'current_mask', 'logits', 'tokenizer'];
  const absent = async () => { for (const id of removed) {
    await expect(page.locator(`.react-flow__node[data-id=${JSON.stringify(id)}]`)).toHaveCount(0);
    await expect(page.locator(`.architecture-browser-row[data-node-id=${JSON.stringify(id)}]`)).toHaveCount(0);
  } };
  await absent();
  await info.attach(`${kind}-interfaces`, { body: await page.screenshot(), contentType: 'image/png' });
  const name = kind === 'visual' ? 'video' : 'positions';
  const search = page.getByRole('searchbox', { name: 'Search components' });
  const before = await camera(page), layouts = await page.locator(panel).getAttribute('data-layout-count');
  await search.fill(name); await search.press('Enter');
  expect(await camera(page)).toBe(before);
  await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layouts!);
  await absent();
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Interface inspection' })).toContainText(name);
  await expect(page.getByRole('dialog')).toContainText('independent.fixture');
  await page.getByRole('button', { name: 'Close inspection' }).click();
  await search.fill('');
  for (const dimensions of [true, false]) {
    await graphPreference(page, 'Show dimensions', dimensions); await ready(page);
    await expect(page.locator('.architecture-port-label').filter({ hasText: name }).first()).toBeAttached();
    await absent();
  }
  await graphAction(page, 'Show all operations'); await ready(page); await absent();
  await expect(page.locator(`.architecture-browser-row[data-node-id=${JSON.stringify(kind === 'visual' ? 'predictor' : 'LM head')}]`)).toHaveCount(1);
  await graphAction(page, 'Collapse all'); await ready(page); await absent();
  await page.getByRole('button', { name: 'Toggle explorer' }).click();
  await page.getByRole('button', { name: 'Toggle explorer' }).click(); await ready(page); await absent();
  expect(calls).toEqual([]);
});

test('boundary hover/focus/pinning preserves layout and source connections; explicit isolation retains ports', async ({ page }) => {
  await page.goto(`${harness}?fixture=interface-hybrid`); await ready(page);
  // Exercise both forwarding boundaries after explicitly opening the child.
  await page.locator('[data-node-id="language"] .architecture-browser-disclosure').click(); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  const port = page.locator('.architecture-port[data-node-id="presentation:model"]').filter({ has: page.locator('.architecture-port-dot') }).nth(1);
  const before = await camera(page), layouts = await page.locator(panel).getAttribute('data-layout-count');
  await port.hover();
  await expect(page.locator('.architecture-connection[data-emphasized="true"]')).toHaveCount(2);
  await port.focus(); await port.press('Enter');
  await page.mouse.move(0, 0);
  await expect(page.locator('.architecture-connection[data-emphasized="true"]')).toHaveCount(2);
  expect(await camera(page)).toBe(before);
  await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layouts!);
  await findComponent(page, 'language');
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await expect(page.locator('.architecture-port[data-node-id="language"][data-port-id="positions"]')).toHaveCount(1);
  await expect(page.locator('.architecture-node[data-presentation="external"]')).toHaveCount(1); // Real LM head.
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
});

test('raised label to terminal traversal retains exact hover at fit and high zoom', async ({ page }) => {
  await page.goto(`${harness}?fixture=interface-hybrid`); await ready(page);
  await page.locator('[data-node-id="language"] .architecture-browser-disclosure').click(); await ready(page);
  await graphAction(page, 'Fit view'); await ready(page);
  const emphasized = () => page.locator('.architecture-connection[data-emphasized="true"]')
    .evaluateAll((edges) => edges.map((edge) => edge.getAttribute('data-edge-id')!).sort());
  const zoom = () => page.locator('.react-flow__viewport').evaluate((element) => new DOMMatrix(getComputedStyle(element).transform).a);
  const traverse = async (port: Locator) => {
    const id = await port.getAttribute('data-port-id');
    const label = port.locator('.architecture-port-label');
    await expect(label).toHaveAttribute('data-raised', 'true');
    const text = await label.boundingBox(), terminal = await port.boundingBox();
    expect(text).toBeTruthy(); expect(terminal).toBeTruthy();
    const start = { x: text!.x + text!.width / 2, y: text!.y + text!.height / 2 };
    const end = { x: terminal!.x + terminal!.width / 2, y: terminal!.y + terminal!.height / 2 };
    await page.mouse.move(start.x, start.y);
    await expect.poll(emphasized).not.toEqual([]);
    const connected = await emphasized();
    for (let step = 1; step <= 16; step++) {
      const point = { x: start.x + (end.x - start.x) * step / 16, y: start.y + (end.y - start.y) * step / 16 };
      await page.mouse.move(point.x, point.y);
      await page.waitForTimeout(100); // Catch the next-frame pointer leave that a fast hover misses.
      const hit = await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.closest('.architecture-port')?.getAttribute('data-port-id') ?? document.elementFromPoint(x, y)?.className, point);
      expect(await emphasized(), `${id} lost emphasis at step ${step}, zoom ${await zoom()}, ${JSON.stringify({ start, end, point, hit })}`).toEqual(connected);
    }
    await page.mouse.move(0, 0);
    await expect.poll(emphasized).toEqual([]);
  };
  const before = await camera(page), layouts = await page.locator(panel).getAttribute('data-layout-count');
  const input = page.locator('.architecture-port[data-node-id="language"][data-port-id="positions"]');
  const output = page.locator('.architecture-port[aria-label^="output port"]:has(.architecture-port-label[data-raised="true"])').first();
  await expect(output).toHaveCount(1);
  await traverse(input);
  await traverse(output);
  expect(await camera(page)).toBe(before);
  await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layouts!);

  const label = page.locator('.architecture-port[data-node-id="language"][data-port-id="positions"] .architecture-port-label');
  await label.hover();
  for (let attempt = 0; attempt < 12 && await zoom() < 3.9; attempt++) await page.mouse.wheel(0, -1000);
  await expect.poll(zoom).toBeGreaterThan(3.9);
  const host = await page.locator('.architecture-flow').boundingBox();
  const highPort = await input.boundingBox();
  expect(host).toBeTruthy(); expect(highPort).toBeTruthy();
  const center = highPort!.x + highPort!.width / 2;
  if (center < host!.x + 12 || center > host!.x + host!.width - 12) {
    // Narrow fit can zoom the terminal offscreen. Pan the actual graph pane so
    // the full pointer path is possible before measuring continuity.
    const shift = host!.x + host!.width * 0.35 - center;
    const from = { x: shift > 0 ? host!.x + 24 : host!.x + host!.width - 24, y: host!.y + 20 };
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    await page.mouse.move(from.x + shift, from.y, { steps: 10 });
    await page.mouse.up(); await ready(page);
  }
  const highCamera = await camera(page);
  await traverse(input);
  expect(await camera(page)).toBe(highCamera);
  // At high zoom the terminal and label are visibly separate. Their gap must
  // resolve to the same port button instead of the graph pane underneath.
  const gap = await label.evaluate((element) => {
    const text = element.getBoundingClientRect(), port = element.closest('button')!.getBoundingClientRect();
    const x = (port.right + text.left) / 2, y = text.top + text.height / 2;
    return { width: text.left - port.right, hit: document.elementFromPoint(x, y)?.closest('.architecture-port')?.getAttribute('data-port-id') };
  });
  expect(gap.width).toBeGreaterThan(2);
  expect(gap.hit).toBe('positions');
  await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layouts!);
});

test('ordinary operation keeps long opposite labels disjoint and bound to their own terminals', async ({ page }) => {
  await page.goto(`${harness}?fixture=interface-long-ports`); await ready(page);
  await findComponent(page, 'Embedding'); await ready(page);
  const node = page.locator('.react-flow__node[data-id="Embedding"]');
  const input = node.locator('.architecture-port[data-port-id="Token IDs"]');
  const output = node.locator('.architecture-port[data-port-id="out"]');
  const inputLabel = input.locator('.architecture-port-label');
  const outputLabel = output.locator('.architecture-port-label');
  const boundary = page.locator('.architecture-port[data-node-id="language"][data-port-id="Token IDs"]');
  await expect(boundary.locator('.architecture-port-label')).toHaveText('An unusually long attention mask input name');
  await expect(boundary.locator('.architecture-port-label')).toHaveAttribute('title', 'input: An unusually long attention mask input name');
  await expect(inputLabel).toHaveAttribute('title', 'input: An unusually long attention mask input name');
  await expect(inputLabel).toHaveText('An unusually long attention mask input name');
  await expect(input).toHaveAttribute('title', 'input: An unusually long attention mask input name');
  await expect(outputLabel).toHaveAttribute('title', 'output: An unusually long hidden state output name');
  const bounds = await node.evaluate((card) => {
    const first = card.querySelector('.architecture-port[data-port-id="Token IDs"] .architecture-port-label')!;
    const last = card.querySelector('.architecture-port[data-port-id="out"] .architecture-port-label')!;
    const left = first.getBoundingClientRect(), right = last.getBoundingClientRect();
    const scale = new DOMMatrix(getComputedStyle(document.querySelector('.react-flow__viewport')!).transform).a;
    return { gap: (right.left - left.right) / scale, rowGap: Math.abs(left.top - right.top) / scale,
      cardWidth: card.getBoundingClientRect().width / scale,
      inputWidth: left.width / scale, outputWidth: right.width / scale };
  });
  expect(bounds.cardWidth).toBeGreaterThanOrEqual(288);
  expect(bounds.inputWidth).toBeCloseTo(120, 0);
  expect(bounds.outputWidth).toBeCloseTo(120, 0);
  expect(bounds.rowGap).toBeLessThan(1);
  expect(bounds.gap).toBeGreaterThanOrEqual(8);
  const emphasized = () => page.locator('.architecture-connection[data-emphasized="true"]')
    .evaluateAll((edges) => edges.map((edge) => edge.getAttribute('data-edge-id')!).sort());
  const layout = await page.locator(panel).getAttribute('data-layout-count');
  const before = await camera(page);
  await inputLabel.hover();
  await expect(input).toHaveAttribute('data-emphasized', 'true');
  const inputEdges = await emphasized();
  expect(inputEdges.length).toBeGreaterThan(0);
  await outputLabel.hover();
  await expect(output).toHaveAttribute('data-emphasized', 'true');
  const outputEdges = await emphasized();
  expect(outputEdges.length).toBeGreaterThan(0);
  expect(outputEdges).not.toEqual(inputEdges);
  await output.focus();
  await expect.poll(emphasized).toEqual(outputEdges);
  expect(await camera(page)).toBe(before);
  await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layout!);
});

test('many interfaces stay targetable without document overflow', async ({ page }) => {
  await page.goto(`${harness}?fixture=interface-many`); await ready(page);
  await viewOptions(page); await page.keyboard.press('Escape');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true);
  const labels = page.locator('.react-flow__node[data-id="model"] .architecture-port-label');
  await expect(labels.filter({ hasText: 'auxiliary_17' })).toHaveCount(1);
  const unconnected = page.locator('.architecture-port[data-node-id="model"][aria-label*="auxiliary_17"]');
  await unconnected.focus(); await unconnected.press('Enter');
  await expect(unconnected).toHaveAttribute('data-emphasized', 'true');
  await expect(page.locator('.architecture-connection[data-emphasized="true"]')).toHaveCount(0);
  const ports = await page.locator('.architecture-port').evaluateAll((items) => items.map((e) => ({ label: e.getAttribute('aria-label'), top: (e as HTMLElement).style.top })));
  expect(ports.length).toBeGreaterThanOrEqual(23);
  expect(new Set(ports.map((p) => p.label)).size).toBe(ports.length);
});

test('expanded isolation retains computational output and disconnected boundary hit targets', async ({ page }) => {
  await page.goto(`${harness}?fixture=interface-hybrid-many`); await ready(page);
  await findComponent(page, 'language');
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  await expect(page.locator('.react-flow__node[data-id="Embedding"]')).toHaveCount(1);
  const before = await camera(page), layouts = await page.locator(panel).getAttribute('data-layout-count');
  for (const [id, connections] of [['out', 2], ['auxiliary_17', 0]] as const) {
    const port = page.locator(`.architecture-port[data-node-id="language"][data-port-id="${id}"]`);
    await port.focus(); await port.press('Enter');
    await expect(port).toHaveAttribute('data-emphasized', 'true');
    await expect(page.locator('.architecture-connection[data-emphasized="true"]')).toHaveCount(connections);
    await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
    await expect(page.getByRole('dialog')).toContainText(id);
    await page.getByRole('button', { name: 'Close inspection' }).click();
    expect(await camera(page)).toBe(before);
    await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layouts!);
  }
});

test('concrete shared ports inspect original declarations after nonzero instance rebinding', async ({ page }) => {
  const requests: string[] = []; page.on('request', (r) => { if (/\/sessions\//.test(r.url())) requests.push(r.url()); });
  await page.goto(`${harness}?fixture=templates&native-inspection`); await ready(page);
  await openShared(page, 'shared-full-attention'); await ready(page);
  await page.getByLabel('Shared structure instance', { exact: true }).selectOption('layer-2.attention'); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  const before = await camera(page), layouts = await page.locator(panel).getAttribute('data-layout-count');
  const port = page.locator('.architecture-port[data-node-id="layer-0.attention"][data-port-id="positions"]');
  await port.focus(); await port.press('Enter');
  for (const instance of ['layer-2.attention', 'layer-0.attention', 'layer-2.attention']) {
    await page.getByLabel('Shared structure instance', { exact: true }).selectOption(instance); await ready(page);
    await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Interface inspection' });
    await expect(dialog).toContainText(`${instance} · group`);
    await expect(dialog).toContainText('positions · input');
    await expect(dialog).toContainText('positions:out → model:positions');
    await expect(dialog.getByLabel('Inspect parameter')).toHaveCount(0);
    await page.getByRole('button', { name: 'Close inspection' }).click();
    expect(await camera(page)).toBe(before);
    await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layouts!);
  }
  await page.getByLabel('Shared structure instance', { exact: true }).selectOption(''); await ready(page);
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  await expect(page.getByRole('dialog')).not.toContainText('positions:out');
  await expect(page.getByRole('dialog').getByLabel('Inspect parameter')).toHaveCount(0);
  expect(requests).toEqual([]);
});

test('outside-component navigation excludes declarations and tools but retains real components', async ({ page }) => {
  await page.goto(`${harness}?fixture=interface-hybrid`); await ready(page);
  await findComponent(page, 'language');
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  const browser = page.getByRole('tree', { name: 'Model components', exact: true });
  for (const name of ['Token IDs', 'positions', 'mask', 'current_mask', 'logits', 'Tokenizer capability']) {
    await expect(browser.locator('[data-browser-name]').filter({ hasText: new RegExp(`^${name}$`) })).toHaveCount(0);
  }
  await findComponent(page, 'LM head'); await ready(page);
  await expect(page.locator(panel)).toHaveAttribute('data-scope-id', '');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'LM head');
});

test('neutral shared port stays pinned after blur and restoration, then binds to the chosen instance', async ({ page }) => {
  const requests: string[] = []; page.on('request', (r) => { if (/\/sessions\//.test(r.url())) requests.push(r.url()); });
  await page.goto(`${harness}?fixture=templates&native-inspection`); await ready(page);
  await openShared(page, 'shared-full-attention'); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  const port = page.locator('.architecture-port[data-node-id="layer-0.attention"][data-port-id="positions"]');
  const emphasized = page.locator('.architecture-connection[data-emphasized="true"]');
  const before = await camera(page), layouts = await page.locator(panel).getAttribute('data-layout-count');
  await port.focus(); await port.press('Enter');
  await expect(emphasized).toHaveCount(2);
  const pinned = await emphasized.evaluateAll((edges) => edges.map((e) => e.getAttribute('data-edge-id')).sort());
  await page.getByRole('button', { name: 'Fit view', exact: true }).focus();
  await ready(page);
  await expect(emphasized).toHaveCount(2);
  expect(await camera(page)).toBe(before);
  await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layouts!);
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  await expect(page.getByRole('dialog')).toContainText('root.positions');
  await expect(page.getByRole('dialog')).toContainText('No instance selected');
  await expect(page.getByRole('dialog')).not.toContainText('positions:out');
  await expect(page.getByRole('dialog').getByLabel('Inspect parameter')).toHaveCount(0);
  await page.getByRole('button', { name: 'Close inspection' }).click();
  await page.getByRole('button', { name: 'Toggle explorer' }).click();
  await page.getByRole('button', { name: 'Toggle explorer' }).click(); await ready(page);
  await expect(page.locator(panel)).toHaveAttribute('data-template-instance-id', '');
  await expect(emphasized).toHaveCount(2);
  expect(await emphasized.evaluateAll((edges) => edges.map((e) => e.getAttribute('data-edge-id')).sort())).toEqual(pinned);
  const restoredLayouts = await page.locator(panel).getAttribute('data-layout-count'), restoredCamera = await camera(page);
  for (const instance of ['layer-2.attention', '', 'layer-0.attention', 'layer-2.attention']) {
    await page.getByLabel('Shared structure instance', { exact: true }).selectOption(instance); await ready(page);
    await expect(emphasized).toHaveCount(2);
    await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
    if (instance) {
      await expect(page.getByRole('dialog')).toContainText(`${instance} · group`);
      await expect(page.getByRole('dialog')).toContainText('positions:out → model:positions');
    } else {
      await expect(page.getByRole('dialog')).toContainText('root.positions');
      await expect(page.getByRole('dialog')).not.toContainText('positions:out');
    }
    await page.getByRole('button', { name: 'Close inspection' }).click();
    expect(await camera(page)).toBe(restoredCamera);
    await expect(page.locator(panel)).toHaveAttribute('data-layout-count', restoredLayouts!);
  }
  expect(requests).toEqual([]);
});

test('source interface search remains inspectable from a neutral shared view', async ({ page }) => {
  const requests: string[] = []; page.on('request', (r) => { if (/\/sessions\//.test(r.url())) requests.push(r.url()); });
  await page.goto(`${harness}?fixture=templates&native-inspection`); await ready(page);
  await openShared(page, 'shared-full-attention'); await ready(page);
  const before = await camera(page), layouts = await page.locator(panel).getAttribute('data-layout-count');
  const search = page.getByRole('searchbox', { name: 'Search components', exact: true });
  await search.fill('positions'); await search.press('Enter');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  const inspect = page.getByRole('button', { name: 'Inspect selected', exact: true });
  for (const clear of [false, true]) {
    if (clear) await search.fill('');
    await inspect.click();
    const dialog = page.getByRole('dialog', { name: 'Interface inspection' });
    await expect(dialog).toContainText('positions · input');
    await expect(dialog).toContainText('positions:out → model:positions');
    await expect(dialog).toContainText('authored-projection-fixture');
    await expect(dialog.getByLabel('Inspect parameter')).toHaveCount(0);
    await page.getByRole('button', { name: 'Close inspection' }).click();
    await expect(inspect).toBeFocused();
    await expect(page.locator(panel)).toHaveAttribute('data-template-instance-id', '');
    await expect(page.locator(panel)).toHaveAttribute('data-scope-id', 'layer-0.attention');
    await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layouts!);
    expect(await camera(page)).toBe(before);
  }
  expect(requests).toEqual([]);
});
