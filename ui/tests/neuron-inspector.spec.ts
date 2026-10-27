import { expect, test } from '@playwright/test';
import { findComponent } from './architecture-controls';
import { nativeCamera } from './native-camera';

test('graph arrival preserves an active neuron weight inspection and its streams', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 960 });
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html?architecture&neurons`);
  await page.evaluate(() => { window.explorerFixture.responses.architecture.paused = true; });
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('lab/alpha');
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Neurons', exact: true }).click();
  await page.getByLabel('Concrete MLP').selectOption('mlp-7');
  await page.getByRole('spinbutton', { name: 'Intermediate index' }).fill('2');
  await page.getByRole('button', { name: 'Open gate weight' }).click();
  const dialog = page.getByRole('dialog', { name: 'Layer 7 · gate weight' });
  await expect(dialog).toBeVisible();
  await expect.poll(() => page.evaluate(() => window.explorerFixture.requests.length)).toBe(3);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    f.emit(0, 1, f.metadata(0)); f.data(0, [1, -2]);
  });
  const canvas = dialog.locator('.matrix-scroll canvas');
  await expect(canvas).toBeVisible();
  const elements = await Promise.all([dialog.elementHandle(), canvas.elementHandle(),
    page.getByLabel('Neuron inspector', { exact: true }).elementHandle(),
    page.getByLabel('Neuron experiment', { exact: true }).elementHandle()]);
  const requestIds = await page.evaluate(() => window.explorerFixture.requests.map(r => r.id));
  await page.evaluate(() => { window.explorerFixture.responses.architecture.resume(); });
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('aria-busy', 'false');
  await expect(dialog).toBeVisible();
  for (const element of elements) expect(await element!.evaluate(node => node.isConnected)).toBe(true);
  expect(await page.evaluate(() => ({ ids: window.explorerFixture.requests.map(r => r.id),
    cancelled: window.explorerFixture.cancelled }))).toEqual({ ids: requestIds, cancelled: [] });
  await page.evaluate(() => { window.explorerFixture.data(0, [3, -4, 5, -6]); });
  await expect.poll(() => page.evaluate(() => window.explorerFixture.renderers.findLast(r => r.state === 'ready' && r.geometry.rows === 3 && r.geometry.columns === 2)?.readCell(2, 1)))
    .toEqual({ state: 'finite', row: 2, column: 1, value: -6 });
  await page.keyboard.press('Escape');
  await expect.poll(() => page.evaluate(() => [...window.explorerFixture.cancelled].sort())).toEqual([...requestIds].sort());
});

test('delayed capabilities restore neuron inspector scroll after explorer switching', async ({ page }) => {
  await page.setViewportSize({ width: 760, height: 960 });
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html?architecture&neurons`);
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('lab/alpha');
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const graph = page.getByLabel('Architecture graph', { exact: true });
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await page.getByRole('button', { name: 'Neurons', exact: true }).click();
  await page.getByLabel('Concrete MLP').selectOption('mlp-7');
  const index = page.getByRole('spinbutton', { name: 'Intermediate index' });
  await index.fill('2');
  const scroller = page.locator('.neuron-inspector-scroll');
  await scroller.evaluate(node => { node.scrollTop = 140; });
  await expect.poll(() => scroller.evaluate(node => node.scrollTop)).toBe(140);
  await page.evaluate(() => { window.explorerFixture.responses.capabilities.paused = true; });
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await expect(scroller).toContainText('Retrieving neuron capabilities');
  expect(await scroller.evaluate(node => node.scrollTop)).toBe(0);
  await page.evaluate(() => { window.explorerFixture.responses.capabilities.resume(); });
  await expect(page.getByLabel('Concrete MLP')).toHaveValue('mlp-7');
  await expect(index).toHaveValue('2');
  await expect.poll(() => scroller.evaluate(node => node.scrollTop)).toBe(140);
  // Subsequent user scroll becomes the saved offset; restoration is not repeated.
  await scroller.evaluate(node => { node.scrollTop = 80; });
  await expect.poll(() => scroller.evaluate(node => node.scrollTop)).toBe(80);
  await page.getByRole('button', { name: 'Neurons', exact: true }).click();
  expect(await scroller.evaluate(node => node.scrollTop)).toBe(80);
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByLabel('Concrete MLP')).toHaveValue('mlp-7');
  await expect(index).toHaveValue('2');
  await expect.poll(() => scroller.evaluate(node => node.scrollTop)).toBe(80);
});

for (const width of [1440, 760]) test(`neuron inspector preserves a concrete binding and graph context at ${width}px`, async ({ page }, info) => {
  await page.setViewportSize({ width, height: 960 });
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html?architecture&neurons`);
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('lab/alpha');
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const graph = page.getByLabel('Architecture graph', { exact: true });
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await findComponent(page, 'linear1');
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  const camera = await page.locator('.react-flow__viewport').getAttribute('style');
  const layout = await graph.getAttribute('data-layout-count');
  const expanded = await graph.getAttribute('data-visible-nodes');
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  const entry = page.getByRole('button', { name: 'Explore neurons', exact: true });
  await entry.focus(); await page.keyboard.press('Enter');
  await expect(page.getByLabel('Concrete MLP')).toHaveValue('mlp-7');
  const input = page.getByRole('spinbutton', { name: 'Intermediate index' });
  await input.fill('3');
  await expect(page.getByRole('alert')).toContainText('0 to 2');
  await expect(page.getByRole('button', { name: 'Open gate weight' })).toBeDisabled();
  await input.fill('1'); await input.press('ArrowUp');
  await expect(input).toHaveValue('2');
  const inspector = page.getByLabel('Neuron inspector', { exact: true });
  await expect(inspector).toContainText('a[j] = SiLU(g[j]) * u[j]');
  expect(await graph.getAttribute('data-layout-count')).toBe(layout);
  await page.screenshot({ path: info.outputPath(`neuron-inspector-${width}.png`) });
  for (const [projection, axis, shape] of [['gate', 'rows', [3, 2]], ['up', 'rows', [3, 2]], ['down', 'columns', [2, 3]]] as const) {
    const trigger = page.getByRole('button', { name: `Open ${projection} weight` });
    await trigger.focus(); await page.keyboard.press('Enter');
    const dialog = page.getByRole('dialog');
    await expect(dialog).toHaveAccessibleName(`Layer 7 · ${projection} weight`);
    await expect.poll(() => page.evaluate(() => window.explorerFixture.requests.length)).toBe(projection === 'gate' ? 3 : projection === 'up' ? 6 : 9);
    await page.evaluate(({ projection }) => {
      const f = window.explorerFixture;
      const index = f.requests.findLastIndex(r => r.kind === 'data');
      if (f.requests[index]!.tensor !== `${projection}7`) throw new Error('Wrong concrete tensor');
      f.emit(index, 1, f.metadata(index)); f.data(index, [1, -2, 3, -4, 5, -6]); f.end(index);
      f.emit(index + 1, 1, f.metadata(index + 1)); f.end(index + 1);
    }, { projection });
    await expect(page.locator('[data-result="tensor"]')).toHaveCount(0);
    await nativeCamera(page);
    const guide = dialog.locator(`.matrix-scroll .matrix-selected-${axis}`);
    await expect(guide).toHaveAttribute(`data-${axis}`, '2:3');
    await expect(dialog.locator(`.${axis === 'rows' ? 'row' : 'column'}-distributions .matrix-selected-${axis}`)).toHaveCount(1);
    const values = await page.evaluate(() => {
      const matrix = window.explorerFixture.renderers.findLast(r => r.state === 'ready' && r.geometry.rows !== 100 && r.geometry.columns !== 100)!;
      return { rows: matrix.geometry.rows, columns: matrix.geometry.columns, value: matrix.readCell(1, 1), scale: matrix.view?.scaleX };
    });
    expect(values).toMatchObject({ rows: shape[0], columns: shape[1], value: { value: projection === 'down' ? 5 : -4 }, scale: 1 });
    if (projection === 'down') {
      await dialog.getByRole('button', { name: 'Fit width' }).click();
      await page.screenshot({ path: info.outputPath(`neuron-weight-${width}.png`) });
    }
    await page.keyboard.press('Escape');
    await expect(trigger).toBeFocused();
  }
  await page.getByRole('button', { name: 'Back to graph' }).click();
  await expect(inspector).toHaveCount(0);
  expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
  expect(await graph.getAttribute('data-layout-count')).toBe(layout);
  expect(await graph.getAttribute('data-visible-nodes')).toBe(expanded);
  // Explorer remount retains the same current neuron identity; no old revision is revived.
  await page.getByRole('button', { name: 'Neurons', exact: true }).click();
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await expect(page.getByLabel('Concrete MLP')).toHaveValue('mlp-7');
  await expect(input).toHaveValue('2');
  expect(await page.evaluate(() => ({ width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight }))).toEqual({ width, height: 960 });
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('lab/beta');
  await expect(inspector).toHaveCount(0);
});
