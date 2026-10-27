import { expect, test } from '@playwright/test';
import { findComponent } from './architecture-controls';
import { nativeCamera } from './native-camera';

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
