import { expect, test } from '@playwright/test';
import { findComponent, graphPreference } from './architecture-controls';

test('compact summaries retain values, exact port hits and bounded geometry at either width', async ({ page }, info) => {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html?fixture=summaries`);
  const graph = page.getByLabel('Architecture graph', { exact: true });
  const ready = () => expect(graph).toHaveAttribute('aria-busy', 'false');
  await ready();
  const card = (id: string) => page.locator(`.react-flow__node[data-id="${id}"]`);
  for (const dimensions of [false, true]) {
    await graphPreference(page, 'Show dimensions', dimensions); await ready();
    for (const id of ['norm', 'linear']) {
      await findComponent(page, id); await ready();
      await expect(card(id).locator('.architecture-tensor-row')).toHaveCount(2);
      await expect(card(id).locator('.architecture-card-shape')).toHaveCount(dimensions ? 2 : 0);
      await expect(card(id).locator('.architecture-port-shape')).toHaveCount(dimensions ? 2 : 0);
      await expect(card(id).locator('.architecture-node-type > .architecture-summary-text')).toHaveAccessibleName(
        id === 'norm' ? 'LayerNorm(x; weight, bias, epsilon)' : 'x W^T + bias');
      if (id === 'norm') await expect(card(id).getByLabel('epsilon = 0.00003', { exact: true })).toBeVisible();
      const before = await page.locator('.react-flow__viewport').getAttribute('style');
      const count = await graph.getAttribute('data-layout-count');
      for (const portId of ['x', 'out']) {
        const port = card(id).locator(`[data-port-id="${portId}"].architecture-port`);
        await port.hover(); await expect(port).toHaveAttribute('data-emphasized', 'true');
        // Unconnected ports remain interactive but must never invent a route.
        await expect(page.locator('.architecture-connection[data-emphasized="true"]')).toHaveCount(
          (id === 'norm' ? portId === 'out' : portId === 'x') ? 1 : 0);
        // Actual hit test: summary rows must never occlude a port's pointer target.
        expect(await port.evaluate((element) => {
          const b = element.getBoundingClientRect(); return element.contains(document.elementFromPoint(b.x + b.width / 2, b.y + b.height / 2));
        })).toBe(true);
      }
      const name = card(id).getByLabel(`layers.7.${id}.weight`, { exact: true });
      await name.focus();
      await expect(name.locator('.architecture-summary-tooltip')).toBeVisible();
      expect(await graph.getAttribute('data-layout-count')).toBe(count);
      expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(before);
      const geometry = await card(id).evaluate((element) => {
        const box = element.getBoundingClientRect();
        const parts = [...element.querySelectorAll('.architecture-node-heading, .architecture-node-type, .architecture-tensor-row, .architecture-constant, .architecture-port-label')]
          .map((p) => ({ kind: p.className, box: p.getBoundingClientRect().toJSON() as { x: number; y: number; width: number; height: number } }));
        return { box: box.toJSON(), parts };
      });
      for (const part of geometry.parts) {
        expect(part.box.y).toBeGreaterThanOrEqual(geometry.box.y);
        expect(part.box.y + part.box.height).toBeLessThanOrEqual(geometry.box.y + geometry.box.height + 1);
        expect(part.box.x + part.box.width).toBeLessThanOrEqual(geometry.box.x + geometry.box.width + 1);
      }
      const rows = geometry.parts.filter((p) => p.kind === 'architecture-tensor-row');
      const ports = geometry.parts.filter((p) => p.kind === 'architecture-port-label');
      expect(rows[0]!.box.y).toBeGreaterThanOrEqual(Math.max(...ports.map((p) => p.box.y + p.box.height)));
      await page.mouse.move(0, 0); await name.blur();
      await page.screenshot({ path: info.outputPath(`${id}-dimensions-${dimensions}.png`) });
      await info.attach(`${id}-geometry-${dimensions}`, { body: JSON.stringify(geometry), contentType: 'application/json' });
    }
  }
  await findComponent(page, 'block'); await ready();
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  const expanded = await page.evaluate(() => {
    const box = (id: string, part = '') => document.querySelector(`.react-flow__node[data-id="${id}"] ${part}`)!.getBoundingClientRect().toJSON();
    return { parent: box('block'), header: box('block', '.architecture-node-heading'), children: ['norm', 'linear'].map((id) => box(id)) };
  });
  for (const child of expanded.children) {
    expect(child.y).toBeGreaterThan(expanded.header.y + expanded.header.height);
    expect(child.x + child.width).toBeLessThanOrEqual(expanded.parent.x + expanded.parent.width);
    expect(child.y + child.height).toBeLessThanOrEqual(expanded.parent.y + expanded.parent.height);
  }
  await page.screenshot({ path: info.outputPath('expanded-group.png') });
});

test('long summaries keep full text, truthful overflow and expanded group-owned rows clear of children', async ({ page }) => {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html?fixture=summaries&long`);
  const graph = page.getByLabel('Architecture graph', { exact: true });
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await graphPreference(page, 'Show dimensions', true);
  await findComponent(page, 'norm');
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  const norm = page.locator('.react-flow__node[data-id="norm"]');
  expect(await norm.evaluate((element) => parseFloat((element as HTMLElement).style.width))).toBeLessThanOrEqual(300);
  await expect(norm.locator('.architecture-tensor-row')).toHaveCount(6);
  const camera = await page.locator('.react-flow__viewport').getAttribute('style');
  const layout = await graph.getAttribute('data-layout-count');
  const formula = norm.locator('.architecture-node-type > .architecture-summary-text');
  await formula.focus();
  await expect(formula.locator('.architecture-summary-tooltip')).toBeVisible();
  await expect(formula).toHaveAccessibleName('<script>display only</script> ' + 'supplied long formula '.repeat(30));
  await expect(norm.locator('script')).toHaveCount(0);
  const name = norm.locator('.architecture-tensor-name > .architecture-summary-text').first();
  await name.hover();
  await expect(name.locator('.architecture-summary-tooltip')).toBeVisible();
  await expect(name.locator('span').first()).toHaveText('q_proj.' + 'submodule.'.repeat(35) + 'weight');
  const more = norm.getByRole('button', { name: 'Inspect 3 more tensors of layer norm' });
  await more.click();
  await expect(page.locator('output')).toHaveText('card-summaries: norm');
  expect(await graph.getAttribute('data-layout-count')).toBe(layout);
  expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
  await findComponent(page, 'block');
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  const group = page.locator('.react-flow__node[data-id="block"]');
  await expect(group.locator('.architecture-tensor-row')).toHaveCount(1);
  const owned = (await group.locator('.architecture-tensor-row').boundingBox())!;
  const child = (await norm.boundingBox())!;
  expect(owned.y + owned.height).toBeLessThan(child.y);
  for (const label of await group.locator('.architecture-port-label').all()) {
    const b = (await label.boundingBox())!;
    for (const id of ['norm', 'linear']) {
      const c = (await page.locator(`.react-flow__node[data-id="${id}"]`).boundingBox())!;
      expect(b.x + b.width <= c.x || b.x >= c.x + c.width || b.y + b.height <= c.y || b.y >= c.y + c.height).toBe(true);
    }
  }
});
