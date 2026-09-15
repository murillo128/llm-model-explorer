import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { models, sessionA, tensors } from '../src/test/shell-fixtures';
import { revealTensor } from './tensor-tree-helpers';
import { nativeCamera } from './native-camera';

const path = ['model', 'layers', '0', 'mlp', 'down_proj', 'weight'];
async function openInventory(page: Page) {
  await page.route('**/runtime-config.json', (route) => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  await page.route('https://backend.example/**', (route) => route.fulfill({
    status: route.request().method() === 'POST' ? 201 : 200,
    json: route.request().url().endsWith('/models') ? { models } : route.request().url().endsWith('/tensors') ? {
      coverage: 'complete', diagnostics: [],
      tensors: [
        { ...tensors[2], id: 'deep', name: path.join('.'), path },
        { ...tensors[2], id: 'second-layer', path: ['model', 'layers', '1', 'self_attn', 'q_proj', 'weight'] },
      ],
    } : sessionA,
  }));
  await page.goto('/');
  await page.getByRole('combobox').selectOption(models[0]!.id);
  await expect(page.locator('.tensor-tree').first()).toBeVisible();
}
async function noDocumentOverflow(page: Page) {
  const { width, height } = page.viewportSize()!;
  expect(await page.evaluate(() => ({
    document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
    body: [document.body.scrollWidth, document.body.scrollHeight], viewport: [innerWidth, innerHeight],
    scroll: [scrollX, scrollY],
  }))).toEqual({ document: [width, height], body: [width, height],
    viewport: [width, height], scroll: [0, 0] });
}

test('quiet branches persist explicit keyboard and pointer choices across explorer visits and reload', async ({ page }) => {
  await openInventory(page);
  const branches = page.locator('.tensor-tree details');
  await expect(branches.first()).toHaveAttribute('open', '');
  for (const branch of (await branches.all()).slice(1)) await expect(branch).not.toHaveAttribute('open');
  const root = branches.first().locator(':scope > summary');
  await root.focus();
  await page.keyboard.press('ArrowRight');
  const layers = page.locator('summary').filter({ hasText: /^layers$/ });
  await expect(layers).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(layers.locator('..')).toHaveAttribute('open');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('Space');
  const leaf = page.getByRole('button', { includeHidden: true, name: /^model.layers.0.mlp.down_proj.weight/ });
  await (await revealTensor(leaf)).click();
  await expect(leaf).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  await expect(leaf).toBeVisible();
  await page.reload();
  await expect(leaf).toBeVisible();
  const mlp = page.locator('summary').filter({ hasText: /^mlp$/ });
  await mlp.click();
  await page.reload();
  await expect(mlp).toBeVisible();
  await expect(mlp.locator('..')).not.toHaveAttribute('open');
  await expect(leaf).not.toBeVisible();
  await root.focus(); await page.keyboard.press('ArrowLeft');
  await page.reload();
  await expect(branches.first()).not.toHaveAttribute('open');
  await noDocumentOverflow(page);
});

test('drawer and icon-only rail resize the workspace, retain selection, transfer focus and survive reload', async ({ page }) => {
  await openInventory(page);
  const leaf = page.getByRole('button', { includeHidden: true, name: /^model.layers.0.mlp.down_proj.weight/ });
  await (await revealTensor(leaf)).click();
  const pane = page.getByRole('region', { name: 'Tensor Explorer workspace', exact: true });
  const inventory = page.getByRole('complementary', { name: 'Tensor inventory' });
  await expect(inventory.getByRole('heading', { name: 'Inventory' })).toBeVisible();
  await expect(inventory.locator('.tensor-folder-icon').first()).toBeVisible();
  await expect(leaf.locator('.tensor-leaf-icon')).toBeVisible();
  const initial = (await pane.boundingBox())!;
  const inventoryBox = (await inventory.boundingBox())!;
  if (process.env.CAPTURE_INVENTORY_EVIDENCE) await page.screenshot({ path: `evidence/inventory-expanded-${page.viewportSize()!.width}.png` });
  const hide = page.getByRole('button', { name: 'Collapse inventory', exact: true });
  const show = page.getByRole('button', { name: 'Expand inventory', exact: true });
  await hide.focus(); await page.keyboard.press('Enter');
  await expect(show).toBeFocused();
  await expect(show).toHaveAttribute('aria-expanded', 'false');
  await expect(inventory).not.toBeVisible();
  await expect(page.getByRole('separator')).toHaveCount(0);
  const hidden = (await pane.boundingBox())!;
  const workspace = (await page.locator('#workspace').boundingBox())!;
  expect(hidden.x).toBe(workspace.x + 40);
  expect(hidden.width).toBe(workspace.width - 40);
  expect(hidden.height).toBe(workspace.height);
  const rail = page.getByRole('navigation', { name: 'Inventory navigation' });
  expect((await rail.boundingBox())!.width).toBe(40);
  await expect(rail.locator('button')).toHaveCount(1);
  await expect(show).toHaveText('');
  await expect(show.locator('svg')).toHaveCount(1);
  await expect(pane.getByRole('button', { name: 'Expand inventory' })).toHaveCount(0);
  const tooltip = rail.getByRole('tooltip');
  await expect(tooltip).toBeVisible();
  await expect(show).toHaveAccessibleDescription('Expand inventory');
  await show.blur(); await page.mouse.move(0, 0);
  await expect(tooltip).not.toBeVisible();
  expect(await rail.innerText()).toBe('');
  expect(await show.evaluate((node) => [getComputedStyle(node, '::before').content, getComputedStyle(node, '::after').content]))
    .toEqual(['none', 'none']);
  expect(await show.locator('svg path').getAttribute('d')).toBe('M6 2v12M9 5h3M9 8h3M9 11h3');
  await show.hover(); await expect(tooltip).toBeVisible();
  await page.mouse.move(0, 0); await expect(tooltip).not.toBeVisible();
  if (process.env.CAPTURE_INVENTORY_EVIDENCE) await page.screenshot({ path: `evidence/inventory-collapsed-${page.viewportSize()!.width}.png` });
  if (page.viewportSize()!.width > 760) expect(hidden.width - initial.width).toBe(inventoryBox.width + 16 - 40);
  else expect(hidden.height - initial.height).toBeCloseTo(inventoryBox.height + 8, 1);
  await expect(pane.locator('h2')).toHaveText(path.join(' › '));
  await noDocumentOverflow(page);
  await page.getByRole('button', { name: 'Tensor information' }).click();
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  await expect(dialog).toBeVisible();
  expect((await dialog.boundingBox())!.y).toBeGreaterThan((await pane.locator('.matrix-panel-header').boundingBox())!.y);
  await page.keyboard.press('Escape');
  await show.focus();
  await page.keyboard.press('Space');
  await expect(hide).toBeFocused();
  await expect(leaf).toHaveAttribute('aria-pressed', 'true');
  expect(await pane.boundingBox()).toEqual(initial);
  await hide.click(); await page.reload();
  await expect(show).toBeVisible();
  await expect(inventory).not.toBeVisible();
  await show.focus(); await page.keyboard.press('Enter');
  await expect(hide).toBeFocused();
  await expect(leaf).toBeVisible();
  await noDocumentOverflow(page);
});

test('pointer and keyboard resizing clamp, persist width and preserve scientific consumers', async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 800 });
  const port = Number(process.env.UI_TEST_PORT ?? 4173) + 1;
  await page.goto(`http://127.0.0.1:${port}/tests/tensor-explorer.html`);
  await page.getByRole('combobox').selectOption(models[0]!.id);
  await page.getByRole('button', { name: /^reference \[/ }).click();
  await expect.poll(() => page.evaluate(() => window.explorerFixture.requests.length)).toBe(3);
  const canvas = await page.locator('.matrix-scroll canvas').elementHandle();
  const divider = page.getByRole('separator', { name: 'Resize tensor inventory' });
  const inventory = page.getByRole('complementary', { name: 'Tensor inventory' });
  const pane = page.getByRole('region', { name: 'Tensor Explorer workspace', exact: true });
  await expect(divider).toHaveAttribute('aria-valuenow', '280');
  await divider.focus(); await page.keyboard.press('Home');
  await expect(divider).toHaveAttribute('aria-valuenow', '200');
  await page.keyboard.press('ArrowLeft');
  expect((await inventory.boundingBox())!.width).toBe(200);
  await page.keyboard.press('End'); await page.keyboard.press('ArrowRight');
  expect((await inventory.boundingBox())!.width).toBe(480);
  const box = (await divider.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + 50); await page.mouse.down();
  await page.mouse.move(0, box.y + 50);
  await expect(divider).toHaveAttribute('aria-valuenow', '200');
  await page.mouse.move(1190, box.y + 50);
  await expect(divider).toHaveAttribute('aria-valuenow', '480');
  await page.mouse.move(box.x - 120, box.y + 50); await page.mouse.up();
  const preferred = (await inventory.boundingBox())!.width;
  expect(preferred).toBe(352);
  await page.setViewportSize({ width: 761, height: 600 });
  await expect(divider).toHaveAttribute('aria-valuenow', '352');
  // The chosen 352px still fits, so use End to choose the constrained maximum.
  await divider.focus(); await page.keyboard.press('End');
  expect((await pane.boundingBox())!.width).toBe(360);
  await noDocumentOverflow(page);
  await page.setViewportSize({ width: 1200, height: 800 });
  await expect(divider).toHaveAttribute('aria-valuemax', '480');
  await divider.focus(); await page.keyboard.press('End');
  await expect(divider).toHaveAttribute('aria-valuenow', '480');
  await page.setViewportSize({ width: 761, height: 600 });
  await expect(divider).toHaveAttribute('aria-valuenow', '353');
  await page.setViewportSize({ width: 280, height: 400 });
  await expect(divider).not.toBeVisible();
  await noDocumentOverflow(page);
  await page.getByRole('button', { name: 'Collapse inventory' }).click();
  await expect(page.getByRole('button', { name: 'Expand inventory' })).toBeFocused();
  await noDocumentOverflow(page);
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'Collapse inventory' })).toBeFocused();
  await noDocumentOverflow(page);
  await page.setViewportSize({ width: 1200, height: 800 });
  await expect(divider).toHaveAttribute('aria-valuenow', '480');
  await page.evaluate(() => {
    const f = window.explorerFixture;
    f.emit(0, 1, f.metadata(0)); f.data(0, [-2, 0, 2]);
  });
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'streaming');
  await nativeCamera(page);
  const scroller = page.locator('.matrix-scroll');
  await scroller.evaluate((host) => {
    host.dispatchEvent(new WheelEvent('wheel', { deltaY: -300, cancelable: true,
      clientX: host.getBoundingClientRect().left, clientY: host.getBoundingClientRect().top }));
    host.scrollLeft = 32; host.scrollTop = 24;
  });
  const camera = () => scroller.evaluate((host) => ({
    extent: (host.firstElementChild as HTMLElement).style.width, x: host.scrollLeft, y: host.scrollTop,
  }));
  const previousCamera = await camera();
  const allocations = await page.evaluate(() => window.explorerFixture.metrics.scalarAllocations);
  await page.getByRole('button', { name: 'Collapse inventory' }).click();
  expect(await camera()).toEqual(previousCamera);
  await page.evaluate(() => window.explorerFixture.data(0, [1, -1, 0]));
  await expect.poll(() => page.evaluate(() => window.explorerFixture.renderers[0]!.populatedPrefix)).toBe(6);
  await page.getByRole('button', { name: 'Expand inventory' }).click();
  expect(await camera()).toEqual(previousCamera);
  expect((await inventory.boundingBox())!.width).toBe(480);
  expect(await page.evaluate(() => window.explorerFixture.metrics.scalarAllocations)).toBe(allocations);
  expect(await canvas!.evaluate((node) => node === document.querySelector('.matrix-scroll canvas'))).toBe(true);
  expect(await page.evaluate(() => ({ requests: window.explorerFixture.requests.length, cancelled: window.explorerFixture.cancelled })))
    .toEqual({ requests: 3, cancelled: [] });
  await page.reload();
  await expect(divider).toHaveAttribute('aria-valuenow', '480');
  expect((await inventory.boundingBox())!.width).toBe(480);
  await noDocumentOverflow(page);
});
