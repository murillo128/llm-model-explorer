import { expect, test } from '@playwright/test';
import { models, sessionA, tensors } from '../src/test/shell-fixtures';
import { revealTensor } from './tensor-tree-helpers';

const path = ['model', 'layers', '12', 'attention', 'projection', 'nested', 'deep', 'long_module_name_for_truncation'.repeat(5), 'weight'];

test('compact deep navigator and contextual metadata remain accessible in narrow panels', async ({ page }, testInfo) => {
  const inventory = [
    { ...tensors[2]!, id: 'deep', name: path.join('.'), path },
    { ...tensors[2]!, id: 'other', name: 'other.weight', path: ['other', 'weight'] },
  ];
  await page.route('**/runtime-config.json', (route) => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  await page.route('https://backend.example/**', (route) => {
    const url = route.request().url();
    return route.fulfill({ status: route.request().method() === 'POST' ? 201 : 200,
      json: url.endsWith('/models') ? { models } : url.endsWith('/tensors') ? { coverage: 'complete', diagnostics: [], tensors: inventory } : sessionA });
  });
  await page.goto('/');
  await page.getByRole('combobox').selectOption(models[0]!.id);
  const tree = page.getByRole('complementary', { name: 'Tensor inventory' });
  const leaf = tree.getByRole('button', { includeHidden: true, name: /model.layers/ });
  const other = tree.getByRole('button', { name: /other.weight/ });
  await (await revealTensor(leaf)).click();
  await expect(leaf).toHaveAttribute('aria-pressed', 'true');
  await expect(leaf).toHaveText('weight[1 × 2 × 3] · int8');
  expect(await leaf.evaluate((node) => node.getBoundingClientRect().height)).toBe(28);
  await expect(leaf.locator('svg')).toHaveCount(1);
  await expect(leaf.locator('summary')).toHaveCount(0);
  await expect(tree.locator('summary')).toHaveCount(9);
  const workspace = page.getByRole('region', { name: 'Tensor Explorer workspace', exact: true });
  await expect(workspace.getByText(path.join(' › '), { exact: true })).toHaveCount(1);
  await expect(page.locator('.app-bar')).not.toContainText('weight');
  await expect(workspace.locator('h2')).toHaveText(path.join(' › '));
  await expect(workspace.getByText('[1 × 2 × 3] · int8', { exact: true })).toBeVisible();
  await expect(workspace.getByText('Logical path')).toHaveCount(0);
  expect(await workspace.locator('h2').evaluate((node) => node.scrollWidth > node.clientWidth)).toBe(true);
  const info = workspace.getByRole('button', { name: 'Tensor information' });
  await info.focus(); await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button')).toBeFocused();
  await expect(dialog.getByText(path.join(' › '), { exact: true })).toBeVisible();
  await expect(dialog.getByText('Logical dtype')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(info).toBeFocused();
  await info.click();
  await tree.getByRole('heading', { name: 'Tensors' }).click();
  await expect(dialog).toHaveCount(0);
  await leaf.focus(); await page.keyboard.press('ArrowLeft');
  await expect(tree.locator('summary').filter({ hasText: 'long_module_name' })).toBeFocused();
  await page.keyboard.press('ArrowLeft');
  await expect(leaf).not.toBeVisible();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('End');
  await expect(other).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(other).toHaveAttribute('aria-pressed', 'true');
  await expect(workspace.locator('h2')).toHaveText('other › weight');
  // Constrain the inventory independently of the browser viewport.
  await tree.evaluate((node) => { node.style.width = '180px'; });
  await leaf.scrollIntoViewIfNeeded();
  expect(await tree.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
  await expect(leaf.locator('.tensor-leaf-metadata')).not.toBeVisible();
  await leaf.click();
  await expect(info).toBeInViewport();
  await expect(workspace.getByText('[1 × 2 × 3] · int8', { exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('compact-tensor-navigation.png') });
  await testInfo.attach('compact tensor navigation', { path: testInfo.outputPath('compact-tensor-navigation.png'), contentType: 'image/png' });
  // Pane containment must not clip the bottom of the on-demand metadata dialog.
  await page.setViewportSize({ width: 280, height: 400 });
  await info.click();
  await expect(dialog).toBeVisible();
  expect(await dialog.evaluate((node) => {
    const pane = node.closest('.working-surface')!.getBoundingClientRect();
    const rect = node.getBoundingClientRect();
    return rect.top >= pane.top && rect.bottom <= pane.bottom && rect.left >= pane.left && rect.right <= pane.right;
  })).toBe(true);
  await dialog.evaluate((node) => { node.scrollTop = node.scrollHeight; });
  await expect(dialog.getByText('Logical dtype', { exact: true })).toBeInViewport({ ratio: 1 });
  await page.keyboard.press('Escape');
  await expect(info).toBeFocused();
});
