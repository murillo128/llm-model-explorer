import { expect, test } from '@playwright/test';
import { chooseInstance, findComponent, graphAction } from './architecture-controls';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;

test('authored components retain labels, focus, raw inspection and reversible detail', async ({ page }) => {
  await page.goto(harness);
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('components');
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await chooseInstance(page, 'Decoder layers', 'layer-3');
  await graphAction(page, 'Focus layer');
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await expect(page.locator('.react-flow__node[data-id="layer-3.mlp"]')).toContainText('MLP');
  await expect(page.locator('.react-flow__node[data-id^="mlp:"]')).toHaveCount(0);
  await graphAction(page, 'Focus MLP');
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByRole('button', { name: 'MLP', exact: true })).toBeVisible();
  await expect(page.getByText('MLP (derived)', { exact: true })).toHaveCount(0);
  await expect(page.locator('.react-flow__node[data-id="layer-3.gate"]')).toBeVisible();
  await findComponent(page, 'layer-3.mlp');
  await expect(page.getByLabel('Graph selection')).toContainText('MLP');
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  await expect(page.locator('output')).toContainText(': layer-3.mlp');
  await graphAction(page, 'Toggle selected group');
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await expect(page.locator('.react-flow__node[data-id="layer-3.gate"]')).toHaveCount(0);
  await graphAction(page, 'Show all operations');
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await expect(page.locator('.react-flow__node[data-id="layer-3.gate"]')).toHaveCount(1);
  await expect(page.locator('.react-flow__node[data-id^="mlp:"]')).toHaveCount(0);
});
