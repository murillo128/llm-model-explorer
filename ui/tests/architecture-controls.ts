import { expect } from '@playwright/test';
import type { Page } from '@playwright/test';

export async function viewOptions(page: Page) {
  const trigger = page.getByRole('button', { name: 'View options', exact: true });
  if (await trigger.getAttribute('aria-expanded') !== 'true') await trigger.click();
  return page.getByRole('dialog', { name: 'View options', exact: true });
}
export async function graphAction(page: Page, name: string | RegExp) {
  const options = await viewOptions(page);
  await options.getByRole('button', { name, exact: true }).click();
  if (await options.isVisible()) await page.keyboard.press('Escape');
}
export async function graphPreference(page: Page, name: string, checked: boolean) {
  const options = await viewOptions(page);
  await options.getByLabel(name, { exact: true }).setChecked(checked);
  await page.keyboard.press('Escape');
}
export async function findComponent(page: Page, id: string) {
  await page.getByRole('button', { name: 'Find component', exact: true }).click();
  await page.getByRole('combobox', { name: 'Search components', exact: true }).fill(id);
  // IDs locate the exact authored target; user-facing result content is a concise
  // label plus containment context. All results come from received graph records.
  await page.locator(`[role="option"][data-node-id=${JSON.stringify(id)}]`).click();
  await expect(page.getByRole('button', { name: 'Find component', exact: true })).toBeFocused();
}
export async function chooseInstance(page: Page, stack: string, id: string) {
  const picker = page.getByRole('combobox', { name: `Expand instance of ${stack}`, exact: true });
  if (!await picker.count()) {
    await page.getByRole('button', { name: 'Model overview', exact: true }).click();
    await page.getByRole('button', { name: `Explore stack ${stack}`, exact: true }).click();
  }
  await picker.selectOption(id);
}
