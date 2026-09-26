import type { Page } from '@playwright/test';

export async function viewOptions(page: Page) {
  const trigger = page.getByRole('button', { name: 'View options', exact: true });
  if (await trigger.getAttribute('aria-expanded') !== 'true') await trigger.click();
  return page.getByRole('dialog', { name: 'View options', exact: true });
}
export async function graphAction(page: Page, name: string | RegExp) {
  await page.getByRole('button', { name, exact: true }).click();
}
export async function graphPreference(page: Page, name: string, checked: boolean) {
  const options = await viewOptions(page);
  await options.getByLabel(name, { exact: true }).setChecked(checked);
  await page.keyboard.press('Escape');
}
export async function selectComponent(page: Page, id: string) {
  const restore = page.getByRole('button', { name: 'Expand browser', exact: true });
  if (await restore.isVisible()) await restore.click();
  await page.getByRole('searchbox', { name: 'Search components', exact: true }).fill(id);
  await page.getByRole('group', { name: 'Model search results', exact: true }).locator(`[data-node-id=${JSON.stringify(id)}] [data-browser-name]`).click();
}
/** Existing navigation scenarios explicitly opt into reveal after selection. */
export async function findComponent(page: Page, id: string) {
  await selectComponent(page, id);
  await page.getByRole('button', { name: 'Center selected', exact: true }).click();
  await page.getByRole('searchbox', { name: 'Search components', exact: true }).fill('');
}
export async function openShared(page: Page, id: string) {
  const restore = page.getByRole('button', { name: 'Expand browser', exact: true });
  if (await restore.isVisible()) await restore.click();
  await page.getByRole('searchbox', { name: 'Search components', exact: true }).fill('');
  await page.locator(`[data-family-id=${JSON.stringify(id)}]`).getByRole('button', { name: /^Select shared family/ }).click();
  await page.getByLabel('Graph selection', { exact: true }).getByRole('button', { name: 'Explore structure', exact: true }).click();
}
export async function chooseInstance(page: Page, stack: string, id: string) {
  const picker = page.getByRole('combobox', { name: `Expand instance of ${stack}`, exact: true });
  if (!await picker.count()) {
    await page.getByRole('button', { name: 'Model overview', exact: true }).click();
    await page.getByRole('button', { name: `Explore stack ${stack}`, exact: true }).click();
  }
  await picker.selectOption(id);
}
