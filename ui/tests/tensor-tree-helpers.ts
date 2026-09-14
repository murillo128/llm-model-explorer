import type { Locator } from '@playwright/test';

/** Follow actual disclosures before selecting a leaf under the quiet defaults. */
export async function revealTensor(leaf: Locator) {
  await leaf.waitFor({ state: 'attached' });
  const ancestors = leaf.locator('xpath=ancestor::details');
  for (const branch of await ancestors.all()) {
    if (await branch.getAttribute('open') === null) await branch.locator(':scope > summary').click();
  }
  return leaf;
}
