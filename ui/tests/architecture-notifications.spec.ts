import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { contractInventory, contractResponse } from './architecture-fixtures';

async function ready(page: Page) {
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('aria-busy', 'false');
  await expect(page.locator('.architecture-node').first()).toBeAttached();
}
async function unobstructed(page: Page) {
  await expect.poll(() => page.evaluate(() => {
    const dock = document.querySelector('.architecture-camera-dock')!.getBoundingClientRect();
    const viewport = document.querySelector('.architecture-flow')!.getBoundingClientRect();
    const toasts = [...document.querySelectorAll('.app-toast')];
    return dock.top >= viewport.top && dock.bottom <= viewport.bottom && toasts.every((toast) => {
      const rect = toast.getBoundingClientRect();
      return dock.right <= rect.left || dock.left >= rect.right || dock.bottom <= rect.top || dock.top >= rect.bottom;
    }) && [...document.querySelectorAll('.architecture-camera-dock button, .toast-actions button')].every((button) => {
      const rect = button.getBoundingClientRect();
      return button.contains(document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2));
    });
  })).toBe(true);
}

test('actionable session errors leave the camera dock and notifications pointer reachable', async ({ page }, testInfo) => {
  let deletes = 0;
  const requests: string[] = [];
  await page.route('**/runtime-config.json', r => r.fulfill({ json: { backend_base_url: 'https://notifications.example' } }));
  await page.route('https://notifications.example/**', r => {
    const path = new URL(r.request().url()).pathname;
    requests.push(path);
    if (path === '/models') return r.fulfill({ json: { models: [{ id: contractResponse.model_id, display_name: 'Control fixture', architectures: [], tokenizer_available: true }] } });
    if (r.request().method() === 'DELETE') { deletes++; return r.abort('connectionrefused'); }
    if (path === '/sessions') return r.fulfill({ status: 201, json: { id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', model_id: contractResponse.model_id } });
    if (path.endsWith('/tensors')) return r.fulfill({ json: contractInventory });
    if (path.endsWith('/architecture')) return r.fulfill({ json: contractResponse });
    return r.fulfill({ status: 204 });
  });
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(contractResponse.model_id);
  await expect(page.getByText('Session active.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await ready(page);
  const originalBounds = await page.locator('.architecture-flow').boundingBox();
  await page.getByRole('button', { name: 'Session options', exact: true }).click();
  await page.getByRole('button', { name: 'Close session', exact: true }).click();
  await expect(page.locator('.app-toast')).toHaveCount(1);
  await ready(page); await unobstructed(page);
  expect(await page.locator('.architecture-flow').boundingBox()).toEqual(originalBounds);
  await page.screenshot({ path: testInfo.outputPath('camera-with-notification.png') });

  // Use real pointer clicks, including the actionable toast, without force.
  const transform = () => page.locator('.react-flow__viewport').getAttribute('style');
  for (const label of ['Zoom in', 'Zoom out']) {
    const before = await transform();
    await page.getByRole('button', { name: label, exact: true }).click();
    await expect.poll(transform).not.toBe(before);
  }
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  await page.getByRole('button', { name: 'Retry close', exact: true }).click();
  await expect.poll(() => deletes).toBe(2);
  await expect(page.locator('.app-toast')).toHaveCount(1);
  await ready(page); await unobstructed(page);

  // A live resize re-evaluates horizontal collision as well as toast height.
  const initialSize = page.viewportSize()!;
  await page.setViewportSize(initialSize.width < 600 ? { width: 1440, height: 900 } : { width: 390, height: 844 });
  await unobstructed(page);
  await page.setViewportSize(initialSize); await unobstructed(page);
  const element = await page.locator('.react-flow').elementHandle();
  const graph = page.getByLabel('Architecture graph', { exact: true });
  await ready(page);
  const count = await graph.getAttribute('data-layout-count');
  const camera = await transform();
  const bounds = await page.locator('.architecture-flow').boundingBox();
  const beforeRequests = [...requests];
  await page.getByRole('button', { name: 'Dismiss notification', exact: true }).click();
  await expect(page.locator('.app-toast')).toHaveCount(0);
  await unobstructed(page);
  await expect.poll(() => page.locator('.architecture-camera-dock').evaluate(el => getComputedStyle(el).bottom)).toBe('10px');
  expect(await page.locator('.architecture-flow').boundingBox()).toEqual(bounds);
  expect(await transform()).toBe(camera);
  await expect(graph).toHaveAttribute('data-layout-count', count!);
  expect(await element!.evaluate(el => el === document.querySelector('.react-flow'))).toBe(true);
  expect(requests).toEqual(beforeRequests);
});
