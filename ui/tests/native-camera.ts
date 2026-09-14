import type { Page } from '@playwright/test';

/** Keep native-resolution scalar/guide oracles explicit after fit-width became the default. */
export async function nativeCamera(page: Page) {
  await page.locator('.matrix-scroll').evaluate((host) => {
    for (let i = 0; i < 20; i++) host.dispatchEvent(new WheelEvent('wheel', {
      deltaY: 10000, clientX: host.getBoundingClientRect().left,
      clientY: host.getBoundingClientRect().top, cancelable: true,
    }));
  });
}
