import type { Page } from '@playwright/test';

/** Keep native-resolution scalar/guide oracles explicit after fit-width became the default. */
export async function nativeCamera(page: Page) {
  await page.locator('.matrix-scroll').evaluate((host) => {
    const extent = host.firstElementChild as HTMLElement;
    for (let i = 0; i < 20; i++) {
      const width = extent.style.width;
      host.dispatchEvent(new WheelEvent('wheel', {
        deltaY: 10000, clientX: host.getBoundingClientRect().left,
        clientY: host.getBoundingClientRect().top, cancelable: true,
      }));
      // Native scale clamps the extent. Extra events at that bound still trigger
      // synchronous scientific redraws, which dominated the large DPR-2 CI case.
      if (width && extent.style.width === width) return;
    }
    throw new Error('Matrix camera did not reach its native lower bound');
  });
}
