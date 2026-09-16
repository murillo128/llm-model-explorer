import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { cameraProbe, releaseCompletions, releaseLayouts, releaseSizes } from './architecture-camera-probe';
import { findComponent, graphAction } from './architecture-controls';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;
const panel = (page: Page) => page.getByLabel('Architecture graph', { exact: true });
const sample = (page: Page) => page.evaluate(() => window.cameraProbe.sample());
async function ready(page: Page) {
  await expect(panel(page)).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByRole('alert')).toHaveCount(0);
  const state = await sample(page);
  expect(state.consumed, JSON.stringify(state)).toEqual(state.actual);
}
async function pending(page: Page) {
  await page.evaluate(() => { window.cameraProbe.holdLayout = true; window.cameraProbe.holdSizes = true; window.cameraProbe.events = []; });
}
async function resizeWhilePending(page: Page) {
  // A real container resize with controlled notification ordering, not a fabricated renderer size.
  await expect.poll(() => page.evaluate(() => window.cameraProbe.layouts.length)).toBeGreaterThan(0);
  await page.evaluate(() => { document.querySelector<HTMLElement>('.architecture-flow')!.style.marginBottom = '40px'; });
  await expect.poll(async () => { const s = await sample(page); return s.actual[1] !== s.consumed[1]; }).toBe(true);
  await releaseLayouts(page);
  await expect(panel(page)).toHaveAttribute('data-layout-count', /^[1-9]\d*$/);
}

test.beforeEach(async ({ page }) => { await cameraProbe(page); });

for (const fixture of ['templates-absent', 'templates']) {
  test(`${fixture}: current viewport and committed camera gate cold/remounted readiness`, async ({ page }, info) => {
    await page.goto(`${harness}?fixture=${fixture}`); await ready(page);
    // These tests retain the project's actual 390px viewport in narrow runs.
    expect((await page.locator('.architecture-flow').boundingBox())!.width).toBeLessThanOrEqual(info.project.use.viewport!.width);
    const initial = await sample(page);
    await pending(page);
    // Remount with a saved camera. Parent geometry changes before the worker's reply.
    await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click();
    await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click();
    await resizeWhilePending(page);
    await expect(panel(page)).toHaveAttribute('aria-busy', 'true');
    expect(await page.evaluate(() => window.cameraProbe.events.filter((e) => e.event.endsWith('requested')))).toEqual([]);
    await page.evaluate(() => { window.cameraProbe.holdCompletion = true; });
    await releaseSizes(page);
    await expect.poll(() => page.evaluate(() => window.cameraProbe.completions.length)).toBeGreaterThan(0);
    await expect(panel(page)).toHaveAttribute('aria-busy', 'true');
    await releaseCompletions(page); await ready(page);
    expect((await sample(page)).camera).toEqual(initial.camera);
    await info.attach('camera-readiness', { body: JSON.stringify(await page.evaluate(() => window.cameraProbe.events), null, 2), contentType: 'application/json' });
  });
}

test('cold fit waits for current geometry, then selection, menus, hover and resize do not refit', async ({ page }, info) => {
  await page.goto(harness); await ready(page); await pending(page);
  await page.getByLabel('Fixture', { exact: true }).selectOption('templates');
  await resizeWhilePending(page);
  await expect(panel(page)).toHaveAttribute('aria-busy', 'true');
  expect(await page.evaluate(() => window.cameraProbe.events.filter((e) => e.event.endsWith('requested')))).toEqual([]);
  await releaseSizes(page); await ready(page);
  const before = await sample(page), count = await page.evaluate(() => window.cameraProbe.events.length);
  await page.locator('.architecture-node-label').first().click();
  await page.locator('.architecture-port').first().hover();
  await page.getByRole('button', { name: 'Find component', exact: true }).click(); await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'View options', exact: true }).click(); await page.keyboard.press('Escape');
  await page.setViewportSize({ width: info.project.use.viewport!.width, height: info.project.use.viewport!.height! - 30 });
  await expect.poll(async () => (await sample(page)).actual[1]).not.toBe(before.actual[1]);
  // An ordinary resize does not reopen initialization; await its own size notification.
  await expect.poll(async () => { const state = await sample(page); return state.actual.every((size, i) => size === state.consumed[i]); }).toBe(true);
  await ready(page);
  expect((await sample(page)).camera).toEqual(before.camera);
  expect((await sample(page)).layout).toBe(before.layout);
  expect(await page.evaluate((start) => window.cameraProbe.events.slice(start).filter((e) => e.event.endsWith('requested')), count)).toEqual([]);
});

test('pending scope camera cannot replace Back, a newer model, or a user camera', async ({ page }) => {
  await page.goto(`${harness}?fixture=templates`); await ready(page);
  await findComponent(page, 'layer-2.attention'); await ready(page);
  const previous = await sample(page);
  await pending(page);
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await resizeWhilePending(page);
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await releaseSizes(page); await ready(page);
  expect((await sample(page)).camera).toEqual(previous.camera);
  await pending(page);
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.cameraProbe.layouts.length)).toBeGreaterThan(0);
  await page.getByLabel('Fixture', { exact: true }).selectOption('partial');
  await releaseLayouts(page); await releaseSizes(page); await ready(page);
  const replacement = await sample(page);
  expect(replacement.graph).not.toBe(previous.graph);
  await pending(page);
  await page.getByLabel('Fixture', { exact: true }).selectOption('templates-absent');
  await resizeWhilePending(page);
  await graphAction(page, 'Zoom graph in');
  const user = await sample(page);
  await releaseSizes(page); await ready(page);
  expect((await sample(page)).camera).toEqual(user.camera);
});

for (const target of ['selected', 'connection'] as const) {
  test(`explicit Center ${target} supersedes pending shared-scope initialization`, async ({ page }, info) => {
    await page.goto(`${harness}?fixture=templates`); await ready(page);
    await findComponent(page, 'layer-2.attention.Q'); await ready(page);
    const previous = await sample(page);
    await pending(page);
    await page.getByRole('button', { name: 'Shared structure', exact: true }).click();
    await resizeWhilePending(page);
    await expect(panel(page)).not.toHaveAttribute('data-layout-count', previous.layout!);
    if (target === 'connection') {
      const edge = page.locator('.architecture-connection[data-source-node="layer-2.attention"][data-source-port="x"][data-target-node="layer-2.attention.Q"][data-target-port="x"]');
      await edge.focus(); await page.keyboard.press('Enter');
      await expect(page.getByRole('dialog', { name: 'Connection inspection', exact: true })).toBeVisible();
      await page.keyboard.press('Escape');
    }
    await expect(panel(page)).toHaveAttribute('aria-busy', 'true');
    await page.evaluate(() => { window.cameraProbe.events = []; });
    const pendingState = await sample(page), method = target === 'selected' ? 'setCenter' : 'fitView';
    await page.getByRole('button', { name: `Center ${target}`, exact: true }).click();
    await expect.poll(() => page.evaluate((name) => window.cameraProbe.events.filter((event) => event.event === `${name} completed`).length, method)).toBe(1);
    const commanded = await sample(page);
    expect(commanded.camera).not.toEqual(pendingState.camera);
    await releaseSizes(page);
    await expect.poll(async () => { const state = await sample(page); return state.actual.every((size, i) => size === state.consumed[i]); }).toBe(true);
    await ready(page);
    const settled = await sample(page), events = await page.evaluate(() => window.cameraProbe.events);
    await info.attach('explicit-center-precedence', { body: JSON.stringify({ pendingState, commanded, settled, events }, null, 2), contentType: 'application/json' });
    expect(settled.camera).toEqual(commanded.camera);
    expect(settled.layout).toBe(commanded.layout);
    expect(events.filter((event) => event.event.endsWith('requested')).map((event) => event.event)).toEqual([`${method} requested`]);
    // Cancelling this generation must leave the prior scope's Back snapshot usable.
    await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
    expect((await sample(page)).camera).toEqual(previous.camera);
  });
}

for (const derived of [false, true]) test(`explicit Center ${derived ? 'derived component' : 'selected'} replaces the pending ordinary-scope fit`, async ({ page }, info) => {
  await page.goto(`${harness}?fixture=${derived ? 'connections' : 'templates'}`); await ready(page);
  await findComponent(page, derived ? 'layer-3.gate' : 'layer-2.attention.Q'); await ready(page);
  if (derived) {
    await graphAction(page, 'Focus MLP'); await ready(page);
    await page.locator('[data-id="mlp:layer-3.gate"] .architecture-node-label').click();
  }
  const previous = await sample(page);
  await pending(page);
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await resizeWhilePending(page);
  await expect(panel(page)).not.toHaveAttribute('data-layout-count', previous.layout!);
  await expect(panel(page)).toHaveAttribute('aria-busy', 'true');
  await page.evaluate(() => { window.cameraProbe.events = []; });
  await page.getByRole('button', { name: 'Center selected', exact: true }).click();
  await releaseSizes(page); await ready(page);
  const events = await page.evaluate(() => window.cameraProbe.events);
  await info.attach('deferred-center-precedence', { body: JSON.stringify(events, null, 2), contentType: 'application/json' });
  expect(events.filter((event) => event.event.endsWith('requested')).map((event) => event.event)).toEqual(['setCenter requested']);
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  expect((await sample(page)).camera).toEqual(previous.camera);
});

test('late camera completion cannot commit after Back or model replacement', async ({ page }) => {
  await page.goto(`${harness}?fixture=templates`); await ready(page);
  await findComponent(page, 'layer-2.attention'); await ready(page);
  const previous = await sample(page);
  await page.evaluate(() => { window.cameraProbe.holdCompletion = true; });
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.cameraProbe.completions.length)).toBeGreaterThan(0);
  await expect(panel(page)).toHaveAttribute('aria-busy', 'true');
  await page.evaluate(() => { window.cameraProbe.holdCompletion = false; });
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  const back = await sample(page);
  expect(back.camera).toEqual(previous.camera);
  await releaseCompletions(page); await ready(page);
  expect(await sample(page)).toEqual(back);
  await page.evaluate(() => { window.cameraProbe.holdCompletion = true; });
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.cameraProbe.completions.length)).toBeGreaterThan(0);
  await page.evaluate(() => { window.cameraProbe.holdCompletion = false; });
  await page.getByLabel('Fixture', { exact: true }).selectOption('partial'); await ready(page);
  const replacement = await sample(page);
  await releaseCompletions(page); await ready(page);
  expect(await sample(page)).toEqual(replacement);
});
