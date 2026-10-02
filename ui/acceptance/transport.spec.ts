/* eslint-disable @typescript-eslint/no-explicit-any -- Test-only JSON evidence. */
import { expect } from '@playwright/test';
import { test, matrix } from './product-harness';
import { revealTensor } from '../tests/tensor-tree-helpers';

// Logical transport cases use headless Chromium; native focus/geometry stays headed.
test.use({ headless: true });

test('live tokenizer uses real Unicode IDs/spans and suppresses delayed old responses', async ({ page, product }, testInfo) => {
  const { metrics, embeddingDone } = product;
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('acceptance/fixture');
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  const input = page.getByRole('textbox', { name: 'Prompt', exact: true });
  const text = 'A😀e\u0301<special>  café 你好';
  const responsePromise = page.waitForResponse((r) => r.url().endsWith('/tokenize') && r.request().postDataJSON().text === text);
  await input.fill(text);
  const response = await responsePromise;
  const result = await response.json();
  await expect(page.locator('.tokenizer-status')).toContainText(`${result.tokens.length} tokens`);
  expect(result.text).toBe(text);
  // Strip annotation widgets to prove source appears once, with exact Unicode.
  const source = await input.evaluate((node) => {
    const clone = node.cloneNode(true) as HTMLElement;
    clone.querySelectorAll('.token-opening,.token-closing,.unmapped-annotation').forEach((n) => n.remove());
    return clone.textContent;
  });
  expect(source).toBe(text);
  const ids = await page.locator('.token-ids').allTextContents();
  expect(ids.flatMap((id) => id.split(', ').map(Number))).toEqual(result.tokens.map((t: any) => t.id));
  const emoji = result.tokens.filter((t: any) => t.start === 1 && t.end === 2);
  expect(emoji.length).toBeGreaterThan(1); // byte tokens overlap one Unicode point
  await page.screenshot({ path: testInfo.outputPath('unicode-tokenizer.png') });
  let release!: () => void;
  const barrier = new Promise<void>((resolve) => { release = resolve; });
  let entered = false;
  await page.route('**/tokenize', async (route) => {
    // Only delay delivery; response bytes still come from the real tokenizer.
    const response = await route.fetch();
    if (route.request().postDataJSON().text === 'old') { entered = true; await barrier; }
    await route.fulfill({ response });
  });
  await input.fill('old');
  await expect.poll(() => entered).toBe(true);
  await input.fill('new😀');
  await expect(page.locator('.tokenizer-status')).toContainText('current prompt');
  await embeddingDone(page, 8);
  const current = await page.locator('.token-ids').allTextContents();
  const settled = await metrics(page);
  release();
  await page.unrouteAll({ behavior: 'wait' });
  await expect(page.locator('.token-ids')).toHaveText(current);
  expect(await metrics(page)).toMatchObject({ uploads: settled.uploads, createdTextures: settled.createdTextures });
  const persisted = await page.evaluate(() => sessionStorage.getItem(Object.keys(sessionStorage)[0]!));
  await page.reload();
  await expect(page.getByRole('contentinfo').locator('.session-status')).toHaveAttribute('data-state', 'ready');
  expect(await page.evaluate(() => sessionStorage.getItem(Object.keys(sessionStorage)[0]!))).toBe(persisted);
});

test('cancel and network disconnect preserve incomplete status and return resources to baseline', async ({ page, product }) => {
  const { control, metrics, idle, open } = product;
  for (const [name, disconnect] of [[matrix, false], ['model.layers.0.mlp.up_proj.weight', true]] as const) {
    await control('arm', { kind: 'logical_tensor' });
    if (name === matrix) await open(page, name);
    else await (await revealTensor(page.getByRole('button', { includeHidden: true, name: new RegExp(name.replaceAll('.', '\\.')) }))).click();
    await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'streaming');
    if (disconnect) {
      await page.context().setOffline(true);
      await page.getByRole('button', { name: 'Cancel loading', exact: true }).click();
      await page.context().setOffline(false);
    } else await page.getByRole('button', { name: 'Cancel loading', exact: true }).click();
    await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'cancelled');
    await idle();
    expect(Object.keys((await control()).artifacts)).toHaveLength(0);
  }
  await page.getByRole('button', { name: 'Session options', exact: true }).click();
  await page.getByRole('button', { name: 'Close session', exact: true }).click();
  await expect.poll(async () => (await metrics(page)).textures).toBe(0);
  await expect.poll(async () => (await metrics(page)).readers).toBe(0);
});

test('real producer errors are distinct from cancellation in both primary and auxiliary UI results', async ({ page, product }) => {
  const { control, idle, open } = product;
  await control('arm', { kind: 'tensor_statistics', mode: 'pre-meta-error' });
  await open(page);
  await expect(page.locator('[data-result=statistics]')).toHaveAttribute('data-state', 'failed');
  await expect(page.locator('[data-result=tensor]')).toHaveCount(0);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  await idle();
  const before = Object.keys((await control()).artifacts);
  await control('arm', { kind: 'logical_tensor', mode: 'midstream-error' });
  await (await revealTensor(page.getByRole('button', { includeHidden: true, name: /model\.layers\.0\.mlp\.up_proj\.weight/ }))).click();
  await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'streaming');
  await control('release', {});
  await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'failed');
  await idle();
  expect(Object.keys((await control()).artifacts)).toEqual(before);
});

test('real A→B→A response reordering and model/session changes show only the latest generation', async ({ page, product }) => {
  const { control, metrics, idle, tokenizer, closeSession, embeddingDone } = product;
  const input = await tokenizer(page);
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  let entered = 0;
  await page.route('**/embeddings', async route => {
    const response = await route.fetch();
    const index = ++entered;
    if (index <= 2) await held;
    await route.fulfill({ response });
  });
  await input.fill('A'); await expect.poll(() => entered).toBe(1);
  await input.fill('B'); await expect.poll(() => entered).toBe(2);
  await expect(page.locator('.matrix-scroll:visible')).toBeVisible();
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
  await input.fill('A'); await expect.poll(() => entered).toBe(3);
  await embeddingDone(page, 2);
  const canvas = page.locator('.matrix-scroll'); await canvas.focus(); await canvas.press('ArrowDown');
  const newest = await page.locator('.inspection-readout').textContent();
  const settled = await metrics(page);
  release(); await page.unrouteAll({ behavior: 'wait' });
  await expect(page.locator('.inspection-readout')).toHaveText(newest!);
  expect(await metrics(page)).toMatchObject({ uploads: settled.uploads, createdTextures: settled.createdTextures });
  await expect(page.locator('.tokenizer-status')).toContainText('2 tokens');
  // Keep a real stream open, then replace its owning model/session twice.
  await control('arm', { kind: 'input_embeddings' });
  await input.fill('switch😀');
  await expect(page.getByRole('status').filter({ hasText: 'Streaming input embeddings…' })).toBeVisible();
  await page.getByRole('combobox').selectOption('acceptance/unsupported');
  await expect(page.getByText(/Input embeddings are unavailable/)).toBeVisible();
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  await input.fill('still editable😀');
  await expect(page.locator('.tokenizer-status')).toContainText('current prompt');
  await expect(page.getByText(/Input embeddings are unavailable/)).toBeVisible();
  await idle();
  await control('release', {});
  await page.getByRole('combobox').selectOption('acceptance/fixture');
  await expect(page.locator('.input-embeddings .embedding-shape')).toContainText('× 576] · float32');
  await closeSession(page);
});

test('repeated prompt edits and explorer unmounts cancel real embedding readers and GPU owners', async ({ page, context , product }) => {
  const { control, metrics, idle, tokenizer, closeSession } = product;
  for (let cycle = 0; cycle < 3; cycle++) {
    const input = await tokenizer(page);
    await control('arm', { kind: 'input_embeddings' });
    await input.fill(`cancel-${cycle}😀`);
    await expect(page.getByRole('status').filter({ hasText: 'Streaming input embeddings…' })).toBeVisible();
    await expect.poll(async () => (await control()).control.entered).toBe(true);
    await input.fill(`replacement-${cycle}`);
    await expect(page.locator('[data-staging]')).toHaveCount(0);
    await expect(page.locator('.matrix-scroll:visible')).toBeVisible();
    await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
    await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
    await idle();
    await expect.poll(async () => (await metrics(page)).textures).toBe(0);
    await expect.poll(async () => (await metrics(page)).readers).toBe(0);
    await control('release', {});
    await closeSession(page);
    const cdp = await context.newCDPSession(page); await cdp.send('HeapProfiler.collectGarbage'); await cdp.detach();
    expect((await metrics(page)).arrays).toEqual([]);
  }
});
