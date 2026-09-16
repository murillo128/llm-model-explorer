import { nativeCamera } from './native-camera';
import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type { Tokenization } from '../src/tokenizer/annotations';

import { color } from './scalar-oracle';
import { frame, meta, data, values, analysis } from './embedding-fixtures';

function tokens(text: string, ids = [2, 0, 2], special = true): Tokenization {
  return { text, add_special_tokens: special, tokens: text ? ids.map((id, index) => ({ index, id, token: 'fixture', decoded: '', special: false,
    ...(index < [...text].length ? { start: index, end: index + 1 } : {}) })) : [] };
}
async function count(page: Page, n: number) {
  await expect.poll(() => page.evaluate(() => window.tokenizerHarness.requests.length)).toBe(n);
}
async function embeddingCount(page: Page, n: number) {
  await expect.poll(() => page.evaluate(() => window.embeddingHarness.requests.length)).toBe(n);
}
async function tokenize(page: Page, index: number, result: Tokenization) {
  await page.evaluate(({ index, result }) => window.tokenizerHarness.complete(index, result), { index, result });
}
async function send(page: Page, index: number, bytes: number[], close = false) {
  await page.evaluate(({ index, bytes, close }) => window.embeddingHarness.send(index, bytes, close), { index, bytes, close });
}
async function stream(page: Page, index: number, ids: number[], width = 7) {
  await page.evaluate(index => window.embeddingHarness.headers(index), index);
  await send(page, index, [...meta(ids, width), ...data(ids, width), ...frame(4)], true);
}
async function start(page: Page) {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tokenizer.html?embeddings`);
  await count(page, 1); await tokenize(page, 0, tokens(''));
  await expect(page.getByText('No tokens to embed.')).toBeVisible();
  return page.getByRole('textbox', { name: 'Prompt', exact: true });
}

test('empty and unavailable embeddings share the full-width title/body composition', async ({ page }) => {
  const editor = await start(page);
  async function fullWidthHeader() {
    const panel = page.locator('.input-embeddings');
    await expect(panel.locator('.matrix-panel-header')).toHaveCount(1);
    const region = (await panel.boundingBox())!, title = (await panel.locator('.matrix-panel-header').boundingBox())!;
    const body = (await panel.locator('.viewer-panel-body').boundingBox())!;
    expect(title).toEqual({ x: region.x, y: region.y, width: region.width, height: 40 });
    expect(body.x).toBe(title.x); expect(body.width).toBe(title.width);
    expect(body.y).toBe(title.y + title.height);
  }
  await fullWidthHeader();
  await editor.fill('A'); await count(page, 2); await tokenize(page, 1, tokens('A'));
  await embeddingCount(page, 1);
  await fullWidthHeader();
  await page.evaluate(() => window.embeddingHarness.headers(0, 422));
  await expect(page.getByText('Input embeddings are unavailable for this model. Tokenization remains usable.')).toBeVisible();
  await fullWidthHeader();
  expect(await page.evaluate(() => window.embeddingHarness.auxiliary.length)).toBe(0);
});

for (const dpr of [1, 2]) test(`exact progressive rows, linked annotations and frozen editor at DPR ${dpr}`, async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: page.viewportSize()!.width, height: page.viewportSize()!.height, deviceScaleFactor: dpr, mobile: false });
  const editor = await start(page);
  await editor.fill('A😀A'); await count(page, 2); await tokenize(page, 1, tokens('A😀A'));
  await embeddingCount(page, 1);
  expect(await page.evaluate(() => window.embeddingHarness.requests[0]!.token_ids)).toEqual([2, 0, 2]);
  const before = await page.locator('.tokenizer-editor').boundingBox();
  await page.evaluate(() => window.embeddingHarness.headers(0));
  // META and a split first float share the network chunk. The matrix must mount
  // before DATA, and become useful before COMPLETE or even a complete row.
  const payload = data([2, 0, 2]);
  await send(page, 0, [...meta([2, 0, 2]), ...payload.slice(0, 15)]);
  const matrix = page.locator('.matrix-scroll');
  await expect(matrix).toBeVisible();
  await matrix.focus();
  await expect(page.locator('.inspection-readout')).toContainText('Unavailable');
  await send(page, 0, payload.slice(15, 16));
  await expect(page.locator('.inspection-readout')).toContainText('0.4375');
  await expect(page.getByText('Streaming input embeddings…')).toBeVisible();
  await send(page, 0, [...payload.slice(16), ...frame(4)], true);
  await expect(page.getByText('[3 × 7] · float32')).toBeVisible();
  const expected = values([2, 0, 2]);
  for (let row = 0; row < 3; row++) {
    if (row) { for (let i = 0; i < 6; i++) await matrix.press('ArrowLeft'); await matrix.press('ArrowDown'); }
    for (let col = 0; col < 7; col++) {
      if (col) await matrix.press('ArrowRight');
      await expect(page.locator('.inspection-readout span').first()).toHaveText(`row ${row} · column ${col}`);
      await expect(page.locator('.inspection-readout span').last()).toHaveText(String(expected[row * 7 + col]));
      await expect(page.locator(`[data-token-index="${row}"]`)).toHaveAttribute('data-active-token', '');
    }
  }
  await nativeCamera(page);
  const canvas = matrix.locator('canvas');
  const box = (await canvas.boundingBox())!;
  expect(box.width * dpr).toBeCloseTo(7, 5); expect(box.height * dpr).toBeCloseTo(3, 5);
  await editor.focus(); await editor.press('Home'); await editor.press('ArrowRight');
  const selection = await page.evaluate(() => window.tokenizerHarness.selection());
  await page.mouse.move(0, 0);
  const beforeLink = await page.evaluate(() => {
    const r = window.embeddingHarness.renderers.at(-1)!; r.draw();
    const gl = r.canvas.getContext('webgl2')!;
    const pixels = new Uint8Array(7 * 3 * 4); gl.readPixels(0, 0, 7, 3, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return { pixels: [...pixels], uploads: r.diagnostics.scalarUploadCalls };
  });
  await page.locator('[data-token-index="2"]').hover();
  await expect(page.locator('[data-token-index="2"]')).toHaveAttribute('data-active-token', '');
  const afterLink = await page.evaluate(() => {
    const r = window.embeddingHarness.renderers.at(-1)!; r.draw();
    const gl = r.canvas.getContext('webgl2')!;
    const pixels = new Uint8Array(7 * 3 * 4); gl.readPixels(0, 0, 7, 3, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return { pixels: [...pixels], uploads: r.diagnostics.scalarUploadCalls };
  });
  // GL reads bottom-up: only logical row 2 gets amber; scalar storage is untouched.
  expect(afterLink.pixels.slice(0, 28)).not.toEqual(beforeLink.pixels.slice(0, 28));
  expect(afterLink.pixels.slice(28)).toEqual(beforeLink.pixels.slice(28));
  expect(afterLink.uploads).toBe(beforeLink.uploads);
  // Both copies of ID 2 remain separate sequence positions.
  await expect(page.locator('[data-token-index="0"]')).not.toHaveAttribute('data-active-token');
  await page.locator('[data-token-index="2"]').click();
  expect(await page.evaluate(() => window.tokenizerHarness.selection())).toEqual(selection);
  await page.locator('[data-token-index="1"]').focus();
  await page.locator('[data-token-index="1"]').press('Enter');
  await expect(page.locator('[data-token-index="1"]')).toHaveAttribute('data-active-token', '');
  expect(await page.evaluate(() => window.tokenizerHarness.selection())).toEqual(selection);
  expect(await page.evaluate(() => window.tokenizerHarness.source())).toBe('A😀A');
  expect((await page.locator('.tokenizer-editor').boundingBox())!.height).toBe(before!.height);
  await expect(page.locator('.row-distributions, .column-distributions')).toHaveCount(2);
  await expect(page.getByRole('textbox')).toHaveCount(1);
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight && document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('A→B→A, late headers/data, options and session changes cancel superseded generations', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('A'); await count(page, 2); await tokenize(page, 1, tokens('A', [2])); await embeddingCount(page, 1);
  await page.evaluate(() => window.embeddingHarness.headers(0));
  await send(page, 0, [...meta([2]), ...data([2])]);
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await editor.fill('B'); await count(page, 3);
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  await tokenize(page, 2, tokens('B', [0])); await embeddingCount(page, 2);
  await editor.fill('A'); await count(page, 4); await tokenize(page, 3, tokens('A', [2])); await embeddingCount(page, 3);
  await stream(page, 2, [2]);
  await expect(page.getByText('[1 × 7] · float32')).toBeVisible();
  await stream(page, 1, [0]); await send(page, 0, frame(4), true);
  expect(await page.evaluate(() => window.embeddingHarness.requests.slice(0, 2).map(r => r.aborted))).toEqual([true, true]);
  expect(await page.evaluate(() => window.embeddingHarness.cancelled)).toContain('/operations/00000000-0000-4000-8000-000000000000');
  await page.getByRole('checkbox', { name: 'Add special tokens' }).uncheck(); await count(page, 5);
  await page.getByRole('checkbox', { name: 'Add special tokens' }).check(); await count(page, 6);
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
  await tokenize(page, 4, tokens('A', [0], false)); await embeddingCount(page, 3);
  await tokenize(page, 5, tokens('A', [2])); await embeddingCount(page, 4);
  await page.getByRole('button', { name: 'Change session' }).click(); await count(page, 7);
  await stream(page, 3, [2]); await expect(page.locator('.matrix-scroll')).toBeVisible();
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
  await tokenize(page, 6, tokens('A', [0])); await embeddingCount(page, 5); await stream(page, 4, [0]);
  expect(await page.evaluate(() => window.embeddingHarness.requests[4]!.session)).toMatch(/^bbbb/);
  await expect(page.getByText('[1 × 7] · float32')).toBeVisible();
});

test('unsupported capability, cancellation, invalid echo and empty input stay bounded below the editable prompt', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('A'); await count(page, 2); await tokenize(page, 1, tokens('A', [2])); await embeddingCount(page, 1);
  await page.evaluate(() => window.embeddingHarness.headers(0, 422));
  await expect(page.getByText(/Input embeddings are unavailable/)).toBeVisible();
  await expect(page.locator('.token-opening .token-ids')).toHaveText('2');
  await editor.fill('B'); await count(page, 3); await tokenize(page, 2, tokens('B', [0])); await embeddingCount(page, 2);
  await page.evaluate(() => window.embeddingHarness.headers(1)); await send(page, 1, frame(6), true);
  await expect(page.getByText('Input embedding lookup cancelled.')).toBeVisible();
  await editor.fill('C'); await count(page, 4); await tokenize(page, 3, tokens('C', [2])); await embeddingCount(page, 3);
  await stream(page, 2, [0]); await expect(page.getByText(/Could not load input embeddings/)).toBeVisible();
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  await editor.fill(''); await count(page, 5); await tokenize(page, 4, tokens(''));
  await expect(page.getByText('No tokens to embed.')).toBeVisible(); await embeddingCount(page, 3);
});

test('overlapping Unicode IDs and inserted specials link individual sequence rows', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('😀'); await count(page, 2);
  await tokenize(page, 1, { text: '😀', add_special_tokens: true, tokens: [
    { index: 0, id: 1, token: '<bos>', decoded: '', special: true },
    { index: 1, id: 2, token: 'byte-a', decoded: '�', special: false, start: 0, end: 1 },
    { index: 2, id: 3, token: 'byte-b', decoded: '�', special: false, start: 0, end: 1 },
  ] });
  await embeddingCount(page, 1); await stream(page, 0, [1, 2, 3]);
  await expect(page.getByText('[3 × 7] · float32')).toBeVisible();
  await expect(page.locator('.source-annotation')).toHaveCount(1);
  await expect(page.locator('.token-opening .token-ids')).toHaveText('2, 3');
  for (const row of [0, 1, 2]) {
    await page.locator(`[data-token-index="${row}"]`).focus();
    await expect(page.locator('[data-token-index][data-active-token]')).toHaveCount(1);
    await expect(page.locator(`[data-token-index="${row}"]`)).toHaveAttribute('data-active-token', '');
  }
  const matrix = page.locator('.matrix-scroll'); await matrix.focus();
  await matrix.press('ArrowDown'); await matrix.press('ArrowDown');
  await expect(page.locator('[data-token-index="2"]')).toHaveAttribute('data-active-token', '');
  expect(await page.evaluate(() => window.tokenizerHarness.source())).toBe('😀');
});

test('oversized embeddings use native internal scroll and release scalar resources on edit', async ({ page }) => {
  const editor = await start(page);
  // Reserve a deliberately small viewport: automatic allocation can now show
  // most of this fixture without vertical scrolling.
  await page.getByRole('separator', { name: 'Resize prompt and embeddings' }).press('End');
  const ids = Array.from({ length: 400 }, (_, i) => i % 4);
  const text = 'A'.repeat(400);
  await editor.fill(text); await count(page, 2); await tokenize(page, 1, tokens(text, ids)); await embeddingCount(page, 1);
  await page.evaluate(() => window.embeddingHarness.headers(0)); await send(page, 0, meta(ids, 2048));
  const matrix = page.locator('.matrix-scroll'); await matrix.scrollIntoViewIfNeeded(); await matrix.focus();
  const promptHeight = (await page.locator('.tokenizer-editor').boundingBox())!.height;
  await matrix.evaluate(node => { node.scrollLeft = 100; node.scrollTop = 100; });
  await expect.poll(() => page.evaluate(() => window.embeddingHarness.renderers.at(-1)!.view!.y)).toBe(100);
  expect(await page.evaluate(() => {
    const r = window.embeddingHarness.renderers.at(-1)!;
    return { rows: r.geometry.rows, columns: r.geometry.columns, x: r.view!.x };
  })).toEqual({ rows: 400, columns: 2048, x: 100 });
  expect((await page.locator('.tokenizer-editor').boundingBox())!.height).toBe(promptHeight);
  expect(await page.evaluate(() => [document.body.scrollHeight, document.documentElement.scrollHeight, scrollY])).toEqual([page.viewportSize()!.height, page.viewportSize()!.height, 0]);
  await editor.fill('new');
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  expect(await page.evaluate(() => window.embeddingHarness.renderers.every(r => r.diagnostics.cpuBytes === 0 && r.state === 'disposed'))).toBe(true);
});

test('token hover and activation take over keyboard matrix inspection without moving focus or editor selection', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('ABC'); await count(page, 2); await tokenize(page, 1, tokens('ABC'));
  await embeddingCount(page, 1); await stream(page, 0, [2, 0, 2]);
  await expect(page.getByText('[3 × 7] · float32')).toBeVisible();
  await editor.press('Home'); await editor.press('ArrowRight');
  const selection = await page.evaluate(() => window.tokenizerHarness.selection());
  await page.mouse.move(0, 0);
  await nativeCamera(page);
  const matrix = page.locator('.matrix-scroll');
  await matrix.focus();
  await expect(page.locator('.inspection-readout')).toContainText('row 0 · column 0');
  const before = await page.evaluate(() => {
    const r = window.embeddingHarness.renderers.at(-1)!;
    return { uploads: r.diagnostics.scalarUploadCalls, cpu: r.diagnostics.cpuBytes, requests: window.embeddingHarness.requests.length };
  });
  const token = page.locator('[data-token-index="2"]');
  await token.hover();
  await expect(token).toHaveAttribute('data-active-token', '');
  await expect(page.locator('[data-token-index="0"]')).not.toHaveAttribute('data-active-token');
  await expect(page.locator('.inspection-readout')).toHaveCount(0);
  await token.click();
  await expect(matrix).toBeFocused();
  await expect(token).toHaveAttribute('data-active-token', '');
  // The still-focused matrix can take over again, then another click on the same
  // stationary token must reclaim its row (no intervening pointer-enter event).
  await page.keyboard.press('ArrowDown');
  await expect(page.locator('.inspection-readout')).toContainText('row 1 · column 0');
  await expect(page.locator('[data-token-index="1"]')).toHaveAttribute('data-active-token', '');
  await token.click();
  await expect(token).toHaveAttribute('data-active-token', '');
  await expect(page.locator('.inspection-readout')).toHaveCount(0);
  await expect(matrix).toBeFocused();
  await nativeCamera(page);
  const pixels = await page.evaluate(() => {
    const r = window.embeddingHarness.renderers.at(-1)!; r.draw();
    const gl = r.canvas.getContext('webgl2')!;
    const bytes = new Uint8Array(7 * 3 * 4);
    gl.readPixels(0, 0, 7, 3, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
    return [...bytes];
  });
  for (let row = 0; row < 3; row++) for (let column = 0; column < 7; column++) {
    const value = values([2, 0, 2])[row * 7 + column]!;
    const expected = color(1 / (1 + Math.exp(-12 * value)), row === 2 ? 0.65 : 0);
    const offset = ((2 - row) * 7 + column) * 4;
    for (let channel = 0; channel < 3; channel++) expect(Math.abs(pixels[offset + channel]! - expected[channel]!)).toBeLessThanOrEqual(1);
  }
  expect(await page.evaluate(() => window.tokenizerHarness.selection())).toEqual(selection);
  expect(await page.evaluate(() => window.tokenizerHarness.source())).toBe('ABC');
  expect(await page.evaluate(() => {
    const r = window.embeddingHarness.renderers.at(-1)!;
    return { uploads: r.diagnostics.scalarUploadCalls, cpu: r.diagnostics.cpuBytes, requests: window.embeddingHarness.requests.length };
  })).toEqual(before);
});

for (const dpr of [1, 2]) test(`panel cameras and offscreen token reveal stay independent at DPR ${dpr}`, async ({ page }) => {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: page.viewportSize()!.width, height: page.viewportSize()!.height, deviceScaleFactor: dpr, mobile: false });
  const editor = await start(page);
  // Reserve enough room for the full fixed histogram tracks and the 65px
  // camera gesture, while keeping the last row offscreen at native DPR 1/2.
  const split = page.getByRole('separator', { name: 'Resize prompt and embeddings' });
  await split.press('End');
  for (let step = 0; step < 8; step++) await split.press('ArrowUp');
  const text = 'A'.repeat(400);
  const ids = Array.from({ length: 400 }, (_, i) => i % 4);
  await editor.fill(text); await count(page, 2); await tokenize(page, 1, tokens(text, ids));
  await embeddingCount(page, 1); await stream(page, 0, ids, 64);
  const prompt = page.getByRole('region', { name: 'Prompt / Tokens', exact: true });
  const embeddings = page.getByRole('region', { name: 'Input embeddings', exact: true });
  await expect(prompt.locator('.matrix-panel-header')).toHaveCount(1);
  await expect(embeddings.locator('.matrix-panel-header')).toHaveCount(1);
  await expect(embeddings.locator('.matrix-explorer')).toHaveCount(1);
  await expect(embeddings.getByText('[400 × 64] · float32')).toBeVisible();
  await expect(embeddings.getByText(/token rows/)).toHaveCount(0);
  await expect(embeddings.getByRole('button', { name: 'Fit width' })).toBeVisible();

  const matrix = embeddings.locator('.matrix-scroll');
  await matrix.scrollIntoViewIfNeeded();
  // Set the caret without asking the browser to reveal the editor's ancestors.
  await editor.evaluate(node => (node as HTMLElement).focus({ preventScroll: true }));
  await editor.press('Home'); await editor.press('ArrowRight');
  await matrix.scrollIntoViewIfNeeded();
  const promptState = () => page.evaluate(() => {
    const editor = document.querySelector<HTMLElement>('.tokenizer-editor')!;
    const prompt = document.querySelector<HTMLElement>('.prompt-panel')!;
    return { editor: editor.getBoundingClientRect().toJSON(), prompt: prompt.getBoundingClientRect().toJSON(),
      scroll: [editor.scrollLeft, editor.scrollTop], caret: window.tokenizerHarness.selection(),
      source: window.tokenizerHarness.source(), document: [scrollX, scrollY] };
  });
  await nativeCamera(page);
  const before = await promptState();
  const camera = () => page.evaluate(() => {
    const view = window.embeddingHarness.renderers.at(-1)!.view!;
    return { x: view.x, y: view.y, scale: view.scaleX, height: view.height / view.scaleY };
  });
  const canvas = matrix.locator('canvas');
  const box = (await canvas.boundingBox())!;
  const initial = await camera();
  await page.mouse.move(box.x + 20, box.y + 25); await page.mouse.wheel(0, -400);
  await expect.poll(async () => (await camera()).scale).toBeGreaterThan(initial.scale);
  expect(await promptState()).toEqual(before);
  await matrix.evaluate(node => { node.scrollTop += 120; node.scrollLeft += 20; });
  await expect.poll(async () => (await camera()).y).toBeGreaterThan(initial.y);
  expect(await promptState()).toEqual(before);
  const zoomed = await camera();
  await page.mouse.move(box.x + 12, box.y + 12); await page.mouse.down();
  await page.mouse.move(box.x + 65, box.y + 65, { steps: 5 });
  await expect(page.locator('.matrix-zoom-preview')).toHaveCount(3);
  await expect(page.locator('.matrix-zoom-preview[data-surface=matrix]')).toBeVisible();
  await expect(page.locator('.matrix-zoom-preview[data-surface=rows]')).toBeVisible();
  await expect(page.locator('.matrix-zoom-preview[data-surface=columns]')).toBeVisible();
  await page.mouse.up();
  await expect(page.locator('.matrix-zoom-preview')).toHaveCount(0);
  await expect.poll(async () => (await camera()).scale).toBeGreaterThan(zoomed.scale);
  expect(await promptState()).toEqual(before);
  await embeddings.getByRole('button', { name: 'Fit width' }).click();
  await expect.poll(async () => (await camera()).y).toBe(0);
  expect(await promptState()).toEqual(before);

  // Synthetic events avoid Playwright's own automatic ancestor scrolling and
  // exercise the same annotation handlers for a horizontally offscreen token.
  const token = page.locator('[data-token-index="399"]');
  const fitted = await camera();
  await token.dispatchEvent('pointerover');
  expect(await camera()).toEqual(fitted); // hover is transient, never reveal
  await token.dispatchEvent('click');
  await expect.poll(async () => (await camera()).y).toBeGreaterThan(0);
  let revealed = await camera();
  expect(revealed.scale).toBe(fitted.scale); expect(revealed.x).toBe(fitted.x);
  expect(revealed.y).toBeLessThanOrEqual(399); expect(revealed.y + revealed.height).toBeGreaterThanOrEqual(399.99);
  expect(await promptState()).toEqual(before);
  await token.dispatchEvent('pointerout');
  await expect(page.locator('[data-token-index][data-active-token]')).toHaveCount(0);
  // A fresh activation of the same row is a new reveal intent.
  await matrix.evaluate(node => { node.scrollTop = 0; });
  await expect.poll(async () => (await camera()).y).toBe(0);
  await token.dispatchEvent('keydown', { key: 'Enter' });
  await expect.poll(async () => (await camera()).y).toBeGreaterThan(0);
  revealed = await camera(); expect(revealed.scale).toBe(fitted.scale);
  expect(await promptState()).toEqual(before);
});

test('session/source replacement clears old linkage before and after the new matrix mounts', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('ABC'); await count(page, 2); await tokenize(page, 1, tokens('ABC'));
  await embeddingCount(page, 1); await stream(page, 0, [2, 0, 2]);
  await page.locator('[data-token-index="2"]').dispatchEvent('click');
  await expect(page.locator('[data-token-index="2"]')).toHaveAttribute('data-active-token', '');
  await page.getByRole('button', { name: 'Change session' }).click(); await count(page, 3);
  await tokenize(page, 2, tokens('ABC'));
  await expect(page.locator('[data-token-index][data-active-token]')).toHaveCount(0);
  await embeddingCount(page, 2); await stream(page, 1, [2, 0, 2]);
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await expect(page.locator('[data-token-index][data-active-token]')).toHaveCount(0);
  expect(await page.evaluate(() => window.embeddingHarness.renderers[0]!.state)).toBe('disposed');
  await page.locator('.matrix-scroll').focus();
  await expect(page.locator('[data-token-index="0"]')).toHaveAttribute('data-active-token', '');
  await editor.fill('DEF'); await count(page, 4); await tokenize(page, 3, tokens('DEF'));
  await embeddingCount(page, 3); await stream(page, 2, [2, 0, 2]);
  await expect(page.locator('[data-token-index][data-active-token]')).toHaveCount(0);
});

test('completed matrices stay mounted through staged replacements, failures and generation mismatches', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('ABC'); await count(page, 2); await tokenize(page, 1, tokens('ABC'));
  await embeddingCount(page, 1); await stream(page, 0, [2, 0, 2]);
  const visibleMatrix = page.locator('.embedding-layer:not([data-staging]) .matrix-scroll');
  await expect(visibleMatrix).toBeVisible();
  await visibleMatrix.evaluate(node => { node.setAttribute('data-original-matrix', ''); });
  const originalRenderer = await page.evaluate(() => window.embeddingHarness.renderers.findIndex(r => r.state !== 'disposed'));
  await page.evaluate(() => {
    const frames: number[] = [];
    Object.assign(window, { matrixFrames: frames, matrixSampling: true });
    const sample = () => {
      frames.push(document.querySelectorAll('.embedding-layer:not([data-staging]) .matrix-scroll').length);
      if (Reflect.get(window, 'matrixSampling')) requestAnimationFrame(sample);
    };
    requestAnimationFrame(sample);
  });
  const geometry = () => page.locator('.input-embeddings').evaluate(node => ({
    height: node.getBoundingClientRect().height,
    header: node.querySelector('.matrix-panel-header')!.getBoundingClientRect().height,
    matrix: node.querySelector('.matrix-scroll')!.getBoundingClientRect().height,
  }));
  const before = await geometry();
  await editor.press('End'); await editor.pressSequentially('D'); await count(page, 3);
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'stale');
  await expect(visibleMatrix).toHaveAttribute('data-original-matrix', '');
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
  await page.locator('[data-token-index="2"]').dispatchEvent('click');
  await expect(page.locator('[data-active-token]')).toHaveCount(0);
  await tokenize(page, 2, tokens('ABCD', [3, 1, 0, 2])); await embeddingCount(page, 2);
  await page.locator('[data-token-index="3"]').dispatchEvent('click');
  await expect(page.locator('[data-active-token]')).toHaveCount(0);
  await visibleMatrix.focus(); await visibleMatrix.press('ArrowDown');
  await expect(page.locator('[data-active-token]')).toHaveCount(0);
  await page.evaluate(() => window.embeddingHarness.headers(1));
  await send(page, 1, [...meta([3, 1, 0, 2]), ...data([3, 1, 0, 2])]);
  await expect(page.locator('[data-staging] .matrix-scroll')).toHaveCount(1);
  await expect(page.locator('[data-staging]')).toHaveAttribute('inert', '');
  await expect(visibleMatrix).toHaveAttribute('data-original-matrix', '');
  expect(await geometry()).toEqual(before);
  expect(await page.evaluate(index => window.embeddingHarness.renderers[index]!.state, originalRenderer)).not.toBe('disposed');

  // Supersede a fully populated but incomplete replacement; late COMPLETE and
  // a late tokenization response cannot promote it or restore its row linkage.
  await editor.press('End'); await editor.pressSequentially('E'); await count(page, 4);
  await expect(page.locator('[data-staging]')).toHaveCount(0);
  await editor.pressSequentially('F'); await count(page, 5);
  await tokenize(page, 4, tokens('ABCDEF', [1])); await embeddingCount(page, 3);
  await tokenize(page, 3, tokens('ABCDE', [9]));
  await send(page, 1, frame(4), true);
  await page.evaluate(() => window.embeddingHarness.headers(2, 422));
  await expect(page.getByText(/Input embeddings are unavailable/)).toBeVisible();
  await expect(page.locator('.token-ids')).toHaveText('1');
  await expect(visibleMatrix).toHaveAttribute('data-original-matrix', '');
  expect(await geometry()).toEqual(before);

  // Tokenizer failure retains both contexts and its error without shifting the panel.
  await editor.pressSequentially('G'); await count(page, 6);
  await page.evaluate(() => window.tokenizerHarness.fail(5));
  await expect(page.getByRole('alert')).toContainText('Previous annotations are stale');
  await expect(visibleMatrix).toHaveAttribute('data-original-matrix', '');
  expect(await geometry()).toEqual(before);
  await page.getByRole('button', { name: 'Retry tokenization' }).click(); await count(page, 7);
  await tokenize(page, 6, tokens('ABCDEFG', [3, 2])); await embeddingCount(page, 4);
  await page.evaluate(() => window.embeddingHarness.headers(3));
  await send(page, 3, [...meta([3, 2]), ...data([3, 2])]);
  await expect(visibleMatrix).toHaveAttribute('data-original-matrix', '');
  await send(page, 3, frame(5, new TextEncoder().encode(JSON.stringify({ code: 'internal_error', message: 'Fixture failure' }))), true);
  await expect(page.getByText(/Could not load input embeddings/)).toBeVisible();
  await expect(page.locator('[data-staging]')).toHaveCount(0);
  await expect(visibleMatrix).toHaveAttribute('data-original-matrix', '');
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'current');

  // A successful replacement swaps the already populated renderer atomically.
  await editor.press('End'); await editor.pressSequentially('H'); await count(page, 8);
  await tokenize(page, 7, tokens('ABCDEFGH', [0, 1])); await embeddingCount(page, 5);
  await page.evaluate(() => window.embeddingHarness.headers(4));
  await send(page, 4, [...meta([0, 1]), ...data([0, 1])]);
  await expect(page.locator('[data-staging] .matrix-scroll')).toHaveCount(1);
  await send(page, 4, frame(4), true);
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
  await expect(page.locator('[data-original-matrix]')).toHaveCount(0);
  await expect(visibleMatrix).toBeVisible();
  await page.locator('[data-token-index="1"]').dispatchEvent('click');
  await expect(page.locator('[data-token-index="1"]')).toHaveAttribute('data-active-token', '');
  const allocated = await page.evaluate(() => window.embeddingHarness.renderers.map(r => ({ state: r.state, bytes: r.diagnostics.cpuBytes })));
  expect(allocated.filter(r => r.state !== 'disposed')).toHaveLength(1);
  expect(allocated.filter(r => r.state === 'disposed').every(r => r.bytes === 0)).toBe(true);
  expect(allocated[originalRenderer]!.state).toBe('disposed');
  const frames = await page.evaluate(() => {
    Reflect.set(window, 'matrixSampling', false);
    return Reflect.get(window, 'matrixFrames') as number[];
  });
  expect(frames.length).toBeGreaterThan(5);
  expect(frames.every(count => count === 1)).toBe(true);
  // Empty input remains explicit, with only the previous matrix as stale context.
  await editor.fill(''); await count(page, 9); await tokenize(page, 8, tokens(''));
  await expect(page.getByText(/No tokens to embed/)).toBeVisible();
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
  await expect(visibleMatrix).toBeVisible();
});


test('disposing a workspace releases both completed and staged matrix resources', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('ABC'); await count(page, 2); await tokenize(page, 1, tokens('ABC'));
  await embeddingCount(page, 1); await stream(page, 0, [2, 0, 2]);
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await editor.press('End'); await editor.pressSequentially('D'); await count(page, 3);
  await tokenize(page, 2, tokens('ABCD', [2, 0])); await embeddingCount(page, 2);
  await page.evaluate(() => window.embeddingHarness.headers(1));
  await send(page, 1, [...meta([2, 0]), ...data([2, 0])]);
  await expect(page.locator('.matrix-scroll')).toHaveCount(2);
  await page.getByRole('button', { name: 'Close workspace' }).click();
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  expect(await page.evaluate(() => window.embeddingHarness.renderers.every(r => r.state === 'disposed' && r.diagnostics.cpuBytes === 0))).toBe(true);
  await send(page, 1, frame(4), true);
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
});

test('a lost staging WebGL context cannot replace the previous completed matrix', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('ABC'); await count(page, 2); await tokenize(page, 1, tokens('ABC'));
  await embeddingCount(page, 1); await stream(page, 0, [2, 0, 2]);
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await page.locator('.matrix-scroll').evaluate(node => node.setAttribute('data-original-matrix', ''));
  await editor.press('End'); await editor.pressSequentially('D'); await count(page, 3);
  await tokenize(page, 2, tokens('ABCD', [2, 0])); await embeddingCount(page, 2);
  await page.evaluate(() => window.embeddingHarness.headers(1));
  await send(page, 1, [...meta([2, 0]), ...data([2, 0])]);
  await expect(page.locator('[data-staging] .matrix-scroll')).toHaveCount(1);
  await page.locator('[data-staging] .matrix-scroll canvas').evaluate(canvas => {
    const gl = (canvas as HTMLCanvasElement).getContext('webgl2')!;
    gl.getExtension('WEBGL_lose_context')!.loseContext();
  });
  await expect(page.getByText(/Could not load input embeddings/)).toBeVisible();
  await expect(page.locator('[data-staging]')).toHaveCount(0);
  await send(page, 1, frame(4), true);
  await expect(page.locator('[data-original-matrix]')).toBeVisible();
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'current');
});

async function auxiliary(page: Page, index: number, bytes: number[], close = true) {
  await page.evaluate(({ index, bytes, close }) => {
    window.embeddingHarness.auxiliaryHeaders(index);
    window.embeddingHarness.auxiliarySend(index, bytes, close);
  }, { index, bytes, close });
}

for (const early of [true, false]) test(`real protocol auxiliary counts ${early ? 'before' : 'after'} values preserve geometry, information and subscriptions`, async ({ page }) => {
  const editor = await start(page);
  await editor.fill('ABC'); await count(page, 2); await tokenize(page, 1, tokens('ABC'));
  await embeddingCount(page, 1);
  const fixture = analysis([2, 0, 2]);
  const deliver = async () => {
    await auxiliary(page, 0, fixture.statisticsStream);
    await auxiliary(page, 1, fixture.distributionStream);
  };
  await page.evaluate(() => window.embeddingHarness.headers(0));
  await send(page, 0, meta([2, 0, 2]));
  if (early) await deliver();
  await send(page, 0, [...data([2, 0, 2]), ...frame(4)], true);
  const matrix = page.locator('.matrix-scroll');
  await expect(matrix).toBeVisible();
  const geometry = () => page.locator('.tokenizer-workspace').evaluate(workspace =>
    [...workspace.querySelectorAll('.prompt-panel, .input-embeddings, .matrix-scroll, .row-distributions, .column-distributions')]
      .map(node => node.getBoundingClientRect().toJSON()));
  const before = await geometry();
  if (!early) await deliver();
  await expect(page.locator('.input-embeddings .matrix-panel-status')).toBeEmpty();
  expect(await geometry()).toEqual(before);
  await nativeCamera(page);
  const actual = await page.evaluate(() => window.embeddingHarness.countRenderers.filter(r => r.state !== 'disposed').map(r => ({
    shape: [r.geometry.rows, r.geometry.columns],
    counts: Array.from({ length: r.geometry.count }, (_, index) => r.readCell(Math.floor(index / r.geometry.columns), index % r.geometry.columns)?.value),
  })));
  expect(actual).toEqual([
    { shape: [3, 100], counts: [...fixture.counts.slice(0, 300)] },
    { shape: [100, 7], counts: [...fixture.counts.slice(300)] },
  ]);
  const resources = await page.evaluate(() => ({
    renderers: window.embeddingHarness.renderers.length, counts: window.embeddingHarness.countRenderers.length,
    requests: window.embeddingHarness.requests.length, auxiliary: window.embeddingHarness.auxiliary.length,
    uploads: window.embeddingHarness.renderers.at(-1)!.diagnostics.scalarUploadCalls,
  }));
  const info = page.getByRole('button', { name: 'Input embeddings information', exact: true });
  await info.click();
  const dialog = page.getByRole('dialog', { name: 'Input embeddings information' });
  await expect(dialog).toContainText('21');
  await expect(dialog).toContainText('Full-range bins');
  await expect(dialog).toContainText('Luminosity anchors');
  await expect(dialog).not.toContainText('Storage dtype');
  await expect(page.getByRole('button', { name: 'Close input embeddings information' })).toBeFocused();
  await page.keyboard.press('Escape'); await expect(info).toBeFocused();
  await page.getByRole('separator').press('ArrowDown');
  await page.getByRole('button', { name: 'Fit width' }).click();
  expect(await page.evaluate(() => ({
    renderers: window.embeddingHarness.renderers.length, counts: window.embeddingHarness.countRenderers.length,
    requests: window.embeddingHarness.requests.length, auxiliary: window.embeddingHarness.auxiliary.length,
    uploads: window.embeddingHarness.renderers.at(-1)!.diagnostics.scalarUploadCalls,
  }))).toEqual(resources);
  expect(await page.evaluate(() => window.tokenizerHarness.source())).toBe('ABC');
  await page.getByRole('button', { name: 'Close workspace' }).click();
  expect(await page.evaluate(() => [...window.embeddingHarness.renderers, ...window.embeddingHarness.countRenderers]
    .every(r => r.state === 'disposed' && r.diagnostics.cpuBytes === 0))).toBe(true);
});

test('failed statistics and cancelled distributions leave successful values and current tokens usable', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('ABC'); await count(page, 2); await tokenize(page, 1, tokens('ABC'));
  await embeddingCount(page, 1); await stream(page, 0, [2, 0, 2]);
  await page.evaluate(() => window.embeddingHarness.auxiliaryHeaders(0, 500));
  await expect(page.locator('[data-result="statistics"]')).toHaveAttribute('data-state', 'failed');
  await page.getByRole('button', { name: 'Cancel embedding distributions' }).click();
  await expect(page.locator('[data-result="distributions"]')).toHaveAttribute('data-state', 'cancelled');
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
  await page.locator('.matrix-scroll').focus();
  await expect(page.locator('.inspection-readout')).toContainText('0.4375');
  await expect(page.locator('[data-token-index="0"]')).toHaveAttribute('data-active-token', '');
  await expect(page.getByText(/unavailable for this model/)).toHaveCount(0);
  expect(await page.evaluate(() => window.embeddingHarness.auxiliary.map(r => r.aborted))).toEqual([false, true]);
});

test('old histograms stay with stale values while equal-shaped replacement promotes with its own pending analysis', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('ABC'); await count(page, 2); await tokenize(page, 1, tokens('ABC'));
  await embeddingCount(page, 1); await stream(page, 0, [2, 0, 2]);
  await auxiliary(page, 0, analysis([2, 0, 2]).statisticsStream);
  await auxiliary(page, 1, analysis([2, 0, 2]).distributionStream);
  await expect(page.locator('.input-embeddings .matrix-panel-status')).toBeEmpty();
  const oldDomain = await page.locator('.distribution-scale-rows').getAttribute('aria-label');
  await editor.fill('DEF'); await count(page, 3); await tokenize(page, 2, tokens('DEF', [0, 0, 0]));
  await embeddingCount(page, 2);
  await page.evaluate(() => window.embeddingHarness.headers(1));
  await send(page, 1, [...meta([0, 0, 0]), ...data([0, 0, 0])]);
  await expect(page.locator('[data-staging]')).toHaveCount(1);
  expect(await page.locator('.embedding-layer:not([data-staging]) .distribution-scale-rows').getAttribute('aria-label')).toBe(oldDomain);
  await send(page, 1, frame(4), true);
  await expect(page.locator('[data-staging]')).toHaveCount(0);
  await expect(page.locator('.distribution-scale-rows')).not.toHaveAttribute('data-minimum');
  expect(await page.evaluate(() => window.embeddingHarness.countRenderers.filter(r => r.state !== 'disposed').map(r => r.populatedPrefix))).toEqual([0, 0]);
  await auxiliary(page, 2, analysis([0, 0, 0]).statisticsStream);
  await auxiliary(page, 3, analysis([0, 0, 0]).distributionStream);
  await expect(page.locator('.input-embeddings .matrix-panel-status')).toBeEmpty();
  expect(await page.locator('.distribution-scale-rows').getAttribute('aria-label')).not.toBe(oldDomain);
  expect(await page.evaluate(() => window.embeddingHarness.countRenderers.filter(r => r.state !== 'disposed').length)).toBe(2);
  expect(await page.evaluate(() => window.embeddingHarness.countRenderers.filter(r => r.state === 'disposed').every(r => r.diagnostics.cpuBytes === 0))).toBe(true);
});

const selectionText = 'Hi 中😀\nHi <s>';
function selectionTokens(): Tokenization {
  const spans = [null, [0, 2], [2, 3], [3, 4], [4, 5], [4, 5], null, [5, 6], [6, 8], [8, 9], [9, 12], null];
  return { text: selectionText, add_special_tokens: true, tokens: spans.map((span, index) => ({
    index, id: index === 8 ? 1 : index, token: span ? 'source' : '<inserted>', decoded: '',
    special: [0, 10, 11].includes(index), ...(span ? { start: span[0]!, end: span[1]! } : {}),
  })) };
}
const rowGuides = (page: Page) => page.locator('.embedding-layer:not([data-staging]) .matrix-scroll .matrix-selected-rows');
async function guideRows(page: Page) {
  return rowGuides(page).evaluateAll(nodes => nodes.map(node => node.getAttribute('data-rows')));
}
async function selectionSnapshot(page: Page) {
  return page.evaluate(() => {
    const matrix = document.querySelector<HTMLElement>('.embedding-layer:not([data-staging]) .matrix-scroll')!;
    const prompt = document.querySelector('.cm-scroller')!;
    const renderers = [...window.embeddingHarness.renderers, ...window.embeddingHarness.countRenderers];
    return {
      views: renderers.map(r => r.view), resources: renderers.map(r => r.diagnostics),
      samples: renderers.map(r => r.state === 'ready' ? r.readCell(0, 0) : null),
      requests: [window.tokenizerHarness.requests.length, window.embeddingHarness.requests.length, window.embeddingHarness.auxiliary.length],
      history: window.embeddingHarness.historyCalls,
      transfers: window.embeddingHarness.transferCalls,
      scroll: [matrix.scrollLeft, matrix.scrollTop, prompt.scrollLeft, prompt.scrollTop, window.scrollX, window.scrollY],
      split: document.querySelector('[role="separator"]')!.getAttribute('aria-valuenow'),
      domain: [...document.querySelectorAll('.distribution-scale')].map(node => node.textContent),
    };
  });
}
async function alignedGuides(page: Page, ranges: [number, number][]) {
  const geometry = await page.evaluate(() => {
    const r = window.embeddingHarness.renderers.findLast(r => r.state === 'ready' && !r.canvas.closest('[data-staging]'))!;
    const rect = r.canvas.getBoundingClientRect();
    const read = (selector: string) => [...document.querySelectorAll<HTMLElement>(selector)].map(node => {
      const box = node.getBoundingClientRect();
      return { rows: node.dataset.rows, x: box.x, y: box.y, width: box.width, height: box.height,
        events: getComputedStyle(node).pointerEvents };
    });
    const rowCanvas = document.querySelector('.embedding-layer:not([data-staging]) .row-distributions canvas')!.getBoundingClientRect();
    return { view: r.view!, x: rect.x, y: rect.y, width: rect.width, rowX: rowCanvas.x, rowWidth: rowCanvas.width,
      main: read('.embedding-layer:not([data-staging]) .matrix-scroll .matrix-selected-rows'),
      rows: read('.embedding-layer:not([data-staging]) .row-distributions .matrix-selected-rows'),
      columns: read('.column-distributions .matrix-selected-rows') };
  });
  const v = geometry.view;
  const expected = ranges.map(([from, to]) => ({ from, to,
    top: Math.max(0, Math.ceil((from - v.y) * v.scaleY)),
    bottom: Math.min(v.height, Math.ceil((to - v.y) * v.scaleY)),
  })).filter(({ top, bottom }) => bottom > top);
  expect(geometry.main).toHaveLength(expected.length);
  expect(geometry.rows).toHaveLength(expected.length);
  expect(geometry.columns).toEqual([]);
  expected.forEach(({ from, to, top, bottom }, index) => {
    for (const [guides, x, width] of [[geometry.main, geometry.x, geometry.width], [geometry.rows, geometry.rowX, geometry.rowWidth]] as const) {
      const guide = guides[index]!;
      expect(guide.rows).toBe(`${from}:${to}`);
      expect(guide.events).toBe('none');
      expect(Math.abs(guide.x - x)).toBeLessThan(0.1);
      expect(Math.abs(guide.width - width)).toBeLessThan(0.1);
      expect(Math.abs(guide.y - geometry.y - top / v.dpr)).toBeLessThan(1 / v.dpr + 0.05);
      expect(Math.abs(guide.height - (bottom - top) / v.dpr)).toBeLessThan(1 / v.dpr + 0.05);
    }
  });
}

for (const dpr of [1, 2]) test(`persistent source selection maps exact rows without camera or scientific side effects at DPR ${dpr}`, async ({ page }, testInfo) => {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setDeviceMetricsOverride', { ...page.viewportSize()!, deviceScaleFactor: dpr, mobile: false });
  const editor = await start(page);
  const result = selectionTokens(), ids = result.tokens.map(t => t.id);
  await editor.fill(selectionText); await count(page, 2); await tokenize(page, 1, result);
  await embeddingCount(page, 1); await stream(page, 0, ids, 64);
  await expect(page.getByText('[12 × 64] · float32')).toBeVisible();
  await auxiliary(page, 0, analysis(ids, 64).statisticsStream);
  await auxiliary(page, 1, analysis(ids, 64).distributionStream);
  await editor.press('Control+Home'); await page.mouse.move(0, 0);
  const before = await selectionSnapshot(page);
  const drag = async (from: number, to: number) => {
    const a = await page.evaluate(position => window.tokenizerHarness.coords(position), from);
    const b = await page.evaluate(position => window.tokenizerHarness.coords(position), to);
    await page.mouse.move(a.x + 0.1, a.y); await page.mouse.down();
    await page.mouse.move(b.x + 0.1, b.y, { steps: 12 }); await page.mouse.up();
    await expect.poll(() => page.evaluate(() => window.tokenizerHarness.selection())).toEqual([Math.min(from, to), Math.max(from, to)]);
  };
  await drag(1, 6); // Partial first token, Chinese, and both byte tokens for the emoji.
  await expect.poll(() => guideRows(page)).toEqual(['1:6']);
  await alignedGuides(page, [[1, 6]]);
  await page.mouse.move(0, 0);
  expect(await selectionSnapshot(page)).toEqual(before);
  await page.locator('[data-token-index="8"]').hover();
  await expect.poll(() => guideRows(page)).toEqual(['1:6']);
  await page.locator('.matrix-scroll').focus();
  await expect(page.locator('.inspection-readout')).toBeVisible();
  await expect.poll(() => guideRows(page)).toEqual(['1:6']);
  // An annotation focus/activation keeps native source selection and the group.
  await page.locator('[data-token-index="3"]').focus();
  await expect.poll(() => guideRows(page)).toEqual(['1:6']);
  await drag(6, 1);
  await expect.poll(() => guideRows(page)).toEqual(['1:6']);
  await editor.press('Control+Home'); // Intentional caret clears.
  await expect(rowGuides(page)).toHaveCount(0);
  await editor.press('Shift+ArrowRight'); // Partial token.
  await expect.poll(() => guideRows(page)).toEqual(['1:2']);
  await editor.press('Control+Home');
  for (let i = 0; i < 5; i++) await editor.press('Shift+ArrowRight');
  await expect.poll(() => guideRows(page)).toEqual(['1:6']);
  await editor.press('ArrowRight');
  for (let i = 0; i < 5; i++) await editor.press('Shift+ArrowLeft');
  await expect.poll(() => guideRows(page)).toEqual(['1:6']);
  await editor.press('Control+a');
  await expect.poll(() => guideRows(page)).toEqual(['1:6', '7:11']);
  await alignedGuides(page, [[1, 6], [7, 11]]);
  expect(await page.evaluate(() => window.tokenizerHarness.source())).toBe(selectionText);
  expect(await page.evaluate(() => window.tokenizerHarness.selection())).toEqual([0, selectionText.length]);
  await page.mouse.move(0, 0);
  expect(await selectionSnapshot(page)).toEqual(before);
  if (dpr === 1 && testInfo.project.name === 'desktop') {
    const path = testInfo.outputPath('source-selection.png');
    await page.locator('.tokenizer-workspace').screenshot({ path });
    await testInfo.attach('Source selection and native embedding rows', { path, contentType: 'image/png' });
  }
  const caret = await page.evaluate(() => window.tokenizerHarness.coords(1));
  await page.mouse.click(caret.x + 0.1, caret.y);
  await expect(rowGuides(page)).toHaveCount(0);
  await editor.press('Control+a');
  // Camera changes are intentional here; guides must track underfill and scroll.
  await nativeCamera(page);
  await alignedGuides(page, [[1, 6], [7, 11]]);
  await page.locator('.matrix-scroll').dispatchEvent('wheel', { deltaY: -1200, clientX: 150, clientY: 500 });
  await alignedGuides(page, [[1, 6], [7, 11]]);
  await page.locator('.matrix-scroll').evaluate(node => { node.scrollLeft = 30; node.scrollTop = 15; });
  await page.waitForTimeout(50);
  await alignedGuides(page, [[1, 6], [7, 11]]);
  await page.setViewportSize({ width: page.viewportSize()!.width + 80, height: 780 });
  await page.waitForTimeout(100);
  await alignedGuides(page, [[1, 6], [7, 11]]);
  await cdp.send('Emulation.setDeviceMetricsOverride', { ...page.viewportSize()!, deviceScaleFactor: dpr === 1 ? 2 : 1, mobile: false });
  await expect.poll(() => page.evaluate(() => window.embeddingHarness.renderers.at(-1)!.view!.dpr)).toBe(dpr === 1 ? 2 : 1);
  await alignedGuides(page, [[1, 6], [7, 11]]);
  await page.getByRole('button', { name: 'Close workspace' }).click();
  await expect(page.locator('.matrix-selected-rows')).toHaveCount(0);
});

test('offscreen source selection leaves camera/history and prompt scroll intact until manual navigation', async ({ page }) => {
  const editor = await start(page);
  const text = 'x'.repeat(180), ids = Array.from({ length: 180 }, (_, i) => i % 3);
  await editor.fill(text); await count(page, 2); await tokenize(page, 1, tokens(text, ids));
  await embeddingCount(page, 1); await stream(page, 0, ids, 16);
  await expect(page.getByText('[180 × 16] · float32')).toBeVisible();
  await page.mouse.move(0, 0);
  const before = await selectionSnapshot(page);
  await page.evaluate(() => window.tokenizerHarness.select(170, 179));
  await expect(rowGuides(page)).toHaveCount(0);
  expect(await selectionSnapshot(page)).toEqual(before);
  await page.locator('.matrix-scroll').evaluate(node => { node.scrollTop = node.scrollHeight; });
  await expect.poll(() => guideRows(page)).toEqual(['170:179']);
  await alignedGuides(page, [[170, 179]]);
  const p = await page.evaluate(() => { const n = document.querySelector('.cm-scroller')!; return [n.scrollTop, n.scrollLeft]; });
  expect(p).toEqual(before.scroll.slice(2, 4));
});

test('selected rows are recomputed from current source only after matching generations promote', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('ABC'); await count(page, 2); await tokenize(page, 1, tokens('ABC'));
  await embeddingCount(page, 1); await stream(page, 0, [2, 0, 2], 64);
  await editor.press('Control+a'); await expect.poll(() => guideRows(page)).toEqual(['0:3']);
  // Options retain native selection, but the old visible matrix is immediately fenced.
  await page.getByLabel('Add special tokens').uncheck(); await count(page, 3);
  await expect(page.locator('.matrix-selected-rows')).toHaveCount(0);
  const replacement = tokens('ABC', [2, 0, 2], false);
  delete replacement.tokens[1]!.start; delete replacement.tokens[1]!.end;
  await tokenize(page, 2, replacement); await embeddingCount(page, 2);
  await page.evaluate(() => window.embeddingHarness.headers(1));
  await send(page, 1, [...meta([2, 0, 2], 64), ...data([2, 0, 2], 64)]);
  await expect(page.locator('[data-staging] .matrix-scroll')).toHaveCount(1);
  await expect(page.locator('.matrix-selected-rows')).toHaveCount(0);
  await send(page, 1, frame(4), true);
  await expect.poll(() => guideRows(page)).toEqual(['0:1', '2:3']); // Same shape, different mapping.
  await page.getByRole('button', { name: 'Change session' }).click(); await count(page, 4);
  await expect(page.locator('.matrix-selected-rows')).toHaveCount(0);
  await tokenize(page, 3, tokens('ABC', [2, 0, 2], false)); await embeddingCount(page, 3);
  await page.evaluate(() => window.embeddingHarness.headers(2));
  await send(page, 2, [...meta([2, 0, 2], 64), ...data([2, 0, 2], 64), ...frame(6)], true);
  await expect(page.getByText(/Input embedding lookup cancelled/)).toBeVisible();
  await expect(page.locator('.matrix-selected-rows')).toHaveCount(0);
  await editor.fill('DEF'); await count(page, 5);
  await page.evaluate(() => window.tokenizerHarness.fail(4));
  await expect(page.getByRole('alert')).toBeVisible();
  await expect(page.locator('.matrix-selected-rows')).toHaveCount(0);
  await editor.fill('GHI'); await count(page, 6);
  await editor.fill('JKL'); await count(page, 7);
  await editor.press('Control+a');
  await tokenize(page, 6, tokens('JKL', [2, 0, 2], false)); await embeddingCount(page, 4);
  await tokenize(page, 5, tokens('GHI', [2, 0, 2], false));
  await expect(page.locator('.matrix-selected-rows')).toHaveCount(0);
  await stream(page, 3, [2, 0, 2], 64);
  await expect.poll(() => guideRows(page)).toEqual(['0:3']);
  await page.keyboard.type('M');
  await expect(rowGuides(page)).toHaveCount(0);
  await editor.press('ArrowRight');
  await expect(rowGuides(page)).toHaveCount(0);
});
