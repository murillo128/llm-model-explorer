import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type { Tokenization } from '../src/tokenizer/annotations';

import { color } from './scalar-oracle';
import { frame, meta, data, values } from './embedding-fixtures';

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
  await expect(page.getByText('3 token rows · 7 hidden dimensions')).toBeVisible();
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
  await expect(page.locator('.row-distributions, .column-distributions')).toHaveCount(0);
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
  await expect(page.getByText('1 token rows · 7 hidden dimensions')).toBeVisible();
  await stream(page, 1, [0]); await send(page, 0, frame(4), true);
  expect(await page.evaluate(() => window.embeddingHarness.requests.slice(0, 2).map(r => r.aborted))).toEqual([true, true]);
  expect(await page.evaluate(() => window.embeddingHarness.cancelled)).toContain('/operations/00000000-0000-4000-8000-000000000000');
  await page.getByRole('checkbox', { name: 'Add special tokens' }).uncheck(); await count(page, 5);
  await page.getByRole('checkbox', { name: 'Add special tokens' }).check(); await count(page, 6);
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  await tokenize(page, 4, tokens('A', [0], false)); await embeddingCount(page, 3);
  await tokenize(page, 5, tokens('A', [2])); await embeddingCount(page, 4);
  await page.getByRole('button', { name: 'Change session' }).click(); await count(page, 7);
  await stream(page, 3, [2]); await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  await tokenize(page, 6, tokens('A', [0])); await embeddingCount(page, 5); await stream(page, 4, [0]);
  expect(await page.evaluate(() => window.embeddingHarness.requests[4]!.session)).toMatch(/^bbbb/);
  await expect(page.getByText('1 token rows · 7 hidden dimensions')).toBeVisible();
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
  await expect(page.getByText('3 token rows · 7 hidden dimensions')).toBeVisible();
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
  const ids = Array.from({ length: 400 }, (_, i) => i % 4);
  const text = 'A'.repeat(400);
  await editor.fill(text); await count(page, 2); await tokenize(page, 1, tokens(text, ids)); await embeddingCount(page, 1);
  await page.evaluate(() => window.embeddingHarness.headers(0)); await send(page, 0, meta(ids, 2048));
  const matrix = page.locator('.matrix-scroll'); await matrix.scrollIntoViewIfNeeded(); await matrix.focus();
  await matrix.evaluate(node => { node.scrollLeft = 100; node.scrollTop = 100; });
  await expect.poll(() => page.evaluate(() => window.embeddingHarness.renderers.at(-1)!.view!.y)).toBe(100);
  expect(await page.evaluate(() => {
    const r = window.embeddingHarness.renderers.at(-1)!;
    return { rows: r.geometry.rows, columns: r.geometry.columns, x: r.view!.x };
  })).toEqual({ rows: 400, columns: 2048, x: 100 });
  expect((await page.locator('.tokenizer-editor').boundingBox())!.height).toBe(260);
  expect(await page.evaluate(() => [document.body.scrollHeight, document.documentElement.scrollHeight, scrollY])).toEqual([page.viewportSize()!.height, page.viewportSize()!.height, 0]);
  await editor.fill('new');
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  expect(await page.evaluate(() => window.embeddingHarness.renderers.every(r => r.diagnostics.cpuBytes === 0 && r.state === 'disposed'))).toBe(true);
});

test('token hover and activation take over keyboard matrix inspection without moving focus or editor selection', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('ABC'); await count(page, 2); await tokenize(page, 1, tokens('ABC'));
  await embeddingCount(page, 1); await stream(page, 0, [2, 0, 2]);
  await expect(page.getByText('3 token rows · 7 hidden dimensions')).toBeVisible();
  await editor.press('Home'); await editor.press('ArrowRight');
  const selection = await page.evaluate(() => window.tokenizerHarness.selection());
  await page.mouse.move(0, 0);
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
