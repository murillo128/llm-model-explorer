import { expect } from '@playwright/test';
import { test } from './product-harness';
import { zoom, drag, tokenizerGeometry, settledPrompt } from './usability';

// Logical transport cases use headless Chromium; native focus/geometry stays headed.
test.use({ headless: false });

test('production prompt pixels, selection, history and composition survive embedding completion', async ({ page, product }) => {
  const { control, tokenizer, closeSession, embeddingDone } = product;
  const editor = await tokenizer(page);
  await control('arm', { kind: 'input_embeddings' });
  await editor.fill('hello world');
  await expect(page.getByRole('status').filter({ hasText: 'Streaming input embeddings…' })).toBeVisible();
  const prompt = page.getByRole('region', { name: 'Live prompt tokenization' });
  const nav = page.getByRole('button', { name: 'Tokenizer Explorer', exact: true });
  await nav.focus(); await page.mouse.move(0, 0);
  const before = await prompt.screenshot({ animations: 'disabled' });
  const box = await page.locator('.tokenizer-editor').boundingBox();
  await control('release', {});
  await embeddingDone(page, 12);
  expect(await prompt.screenshot({ animations: 'disabled' })).toEqual(before);
  expect(await page.locator('.tokenizer-editor').boundingBox()).toEqual(box);
  // A short prompt now uses compact automatic allocation; embedding completion
  // still must not alter any prompt pixels, selection, or geometry.
  expect(box!.height).toBeLessThan(260);
  // Select source while a real tokenizer response is held; decorating the
  // eventual response must preserve which characters the next key replaces.
  let release!: () => void;
  const barrier = new Promise<void>(resolve => { release = resolve; });
  let entered = false;
  await page.route('**/tokenize', async route => {
    const response = await route.fetch();
    if (route.request().postDataJSON().text === 'heXllo world') { entered = true; await barrier; }
    await route.fulfill({ response });
  });
  await editor.press('Home'); await editor.press('ArrowRight'); await editor.press('ArrowRight');
  await editor.pressSequentially('X'); await expect.poll(() => entered).toBe(true);
  await editor.press('Shift+ArrowRight'); await editor.press('Shift+ArrowRight');
  release(); await page.unrouteAll({ behavior: 'wait' });
  await embeddingDone(page, 13);
  const edited = page.waitForRequest(r => r.url().endsWith('/tokenize') && r.postDataJSON().text === 'heXQo world');
  await page.keyboard.insertText('Q'); await edited;
  await embeddingDone(page, 12);
  const undone = page.waitForRequest(r => r.url().endsWith('/tokenize') && r.postDataJSON().text === 'heXllo world');
  await editor.press('Control+z'); await undone; await embeddingDone(page, 13);
  const redone = page.waitForRequest(r => r.url().endsWith('/tokenize') && r.postDataJSON().text === 'heXQo world');
  await editor.press('Control+Shift+Z'); await redone; await embeddingDone(page, 12);
  const duringComposition: string[] = [];
  page.on('request', r => { if (r.url().endsWith('/tokenize')) duringComposition.push(r.postDataJSON().text); });
  await editor.dispatchEvent('compositionstart'); await editor.fill('に');
  await expect(page.locator('.tokenizer-status')).toContainText('Composing');
  await expect(page.locator('.matrix-scroll:visible')).toBeVisible();
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
  // Deliberately exceed the production debounce while the IME owns the source.
  await page.waitForTimeout(250); expect(duringComposition).toEqual([]);
  await editor.fill('日本'); await editor.dispatchEvent('compositionend', { data: '日本' });
  await expect(page.locator('.tokenizer-status')).toContainText('current prompt');
  await embeddingDone(page, 7);
  expect(duringComposition).toEqual(['日本']);
  await closeSession(page);
});

for (const width of [1178]) {
test(`polish real tokenizer auto sizing and accessible manual split at ${width}px`, { tag: '@extended' }, async ({ page, product }, info) => {
  const { tokenizer, closeSession, embeddingDone, documentFits, polishCapture } = product;
    await page.setViewportSize({ width, height: 900 });
    const editor = await tokenizer(page);
    const fill = async (text: string) => {
      const response = page.waitForResponse(r => r.url().endsWith('/tokenize') && r.request().postDataJSON().text === text);
      await editor.press('ControlOrMeta+A'); await page.keyboard.insertText(text);
      const tokens = (await (await response).json()).tokens;
      // Hundreds of progressive row uploads can exceed the ordinary 15 s wait
      // under DPR 2 SwiftShader. This is a completion/geometry gate, not latency.
      await embeddingDone(page, tokens.length, 45_000); await settledPrompt(page);
      return tokens.length as number;
    };
    const bounded = async () => {
      const g = await tokenizerGeometry(page);
      expect(g.prompt.height).toBeGreaterThanOrEqual(g.min - 1);
      expect(g.prompt.height).toBeLessThanOrEqual(g.max + 1);
      expect(g.divider.top).toBeCloseTo(g.prompt.bottom, 0);
      expect(g.embeddings.top).toBeCloseTo(g.divider.bottom, 0);
      expect(g.embeddings.bottom).toBeCloseTo(g.workspace.bottom, 0);
      await documentFits(page);
      return g;
    };
    const rows = await fill('one two');
    const short = await bounded();
    expect(short.prompt.height).toBeLessThan(short.workspace.height / 2);
    expect(short.scrollHeight).toBeLessThanOrEqual(short.clientHeight + 1);
    await editor.blur(); await polishCapture(page, info, `tokenizer-auto-${width}`);
    const split = page.getByRole('separator', { name: 'Resize prompt and embeddings', exact: true });
    await expect(split).toHaveAttribute('aria-orientation', 'horizontal');
    await split.press('ArrowDown'); await expect(split).toBeFocused();
    await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'manual');
    expect((await bounded()).prompt.height).toBeCloseTo(short.prompt.height + 16, 0);
    await split.press('ArrowUp');
    expect((await bounded()).prompt.height).toBeCloseTo(short.prompt.height, 0);
    const start = (await split.boundingBox())!;
    await drag(page, { x: start.x + start.width / 2, y: start.y + start.height / 2 },
      { x: start.x + start.width / 2, y: start.y + start.height / 2 + 64 });
    await page.mouse.up();
    const manual = await bounded();
    expect(manual.prompt.height).toBeCloseTo(short.prompt.height + 64, 0);
    await split.blur(); await polishCapture(page, info, `tokenizer-manual-${width}`);
    await zoom(page, rows, 576, 4);
    expect((await bounded()).prompt).toEqual(manual.prompt);
    await page.locator('.matrix-scroll:visible').focus();
    await expect(page.locator('[data-token-index="0"]')).toHaveAttribute('data-active-token', '');
    expect((await bounded()).prompt).toEqual(manual.prompt);
    await split.press('Home');
    expect((await bounded()).prompt.height).toBeCloseTo(short.min, 0);
    await split.press('End');
    expect((await bounded()).prompt.height).toBeCloseTo(short.max, 0);
    await split.dblclick(); await settledPrompt(page);
    await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'auto');
    expect((await bounded()).prompt.height).toBeCloseTo(short.prompt.height, 0);
    await fill(Array.from({ length: 40 }, (_, i) => `Line ${i} one two`).join('\n'));
    const long = await bounded();
    expect(long.prompt.height).toBeGreaterThan(short.prompt.height + 30);
    expect(long.prompt.height).toBeLessThanOrEqual((long.workspace.height - long.divider.height) * .46);
    expect(long.scrollHeight).toBeGreaterThan(long.clientHeight * 2);
    await page.locator('.tokenizer-editor').evaluate(n => { n.scrollTop = 100; });
    await expect.poll(() => page.locator('.tokenizer-editor').evaluate(n => n.scrollTop)).toBeGreaterThan(0);
    expect((await bounded()).embeddings).toEqual(long.embeddings);
    await polishCapture(page, info, `tokenizer-long-${width}`);
    await page.setViewportSize({ width, height: 740 }); await settledPrompt(page);
    expect((await bounded()).prompt.height).toBeLessThan(long.prompt.height - 30);
    await split.press('Home'); await split.press('ArrowDown');
    const preferred = (await bounded()).prompt.height;
    await fill('short again');
    expect((await bounded()).prompt.height).toBeCloseTo(preferred, 0);
    await page.setViewportSize({ width: 640, height: 740 }); await settledPrompt(page);
    expect((await bounded()).prompt.height).toBeCloseTo(preferred, 0);
    await info.attach('panel-allocation', { body: JSON.stringify({ short, manual, long }), contentType: 'application/json' });
    await closeSession(page);
  });
}
