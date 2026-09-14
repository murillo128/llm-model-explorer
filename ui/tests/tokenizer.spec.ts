import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type { Token, Tokenization } from '../src/tokenizer/annotations';
import { models, sessionA, tensors } from '../src/test/shell-fixtures';

function response(text: string, id = 123, special = true): Tokenization {
  return { text, add_special_tokens: special, tokens: text ? [{ index: 0, id, token: 'fixture', decoded: text, special: false, start: 0, end: [...text].length }] : [] };
}
async function count(page: Page, expected: number) {
  await expect.poll(() => page.evaluate(() => window.tokenizerHarness.requests.length)).toBe(expected);
}
async function complete(page: Page, index: number, data: Tokenization) {
  await page.evaluate(({ index, data }) => window.tokenizerHarness.complete(index, data), { index, data });
}
async function start(page: Page) {
  const port = Number(process.env.UI_TEST_PORT ?? 4173) + 1;
  await page.goto(`http://127.0.0.1:${port}/tests/tokenizer.html`);
  await count(page, 1); await complete(page, 0, response(''));
  await expect(page.getByRole('status')).toContainText('0 tokens');
  return page.getByRole('textbox', { name: 'Prompt', exact: true });
}

test('reverse responses and A→B→A cannot overwrite the newest source or annotations', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('A'); await count(page, 2);
  await editor.fill('B'); await count(page, 3);
  await editor.fill('A'); await count(page, 4);
  await expect(page.locator('.source-annotation')).toHaveCount(0);
  await complete(page, 3, response('A', 303));
  await expect(page.locator('.token-opening .token-ids')).toHaveText('303');
  await complete(page, 2, response('B', 202)); await complete(page, 1, response('A', 101));
  await expect(page.locator('.token-opening .token-ids')).toHaveText('303');
  expect(await page.evaluate(() => window.tokenizerHarness.requests.slice(1, 3).map((request) => request.aborted))).toEqual([true, true]);
  await expect.poll(() => page.evaluate(() => window.tokenizerHarness.source())).toBe('A');
});

test('typing, mid-string edits, selection, undo/redo and clipboard survive async decoration', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const editor = await start(page);
  await editor.pressSequentially('hello world'); await count(page, 2);
  await editor.press('Home'); await editor.press('ArrowRight'); await editor.press('ArrowRight');
  await editor.pressSequentially('X'); await count(page, 3);
  await complete(page, 2, response('heXllo world', 234));
  expect(await page.evaluate(() => window.tokenizerHarness.selection())).toEqual([3, 3]);
  await editor.press('Shift+ArrowRight'); await editor.press('Shift+ArrowRight');
  await complete(page, 1, response('hello world', 111));
  expect(await page.evaluate(() => window.tokenizerHarness.selection())).toEqual([3, 5]);
  await editor.press('Control+z'); await expect.poll(() => page.evaluate(() => window.tokenizerHarness.source())).toBe('hello world'); await count(page, 4);
  await complete(page, 3, response('hello world', 345));
  await editor.press('Control+Shift+Z');
 await expect.poll(() => page.evaluate(() => window.tokenizerHarness.source())).toBe('heXllo world'); await count(page, 5);
  await complete(page, 4, response('heXllo world', 456));
  await editor.press('Control+a'); await editor.press('Control+c');
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('heXllo world');
  await page.evaluate(() => navigator.clipboard.writeText(' <b>😀</b>\n\t中 '));
  await editor.press('Control+v'); await expect.poll(() => page.evaluate(() => window.tokenizerHarness.source())).toBe(' <b>😀</b>\n\t中 '); await count(page, 6);
  await complete(page, 5, response(' <b>😀</b>\n\t中 '));
  expect(await page.evaluate(() => window.tokenizerHarness.requests.at(-1)!.text)).toBe(' <b>😀</b>\n\t中 ');
  await expect(page.locator('.tokenizer-editor b')).toHaveCount(0);
});

test('composition suppresses intermediate requests and fences pre-composition responses', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('old'); await count(page, 2);
  await editor.dispatchEvent('compositionstart'); await editor.fill('に');
  await page.waitForTimeout(200);
  expect(await page.evaluate(() => window.tokenizerHarness.requests.length)).toBe(2);
  await complete(page, 1, response('old')); await expect(page.locator('.token-opening .token-ids')).toHaveCount(0);
  await editor.fill('日本'); await editor.dispatchEvent('compositionend', { data: '日本' });
  await count(page, 3); await complete(page, 2, response('日本'));
  await expect.poll(() => page.evaluate(() => window.tokenizerHarness.source())).toBe('日本'); await expect(page.getByRole('status')).toContainText('current prompt');
});

test('session/options changes and late errors cannot contaminate the current editor', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('keep me'); await count(page, 2);
  await page.getByRole('button', { name: 'Change session' }).click(); await count(page, 3);
  await page.getByRole('checkbox', { name: 'Add special tokens' }).uncheck(); await count(page, 4);
  await complete(page, 3, response('keep me', 333, false));
  await complete(page, 2, response('keep me', 222));
  await page.evaluate(() => window.tokenizerHarness.fail(1));
  await expect(page.locator('.token-opening .token-ids')).toHaveText('333'); await expect(page.getByRole('alert')).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => window.tokenizerHarness.source())).toBe('keep me');
  await editor.press('End'); await editor.pressSequentially(' error'); await count(page, 5); await page.evaluate(() => window.tokenizerHarness.fail(4));
  await expect(page.getByRole('alert')).toContainText('Could not tokenize');
  await expect(page.locator('.token-opening .token-ids')).toHaveText('333');
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'stale');
  await page.getByRole('button', { name: 'Retry tokenization' }).click(); await count(page, 6);
  await complete(page, 5, response('keep me error', 555, false));
  await expect(page.locator('.token-opening .token-ids')).toHaveText('555');
  await page.getByRole('checkbox', { name: 'Tokenizer available' }).uncheck();
  await expect(page.getByRole('status')).toContainText('Tokenizer unavailable');
  await expect(editor).toHaveAttribute('contenteditable', 'true'); await expect(page.locator('.token-opening .token-ids')).toHaveText('555');
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'stale');
});

test('exact whitespace, Unicode, overlapping and typed versus inserted specials share one source surface', async ({ page, context }, testInfo) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const editor = await start(page);
  const text = '  A😀é<special>\n\t中 café  ';
  await editor.fill(text); await count(page, 2);
  const token = (index: number, id: number, start?: number, end?: number, special = false, native = '�'): Token => ({ index, id, token: native, decoded: '�', special, ...(start === undefined ? {} : { start, end: end! }) });
  await complete(page, 1, { text, add_special_tokens: true, tokens: [
    token(0, 1, undefined, undefined, true, '<bos>'), token(1, 10, 0, 3), token(2, 20, 3, 4), token(3, 21, 3, 4),
    token(4, 30, 4, 6), token(5, 40, 6, 15, true), token(6, 50, 15, 17), token(7, 60, 17, 18),
    token(8, 70, 18, 25), token(9, 2, undefined, undefined, true, '<eos>'),
  ] });
  await expect(page.locator('[data-token-ids="20, 21"]')).toHaveCount(1);
  await expect(page.locator('[data-token-ids="40"]')).toHaveAttribute('aria-label', /source-typed special/);
  await expect(page.locator('.unmapped-annotation')).toHaveCount(2);
  await expect.poll(() => page.evaluate(() => window.tokenizerHarness.source())).toBe(text); await expect(page.getByRole('textbox')).toHaveCount(1);
  expect(await editor.evaluate((node) => getComputedStyle(node).color)).toBe('rgb(0, 0, 0)');
  expect(await page.locator('.token-opening').first().evaluate((node) => getComputedStyle(node).color)).toBe('rgb(116, 113, 107)');
  await editor.focus();
  expect(await page.locator('.tokenizer-editor').evaluate((node) => getComputedStyle(node).outlineStyle)).toBe('solid');
  await page.screenshot({ path: testInfo.outputPath('tokenizer-unicode.png'), fullPage: true });
  await testInfo.attach('Unicode inline annotations', { path: testInfo.outputPath('tokenizer-unicode.png'), contentType: 'image/png' });
  await editor.press('Control+a'); await editor.press('Control+c');
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(text);
});

test('configuration A→B→A does not revive an earlier settled annotation while awaiting fresh results', async ({ page }) => {
  const editor = await start(page);
  await editor.fill('A'); await count(page, 2); await complete(page, 1, response('A', 111));
  await expect(page.locator('.token-opening .token-ids')).toHaveText('111');
  await page.getByRole('checkbox', { name: 'Add special tokens' }).uncheck(); await count(page, 3);
  await page.getByRole('checkbox', { name: 'Add special tokens' }).check(); await count(page, 4);
  await expect(page.locator('.token-opening .token-ids')).toHaveText('111');
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'stale');
  await complete(page, 2, response('A', 222, false));
  await expect(page.locator('.token-opening .token-ids')).toHaveText('111');
  await complete(page, 3, response('A', 333));
  await expect(page.locator('.token-opening .token-ids')).toHaveText('333');
});

test('missing offsets remain explicitly unmapped and HTML-like native/source strings stay inert', async ({ page }) => {
  const editor = await start(page); const text = '<img src=x onerror="window.hacked=1">';
  await editor.fill(text); await count(page, 2);
  await complete(page, 1, { text, add_special_tokens: true, tokens: [{ index: 0, id: 999, token: text, decoded: 'different text', special: false }] });
  await expect(page.locator('.source-annotation')).toHaveCount(0);
  await expect(page.locator('.unmapped-annotation')).toContainText('(no source span)');
  await expect(page.locator('.unmapped-annotation img, .tokenizer-editor img[src="x"]')).toHaveCount(0);
  expect(await page.evaluate(() => 'hacked' in window)).toBe(false);
  await expect.poll(() => page.evaluate(() => window.tokenizerHarness.source())).toBe(text);
});

test('dense annotations and long unmapped text remain scrollable without overlapping source lines', async ({ page }) => {
  const editor = await start(page);
  const text = 'abcdefghij\n\nlast\n\n\n\n\n';
  await editor.fill(text); await count(page, 2);
  const tokens: Token[] = Array.from({ length: 10 }, (_, index) => ({ index, id: 900000000 + index, token: 'fixture', decoded: '', special: false, start: index, end: index + 1 }));
  tokens.push({ index: 10, id: 11, token: '\n', decoded: '\n', special: false, start: 10, end: 11 });
  tokens.push({ index: 11, id: 12, token: '<a-very-long-inserted-special-token-with-native-text>', decoded: '', special: true });
  await complete(page, 1, { text, add_special_tokens: true, tokens });
  await expect(page.locator('.token-opening .token-ids')).toHaveCount(11);
  const boxes = await page.locator('.token-opening .token-ids').evaluateAll((nodes) => nodes.map((node) => { const r = node.getBoundingClientRect(); return { left: r.left, right: r.right, top: r.top, bottom: r.bottom }; }));
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const a = boxes[i]!; const b = boxes[j]!;
    expect(a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top).toBe(false);
  }
  const viewport = page.locator('.tokenizer-editor');
  await viewport.evaluate((node) => { node.scrollLeft = node.scrollWidth; node.scrollTop = node.scrollHeight; });
  await expect.poll(() => page.evaluate(() => window.tokenizerHarness.source())).toBe(text);
  const before = await page.locator('.token-opening .token-ids').first().boundingBox();
  await viewport.evaluate((node) => { node.scrollLeft = 0; node.scrollTop = 0; });
  const after = await page.locator('.token-opening .token-ids').first().boundingBox();
  expect(before!.y).toBeLessThan(after!.y);
  await editor.fill(''); await count(page, 3); await complete(page, 2, response(''));
  await expect(page.locator('.token-opening .token-ids')).toHaveCount(0); await expect.poll(() => page.evaluate(() => window.tokenizerHarness.source())).toBe('');
});

test('standalone explorer displays successful endpoint results in the existing shell', async ({ page }, testInfo) => {
  await page.route('**/runtime-config.json', (route) => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  await page.route('https://backend.example/**', (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/models') return route.fulfill({ json: { models } });
    if (path === '/sessions') return route.fulfill({ status: 201, json: sessionA });
    if (path.endsWith('/tensors')) return route.fulfill({ json: { tensors } });
    if (path.endsWith('/tokenize')) {
      const body = route.request().postDataJSON() as { text: string };
      return route.fulfill({ json: body.text === 'Hello, world!' ? { text: body.text, add_special_tokens: true, tokens: [
        { index: 0, id: 1, token: '<bos>', decoded: '', special: true },
        { index: 1, id: 15496, token: 'Hello', decoded: 'Hello', special: false, start: 0, end: 5 },
        { index: 2, id: 11, token: ',', decoded: ',', special: false, start: 5, end: 6 },
        { index: 3, id: 995, token: 'Ġworld', decoded: ' world', special: false, start: 6, end: 12 },
        { index: 4, id: 0, token: '!', decoded: '!', special: false, start: 12, end: 13 },
      ] } : response(body.text) });
    }
    return route.abort();
  });
  await page.goto('/'); await page.getByRole('combobox').selectOption(models[0]!.id);
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await page.getByRole('textbox', { name: 'Prompt' }).fill('Hello, world!');
  await expect(page.getByText('5 tokens · current prompt')).toBeVisible();
  await expect(page.locator('.token-opening .token-ids')).toHaveCount(4);
  await page.screenshot({ path: testInfo.outputPath('tokenizer-explorer.png'), fullPage: true });
  await testInfo.attach('Standalone Tokenizer Explorer (synthetic API fixture)', { path: testInfo.outputPath('tokenizer-explorer.png'), contentType: 'image/png' });
});

test('standalone shell calls the typed endpoint and reports tokenizer unavailability', async ({ page }) => {
  await page.route('**/runtime-config.json', (route) => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  const requests: unknown[] = [];
  await page.route('https://backend.example/**', (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/models') return route.fulfill({ json: { models } });
    if (path === '/sessions') return route.fulfill({ status: 201, json: sessionA });
    if (path.endsWith('/tensors')) return route.fulfill({ json: { tensors } });
    if (path.endsWith('/tokenize')) {
      requests.push(route.request().postDataJSON());
      return route.fulfill({ status: 422, json: { code: 'unsupported_representation', message: 'No tokenizer' } });
    }
    return route.abort();
  });
  await page.goto('/'); await page.getByRole('combobox').selectOption(models[0]!.id);
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Prompt' })).toBeVisible();
  await expect(page.getByRole('alert')).toContainText('Tokenizer unavailable');
  expect(requests).toEqual([{ text: '', add_special_tokens: true }]);
});


test('mapped stale brackets and IDs survive every animation frame, composition, undo and atomic replacement', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const editor = await start(page);
  await editor.fill('hello'); await count(page, 2); await complete(page, 1, response('hello', 111));
  await expect(page.locator('.token-opening .token-ids')).toHaveText('111');
  await page.evaluate(() => {
    const frames: { ids: string; stale: boolean }[] = [];
    Object.assign(window, { continuityFrames: frames, continuityRunning: true });
    const sample = () => {
      frames.push({ ids: document.querySelector('.token-ids')?.textContent ?? '',
        stale: document.querySelector('[data-annotations]')?.getAttribute('data-annotations') === 'stale' });
      if (Reflect.get(window, 'continuityRunning')) requestAnimationFrame(sample);
    };
    requestAnimationFrame(sample);
  });
  await editor.press('Home'); await editor.press('ArrowRight'); await editor.pressSequentially('X'); await count(page, 3);
  await expect(page.locator('.source-annotation')).toHaveText('hXello');
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'stale');
  expect(await page.locator('.token-opening').evaluate(node => getComputedStyle(node).opacity)).toBe('0.55');
  await editor.dispatchEvent('compositionstart');
  await expect(page.getByRole('status')).toContainText('Previous annotations are stale');
  await complete(page, 2, response('hXello', 222)); // pre-composition result is obsolete
  await expect(page.locator('.token-ids')).toHaveText('111');
  await editor.dispatchEvent('compositionend'); await count(page, 4);
  await editor.press('Control+z'); await count(page, 5);
  expect(await page.evaluate(() => window.tokenizerHarness.source())).toBe('hello');
  await editor.press('Control+Shift+Z'); await count(page, 6);
  expect(await page.evaluate(() => window.tokenizerHarness.source())).toBe('hXello');
  await editor.press('Control+a'); await editor.press('Control+c');
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('hXello');
  await complete(page, 5, response('hXello', 666));
  await expect(page.locator('.token-ids')).toHaveText('666');
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'current');
  await complete(page, 3, response('hXello', 444)); await complete(page, 4, response('hello', 555));
  await expect(page.locator('.token-ids')).toHaveText('666');
  const frames = await page.evaluate(() => {
    Reflect.set(window, 'continuityRunning', false);
    return Reflect.get(window, 'continuityFrames') as { ids: string; stale: boolean }[];
  });
  expect(frames.length).toBeGreaterThan(5);
  expect(frames.every(frame => frame.ids === '111' || frame.ids === '666')).toBe(true);
  expect(frames.some(frame => frame.stale)).toBe(true);
});
