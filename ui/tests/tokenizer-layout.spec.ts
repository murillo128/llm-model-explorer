import { expect, test } from '@playwright/test';
import type { Page, Route, TestInfo } from '@playwright/test';
import type { Tokenization } from '../src/tokenizer/annotations';
import { models, sessionA } from '../src/test/shell-fixtures';
import { data, frame, meta } from './embedding-fixtures';
import { nativeCamera } from './native-camera';

// Exercise the built application shell with deterministic transport responses.
// Word spans are a fixture, not a browser implementation of tokenization.
function tokenization(text: string, addSpecialTokens: boolean): Tokenization {
  return { text, add_special_tokens: addSpecialTokens, tokens: [...text.matchAll(/\S+/gu)].map((match, index) => ({
    index, id: index % 4, token: match[0], decoded: match[0], special: false,
    start: [...text.slice(0, match.index)].length,
    end: [...text.slice(0, match.index + match[0].length)].length,
  })) };
}

async function start(page: Page) {
  let paused = false;
  let operation = 0;
  const pending: Route[] = [];
  const complete = (route: Route) => {
    const { text, add_special_tokens } = route.request().postDataJSON() as { text: string; add_special_tokens: boolean };
    return route.fulfill({ json: tokenization(text, add_special_tokens) });
  };
  await page.route('**/runtime-config.json', route => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  await page.route('https://backend.example/**', route => {
    const path = new URL(route.request().url()).pathname;
    if (route.request().method() === 'DELETE') return route.fulfill({ status: 204 });
    if (path === '/models') return route.fulfill({ json: { models } });
    if (path === '/sessions') return route.fulfill({ status: 201, json: sessionA });
    if (path.endsWith('/tensors')) return route.fulfill({ json: { coverage: 'complete', diagnostics: [], tensors: [] } });
    if (path.endsWith('/tokenize')) {
      if (paused) { pending.push(route); return; }
      return complete(route);
    }
    if (path.endsWith('/embeddings')) {
      const { token_ids } = route.request().postDataJSON() as { token_ids: number[] };
      return route.fulfill({ contentType: 'application/vnd.llm-model-explorer.stream',
        headers: { 'Access-Control-Expose-Headers': 'X-Operation-Id',
          'X-Operation-Id': `01234567-89ab-cdef-0123-${String(++operation).padStart(12, '0')}` },
        body: Buffer.from([...meta(token_ids, 64), ...data(token_ids, 64), ...frame(4)]) });
    }
    return route.abort();
  });
  await page.goto('/');
  await page.getByRole('combobox').selectOption(models[0]!.id);
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: '0 tokens' })).toBeVisible();
  return { editor: page.getByRole('textbox', { name: 'Prompt', exact: true }),
    pause: () => { paused = true; }, pending, complete };
}

const divider = (page: Page) => page.getByRole('separator', { name: 'Resize prompt and embeddings', exact: true });

async function geometry(page: Page) {
  return page.locator('.tokenizer-workspace').evaluate(workspace => {
    const rect = (selector: string) => workspace.querySelector(selector)!.getBoundingClientRect().toJSON() as DOMRect;
    const editor = workspace.querySelector<HTMLElement>('.tokenizer-editor')!;
    const splitter = workspace.querySelector('[role="separator"]')!;
    return { workspace: workspace.getBoundingClientRect().toJSON() as DOMRect,
      prompt: rect('.prompt-panel'), embeddings: rect('.input-embeddings'), divider: rect('[role="separator"]'),
      editor: rect('.tokenizer-editor'), scrollHeight: editor.scrollHeight, clientHeight: editor.clientHeight,
      min: Number(splitter.getAttribute('aria-valuemin')), max: Number(splitter.getAttribute('aria-valuemax')),
      now: Number(splitter.getAttribute('aria-valuenow')) };
  });
}

async function layoutSettled(page: Page) {
  // Two matching animation frames are enough for ResizeObserver/CodeMirror's
  // measurement cycle without coupling the test to their scheduling internals.
  await expect.poll(async () => {
    const heights = await page.locator('.prompt-panel').evaluate(async panel => {
      const samples: number[] = [];
      for (let frame = 0; frame < 3; frame++) {
        await new Promise(requestAnimationFrame);
        samples.push(panel.getBoundingClientRect().height);
      }
      return samples;
    });
    return Math.max(...heights) - Math.min(...heights);
  }).toBeLessThan(0.5);
}

async function fill(page: Page, text: string) {
  // Select the full editor document, including CodeMirror's virtualized lines.
  // contenteditable.fill() can only replace the currently rendered DOM range.
  const editor = page.getByRole('textbox', { name: 'Prompt', exact: true });
  await editor.press('ControlOrMeta+A');
  await page.keyboard.insertText(text);
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'current');
  if (text.trim()) await expect(page.locator('.matrix-scroll')).toBeVisible();
  await layoutSettled(page);
}

async function bounded(page: Page) {
  const state = await geometry(page);
  expect(state.prompt.height).toBeGreaterThanOrEqual(state.min - 1);
  expect(state.prompt.height).toBeLessThanOrEqual(state.max + 1);
  expect(state.embeddings.height).toBeGreaterThan(0);
  expect(state.divider.y).toBeCloseTo(state.prompt.bottom, 0);
  expect(state.embeddings.y).toBeCloseTo(state.divider.bottom, 0);
  expect(state.embeddings.bottom).toBeCloseTo(state.workspace.bottom, 0);
  expect(state.prompt.height + state.divider.height + state.embeddings.height).toBeCloseTo(state.workspace.height, 0);
  expect(Math.abs(state.now - state.prompt.height)).toBeLessThanOrEqual(1);
  expect(await page.evaluate(() => ({
    x: scrollX, y: scrollY,
    width: Math.max(document.body.scrollWidth, document.documentElement.scrollWidth) <= innerWidth,
    height: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight) <= innerHeight,
  }))).toEqual({ x: 0, y: 0, width: true, height: true });
}

async function capture(page: Page, info: TestInfo, name: string) {
  await page.mouse.move(0, 0);
  const path = info.outputPath(`${name}.png`);
  await page.locator('.tokenizer-workspace').screenshot({ path });
  await info.attach(name, { path, contentType: 'image/png' });
}

test('short, wrapped and long prompts grow to a workspace cap with prompt-only scrolling', async ({ page }, info) => {
  await start(page);
  await fill(page, 'A short prompt');
  const short = await geometry(page);
  expect(short.prompt.height).toBeLessThan(short.workspace.height * 0.5);
  expect(short.scrollHeight).toBeLessThanOrEqual(short.clientHeight + 1);
  await bounded(page);
  await capture(page, info, 'short-auto');

  await divider(page).press('ArrowDown');
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'manual');
  await capture(page, info, 'short-manual');
  await divider(page).dblclick();
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'auto');

  // No explicit newline: growth comes from actual line wrapping at both widths.
  await fill(page, 'wrapped words '.repeat(Math.ceil(short.editor.width / 50)));
  const wrapped = await geometry(page);
  expect(wrapped.prompt.height).toBeGreaterThan(short.prompt.height + 10);
  await bounded(page);

  const longText = Array.from({ length: 80 }, (_, line) => `Prompt line ${line} with editable source text.`).join('\n');
  await fill(page, longText);
  const long = await geometry(page);
  expect(long.prompt.height).toBeLessThanOrEqual((long.workspace.height - long.divider.height) * 0.46 + 1);
  expect(long.scrollHeight).toBeGreaterThan(long.clientHeight * 2);
  expect(long.embeddings.height).toBeGreaterThan(long.prompt.height);
  expect(await page.locator('.tokenizer-editor').evaluate(node => getComputedStyle(node).resize)).toBe('none');
  await page.locator('.tokenizer-editor').evaluate(node => { node.scrollTop = 100; });
  await expect.poll(() => page.locator('.tokenizer-editor').evaluate(node => node.scrollTop)).toBeGreaterThan(0);
  const scrolled = await geometry(page);
  expect(scrolled.prompt).toEqual(long.prompt);
  expect(scrolled.embeddings).toEqual(long.embeddings);
  await bounded(page);
  await capture(page, info, 'long-auto');

  await divider(page).press('End');
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'manual');
  await bounded(page);
  await capture(page, info, 'long-manual');
});

test('splitter has keyboard bounds and double activation returns to automatic allocation', async ({ page }) => {
  await start(page);
  await fill(page, 'Editable prompt');
  const split = divider(page);
  await expect(split).toHaveAttribute('aria-orientation', 'horizontal');
  await split.focus();
  await expect(split).toBeFocused();
  const automatic = await geometry(page);
  await split.press('ArrowDown');
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'manual');
  expect((await geometry(page)).prompt.height).toBeCloseTo(automatic.prompt.height + 16, 0);
  await split.press('ArrowUp');
  expect((await geometry(page)).prompt.height).toBeCloseTo(automatic.prompt.height, 0);
  await split.press('Home');
  expect((await geometry(page)).prompt.height).toBeCloseTo(automatic.min, 0);
  await split.press('ArrowUp');
  expect((await geometry(page)).prompt.height).toBeCloseTo(automatic.min, 0);
  await split.press('End');
  expect((await geometry(page)).prompt.height).toBeCloseTo(automatic.max, 0);
  await split.press('ArrowDown');
  expect((await geometry(page)).prompt.height).toBeCloseTo(automatic.max, 0);
  await bounded(page);

  for (const key of ['Enter', 'Space']) {
    await split.press(key);
    await split.press(key);
    await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'auto');
    await layoutSettled(page);
    expect((await geometry(page)).prompt.height).toBeCloseTo(automatic.prompt.height, 0);
    await split.press('End');
  }
  await split.dblclick();
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'auto');
  await bounded(page);
});

test('pointer dragging continuously reallocates space and clamps both panel bounds', async ({ page }) => {
  await start(page);
  await fill(page, 'A short prompt');
  const before = await geometry(page);
  expect(before.divider.height).toBe(12);
  expect(await divider(page).evaluate(node => getComputedStyle(node, '::before').backgroundColor)).toBe('rgba(0, 0, 0, 0)');
  await page.mouse.move(before.divider.x + before.divider.width / 2, before.divider.y + before.divider.height / 2);
  await page.mouse.down();
  await page.mouse.move(before.divider.x + before.divider.width / 2, before.divider.y + before.divider.height / 2 + 35, { steps: 4 });
  const moved = await geometry(page);
  await expect(divider(page)).toHaveAttribute('data-resizing', 'true');
  expect(await divider(page).evaluate(node => getComputedStyle(node).cursor)).toBe('row-resize');
  expect(moved.prompt.height).toBeCloseTo(before.prompt.height + 35, 0);
  expect(moved.embeddings.height).toBeCloseTo(before.embeddings.height - 35, 0);
  await page.mouse.move(before.divider.x + before.divider.width / 2, before.divider.y + before.divider.height / 2);
  expect((await geometry(page)).prompt.height).toBeCloseTo(before.prompt.height, 0);
  await page.mouse.move(before.divider.x + 10, page.viewportSize()!.height - 1, { steps: 4 });
  expect((await geometry(page)).prompt.height).toBeCloseTo(before.max, 0);
  await page.mouse.move(before.divider.x + 10, 0, { steps: 4 });
  expect((await geometry(page)).prompt.height).toBeCloseTo(before.min, 0);
  await page.mouse.up();
  await expect(divider(page)).not.toHaveAttribute('data-resizing');
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'manual');
  await bounded(page);
});

test('browser resizing recomputes automatic height and preserves a bounded manual preference', async ({ page }) => {
  await start(page);
  const original = page.viewportSize()!;
  await page.setViewportSize({ width: 1440, height: 900 });
  await fill(page, 'wrapped words '.repeat(6));
  const wide = await geometry(page);
  await page.setViewportSize({ width: 390, height: 900 });
  await expect.poll(async () => (await geometry(page)).prompt.height).toBeGreaterThan(wide.prompt.height + 10);
  await bounded(page);

  await page.setViewportSize(original);
  await fill(page, 'A line of source text\n'.repeat(60));
  const automatic = await geometry(page);
  await page.setViewportSize({ ...original, height: original.height - 140 });
  await layoutSettled(page);
  const smaller = await geometry(page);
  expect(smaller.prompt.height).toBeLessThan(automatic.prompt.height - 30);
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'auto');
  await bounded(page);

  await page.setViewportSize(original);
  await layoutSettled(page);
  await divider(page).press('Home');
  for (let step = 0; step < 4; step++) await divider(page).press('ArrowDown');
  const manual = (await geometry(page)).prompt.height;
  await fill(page, 'Short again');
  expect((await geometry(page)).prompt.height).toBeCloseTo(manual, 0);
  await page.setViewportSize({ ...original, height: original.height - 100 });
  await layoutSettled(page);
  expect((await geometry(page)).prompt.height).toBeCloseTo(manual, 0);
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'manual');
  await bounded(page);

  // A severely constrained workspace must relax minimums without page overflow.
  await page.setViewportSize({ width: original.width, height: 400 });
  await layoutSettled(page);
  expect((await geometry(page)).prompt.height).toBeLessThan(manual);
  await bounded(page);
  await page.setViewportSize(original);
  await layoutSettled(page);
  // Temporary clamping must not overwrite the user's preferred manual height.
  expect((await geometry(page)).prompt.height).toBeCloseTo(manual, 0);
  await divider(page).dblclick();
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'auto');
  await layoutSettled(page);
  expect((await geometry(page)).prompt.height).toBeLessThan(manual);
  await bounded(page);

  await divider(page).press('End');
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'auto');
  await bounded(page);
});

test('stale annotations and current replacement never flicker prompt allocation', async ({ page }) => {
  const api = await start(page);
  await fill(page, 'Alpha beta');
  api.pause();
  await api.editor.fill('Alpha zeta');
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'stale');
  await expect.poll(() => api.pending.length).toBe(1);
  await layoutSettled(page);
  const stale = await geometry(page);
  const samples = page.locator('.prompt-panel').evaluate(async panel => {
    const heights: number[] = [];
    for (let frame = 0; frame < 24; frame++) {
      await new Promise(requestAnimationFrame);
      heights.push(panel.getBoundingClientRect().height);
    }
    return heights;
  });
  await api.complete(api.pending[0]!);
  await expect(page.locator('[data-annotations]')).toHaveAttribute('data-annotations', 'current');
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
  const heights = await samples;
  expect(heights.every(height => Math.abs(height - stale.prompt.height) < 0.5)).toBe(true);
  expect((await geometry(page)).prompt).toEqual(stale.prompt);
  expect(await api.editor.evaluate(node => {
    const source = node.cloneNode(true) as HTMLElement;
    source.querySelectorAll('.token-opening, .token-closing, .unmapped-annotation').forEach(annotation => annotation.remove());
    return source.textContent;
  })).toBe('Alpha zeta');
  await bounded(page);
});

test('matrix zoom, scroll and linked row inspection leave prompt geometry and source intact', async ({ page }) => {
  await start(page);
  await fill(page, 'Alpha beta gamma delta');
  const matrix = page.locator('.matrix-scroll');
  const before = await geometry(page);
  const promptState = () => page.locator('.tokenizer-editor').evaluate(editor => ({
    rect: editor.getBoundingClientRect().toJSON(), scroll: [editor.scrollLeft, editor.scrollTop],
    text: editor.querySelector('.cm-content')!.textContent,
  }));
  const original = await promptState();
  await matrix.focus();
  await matrix.press('ArrowDown');
  await expect(page.locator('[data-token-index="1"]')).toHaveAttribute('data-active-token', '');
  await expect(page.locator('.inspection-readout')).toContainText('row 1');
  const token = page.locator('[data-token-index="3"]');
  await token.hover();
  await expect(token).toHaveAttribute('data-active-token', '');
  await token.click();
  await nativeCamera(page);
  const extentWidth = () => matrix.evaluate(node => (node.firstElementChild as HTMLElement).style.width);
  const initialWidth = await extentWidth();
  await matrix.evaluate(node => {
    const rect = node.getBoundingClientRect();
    for (let step = 0; step < 6; step++) {
      node.dispatchEvent(new WheelEvent('wheel', { deltaY: -500, clientX: rect.x + 20, clientY: rect.y + 20, cancelable: true }));
    }
  });
  await expect.poll(extentWidth).not.toBe(initialWidth);
  await matrix.evaluate(node => { node.scrollLeft = 40; node.scrollTop = 20; });
  await expect.poll(() => matrix.evaluate(node => node.scrollLeft)).toBeGreaterThan(0);
  await expect.poll(() => matrix.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Fit width', exact: true }).click();
  expect(await promptState()).toEqual(original);
  expect((await geometry(page)).prompt).toEqual(before.prompt);
  await expect(page.locator('.tokenizer-workspace')).toHaveAttribute('data-sizing', 'auto');
  await bounded(page);
});
