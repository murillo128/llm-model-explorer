/* eslint-disable @typescript-eslint/no-explicit-any -- Test-only browser observer and JSON evidence. */
import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import { installProbe } from './probe';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const backend = 'http://127.0.0.1:8765';
const matrix = 'model.layers.0.mlp.down_proj.weight';
const value = (index: number) => ((index * 17) % 257 - 128) / 128;
let service: ReturnType<typeof spawn>;
let log = '';
let referenceRoot: string | undefined;
let referenceSamples: any;

async function control(path = 'state', body?: object) {
  const response = await fetch(`${backend}/__test/${path}`, body ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  } : undefined);
  expect(response.ok).toBeTruthy();
  return response.json();
}
async function metrics(page: Page) { return page.evaluate(() => (window as any).__acceptance.metrics()); }
async function idle() {
  await expect.poll(async () => {
    const state = await control();
    return ['operations', 'consumers', 'readers', 'flights', 'tasks', 'temporary'].map((key) => state[key]);
  }).toEqual([0, 0, 0, 0, 0, 0]);
}
async function open(page: Page, name = matrix) {
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('acceptance/fixture');
  await page.getByRole('button', { name: new RegExp(name.replaceAll('.', '\\.')) }).click();
}
async function complete(page: Page) {
  await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'complete');
  await expect(page.locator('[data-result=statistics]')).toHaveAttribute('data-state', 'complete');
}
async function pixel(page: Page, x: number, y: number) {
  return page.evaluate(({ x, y }) => (window as any).__acceptance.pixel('.matrix-scroll canvas', x, y), { x, y });
}
function luminance(rgb: number[]) {
  const linear = rgb.slice(0, 3).map((v) => v / 255 <= .04045 ? v / 255 / 12.92 : ((v / 255 + .055) / 1.055) ** 2.4);
  return .2126 * linear[0]! + .7152 * linear[1]! + .0722 * linear[2]!;
}

test.beforeEach(async ({ page }, testInfo) => {
  log = '';
  const isReference = testInfo.title.startsWith('local reference');
  test.skip(isReference && !process.env.LMEX_REFERENCE_MODEL_DIR,
    'LMEX_REFERENCE_MODEL_DIR not supplied; local SmolLM2-135M Base UI not tested');
  let command = ['-m', 'acceptance.server'];
  if (isReference) {
    const directory = process.env.LMEX_REFERENCE_MODEL_DIR!;
    referenceSamples = JSON.parse(execFileSync(`${repo}backend/.venv/bin/python`,
      ['-m', 'acceptance.reference', directory], { cwd: repo, encoding: 'utf8' }));
    referenceRoot = mkdtempSync(join(tmpdir(), 'lmex-reference-'));
    command = ['-m', 'llm_model_explorer', '--model-root', dirname(directory),
      '--cache-dir', referenceRoot, '--port', '8765', '--cors-origin', 'http://127.0.0.1:4175',
      '--device', process.env.LMEX_REFERENCE_DEVICE ?? 'cpu'];
  }
  service = spawn(`${repo}backend/.venv/bin/python`, command, {
    cwd: repo, env: { ...process.env, HF_HUB_OFFLINE: '1', TOKENIZERS_PARALLELISM: 'false' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  service.stdout!.on('data', (data) => { log += data; });
  service.stderr!.on('data', (data) => { log += data; });
  await expect.poll(async () => {
    if (service.exitCode !== null) throw new Error(log);
    try { return (await fetch(`${backend}/models`)).status; } catch { return 0; }
  }, { timeout: 30_000 }).toBe(200);
  page.on('console', (message) => { if (message.type() === 'error' || message.type() === 'warning') log += `\nBrowser: ${message.text()}`; });
  page.on('pageerror', (error) => { log += `\nPage: ${error.message}`; });
  await page.addInitScript(installProbe);
  if (isReference) await page.addInitScript(() => { (window as any).__acceptance.capture = false; });
  await page.goto('/');
  await expect(page.getByTestId('backend-url')).toHaveText(backend);
});

test.afterEach(async ({ page }, testInfo) => {
  if (testInfo.status === 'skipped') return;
  if (!page.isClosed() && testInfo.status !== testInfo.expectedStatus) {
    await testInfo.attach('resource-state', { body: JSON.stringify(await metrics(page)), contentType: 'application/json' });
  }
  await page.close();
  if (testInfo.status !== testInfo.expectedStatus) await testInfo.attach('backend-log', { body: log, contentType: 'text/plain' });
  if (service?.exitCode === null) {
    service.kill('SIGTERM');
    const timer = setTimeout(() => service.kill('SIGKILL'), 10_000);
    try { await once(service, 'exit'); } finally { clearTimeout(timer); }
  }
  if (referenceRoot) { rmSync(referenceRoot, { recursive: true }); referenceRoot = undefined; }
});

test('production UI renders before producer completes; native geometry, inspection and cleanup', async ({ page, context }, testInfo) => {
  await page.setViewportSize({ width: 1000, height: 500 });
  await control('arm', { kind: 'logical_tensor' });
  const started = await page.evaluate(() => performance.now());
  await open(page);
  await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'streaming');
  const state = await control();
  expect(state.control.entered).toBe(true);
  expect(state.control.released).toBe(false);
  expect(Object.keys(state.artifacts)).toHaveLength(0);
  const first = await metrics(page);
  expect(first.firstRender).toBeGreaterThan(started);
  const known = await pixel(page, 8, 8);
  // First pixel read is real grayscale, missing data is the explicit pending color.
  expect(known[0]).toBe(known[1]);
  expect(known[1]).toBe(known[2]);
  const beforeRelease = await page.evaluate(() => performance.now());
  expect(first.firstRender).toBeLessThan(beforeRelease);
  await control('release', {});
  await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveAttribute('data-state', 'complete');
  const ended = await page.evaluate(() => performance.now());
  const limits = await page.evaluate(() => (window as any).__acceptance.limits());
  const canvas = page.locator('.matrix-scroll canvas');
  const geometry = await canvas.evaluate((node) => {
    const c = node as HTMLCanvasElement, box = c.getBoundingClientRect();
    const extent = c.closest('.matrix-scroll')!.firstElementChild as HTMLElement;
    return { width: c.width, height: c.height, cssWidth: box.width, cssHeight: box.height,
      extentWidth: extent.style.width, extentHeight: extent.style.height };
  });
  expect(geometry.width / limits.dpr).toBeCloseTo(geometry.cssWidth, 5);
  expect(geometry.height / limits.dpr).toBeCloseTo(geometry.cssHeight, 5);
  // Main extent must preserve 1536 columns and 576 rows regardless of viewport/DPR.
  expect(parseFloat(geometry.extentWidth) * limits.dpr).toBe(1536);
  expect(parseFloat(geometry.extentHeight) * limits.dpr).toBe(576);
  await canvas.scrollIntoViewIfNeeded();
  await page.locator('.matrix-scroll').evaluate((node) => { node.scrollLeft = 53; node.scrollTop = 27; });
  await expect.poll(async () => canvas.getAttribute('data-origin')).not.toBe('0,0');
  const origin = (await canvas.getAttribute('data-origin'))!.split(',').map(Number);
  await expect(page.locator('.row-distributions canvas')).toHaveAttribute('data-origin', `0,${origin[1]}`);
  await expect(page.locator('.column-distributions canvas')).toHaveAttribute('data-origin', `${origin[0]},0`);
  expect(origin[0]).toBeGreaterThan(0);
  expect(origin[1]).toBeGreaterThan(0);
  const neutral = await pixel(page, 8, 8);
  const beforeHover = await metrics(page);
  expect(beforeHover.textures).toBe(3);
  expect(beforeHover.gpuBytes).toBe(4 * (576 * 1536 + (576 + 1536) * 100));
  expect(beforeHover.maxUploadBytes).toBeLessThanOrEqual(262144);
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + 8.5 / limits.dpr, box.y + 8.5 / limits.dpr);
  const row = origin[1]! + 8, column = origin[0]! + 8;
  await expect(page.locator('.inspection-readout')).toHaveText(`row ${row} · column ${column}${value(row * 1536 + column)}`);
  const selected = await pixel(page, 8, 8);
  expect(Math.abs(luminance(neutral) - luminance(selected))).toBeLessThan(.0045);
  expect(await metrics(page)).toMatchObject({ uploads: beforeHover.uploads, createdTextures: beforeHover.createdTextures });
  const neighborhood = page.getByLabel('9 by 9 tensor neighborhood');
  expect(await neighborhood.evaluate((c) => [(c as HTMLCanvasElement).width, (c as HTMLCanvasElement).height])).toEqual([9, 9]);
  const center = await neighborhood.evaluate((c) => [...(c as HTMLCanvasElement).getContext('2d')!.getImageData(4, 4, 1, 1).data]);
  center.forEach((v, i) => expect(Math.abs(v - selected[i])).toBeLessThanOrEqual(1));
  const comparison = await page.evaluate(() => {
    const c = document.querySelector<HTMLCanvasElement>('[aria-label="9 by 9 tensor neighborhood"]')!;
    const bytes = c.getContext('2d')!.getImageData(0, 0, 9, 9).data;
    let maximumDifference = 0;
    for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
      const expected = (window as any).__acceptance.pixel('.matrix-scroll canvas', x + 4, y + 4);
      for (let channel = 0; channel < 4; channel++) maximumDifference = Math.max(maximumDifference,
        Math.abs(bytes[(y * 9 + x) * 4 + channel]! - expected[channel]));
    }
    return maximumDifference;
  });
  expect(comparison).toBeLessThanOrEqual(1);
  await page.screenshot({ path: testInfo.outputPath('matrix-inspection.png') });
  await testInfo.attach('measurements', { body: JSON.stringify({ browser: context.browser()!.version(), backendDevice: 'cpu', limits, firstUploadMs: first.firstUpload - started,
    firstPopulatedRenderMs: first.firstRender - started, completionMs: ended - started,
    producerHeldUntilMs: beforeRelease - started, resources: await metrics(page) }, null, 2), contentType: 'application/json' });
  // Repeated real navigation must release GL allocations, readers and CPU owners.
  for (const name of ['model.norm.weight', 'model.layers.0.mlp.up_proj.weight', 'model.embed_tokens.weight']) {
    await page.getByRole('button', { name: new RegExp(name.replaceAll('.', '\\.')) }).click();
    await complete(page);
    const dimensions = name === 'model.norm.weight' ? [576, 1] : name.includes('up_proj') ? [576, 1536] : [576, 1025];
    expect(await canvas.evaluate((c) => {
      const extent = c.closest('.matrix-scroll')!.firstElementChild as HTMLElement;
      return [parseFloat(extent.style.width) * devicePixelRatio, parseFloat(extent.style.height) * devicePixelRatio];
    })).toEqual(dimensions);
    if (name === 'model.norm.weight') {
      expect(await canvas.evaluate((c) => (c as HTMLCanvasElement).height)).toBe(1);
      await expect(page.locator('.row-distributions')).toHaveCount(0);
    }
    await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
    const cdp = await context.newCDPSession(page);
    await cdp.send('HeapProfiler.collectGarbage');
    await cdp.detach();
    await expect.poll(async () => (await metrics(page)).textures).toBe(0);
    await expect.poll(async () => (await metrics(page)).readers).toBe(0);
    expect((await metrics(page)).arrays).toEqual([]);
    await idle();
    await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  }
  await page.getByRole('button', { name: 'Close session', exact: true }).click();
  await idle();
});

test('live tokenizer uses real Unicode IDs/spans and suppresses delayed old responses', async ({ page }, testInfo) => {
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
  const current = await page.locator('.token-ids').allTextContents();
  release();
  await page.unrouteAll({ behavior: 'wait' });
  await expect(page.locator('.token-ids')).toHaveText(current);
  const persisted = await page.evaluate(() => sessionStorage.getItem(Object.keys(sessionStorage)[0]!));
  await page.reload();
  await expect(page.locator('.session-feedback')).toHaveAttribute('data-state', 'ready');
  expect(await page.evaluate(() => sessionStorage.getItem(Object.keys(sessionStorage)[0]!))).toBe(persisted);
});

test('cancel and network disconnect preserve incomplete status and return resources to baseline', async ({ page }) => {
  for (const [name, disconnect] of [[matrix, false], ['model.layers.0.mlp.up_proj.weight', true]] as const) {
    await control('arm', { kind: 'logical_tensor' });
    if (name === matrix) await open(page, name);
    else await page.getByRole('button', { name: new RegExp(name.replaceAll('.', '\\.')) }).click();
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
  await page.getByRole('button', { name: 'Close session', exact: true }).click();
  await expect.poll(async () => (await metrics(page)).textures).toBe(0);
  await expect.poll(async () => (await metrics(page)).readers).toBe(0);
});


test('real producer errors are distinct from cancellation in both primary and auxiliary UI results', async ({ page }) => {
  await control('arm', { kind: 'tensor_statistics', mode: 'pre-meta-error' });
  await open(page);
  await expect(page.locator('[data-result=statistics]')).toHaveAttribute('data-state', 'failed');
  await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'complete');
  await expect(page.locator('[data-result=distributions]')).toHaveAttribute('data-state', 'complete');
  await idle();
  const before = Object.keys((await control()).artifacts);
  await control('arm', { kind: 'logical_tensor', mode: 'midstream-error' });
  await page.getByRole('button', { name: /model\.layers\.0\.mlp\.up_proj\.weight/ }).click();
  await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'streaming');
  await control('release', {});
  await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'failed');
  await idle();
  expect(Object.keys((await control()).artifacts)).toEqual(before);
});

test('local reference Base opens normalization, both MLP orientations and embedding with exact samples', async ({ page }, testInfo) => {
  test.setTimeout(300_000);
  const models = await (await fetch(`${backend}/models`)).json();
  const model = models.models.find((m: any) => /SmolLM2-135M/.test(m.id) && !/instruct/i.test(m.id));
  expect(model).toBeTruthy();
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(model.id);
  for (const tensor of referenceSamples.selected) {
    await page.getByRole('button', { name: new RegExp(tensor.name.replaceAll('.', '\\.')) }).click();
    await expect.poll(async () => {
      const state = await page.locator('[data-result=tensor]').getAttribute('data-state');
      if (state === 'failed') throw new Error(`Reference tensor failed: ${JSON.stringify(await metrics(page))} ${log}`);
      return state;
    }, { timeout: 180_000 }).toBe('complete');
    if (tensor.shape.length === 2) {
      const canvas = page.locator('.matrix-scroll canvas');
      await canvas.scrollIntoViewIfNeeded();
      const box = (await canvas.boundingBox())!;
      const dpr = await page.evaluate(() => devicePixelRatio);
      await page.mouse.move(box.x + .5 / dpr, box.y + .5 / dpr);
      await expect(page.locator('.inspection-readout')).toHaveText(`row 0 · column 0${tensor.samples[0].value}`);
    } else expect(await page.locator('.matrix-scroll canvas').evaluate((c) => (c as HTMLCanvasElement).height)).toBe(1);
  }
  await testInfo.attach('reference-webgl', { body: JSON.stringify(await page.evaluate(() => (window as any).__acceptance.limits())), contentType: 'application/json' });
  await testInfo.attach('actual-reference-descriptors-and-samples', {
    body: JSON.stringify(referenceSamples), contentType: 'application/json',
  });
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await page.getByRole('textbox', { name: 'Prompt', exact: true }).fill('A😀e\u0301 café 你好');
  await expect(page.locator('.tokenizer-status')).toContainText('current prompt');
});
