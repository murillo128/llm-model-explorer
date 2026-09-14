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
import { nativeCamera } from '../tests/native-camera';
import { revealTensor } from '../tests/tensor-tree-helpers';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const componentPort = Number(process.env.UI_TEST_PORT ?? 4173);
const isolatedPorts = Boolean(process.env.UI_TEST_PORT);
function acceptancePorts(project: string) {
  if (project === 'dpr2') return {
    ui: isolatedPorts ? componentPort + 4 : 4177,
    backend: isolatedPorts ? componentPort + 5 : 8767,
  };
  return {
    ui: isolatedPorts ? componentPort + 2 : 4175,
    backend: isolatedPorts ? componentPort + 3 : 8765,
  };
}
let backend = 'http://127.0.0.1:8765';
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
  await (await revealTensor(page.getByRole('button', { includeHidden: true, name: new RegExp(name.replaceAll('.', '\\.')) }))).click();
}
async function complete(page: Page) {
  await expect(page.locator('[data-result=tensor]')).toHaveCount(0);
  await expect(page.locator('[data-result=statistics]')).toHaveCount(0);
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
  const ports = acceptancePorts(testInfo.project.name);
  backend = `http://127.0.0.1:${ports.backend}`;
  const uiOrigin = `http://127.0.0.1:${ports.ui}`;
  const isReference = testInfo.title.startsWith('local reference');
  test.skip(isReference && !process.env.LMEX_REFERENCE_MODEL_DIR,
    'LMEX_REFERENCE_MODEL_DIR not supplied; local SmolLM2-135M Base UI not tested');
  let command = ['-m', 'acceptance.server', '--port', String(ports.backend), '--origin', uiOrigin];
  if (isReference) {
    const directory = process.env.LMEX_REFERENCE_MODEL_DIR!;
    referenceSamples = JSON.parse(execFileSync(`${repo}backend/.venv/bin/python`,
      ['-m', 'acceptance.reference', directory], { cwd: repo, encoding: 'utf8' }));
    referenceRoot = mkdtempSync(join(tmpdir(), 'lmex-reference-'));
    command = ['-m', 'llm_model_explorer', '--model-root', dirname(directory),
      '--cache-dir', referenceRoot, '--port', String(ports.backend), '--cors-origin', uiOrigin,
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
  // First pixel read uses the green scalar family; pending data has its own color.
  expect(known[1]).toBeGreaterThan(known[0]);
  expect(known[1]).toBeGreaterThan(known[2]);
  const beforeRelease = await page.evaluate(() => performance.now());
  expect(first.firstRender).toBeLessThan(beforeRelease);
  await control('release', {});
  await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
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
  // Reserve room for the full card at DPR 1. At DPR 2 the original pane
  // already fits it and retains the vertical overflow asserted below.
  if (limits.dpr === 1) await page.setViewportSize({ width: 1000, height: 700 });
  await canvas.scrollIntoViewIfNeeded();
  await page.locator('.matrix-scroll').evaluate((node) => { node.scrollLeft = 53; node.scrollTop = 27; });
  await expect.poll(async () => canvas.getAttribute('data-origin')).not.toBe('0,0');
  const origin = (await canvas.getAttribute('data-origin'))!.split(',').map(Number);
  await expect(page.locator('.row-distributions canvas')).toHaveAttribute('data-origin', `0,${origin[1]}`);
  await expect(page.locator('.column-distributions canvas')).toHaveAttribute('data-origin', `${origin[0]},0`);
  expect(origin[0]).toBeGreaterThan(0);
  expect(origin[1]).toBeGreaterThan(0);
  const neutral = await pixel(page, 8, 8);
  const neutralNeighborhood = await page.evaluate(() => Array.from({ length: 81 }, (_, i) =>
    (window as any).__acceptance.pixel('.matrix-scroll canvas', i % 9 + 4, Math.floor(i / 9) + 4)));
  const profileBefore = await page.evaluate(() => ['.row-distributions canvas', '.column-distributions canvas']
    .map(selector => (window as any).__acceptance.pixel(selector, 8, 8)));
  for (const rgb of profileBefore) {
    expect(rgb[1]).toBeGreaterThan(rgb[0]); expect(rgb[1]).toBeGreaterThan(rgb[2]);
  }
  const scalarSamples: number[][] = await page.evaluate(() => Array.from({ length: 128 }, (_, x) =>
    (window as any).__acceptance.pixel('.matrix-scroll canvas', x, 0)));
  const ordered = scalarSamples.map((rgb, x) => ({ scalar: value(origin[1]! * 1536 + origin[0]! + x), rgb }))
    .sort((a, b) => a.scalar - b.scalar);
  for (let i = 1; i < ordered.length; i++) expect(luminance(ordered[i]!.rgb)).toBeGreaterThanOrEqual(luminance(ordered[i - 1]!.rgb));
  expect(luminance(ordered.at(-1)!.rgb) - luminance(ordered[0]!.rgb)).toBeGreaterThan(.7);
  const beforeHover = await metrics(page);
  expect(beforeHover.textures).toBe(3);
  expect(beforeHover.gpuBytes).toBe(4 * (576 * 1536 + (576 + 1536) * 100));
  expect(beforeHover.maxUploadBytes).toBeLessThanOrEqual(262144);
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + 8.5 / limits.dpr, box.y + 8.5 / limits.dpr);
  const row = origin[1]! + 8, column = origin[0]! + 8;
  await expect(page.locator('.inspection-readout')).toHaveText(`row ${row} · column ${column}${value(row * 1536 + column)}`);
  const selected = await pixel(page, 8, 8);
  expect(neutral[1]).toBeGreaterThan(neutral[0]);
  expect(selected[0] - selected[1]).toBeGreaterThan(30);
  expect(luminance(selected)).toBeGreaterThan(0);
  const profileAfter = await page.evaluate(() => ['.row-distributions canvas', '.column-distributions canvas']
    .map(selector => (window as any).__acceptance.pixel(selector, 8, 8)));
  for (let i = 0; i < 2; i++) {
    expect(profileAfter[i]).not.toEqual(profileBefore[i]);
    expect(profileAfter[i][0] - profileAfter[i][2]).toBeGreaterThan(60);
  }
  expect(await metrics(page)).toMatchObject({ uploads: beforeHover.uploads, createdTextures: beforeHover.createdTextures });
  const neighborhood = page.getByLabel('9 by 9 matrix neighborhood');
  expect(await neighborhood.evaluate((c) => [(c as HTMLCanvasElement).width, (c as HTMLCanvasElement).height])).toEqual([9, 9]);
  const center = await neighborhood.evaluate((c) => [...(c as HTMLCanvasElement).getContext('2d')!.getImageData(4, 4, 1, 1).data]);
  center.forEach((v, i) => expect(Math.abs(v - neutral[i])).toBeLessThanOrEqual(1));
  const comparison = await page.evaluate((neutralNeighborhood) => {
    const c = document.querySelector<HTMLCanvasElement>('[aria-label="9 by 9 matrix neighborhood"]')!;
    const bytes = c.getContext('2d')!.getImageData(0, 0, 9, 9).data;
    let maximumDifference = 0;
    for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
      const expected = neutralNeighborhood[y * 9 + x];
      for (let channel = 0; channel < 4; channel++) maximumDifference = Math.max(maximumDifference,
        Math.abs(bytes[(y * 9 + x) * 4 + channel]! - expected[channel]));
    }
    return maximumDifference;
  }, neutralNeighborhood);
  expect(comparison).toBeLessThanOrEqual(1);
  await page.screenshot({ path: testInfo.outputPath('matrix-inspection.png') });
  await testInfo.attach('measurements', { body: JSON.stringify({ browser: context.browser()!.version(), viewport: page.viewportSize(), backendDevice: 'cpu', limits, firstUploadMs: first.firstUpload - started,
    firstPopulatedRenderMs: first.firstRender - started, completionMs: ended - started,
    producerHeldUntilMs: beforeRelease - started, resources: await metrics(page) }, null, 2), contentType: 'application/json' });
  // Repeated real navigation must release GL allocations, readers and CPU owners.
  for (const name of ['model.norm.weight', 'model.layers.0.mlp.up_proj.weight', 'model.embed_tokens.weight']) {
    await (await revealTensor(page.getByRole('button', { includeHidden: true, name: new RegExp(name.replaceAll('.', '\\.')) }))).click();
    await complete(page);
    await nativeCamera(page);
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
    // The real empty prompt includes BOS, so the embedding view owns one row.
    await expect(page.getByText('[1 × 576] · float32')).toBeVisible();
    await expect.poll(async () => (await metrics(page)).textures).toBe(1);
    await expect.poll(async () => (await metrics(page)).readers).toBe(0);
    const cdp = await context.newCDPSession(page);
    await cdp.send('HeapProfiler.collectGarbage');
    await cdp.detach();
    expect((await metrics(page)).arrays).toEqual([576 * 4]);
    await idle();
    await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  }
  await page.getByRole('button', { name: 'Session options', exact: true }).click();
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
  await embeddingDone(page, 8);
  const current = await page.locator('.token-ids').allTextContents();
  const settled = await metrics(page);
  release();
  await page.unrouteAll({ behavior: 'wait' });
  await expect(page.locator('.token-ids')).toHaveText(current);
  expect(await metrics(page)).toMatchObject({ uploads: settled.uploads, createdTextures: settled.createdTextures });
  const persisted = await page.evaluate(() => sessionStorage.getItem(Object.keys(sessionStorage)[0]!));
  await page.reload();
  await expect(page.getByRole('contentinfo').getByRole('status')).toHaveAttribute('data-state', 'ready');
  expect(await page.evaluate(() => sessionStorage.getItem(Object.keys(sessionStorage)[0]!))).toBe(persisted);
});

test('cancel and network disconnect preserve incomplete status and return resources to baseline', async ({ page }) => {
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


test('real producer errors are distinct from cancellation in both primary and auxiliary UI results', async ({ page }) => {
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

test('local reference Base opens normalization, both MLP orientations and embedding with exact samples', async ({ page }, testInfo) => {
  test.setTimeout(300_000);
  const models = await (await fetch(`${backend}/models`)).json();
  const model = models.models.find((m: any) => /SmolLM2-135M/.test(m.id) && !/instruct/i.test(m.id));
  expect(model).toBeTruthy();
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(model.id);
  for (const tensor of referenceSamples.selected) {
    await (await revealTensor(page.getByRole('button', { includeHidden: true, name: new RegExp(tensor.name.replaceAll('.', '\\.')) }))).click();
    await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
    await expect(page.locator('[data-result=tensor]')).toHaveCount(0, { timeout: 180_000 });
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

// All routes below still reach the production service. Barriers delay actual
// responses; no numeric/tokenizer response is synthesized in this suite.
async function tokenizer(page: Page) {
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('acceptance/fixture');
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await expect(page.getByText('[1 × 576] · float32')).toBeVisible();
  return page.getByRole('textbox', { name: 'Prompt', exact: true });
}
async function closeSession(page: Page) {
  await page.getByRole('button', { name: 'Session options', exact: true }).click();
  await page.getByRole('button', { name: 'Close session', exact: true }).click();
  await idle();
  await expect.poll(async () => (await metrics(page)).textures).toBe(0);
  await expect.poll(async () => (await metrics(page)).readers).toBe(0);
}
async function embeddingDone(page: Page, rows: number) {
  await expect(page.getByText(`[${rows} × 576] · float32`)).toBeVisible();
  await expect(page.locator('.input-embeddings .matrix-panel-status')).toBeEmpty();
}
async function documentFits(page: Page) {
  expect(await page.evaluate(() => ({
    document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
    body: [document.body.scrollWidth, document.body.scrollHeight], scroll: [scrollX, scrollY],
  }))).toEqual({ document: [page.viewportSize()!.width, page.viewportSize()!.height],
    body: [page.viewportSize()!.width, page.viewportSize()!.height], scroll: [0, 0] });
}

test('real ordered embeddings render progressively with exact duplicate rows and linked annotations', async ({ page, context }, testInfo) => {
  const input = await tokenizer(page);
  const readsBefore = (await control()).source_reads.row_elements;
  await page.evaluate(() => { (window as any).__acceptance.captureScalars = true; });
  await control('arm', { kind: 'input_embeddings' });
  const text = 'A😀A';
  const tokenized = page.waitForResponse(r => r.url().endsWith('/tokenize') && r.request().postDataJSON().text === text);
  const requested = page.waitForRequest(r => r.url().endsWith('/embeddings'));
  await input.fill(text);
  const result = await (await tokenized).json();
  const ids: number[] = result.tokens.map((t: any) => t.id);
  expect((await requested).postDataJSON()).toEqual({ token_ids: ids });
  expect(ids[1]).toBe(ids.at(-1));
  await expect(page.getByText('Streaming input embeddings…')).toBeVisible();
  await expect.poll(async () => (await control()).control.entered).toBe(true);
  const prefix = await page.evaluate(() => [...(window as any).__acceptance.scalarValues]);
  expect(prefix.length).toBeGreaterThan(0); expect(prefix.length).toBeLessThan(ids.length * 576);
  const scroller = page.locator('.matrix-scroll');
  await scroller.focus();
  await expect(page.locator('.inspection-readout')).toHaveText(`row 0 · column 0${value(ids[0]! * 576)}`);
  expect((await control()).control.released).toBe(false);
  await control('release', {});
  await embeddingDone(page, ids.length);
  const uploaded = await page.evaluate(() => [...(window as any).__acceptance.scalarValues]);
  expect(uploaded).toEqual(ids.flatMap(id => Array.from({ length: 576 }, (_, col) => value(id * 576 + col))));
  const reads = (await control()).source_reads;
  expect(reads.row_elements - readsBefore).toBe(ids.length * 576);
  expect(reads.full_tensors).toEqual([]);
  expect((await metrics(page)).gpuBytes).toBe(ids.length * 576 * 4);
  await expect(page.locator('.row-distributions, .column-distributions')).toHaveCount(0);
  const before = await metrics(page);
  for (let row = 0; row < ids.length; row++) {
    if (row) await scroller.press('ArrowDown');
    await expect(page.locator('.inspection-readout')).toHaveText(`row ${row} · column 0${value(ids[row]! * 576)}`);
    await expect(page.locator(`[data-token-index="${row}"]`)).toHaveAttribute('data-active-token', '');
  }
  await page.locator('[data-token-index="1"]').hover();
  await expect(page.locator('[data-token-index="1"]')).toHaveAttribute('data-active-token', '');
  await page.locator(`[data-token-index="${ids.length - 1}"]`).focus();
  await expect(page.locator('[data-token-index][data-active-token]')).toHaveCount(1);
  expect(await metrics(page)).toMatchObject({ uploads: before.uploads, createdTextures: before.createdTextures });
  await expect(page.getByRole('textbox')).toHaveCount(1);
  await documentFits(page);
  await page.screenshot({ path: testInfo.outputPath('tokenizer-embeddings.png') });
  await testInfo.attach('measurements', { body: JSON.stringify({ scenario: 'ordered embeddings',
    viewport: page.viewportSize(), limits: await page.evaluate(() => (window as any).__acceptance.limits()),
    tokenIds: ids, prefixElements: prefix.length, totalElements: uploaded.length, sourceReads: reads,
    resources: await metrics(page) }), contentType: 'application/json' });
  await closeSession(page);
  const cdp = await context.newCDPSession(page); await cdp.send('HeapProfiler.collectGarbage'); await cdp.detach();
  expect((await metrics(page)).arrays).toEqual([]);
});

test('real A→B→A response reordering and model/session changes show only the latest generation', async ({ page }) => {
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
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
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
  await expect(page.getByText('Streaming input embeddings…')).toBeVisible();
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

test('repeated prompt edits and explorer unmounts cancel real embedding readers and GPU owners', async ({ page, context }) => {
  for (let cycle = 0; cycle < 3; cycle++) {
    const input = await tokenizer(page);
    await control('arm', { kind: 'input_embeddings' });
    await input.fill(`cancel-${cycle}😀`);
    await expect(page.getByText('Streaming input embeddings…')).toBeVisible();
    await expect.poll(async () => (await control()).control.entered).toBe(true);
    await input.fill(`replacement-${cycle}`);
    await expect(page.locator('.matrix-scroll')).toHaveCount(0);
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

test.describe('production native pane geometry', () => {
  for (const viewport of [{ width: 1000, height: 700 }, { width: 390, height: 640 }]) {
    test(`compact shell, navigation and four overflow modes at ${viewport.width}×${viewport.height}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      await open(page, 'layout.fits.weight');
      await expect(page.getByRole('banner')).toHaveCount(1);
      await expect(page.getByRole('contentinfo')).toHaveCount(1);
      expect((await page.getByRole('banner').boundingBox())!.height).toBeLessThanOrEqual(56);
      expect((await page.getByRole('contentinfo').boundingBox())!.height).toBeLessThanOrEqual(32);
      const refreshed = page.waitForResponse(r => r.url().endsWith('/models'));
      await page.getByRole('button', { name: 'Refresh models', exact: true }).click(); await refreshed;
      await expect(page.locator('.tensor-identity')).toHaveCount(1);
      await expect(page.getByRole('heading', { level: 1 })).toHaveCount(0);
      await expect(page.locator('.tensor-leaf-name')).toHaveText(Array(90).fill('weight'));
      await expect(page.locator('.tensor-choice summary, .tensor-choice details')).toHaveCount(0);
      const info = page.getByRole('button', { name: 'Tensor information' });
      await info.focus(); await info.press('Enter');
      await expect(page.getByRole('dialog', { name: 'Tensor information' })).toContainText('Logical dtype');
      await page.keyboard.press('Escape'); await expect(info).toBeFocused();
      await expect(page.getByText(/One value per device pixel/)).toHaveCount(0);
      const evidence = [];
      for (const [name, horizontal, vertical] of [['fits', false, false], ['tall', false, true], ['wide', true, false], ['both', true, true]] as const) {
        const leaf = page.getByRole('button', { includeHidden: true, name: new RegExp(`^layout\\.${name}\\.weight`) });
        await revealTensor(leaf); await leaf.focus(); await leaf.press('Enter'); await expect(leaf).toHaveAttribute('aria-pressed', 'true');
        await complete(page); await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
        await nativeCamera(page);
        const geometry = () => page.evaluate(() => {
          const m = document.querySelector<HTMLElement>('.matrix-scroll')!;
          const rect = (s: string) => { const r = document.querySelector(s)!.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; };
          return { horizontal: m.scrollWidth > m.clientWidth, vertical: m.scrollHeight > m.clientHeight,
            gutter: m.offsetWidth - m.clientWidth, scroll: [m.scrollLeft, m.scrollTop],
            extent: [m.scrollWidth, m.scrollHeight], client: [m.clientWidth, m.clientHeight],
            matrix: rect('.matrix-scroll canvas'), row: rect('.row-distributions canvas'), column: rect('.column-distributions canvas'),
            origins: ['.matrix-scroll canvas', '.row-distributions canvas', '.column-distributions canvas'].map(s => document.querySelector(s)!.getAttribute('data-origin')) };
        });
        await expect.poll(async () => { const g = await geometry(); return [g.horizontal, g.vertical]; }).toEqual([horizontal, vertical]);
        const initial = await geometry(); expect(initial.gutter).toBeGreaterThan(0);
        const inventory = page.getByRole('complementary', { name: 'Tensor inventory' });
        await inventory.evaluate(e => { e.scrollTop = e.scrollHeight; });
        expect(await inventory.evaluate(e => e.scrollTop)).toBeGreaterThan(0);
        expect((await geometry()).scroll).toEqual(initial.scroll);
        for (const [x, y] of [[0, 0], [1e9, 0], [0, 1e9], [1e9, 1e9]]) {
          await page.locator('.matrix-scroll').evaluate((e, [x, y]) => { e.scrollLeft = x!; e.scrollTop = y!; }, [x, y]);
          await expect.poll(async () => {
            const g = await geometry();
            return g.origins;
          }).toEqual(await page.locator('.matrix-scroll').evaluate(e => {
            const x = Math.round(e.scrollLeft * devicePixelRatio), y = Math.round(e.scrollTop * devicePixelRatio);
            return [`${x},${y}`, `0,${y}`, `${x},0`];
          }));
          const g = await geometry();
          expect(g.matrix[1]).toBe(g.row[1]); expect(g.matrix[0]).toBe(g.column[0]);
          expect(g.matrix[3]).toBe(g.row[3]); expect(g.matrix[2]).toBe(g.column[2]);
          expect(g.scroll).toEqual([x ? g.extent[0]! - g.client[0]! : 0, y ? g.extent[1]! - g.client[1]! : 0]);
          await documentFits(page); evidence.push({ name, ...g });
        }
      }
      await expect(page.locator('.matrix-panel-header')).not.toContainText('complete');
      await page.screenshot({ path: testInfo.outputPath('compact-tensor.png') });
      await testInfo.attach('measurements', { body: JSON.stringify({ scenario: 'native geometry', viewport,
        limits: await page.evaluate(() => (window as any).__acceptance.limits()), cases: evidence }), contentType: 'application/json' });
      await closeSession(page);
    });
  }
});

test('production prompt pixels, selection, history and composition survive embedding completion', async ({ page }) => {
  const editor = await tokenizer(page);
  await control('arm', { kind: 'input_embeddings' });
  await editor.fill('hello world');
  await expect(page.getByText('Streaming input embeddings…')).toBeVisible();
  const prompt = page.getByRole('region', { name: 'Live prompt tokenization' });
  const nav = page.getByRole('button', { name: 'Tokenizer Explorer', exact: true });
  await nav.focus(); await page.mouse.move(0, 0);
  const before = await prompt.screenshot({ animations: 'disabled' });
  const box = await page.locator('.tokenizer-editor').boundingBox();
  await control('release', {});
  await embeddingDone(page, 12);
  expect(await prompt.screenshot({ animations: 'disabled' })).toEqual(before);
  expect(await page.locator('.tokenizer-editor').boundingBox()).toEqual(box);
  expect(box!.height).toBe(260);
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
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  // Deliberately exceed the production debounce while the IME owns the source.
  await page.waitForTimeout(250); expect(duringComposition).toEqual([]);
  await editor.fill('日本'); await editor.dispatchEvent('compositionend', { data: '日本' });
  await expect(page.locator('.tokenizer-status')).toContainText('current prompt');
  await embeddingDone(page, 7);
  expect(duringComposition).toEqual(['日本']);
  await closeSession(page);
});
