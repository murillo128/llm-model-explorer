/* eslint-disable @typescript-eslint/no-explicit-any -- Test-only browser observer and JSON evidence. */
import { test, expect } from '@playwright/test';
import { integratedCard } from '../tests/viewer-panel';
import type { Page, TestInfo } from '@playwright/test';
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import { installProbe } from './probe';
import { camera, zoom, drag, panelGeometry, promptViewport, tokenizerGeometry, settledPrompt } from './usability';
import { nativeCamera } from '../tests/native-camera';
import { revealTensor } from '../tests/tensor-tree-helpers';
import { findComponent } from '../tests/architecture-controls';
import { makeProjectionFixture } from '../tests/architecture-projection-fixture';

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
  if (isReference && process.env.LMEX_REQUIRE_ARCHITECTURE_REFERENCES === '1') {
    expect(process.env.LMEX_REFERENCE_MODEL_DIR, 'SmolLM2 regression checkpoint is required').toBeTruthy();
  }
  test.skip(isReference && !process.env.LMEX_REFERENCE_MODEL_DIR,
    'LMEX_REFERENCE_MODEL_DIR not supplied; local SmolLM2-135M Base UI not tested');
  let command = ['-m', 'acceptance.server', '--port', String(ports.backend), '--origin', uiOrigin];
  if (testInfo.title.startsWith('expanded model coverage')) command.push('--polish');
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
  // Layout/capture cases inspect DOM, numeric readouts and screenshots. Full
  // framebuffer readback on every streamed draw is reserved for pixel oracles.
  if (isReference || testInfo.title.startsWith('polish ') || testInfo.title.startsWith('expanded model coverage')) {
    await page.addInitScript(() => { (window as any).__acceptance.capture = false; });
  }
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
    await expect.poll(async () => (await metrics(page)).textures).toBe(3);
    await expect.poll(async () => (await metrics(page)).readers).toBe(0);
    const cdp = await context.newCDPSession(page);
    await cdp.send('HeapProfiler.collectGarbage');
    await cdp.detach();
    expect((await metrics(page)).arrays.sort((a: number, b: number) => a - b)).toEqual([576 * 4, 100 * 576 * 4]);
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
    await nativeCamera(page);
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
async function tokenizer(page: Page, waitForInitial = true) {
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('acceptance/fixture');
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  if (waitForInitial) await embeddingDone(page, 1);
  return page.getByRole('textbox', { name: 'Prompt', exact: true });
}
async function closeSession(page: Page) {
  await page.getByRole('button', { name: 'Session options', exact: true }).click();
  await page.getByRole('button', { name: 'Close session', exact: true }).click();
  await idle();
  await expect.poll(async () => (await metrics(page)).textures).toBe(0);
  await expect.poll(async () => (await metrics(page)).readers).toBe(0);
}
async function embeddingDone(page: Page, rows: number, timeout = 15_000) {
  await expect(page.getByText(`[${rows} × 576] · float32`).filter({ visible: true })).toBeVisible({ timeout });
  await expect(page.locator('.input-embeddings [data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
  await expect(page.locator('.input-embeddings .embedding-layer:not([data-staging]) .matrix-panel-status')).toBeEmpty();
}
async function documentFits(page: Page) {
  expect(await page.evaluate(() => ({
    document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
    body: [document.body.scrollWidth, document.body.scrollHeight], scroll: [scrollX, scrollY],
  }))).toEqual({ document: [page.viewportSize()!.width, page.viewportSize()!.height],
    body: [page.viewportSize()!.width, page.viewportSize()!.height], scroll: [0, 0] });
}

test('real ordered embeddings render progressively with exact duplicate rows and linked annotations', async ({ page, context }, testInfo) => {
  // First delivery must be usable progressively, before any completed matrix exists.
  await control('arm', { kind: 'input_embeddings' });
  const input = await tokenizer(page, false);
  await expect(page.getByRole('status').filter({ hasText: 'Streaming input embeddings…' })).toBeVisible();
  await expect.poll(async () => (await control()).control.entered).toBe(true);
  await page.locator('.matrix-scroll').focus();
  const firstId = Number(await page.locator('[data-token-index="0"]').textContent());
  await expect(page.locator('.inspection-readout')).toHaveText(`row 0 · column 0${value(firstId * 576)}`);
  expect((await control()).control.released).toBe(false);
  await control('release', {}); await embeddingDone(page, 1);
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
  await expect(page.getByRole('status').filter({ hasText: 'Streaming input embeddings…' })).toBeVisible();
  await expect.poll(async () => (await control()).control.entered).toBe(true);
  const prefix = await page.evaluate(() => [...(window as any).__acceptance.scalarValues]);
  expect(prefix.length).toBeGreaterThan(0); expect(prefix.length).toBeLessThan(ids.length * 576);
  const scroller = page.locator('.matrix-scroll:visible');
  await scroller.focus();
  await expect(page.locator('.inspection-readout')).toHaveText(`row 0 · column 0${value(ids[0]! * 576)}`);
  expect((await control()).control.released).toBe(false);
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
  await expect(page.locator('[data-token-index][data-active-token]')).toHaveCount(0);
  await expect(page.locator('[data-staging] .matrix-scroll')).toHaveCount(1);
  await control('release', {});
  await embeddingDone(page, ids.length);
  await scroller.focus();
  const uploaded = await page.evaluate(() => [...(window as any).__acceptance.scalarValues]);
  expect(uploaded).toEqual(ids.flatMap(id => Array.from({ length: 576 }, (_, col) => value(id * 576 + col))));
  const reads = (await control()).source_reads;
  expect(reads.row_elements - readsBefore).toBe(3 * ids.length * 576);
  expect(reads.full_tensors).toEqual([]);
  expect((await metrics(page)).gpuBytes).toBe((ids.length * 576 + (ids.length + 576) * 100) * 4);
  await expect(page.locator('.row-distributions, .column-distributions')).toHaveCount(2);
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

for (const family of ['qwen3', 'qwen3_5']) test(`real ${family} input embeddings preserve token linkage and recover after unsupported model`, async ({ page }) => {
  const input = await tokenizer(page);
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(`acceptance/${family}`);
  await embeddingDone(page, 1);
  const tokenized = page.waitForResponse(r => r.url().endsWith('/tokenize') && r.request().postDataJSON().text === 'AA');
  await input.fill('AA');
  const result = await (await tokenized).json();
  await embeddingDone(page, result.tokens.length);
  expect(result.tokens[1].id).toBe(result.tokens[2].id);
  const matrix = page.locator('.matrix-scroll');
  await matrix.focus();
  for (let row = 0; row < result.tokens.length; row++) {
    if (row) await matrix.press('ArrowDown');
    await expect(page.locator('.inspection-readout')).toHaveText(`row ${row} · column 0${value(result.tokens[row].id * 576)}`);
    await expect(page.locator(`[data-token-index="${row}"]`)).toHaveAttribute('data-active-token', '');
  }
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('acceptance/unsupported');
  await expect(page.getByText(/Input embeddings are unavailable/)).toBeVisible();
  await input.fill('still usable');
  await expect(page.locator('.tokenizer-status')).toContainText('current prompt');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(`acceptance/${family}`);
  await expect(page.locator('.embedding-shape')).toContainText('× 576] · float32');
  await expect(page.getByText(/Input embeddings are unavailable/)).toHaveCount(0);
  await closeSession(page);
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

test('repeated prompt edits and explorer unmounts cancel real embedding readers and GPU owners', async ({ page, context }) => {
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
      await expect(page.locator('.tensor-leaf-name')).toHaveText(Array(95).fill('weight'));
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
        const initial = await geometry(); expect(initial.gutter).toBe(0);
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

test('integrated inventory preferences and metadata preserve streaming panel geometry', async ({ page }, testInfo) => {
  const name = 'layout.fits.weight';
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('acceptance/fixture');
  await expect(page.locator('.tensor-tree').first()).toBeVisible();
  expect(await page.locator('.tensor-tree details').evaluateAll(nodes => nodes.every(node =>
    node.hasAttribute('open') === !node.parentElement!.closest('details')))).toBe(true);
  await control('arm', { kind: 'logical_tensor' });
  await open(page, name);
  await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'streaming');
  const initial = await panelGeometry(page);
  expect(initial['.matrix-panel-header']![3]).toBe(40);
  const cardInset = await page.locator('.viewer-panel-body').evaluate(node => {
    const style = getComputedStyle(node);
    return parseFloat(style.borderTopWidth) + parseFloat(style.paddingTop);
  });
  expect(initial['.matrix-surfaces']![1]).toBe(initial['.matrix-panel-header']![1]! + initial['.matrix-panel-header']![3]! + cardInset);
  await expect(page.locator('.matrix-panel-header')).toHaveCount(1);
  await expect(page.locator('.distribution-range')).toHaveCount(0);
  await expect(page.getByText('Full-range bins', { exact: true })).toHaveCount(0);
  const info = page.getByRole('button', { name: 'Tensor information', exact: true });
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  const close = page.getByRole('button', { name: 'Close tensor information' });
  await info.hover(); await expect(dialog).toBeVisible(); await expect(close).toHaveCount(0);
  const icon = (await info.boundingBox())!;
  expect((await dialog.boundingBox())!.y).toBe(icon.y + icon.height);
  await info.click(); await expect(close).toBeFocused();
  await page.mouse.move(0, 0); await expect(dialog).toBeVisible();
  await expect(dialog.locator('dt')).toHaveText(['Logical path', 'Rank', 'Elements', 'Storage dtype', 'Storage format', 'Logical dtype',
    'Distribution domain', 'True finite minimum', 'True finite maximum']);
  expect(await panelGeometry(page)).toEqual(initial);
  await page.keyboard.press('Escape'); await expect(info).toBeFocused();
  await control('release', {}); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  expect(await panelGeometry(page)).toEqual(initial);
  const canvas = await page.locator('.matrix-scroll canvas').elementHandle();
  const resources = await metrics(page);
  const pane = page.getByRole('region', { name: 'Tensor Explorer workspace', exact: true });
  const inventory = page.getByRole('complementary', { name: 'Tensor inventory' });
  const before = (await pane.boundingBox())!, width = (await inventory.boundingBox())!.width;
  if (process.env.CAPTURE_INVENTORY_EVIDENCE) await page.screenshot({ path: `evidence/inventory-matrix-expanded-${testInfo.project.name}.png` });
  await page.getByRole('button', { name: 'Collapse inventory' }).click();
  await expect(page.getByRole('button', { name: 'Expand inventory' })).toBeFocused();
  expect((await pane.boundingBox())!.width - before.width).toBe(width + 16 - 40);
  const workspace = (await page.locator('#workspace').boundingBox())!;
  expect(await pane.boundingBox()).toEqual({ ...workspace, x: workspace.x + 40, width: workspace.width - 40 });
  if (process.env.CAPTURE_INVENTORY_EVIDENCE) {
    await page.getByRole('button', { name: 'Expand inventory' }).blur(); await page.mouse.move(0, 0);
    await page.screenshot({ path: `evidence/inventory-matrix-collapsed-${testInfo.project.name}.png` });
  }
  await page.getByRole('button', { name: 'Expand inventory' }).click();
  const divider = page.getByRole('separator', { name: 'Resize tensor inventory' });
  await divider.focus(); await divider.press('End');
  await expect(divider).toHaveAttribute('aria-valuenow', '480');
  expect(await canvas!.evaluate(node => node === document.querySelector('.matrix-scroll canvas'))).toBe(true);
  expect(await metrics(page)).toMatchObject({ uploads: resources.uploads, createdTextures: resources.createdTextures });
  await page.getByRole('button', { name: 'Collapse inventory' }).click();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Expand inventory' })).toBeVisible();
  await page.getByRole('button', { name: 'Expand inventory' }).click();
  await expect(divider).toHaveAttribute('aria-valuenow', '480');
  await expect(page.getByRole('button', { name: new RegExp(`^${name}`) })).toBeVisible();
  await documentFits(page);
  await testInfo.attach('measurements', { body: JSON.stringify({ initial, reclaimedWidth: width + 16 - 40, restoredWidth: 480 }), contentType: 'application/json' });
  await closeSession(page);
});

test('production matrix navigation centers underfilled data and links zoom selection across fixed profile tracks', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await open(page, 'layout.fits.weight'); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  await nativeCamera(page);
  const before = await camera(page, 32, 32), resources = await metrics(page);
  expect(before.scaleX).toBe(1); expect(before.scaleY).toBe(1);
  const geometry = () => page.evaluate(() => {
    const host = document.querySelector<HTMLElement>('.matrix-scroll')!;
    const scientific = document.querySelector<HTMLElement>('.matrix-surfaces')!;
    const box = (selector: string) => document.querySelector(selector)!.getBoundingClientRect().toJSON();
    const h = host.getBoundingClientRect(), style = getComputedStyle(scientific);
    return { matrix: box('.matrix-scroll canvas'), rows: box('.row-distributions canvas'), columns: box('.column-distributions canvas'),
      viewport: { left: h.left + host.clientLeft, top: h.top + host.clientTop, width: host.clientWidth, height: host.clientHeight },
      scroll: { width: host.scrollWidth, height: host.scrollHeight }, gap: { x: parseFloat(style.columnGap), y: parseFloat(style.rowGap) } };
  });
  const underfilled = await geometry();
  expect(underfilled.matrix.width * before.dpr).toBe(32);
  expect(underfilled.matrix.height * before.dpr).toBe(32);
  expect(underfilled.viewport.width).toBeGreaterThan(underfilled.matrix.width * 10);
  expect(underfilled.viewport.height).toBeGreaterThan(underfilled.matrix.height * 10);
  for (const [axis, size] of [['left', 'width'], ['top', 'height']] as const) {
    const centered = underfilled.viewport[axis] + (underfilled.viewport[size] - underfilled.matrix[size]) / 2;
    expect(Math.abs(underfilled.matrix[axis] - centered)).toBeLessThanOrEqual(1 / before.dpr);
    expect(underfilled.scroll[size]).toBe(underfilled.viewport[size]);
  }
  expect(underfilled.rows.top).toBe(underfilled.matrix.top);
  expect(underfilled.rows.height).toBe(underfilled.matrix.height);
  expect(underfilled.columns.left).toBe(underfilled.matrix.left);
  expect(underfilled.columns.width).toBe(underfilled.matrix.width);
  expect(underfilled.rows.left - underfilled.viewport.left - underfilled.viewport.width).toBeCloseTo(underfilled.gap.x, 1);
  expect(underfilled.columns.top - underfilled.viewport.top - underfilled.viewport.height).toBeCloseTo(underfilled.gap.y, 1);
  expect(underfilled.rows.width * before.dpr).toBe(100);
  expect(underfilled.columns.height * before.dpr).toBe(100);
  await page.mouse.move(0, 0);
  if (process.env.CAPTURE_MATRIX_NAVIGATION_EVIDENCE === '1' && testInfo.project.name === 'dpr1') {
    await page.screenshot({ path: 'evidence/matrix-navigation-underfilled.png' });
  }
  const point = (c: typeof before, column: number, row: number) => ({
    x: c.rect.left + (column - c.x) * c.scaleX / c.dpr,
    y: c.rect.top + (row - c.y) * c.scaleY / c.dpr,
  });
  const bounds = { columns: [4, 24], rows: [8, 24] };
  const previews = page.locator('.matrix-zoom-preview');
  const expectPreview = async (selection: typeof bounds) => {
    await expect(previews).toHaveCount(3);
    await expect(page.locator('.matrix-zoom-preview[data-surface=matrix]')).toHaveAttribute('data-bounds', JSON.stringify(selection));
    for (const axis of ['rows', 'columns'] as const) {
      await expect(page.locator(`.matrix-zoom-preview[data-surface=${axis}]`)).toHaveAttribute('data-bounds', JSON.stringify({ [axis]: selection[axis] }));
    }
    const m = (await page.locator('.matrix-zoom-preview[data-surface=matrix]').boundingBox())!;
    const r = (await page.locator('.matrix-zoom-preview[data-surface=rows]').boundingBox())!;
    const c = (await page.locator('.matrix-zoom-preview[data-surface=columns]').boundingBox())!;
    expect(r.y).toBeCloseTo(m.y, 1); expect(r.height).toBeCloseTo(m.height, 1);
    expect(c.x).toBeCloseTo(m.x, 1); expect(c.width).toBeCloseTo(m.width, 1);
  };
  await drag(page, point(before, 4, 8), point(before, 24, 24));
  await expectPreview(bounds);
  await page.mouse.up(); await expect(previews).toHaveCount(0);
  const selected = await camera(page, 32, 32), zoomed = await geometry();
  expect(selected.scaleX).toBeCloseTo(Math.min(selected.width * before.dpr / 20, selected.height * before.dpr / 16));
  expect(selected.scaleY).toBe(selected.scaleX);
  await expect(page.locator('.row-distributions canvas')).toHaveAttribute('data-origin', `0,${selected.y}`);
  await expect(page.locator('.column-distributions canvas')).toHaveAttribute('data-origin', `${selected.x},0`);
  // A second preview proves synchronized bounds on the enlarged view. Escape
  // cancels this transient range first; the next Escape restores the prior camera.
  const inner = { columns: [8, 20], rows: [10, 22] };
  await drag(page, point(selected, 8, 10), point(selected, 20, 22));
  await expectPreview(inner);
  if (process.env.CAPTURE_MATRIX_NAVIGATION_EVIDENCE === '1' && testInfo.project.name === 'dpr1') {
    await page.screenshot({ path: 'evidence/matrix-navigation-zoomed.png' });
  }
  await page.keyboard.press('Escape'); await page.mouse.up();
  await expect(previews).toHaveCount(0);
  expect(await camera(page, 32, 32)).toEqual(selected);
  await page.locator('.matrix-scroll').focus();
  await page.keyboard.press('Escape');
  expect(await camera(page, 32, 32)).toEqual(before);
  await drag(page, point(before, 4, 8), point(before, 24, 24)); await page.mouse.up();
  await page.locator('.matrix-scroll canvas').click({ button: 'right', position: { x: 10, y: 10 } });
  expect(await camera(page, 32, 32)).toEqual(before);
  expect(await metrics(page)).toMatchObject({ uploads: resources.uploads, createdTextures: resources.createdTextures, gpuBytes: resources.gpuBytes, errors: [] });
  await testInfo.attach('matrix-navigation-geometry', { body: JSON.stringify({ underfilled, zoomed, bounds, selected, resources: await metrics(page) }, null, 2), contentType: 'application/json' });
  await documentFits(page);
  await closeSession(page);
});

test('integrated camera gestures, exact selection, aligned scales and adaptive inspection retain scalar storage', async ({ page }, testInfo) => {
  await open(page, 'layout.fits.weight'); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  let c = await camera(page, 32, 32);
  expect(c.scaleX).toBeCloseTo(Math.max(1, Math.floor(c.width * c.dpr) / 32));
  expect(c.scaleY).toBeCloseTo(c.scaleX);
  await open(page); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  await nativeCamera(page);
  expect((await camera(page, 576, 1536)).scaleX).toBe(1);
  const resource = await metrics(page);
  const rulers = await page.locator('.distribution-scale').evaluateAll(nodes => nodes.map(n => n.getAttribute('aria-label')));
  const canvas = await page.locator('.matrix-scroll canvas').elementHandle();
  await zoom(page, 576, 1536, 3);
  c = await camera(page, 576, 1536);
  const focal = { x: c.rect.left + 91, y: c.rect.top + 71 };
  await page.mouse.move(focal.x, focal.y); await page.mouse.wheel(0, -180);
  await expect.poll(async () => (await camera(page, 576, 1536)).scaleX).toBeGreaterThan(c.scaleX);
  const after = await camera(page, 576, 1536);
  expect(Math.abs(c.x + 91 * c.dpr / c.scaleX - after.x - 91 * c.dpr / after.scaleX)).toBeLessThanOrEqual(2 * c.dpr / after.scaleX);
  expect(Math.abs(c.y + 71 * c.dpr / c.scaleY - after.y - 71 * c.dpr / after.scaleY)).toBeLessThanOrEqual(2 * c.dpr / after.scaleY);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: focal.x - 20, y: focal.y }, { x: focal.x + 20, y: focal.y }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: focal.x - 40, y: focal.y }, { x: focal.x + 40, y: focal.y }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await expect.poll(async () => (await camera(page, 576, 1536)).scaleX).toBeGreaterThan(after.scaleX);
  await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: false }); await cdp.detach();
  for (const axis of ['matrix', 'rows', 'columns'] as const) {
    await page.getByRole('button', { name: 'Fit width', exact: true }).click();
    await zoom(page, 576, 1536, 6);
    c = await camera(page, 576, 1536);
    const selector = axis === 'matrix' ? '.matrix-scroll canvas' : axis === 'rows' ? '.row-distributions canvas' : '.column-distributions canvas';
    const box = (await page.locator(selector).boundingBox())!;
    await drag(page, { x: box.x + 12, y: box.y + 12 },
      { x: box.x + (axis === 'rows' ? 12 : 72), y: box.y + (axis === 'columns' ? 12 : 42) });
    const bounds = axis === 'matrix' ? { columns: [2, 12], rows: [2, 7] } : { [axis]: [2, axis === 'rows' ? 7 : 12] };
    await expect(page.locator('.matrix-zoom-preview[data-surface=matrix]')).toHaveAttribute('data-bounds', JSON.stringify(bounds));
    expect(await page.locator('.matrix-zoom-preview[data-surface=matrix]').evaluate(n => getComputedStyle(n).borderTopColor)).toBe('rgb(245, 154, 56)');
    expect((await camera(page, 576, 1536)).scaleX).toBeCloseTo(c.scaleX);
    await page.mouse.up(); await expect(page.locator('.matrix-zoom-preview')).toHaveCount(0);
    const selected = await camera(page, 576, 1536);
    const expected = axis === 'matrix' ? Math.min(selected.width * c.dpr / 10, selected.height * c.dpr / 5)
      : axis === 'rows' ? selected.height * c.dpr / 5 : selected.width * c.dpr / 10;
    expect(selected.scaleX).toBeCloseTo(expected); expect(selected.scaleY).toBeCloseTo(expected);
    if (axis !== 'matrix') {
      const horizontal = axis === 'columns';
      const oldCenter = horizontal ? c.y + c.height * c.dpr / c.scaleY / 2 : c.x + c.width * c.dpr / c.scaleX / 2;
      const span = (horizontal ? selected.height : selected.width) * c.dpr / expected;
      const wanted = Math.max(0, Math.min((horizontal ? 576 : 1536) - span, oldCenter - span / 2));
      expect(Math.abs((horizontal ? selected.y : selected.x) - wanted)).toBeLessThanOrEqual(2 * c.dpr / expected);
    }
    await expect(page.locator('.row-distributions canvas')).toHaveAttribute('data-origin', `0,${selected.y}`);
    await expect(page.locator('.column-distributions canvas')).toHaveAttribute('data-origin', `${selected.x},0`);
    const row = (await page.locator('.row-distributions canvas').boundingBox())!;
    const column = (await page.locator('.column-distributions canvas').boundingBox())!;
    expect(row.width * c.dpr).toBe(100); expect(column.height * c.dpr).toBe(100);
    expect(row.y).toBe(selected.rect.top); expect(column.x).toBe(selected.rect.left);
    expect(row.height).toBe(selected.rect.height); expect(column.width).toBe(selected.rect.width);
  }
  await page.getByRole('button', { name: 'Fit width', exact: true }).click();
  for (const [size, visible] of [[7, true], [9, true], [10.1, false], [9, false], [7, true]] as const) {
    await zoom(page, 576, 1536, size);
    c = await camera(page, 576, 1536);
    await page.mouse.click(c.rect.left + 3.5 * size, c.rect.top + 3.5 * size);
    await expect(page.locator('.inspection-readout')).toHaveText(`row 3 · column 3${value(3 * 1536 + 3)}`);
    await expect(page.locator('.magnifier-card')).toHaveCount(visible ? 1 : 0);
    if (visible) {
      const card = (await page.locator('.matrix-inspection').boundingBox())!;
      const pane = (await page.locator('.matrix-surfaces').boundingBox())!;
      expect(card.x).toBeGreaterThanOrEqual(pane.x); expect(card.y).toBeGreaterThanOrEqual(pane.y);
      expect(card.x + card.width).toBeLessThanOrEqual(pane.x + pane.width);
      expect(card.y + card.height).toBeLessThanOrEqual(pane.y + pane.height);
      for (const panel of await page.locator('.row-distributions canvas, .column-distributions canvas').all()) {
        const p = (await panel.boundingBox())!;
        expect(card.x + card.width <= p.x || card.x >= p.x + p.width || card.y + card.height <= p.y || card.y >= p.y + p.height).toBe(true);
      }
      expect(await page.locator('.magnifier-guide-horizontal').evaluate(n => n.getBoundingClientRect().height)).toBe(1);
      expect(await page.locator('.magnifier-guide-vertical').evaluate(n => n.getBoundingClientRect().width)).toBe(1);
    }
  }
  expect(await page.locator('.distribution-scale').evaluateAll(nodes => nodes.map(n => n.getAttribute('aria-label')))).toEqual(rulers);
  expect(await metrics(page)).toMatchObject({ uploads: resource.uploads, createdTextures: resource.createdTextures, gpuBytes: resource.gpuBytes, errors: [] });
  expect(await canvas!.evaluate(node => node === document.querySelector('.matrix-scroll canvas'))).toBe(true);
  await documentFits(page);
  await testInfo.attach('measurements', { body: JSON.stringify({ fit: c, focalBefore: after, resources: await metrics(page), rulers }), contentType: 'application/json' });
  await closeSession(page);
});

for (const [name, low, high] of [
  ['science', -2, 6], ['scale.concentrated', Math.fround(-.0001), Math.fround(.0002)],
  ['scale.outliers', -1000, 3000], ['scale.constant', 2, 2], ['scale.nonfinite', null, null],
] as const) test(`production distribution scale is truthful for ${name}`, async ({ page }) => {
  await open(page, `${name}.weight`); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  for (const ruler of await page.locator('.distribution-scale').all()) {
    if (low === null) {
      await expect(ruler).toHaveAttribute('aria-label', /no finite values/);
      await expect(ruler.locator('.distribution-endpoint')).toHaveCount(0);
    } else {
      await expect(ruler).toHaveAttribute('data-minimum', String(low));
      await expect(ruler).toHaveAttribute('data-maximum', String(high));
      if (low === high) await expect(ruler).toHaveAttribute('aria-label', /constant, samples in bin 50/);
      else expect(await ruler.evaluate(n => parseFloat((n as HTMLElement).style.getPropertyValue('--distribution-zero'))))
        .toBeCloseTo(-low / (high! - low) * 100);
    }
  }
  await expect(page.locator('.distribution-range')).toHaveCount(0);
  await expect(page.getByText('Full-range bins', { exact: true })).toHaveCount(0);
  const info = page.getByRole('button', { name: 'Tensor information', exact: true });
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  await info.click();
  if (low === null) {
    await expect(dialog).toContainText('No finite values');
    await expect(dialog.locator('dt').filter({ hasText: /^True finite minimum$/ }).locator('+ dd')).toHaveText('Unavailable');
    await expect(dialog.locator('dt').filter({ hasText: /^True finite maximum$/ }).locator('+ dd')).toHaveText('Unavailable');
  }
  else {
    await expect(dialog).toContainText('Full-range bins');
    await expect(dialog.locator('dt').filter({ hasText: /^True finite minimum$/ }).locator('+ dd')).toHaveAttribute('title', `True finite minimum: ${low}`);
    await expect(dialog.locator('dt').filter({ hasText: /^True finite maximum$/ }).locator('+ dd')).toHaveAttribute('title', `True finite maximum: ${high}`);
  }
  const stable = await dialog.textContent();
  await page.locator('.matrix-scroll').dispatchEvent('wheel', { deltaY: -100, ctrlKey: true });
  expect(await dialog.textContent()).toBe(stable);
  await page.keyboard.press('Escape');
  await closeSession(page);
});

test('production stale results remain visible and generation-fenced while embedding camera leaves prompt intact', async ({ page }, testInfo) => {
  const editor = await tokenizer(page);
  await editor.fill('AAA'); await embeddingDone(page, 4);
  await editor.press('End');
  const prompt = await promptViewport(page);
  const canvas = await page.locator('.matrix-scroll canvas').elementHandle();
  const resource = await metrics(page);
  await zoom(page, 4, 576, 6);
  let c = await camera(page, 4, 576);
  await drag(page, { x: c.rect.left + 12, y: c.rect.top + 6 }, { x: c.rect.left + 72, y: c.rect.top + 18 });
  await expect(page.locator('.matrix-zoom-preview')).toHaveCount(3);
  await expect(page.locator('.matrix-zoom-preview[data-surface=matrix]')).toHaveAttribute('data-bounds', JSON.stringify({ columns: [2, 12], rows: [1, 3] }));
  await expect(page.locator('.matrix-zoom-preview[data-surface=rows]')).toHaveAttribute('data-bounds', JSON.stringify({ rows: [1, 3] }));
  await expect(page.locator('.matrix-zoom-preview[data-surface=columns]')).toHaveAttribute('data-bounds', JSON.stringify({ columns: [2, 12] }));
  await page.mouse.up();
  await page.locator('.matrix-scroll').evaluate(n => { n.scrollLeft = 100; });
  expect(await promptViewport(page)).toEqual(prompt);
  await page.getByRole('button', { name: 'Fit width', exact: true }).click();
  expect(await promptViewport(page)).toEqual(prompt);
  expect(await metrics(page)).toMatchObject({ uploads: resource.uploads, createdTextures: resource.createdTextures });
  const oldIds = await page.locator('.token-ids').allTextContents();
  await page.evaluate(() => {
    const frames = { count: 0, missing: 0, running: true };
    (window as any).__continuity = frames;
    const sample = () => {
      if (!frames.running) return;
      frames.count++;
      if (!document.querySelector('.token-opening') || !document.querySelector('.token-ids')
        || !document.querySelector('.embedding-layer:not([data-staging]) canvas')) frames.missing++;
      requestAnimationFrame(sample);
    };
    requestAnimationFrame(sample);
  });
  let release!: () => void, entered = false;
  const barrier = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/tokenize', async route => {
    const response = await route.fetch();
    if (route.request().postDataJSON().text === 'AAAB') { entered = true; await barrier; }
    await route.fulfill({ response });
  });
  await editor.focus(); await editor.press('End'); await editor.press('B');
  await expect.poll(() => entered).toBe(true);
  await expect(page.locator('.cm-editor')).toHaveAttribute('data-annotations', 'stale');
  await expect(page.locator('.token-ids')).toHaveText(oldIds);
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
  expect(await canvas!.evaluate(n => n === document.querySelector('.matrix-scroll canvas'))).toBe(true);
  await page.locator('[data-token-index="1"]').hover();
  await expect(page.locator('[data-token-index][data-active-token]')).toHaveCount(0);
  await control('arm', { kind: 'input_embeddings' });
  await editor.focus(); await editor.press('End'); await editor.press('C');
  await expect.poll(async () => (await control()).control.entered).toBe(true);
  await expect(page.locator('.cm-editor')).toHaveAttribute('data-annotations', 'current');
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'stale');
  expect(await canvas!.evaluate(n => n === document.querySelector('.embedding-layer:not([data-staging]) .matrix-scroll canvas'))).toBe(true);
  await page.locator('.matrix-scroll:visible').focus();
  await expect(page.locator('[data-token-index][data-active-token]')).toHaveCount(0);
  const newIds = await page.locator('.token-ids').allTextContents();
  release(); await page.unrouteAll({ behavior: 'wait' });
  await expect(page.locator('.token-ids')).toHaveText(newIds);
  await control('release', {}); await embeddingDone(page, 6);
  expect(await canvas!.evaluate(n => n.isConnected)).toBe(false);
  await page.locator('.matrix-scroll').focus();
  await expect(page.locator('[data-token-index="0"]')).toHaveAttribute('data-active-token', '');
  c = await camera(page, 6, 576);
  const frames = await page.evaluate(() => { const frames = (window as any).__continuity; frames.running = false; return frames; });
  expect(frames.count).toBeGreaterThan(2); expect(frames.missing).toBe(0);
  await documentFits(page);
  await testInfo.attach('measurements', { body: JSON.stringify({ prompt, camera: c, frames, resources: await metrics(page) }), contentType: 'application/json' });
  await closeSession(page);
});

test('production inspection tolerates DPR change before viewport resize notification', async ({ page }) => {
  await open(page); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  await page.locator('.matrix-scroll').focus();
  await expect(page.locator('.inspection-readout')).toContainText('row 0 · column 0');
  const before = await metrics(page);
  // A monitor/browser DPR change can precede resize/media-query notification.
  // Force that event order, rather than relying on a timing-sensitive real move.
  const errors = await page.evaluate(() => {
    const messages: string[] = [];
    const record = (event: ErrorEvent) => { messages.push(event.message); };
    window.addEventListener('error', record);
    const original = devicePixelRatio;
    try {
      Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: original === 1 ? 2 : 1 });
      document.querySelector('.matrix-scroll canvas')!.dispatchEvent(new PointerEvent('pointerleave'));
      window.dispatchEvent(new Event('resize'));
    } finally {
      Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: original });
      window.dispatchEvent(new Event('resize'));
      window.removeEventListener('error', record);
    }
    return messages;
  });
  expect(errors).toEqual([]);
  await expect(page.locator('.inspection-readout')).toHaveCount(0);
  await page.locator('.matrix-scroll').press('ArrowRight');
  await expect(page.locator('.inspection-readout')).toHaveText(`row 0 · column 1${value(1)}`);
  expect(await metrics(page)).toMatchObject({ uploads: before.uploads, createdTextures: before.createdTextures });
  await closeSession(page);
});

async function polishCapture(page: Page, info: TestInfo, name: string) {
  await page.mouse.move(0, 0);
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  const path = info.outputPath(`${name}.png`);
  await page.screenshot({ path, animations: 'disabled' });
  await info.attach(name, { path, contentType: 'image/png' });
}

for (const width of [390, 1178, 1440]) test(`architecture safety baseline preserves shell and other explorers at ${width}px`, async ({ page }, info) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width, height: 900 });
  // Only structural architecture is authored. Tensor values, distributions,
  // tokenization and embeddings use the production TCP fixture service.
  await page.route('**/architecture', (route) => route.fulfill({ json: {
    status: 'available', model_id: 'acceptance/fixture', diagnostics: [], graph: makeProjectionFixture({ count: 4 }),
  } }));
  const shell = () => page.evaluate(() => Object.fromEntries(
    ['.app-bar', '.app-status-bar', '.workspace-frame', '.app-bar nav', '.app-bar select'].map((selector) => {
      const element = document.querySelector(selector)!;
      const { x, y, width, height } = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      return [selector, { x, y, width, height, font: style.font, color: style.color, background: style.backgroundColor }];
    })));
  const observations: Record<string, unknown> = {};
  await open(page); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  const baseline = await shell();
  const fixedShell = async () => {
    const current = await shell();
    // The existing active-tab font weight changes the nav's content width.
    // Record each active view; compare its exact geometry on return below.
    for (const [selector, value] of Object.entries(baseline)) {
      if (selector === '.app-bar select' && width === 390) {
        // At narrow widths the selector flexes with the existing active-tab font
        // width. Its styling/height stays fixed; exact same-view geometry is
        // checked during every Architecture action and on Tensor return below.
        expect(current[selector]).toMatchObject({ y: value.y, height: value.height, font: value.font, color: value.color, background: value.background });
      } else if (selector !== '.app-bar nav') expect(current[selector]).toEqual(value);
    }
  };
  await documentFits(page);
  await polishCapture(page, info, `safety-tensor-${width}`);
  observations.tensor = baseline;

  // Keep the selected session: tokenizer() is an independent-test setup helper
  // that deliberately creates a new session, so use navigation directly here.
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await embeddingDone(page, 1);
  const editor = page.getByRole('textbox', { name: 'Prompt', exact: true });
  await editor.fill('Hello, architecture!');
  await expect(page.getByText(/tokens · current prompt/)).toBeVisible();
  await expect(page.locator('.input-embeddings [data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
  await expect(page.locator('.input-embeddings .embedding-layer:not([data-staging]) .matrix-panel-status')).toBeEmpty();
  await settledPrompt(page); await documentFits(page);
  await fixedShell(); observations.tokenizer = await shell();
  await polishCapture(page, info, `safety-tokenizer-${width}`);

  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await expect(canvas).toHaveAttribute('data-layout-count', /^[1-9]\d*$/);
  await documentFits(page); await fixedShell(); observations.architectureShell = await shell();
  await polishCapture(page, info, `safety-architecture-${width}`);
  // Measurements are evidence, not a golden toolbar placement requirement.
  observations.architecture = await page.evaluate(() => Object.fromEntries(
    ['.architecture-workspace', '#architecture-browser', '.architecture-toolbar', '.architecture-flow'].map((selector) => {
      const { x, y, width, height } = document.querySelector(selector)!.getBoundingClientRect();
      return [selector, { x, y, width, height }];
    })));
  const readyGraph = async () => {
    await expect(canvas).toHaveAttribute('aria-busy', 'false');
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  };
  const boundedControls = async () => {
    await readyGraph(); await documentFits(page); await fixedShell();
    expect(await shell()).toEqual(observations.architectureShell);
    // Native horizontal scrollbars add their own height at constrained widths.
    // Bound each content row independently of that platform chrome.
    expect(await page.locator('.architecture-toolbar').evaluate((element) => element.clientHeight)).toBeLessThanOrEqual(50);
    for (const row of await page.locator('.architecture-context-row').all()) {
      expect(await row.evaluate((element) => element.clientHeight)).toBeLessThanOrEqual(45);
    }
    await expect(page.getByRole('button', { name: 'Show all operations', exact: true })).toHaveCount(0);
  };
  await expect(page.getByRole('combobox', { name: /Expand instance of/ })).toHaveCount(0);
  await expect(page.locator('.architecture-context-row')).toHaveCount(0);
  await boundedControls();
  await page.getByRole('button', { name: 'Explore stack Decoder layers', exact: true }).click();
  await expect(page.locator('.architecture-context-row')).toHaveCount(1);
  await boundedControls(); await polishCapture(page, info, `controls-stack-${width}`);
  await findComponent(page, 'layer-3.attention.Q');
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  await boundedControls(); await polishCapture(page, info, `controls-node-${width}`);
  await page.locator('.architecture-connection[data-source-node="layer-3.attention.Q"][data-target-node="layer-3.attention.rope-Q"]').focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'Close connection inspection' })).toBeFocused();
  await page.keyboard.press('Escape');
  await boundedControls(); await polishCapture(page, info, `controls-edge-${width}`);
  await findComponent(page, 'layer-3.attention'); await readyGraph();
  const scopeCamera = await page.locator('.react-flow__viewport').getAttribute('style');
  const scopeResources = await metrics(page);
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await boundedControls();
  await expect(canvas).toHaveAttribute('data-scope-id', 'layer-3.attention');
  await polishCapture(page, info, `isolation-shell-${width}`);
  observations.isolationShell = await shell();
  const isolatedResources = await metrics(page);
  expect([isolatedResources.textures, isolatedResources.readers]).toEqual([scopeResources.textures, scopeResources.readers]);
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await boundedControls();
  expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(scopeCamera);
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  await complete(page); await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  await documentFits(page); expect(await shell()).toEqual(baseline);
  await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
  await info.attach(`safety-geometry-${width}`, { contentType: 'application/json', body: JSON.stringify({
    viewport: page.viewportSize(), browser: page.context().browser()!.version(), project: info.project.name, observations,
  }, null, 2) });
});

for (const width of [1178, 1440]) {
  test(`polish inventory captures retain selected scientific work at ${width}px`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    await open(page); await complete(page);
    await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
    const canvas = await page.locator('.matrix-scroll canvas').elementHandle();
    const aligned = async () => {
      await expect.poll(() => page.evaluate(() => {
        const box = (s: string) => document.querySelector(s)!.getBoundingClientRect();
        const matrix = box('.matrix-scroll canvas'), rows = box('.row-distributions canvas'), columns = box('.column-distributions canvas');
        const style = getComputedStyle(document.querySelector('.matrix-surfaces')!);
        const host = document.querySelector<HTMLElement>('.matrix-scroll')!;
        const viewport = host.getBoundingClientRect();
        return Math.max(Math.abs(rows.left - viewport.left - host.clientWidth - parseFloat(style.columnGap)),
          Math.abs(columns.top - viewport.top - host.clientHeight - parseFloat(style.rowGap)),
          Math.abs(rows.top - matrix.top), Math.abs(columns.left - matrix.left));
      })).toBeLessThanOrEqual(1);
    };
    const resources = await metrics(page);
    const expanded = await page.locator('.tensor-tree details').evaluateAll(nodes => nodes.map(n => n.hasAttribute('open')));
    await aligned(); await polishCapture(page, info, `tensor-expanded-${width}`);
    await page.getByRole('button', { name: 'Collapse inventory' }).click();
    const restore = page.getByRole('button', { name: 'Expand inventory' });
    await expect(restore).toBeFocused();
    expect(await restore.textContent()).toBe('');
    const rail = page.locator('.inventory-rail');
    expect((await rail.boundingBox())!.width).toBe(40);
    expect(await canvas!.evaluate(n => n === document.querySelector('.matrix-scroll canvas'))).toBe(true);
    await restore.blur();
    await aligned(); await polishCapture(page, info, `tensor-collapsed-${width}`);
    await restore.press('Enter');
    expect(await page.locator('.tensor-tree details').evaluateAll(nodes => nodes.map(n => n.hasAttribute('open')))).toEqual(expanded);
    expect(await metrics(page)).toMatchObject({ uploads: resources.uploads, createdTextures: resources.createdTextures });
    await documentFits(page); await closeSession(page);
  });

  test(`polish real tokenizer auto sizing and accessible manual split at ${width}px`, async ({ page }, info) => {
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

test('polish magnifier follows edges after scrolling resize DPR and source replacement', async ({ page }, info) => {
  await open(page); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  await zoom(page, 576, 1536, 2);
  const resources = await metrics(page);
  const captures: object[] = [];
  const sweep = async () => {
    const c = await camera(page, 576, 1536);
    let first: object | undefined;
    for (const [fx, fy] of [[.3, .3], [.35, .35], [0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]]) {
      const x = Math.min(c.rect.width - 2, Math.max(2, c.rect.width * fx!));
      const y = Math.min(c.rect.height - 2, Math.max(2, c.rect.height * fy!));
      await page.mouse.move(c.rect.left + x, c.rect.top + y);
      await expect(page.locator('.inspection-readout')).toBeVisible();
      const g = await page.evaluate(() => {
        const rect = (s: string) => document.querySelector(s)!.getBoundingClientRect().toJSON() as DOMRect;
        return { card: rect('.matrix-inspection'), pane: rect('.matrix-surfaces'),
          profiles: [rect('.row-distributions canvas'), rect('.column-distributions canvas')] };
      });
      expect(g.card.left).toBeGreaterThanOrEqual(g.pane.left);
      expect(g.card.top).toBeGreaterThanOrEqual(g.pane.top);
      expect(g.card.right).toBeLessThanOrEqual(g.pane.right);
      expect(g.card.bottom).toBeLessThanOrEqual(g.pane.bottom);
      for (const p of g.profiles) expect(g.card.right <= p.left || g.card.left >= p.right || g.card.bottom <= p.top || g.card.top >= p.bottom).toBe(true);
      if (fx === .3) first = g.card;
      if (fx === .35) expect(g.card).not.toEqual(first);
      captures.push(g);
    }
  };
  await page.locator('.matrix-scroll').evaluate(n => { n.scrollLeft = 180; n.scrollTop = 140; });
  await expect.poll(async () => (await camera(page, 576, 1536)).y).toBeGreaterThan(0);
  await sweep();
  const before = await camera(page, 576, 1536);
  await page.setViewportSize({ width: 1178, height: 900 }); await sweep();
  let current = await camera(page, 576, 1536);
  expect(current.scaleX).toBeCloseTo(before.scaleX);
  expect(current.x).toBeCloseTo(before.x); expect(current.y).toBeCloseTo(before.y);
  const cdp = await page.context().newCDPSession(page);
  const changedDpr = current.dpr === 1 ? 2 : 1;
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1178, height: 900, deviceScaleFactor: changedDpr, mobile: false });
  await expect.poll(() => page.evaluate(() => devicePixelRatio)).toBe(changedDpr);
  await page.evaluate(() => window.dispatchEvent(new Event('resize')));
  await sweep();
  current = await camera(page, 576, 1536);
  // A DPR increase can make the entire data axis fit. Preserve the previous
  // logical origin only within the new native scrollbar bounds.
  for (const [axis, length, viewport, scale] of [
    ['x', 1536, current.width, current.scaleX], ['y', 576, current.height, current.scaleY],
  ] as const) {
    const expected = Math.min(before[axis], Math.max(0, length - viewport * current.dpr / scale));
    expect(Math.abs(current[axis] - expected)).toBeLessThanOrEqual(current.dpr / scale);
  }
  expect(current.scaleX).toBe(current.scaleY);
  expect(await metrics(page)).toMatchObject({ uploads: resources.uploads, createdTextures: resources.createdTextures, errors: [] });
  await cdp.detach();
  await open(page, 'layout.fits.weight'); await complete(page);
  await expect(page.locator('.inspection-readout')).toHaveCount(0);
  const replacement = await camera(page, 32, 32);
  expect(replacement.scaleX).toBeCloseTo(Math.max(1, Math.floor(replacement.width * replacement.dpr) / 32));
  await page.locator('.matrix-scroll').focus(); await page.keyboard.press('Escape');
  expect(await camera(page, 32, 32)).toEqual(replacement);
  await documentFits(page);
  await info.attach('edge-placements', { body: JSON.stringify(captures), contentType: 'application/json' });
  await closeSession(page);
});

test('expanded model coverage links native embeddings in GPTQ and NVFP4 checkpoints', async ({ page }) => {
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  for (const [family, columns] of [['smollm2', 12], ['qwen3', 128], ['qwen35', 32]] as const) {
    await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(family);
    const editor = page.getByRole('textbox', { name: 'Prompt', exact: true });
    await editor.fill('one two one');
    await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
    await expect(page.getByText(`[4 × ${columns}] · float32`)).toBeVisible();
    await expect(page.locator('.token-ids')).toHaveText(['1', '2', '3', '2']);
    await zoom(page, 4, columns, 8);
    const c = await camera(page, 4, columns);
    await page.mouse.move(c.rect.left + 4, c.rect.top + 3 * 8 + 4);
    await expect(page.locator('[data-token-index="3"]')).toHaveAttribute('data-active-token', '');
    await expect(page.locator('.inspection-readout')).toHaveText(`row 3 · column 0${((2 * columns) % 29 - 14) / 8}`);
    await page.locator('[data-token-index="1"]').hover();
    await expect(page.locator('[data-token-index="1"]')).toHaveAttribute('data-active-token', '');
    await expect(page.locator('[data-token-index="3"]')).not.toHaveAttribute('data-active-token', '');
  }
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('vjepa2');
  await expect(page.getByText('Tokenizer unavailable for this model. You can still edit the prompt.')).toBeVisible();
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  await documentFits(page); await closeSession(page);
});

function embeddingOracle(ids: number[], columns = 576) {
  const values = ids.flatMap(id => Array.from({ length: columns }, (_, column) => value(id * columns + column)));
  const sorted = [...values].sort((a, b) => a - b), minimum = sorted[0]!, maximum = sorted.at(-1)!;
  const rows = Array<number>(ids.length * 100).fill(0), columnCounts = Array<number>(100 * columns).fill(0);
  for (let index = 0; index < values.length; index++) {
    const bin = Math.min(99, Math.floor((values[index]! - minimum) / (maximum - minimum) * 100));
    rows[Math.floor(index / columns) * 100 + bin]!++;
    columnCounts[bin * columns + index % columns]!++;
  }
  return { values, rows, columns: columnCounts, minimum, maximum };
}

async function science(page: Page) {
  return page.evaluate(() => {
    const probe = (window as any).__acceptance;
    return { values: [...probe.scalarValues], counts: structuredClone(probe.countValues), transfer: probe.transfer(),
      domains: [...document.querySelectorAll('.distribution-scale')].map(node => [node.getAttribute('data-minimum'), node.getAttribute('data-maximum')]),
      dimensions: [...document.querySelectorAll('.matrix-surfaces canvas')].map(node => {
        const canvas = node as HTMLCanvasElement;
        return { width: canvas.width, height: canvas.height, origin: canvas.dataset.origin };
      }) };
  });
}
async function captureScience(page: Page) {
  await page.evaluate(() => {
    const probe = (window as any).__acceptance;
    probe.captureScalars = true; probe.captureCounts = true;
    probe.scalarValues.length = 0; probe.countValues.rows.length = 0; probe.countValues.columns.length = 0;
  });
}

test('integrated two-card embeddings have real scientific parity with the same checkpoint matrix', async ({ page }, info) => {
  await captureScience(page);
  await open(page, 'embedding.parity.weight');
  await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  await idle(); await nativeCamera(page);
  const tensor = await science(page);
  expect(tensor.transfer.mode).toEqual([1]);
  await page.getByRole('button', { name: 'Collapse inventory', exact: true }).click();
  for (const width of [1178, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.getByRole('button', { name: 'Expand inventory', exact: true }).hover();
    await expect(page.getByRole('tooltip', { name: 'Expand inventory', exact: true })).toBeVisible();
    await page.screenshot({ path: info.outputPath(`integrated-tensor-${width}.png`) });
  }
  // Hold a real tokenization response to capture the empty prompt / waiting card.
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/tokenize', async route => {
    const response = await route.fetch(); await gate; await route.fulfill({ response });
  });
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await expect(page.getByText('Waiting for current tokenization.')).toBeVisible();
  for (const width of [1178, 1440]) {
    await page.setViewportSize({ width, height: 1000 }); await settledPrompt(page);
    await integratedCard(page.locator('.input-embeddings .viewer-panel'));
    await page.screenshot({ path: info.outputPath(`two-cards-empty-${width}.png`) });
  }
  release(); await embeddingDone(page, 1);
  await page.unroute('**/tokenize');
  await captureScience(page);
  const response = page.waitForResponse(r => r.url().endsWith('/tokenize') && r.request().postDataJSON().text === 'A😀A');
  const orderedRequests: any[] = [];
  page.on('request', request => {
    if (/\/embeddings(?:\/(?:statistics|distributions))?$/.test(new URL(request.url()).pathname)) orderedRequests.push(request.postDataJSON());
  });
  await page.getByRole('textbox', { name: 'Prompt', exact: true }).fill('A😀A');
  const ids: number[] = (await (await response).json()).tokens.map((token: any) => token.id);
  await embeddingDone(page, ids.length); await idle(); await nativeCamera(page);
  expect(orderedRequests).toEqual(Array.from({ length: 3 }, () => ({ token_ids: ids })));
  const expected = embeddingOracle(ids), embeddings = await science(page);
  expect(embeddings.values).toEqual(expected.values);
  expect(embeddings.counts).toEqual({ rows: expected.rows, columns: expected.columns });
  expect(embeddings.domains).toEqual(Array.from({ length: 2 }, () => [String(expected.minimum), String(expected.maximum)]));
  expect(embeddings).toEqual(tensor);
  const prompt = await tokenizerGeometry(page);
  const resources = await metrics(page);
  await page.getByRole('button', { name: 'Input embeddings information', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Input embeddings information' });
  await expect(dialog).toContainText('Full-range bins');
  await expect(dialog).toContainText('Luminosity anchors');
  await expect(dialog).not.toContainText('Storage dtype');
  await page.keyboard.press('Escape');
  for (const width of [1178, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.getByRole('button', { name: 'Fit width', exact: true }).click();
    await settledPrompt(page); await documentFits(page);
    await expect(page.locator('.tokenizer-workspace .viewer-panel')).toHaveCount(2);
    const promptCard = await page.locator('.prompt-panel .viewer-panel').evaluate(panel => {
      const rect = (node: Element) => node.getBoundingClientRect().toJSON();
      return {
        panel: rect(panel), title: rect(panel.querySelector('.matrix-panel-header')!), body: rect(panel.querySelector('.viewer-panel-body')!),
      };
    });
    expect(promptCard.title.width).toBe(promptCard.panel.width);
    expect(promptCard.body.y).toBe(promptCard.title.bottom);
    expect(promptCard.body.width).toBe(promptCard.panel.width);
    await integratedCard(page.locator('.embedding-layer:not([data-staging]) .viewer-panel'));
    await page.screenshot({ path: info.outputPath(`two-cards-populated-${width}.png`) });
  }
  await page.getByRole('separator', { name: 'Resize prompt and embeddings' }).press('ArrowDown');
  await page.screenshot({ path: info.outputPath('two-cards-resize-focus.png') });
  const split = await tokenizerGeometry(page);
  await zoom(page, ids.length, 576, 128);
  await page.locator('.matrix-scroll').evaluate(node => { node.scrollLeft = 300; node.scrollTop = 80; });
  await expect.poll(() => page.locator('.matrix-scroll').evaluate(node => node.scrollTop)).toBeGreaterThan(0);
  await page.locator('.matrix-scroll').focus();
  await page.screenshot({ path: info.outputPath('two-cards-zoom-scroll.png') });
  const after = await tokenizerGeometry(page);
  expect(after.prompt).toEqual(split.prompt);
  expect(after.embeddings).toEqual(split.embeddings);
  expect((await metrics(page)).uploads).toBe(resources.uploads);
  expect((await metrics(page)).createdTextures).toBe(resources.createdTextures);
  expect(orderedRequests).toHaveLength(3);
  await info.attach('integrated-scientific-parity', { contentType: 'application/json', body: JSON.stringify({
    ids, rows: ids.length, columns: 576, domain: [expected.minimum, expected.maximum],
    rowCountSum: expected.rows.reduce((a, b) => a + b), columnCountSum: expected.columns.reduce((a, b) => a + b),
    transfer: embeddings.transfer, nativeDimensions: embeddings.dimensions, prompt, resources: await metrics(page),
  }) });
  await closeSession(page);
});

test('real embedding distribution cancellation preserves completed values and independent statistics', async ({ page }) => {
  await control('arm', { kind: 'input_embeddings_distributions' });
  await tokenizer(page, false);
  await expect.poll(async () => (await control()).control.entered).toBe(true);
  await expect(page.getByText('[1 × 576] · float32')).toBeVisible();
  await expect(page.locator('.input-embeddings [data-result=statistics]')).toHaveCount(0);
  await page.getByRole('button', { name: 'Cancel embedding distributions' }).click();
  await expect(page.locator('.input-embeddings [data-result=distributions]')).toHaveAttribute('data-state', 'cancelled');
  await page.locator('.matrix-scroll').focus();
  await expect(page.locator('.inspection-readout')).toHaveText(`row 0 · column 0${value(576)}`);
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
  await expect(page.locator('[data-token-index="0"]')).toHaveAttribute('data-active-token', '');
  await control('release', {});
  await closeSession(page);
});

test('real embedding statistics failure keeps values, histograms and successful tokenization authoritative', async ({ page }) => {
  await control('arm', { kind: 'input_embeddings_statistics', mode: 'pre-meta-error' });
  await tokenizer(page, false);
  await expect(page.locator('.input-embeddings [data-result=statistics]')).toHaveAttribute('data-state', 'failed');
  await expect(page.locator('.input-embeddings [data-result=distributions]')).toHaveCount(0);
  await expect(page.locator('.distribution-scale-rows')).toHaveAttribute('data-minimum', '-1');
  await expect(page.locator('.tokenizer-status')).toContainText('current prompt');
  await page.locator('.matrix-scroll').focus();
  await expect(page.locator('.inspection-readout')).toHaveText(`row 0 · column 0${value(576)}`);
  await expect(page.locator('[data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
  await expect(page.getByText(/unavailable for this model/)).toHaveCount(0);
  await closeSession(page);
});
