/* eslint-disable @typescript-eslint/no-explicit-any -- Test-only observer and JSON evidence. */
import { test as base, expect } from '@playwright/test';
import type { Page, TestInfo } from '@playwright/test';
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import { HarnessTiming } from './harness-timing';
import { installProbe } from './probe';
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
export const matrix = 'model.layers.0.mlp.down_proj.weight';
export const value = (index: number) => ((index * 17) % 257 - 128) / 128;

// Each test owns its service, cache, timing and closures, including across files.
function makeProduct(page: Page, testInfo: TestInfo) {
  let timing: HarnessTiming;
  let backend = 'http://127.0.0.1:8765';
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
    await expect.poll(async () => {
      const resource = await metrics(page);
      return [resource.textures, resource.readers];
    }).toEqual([0, 0]);
  }

  async function embeddingDone(page: Page, rows: number, timeout = 15_000) {
    await expect(page.getByText(`[${rows} × 576] · float32`).filter({ visible: true })).toBeVisible({ timeout });
    await expect(page.locator('.input-embeddings [data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
    await expect(page.locator('.input-embeddings .embedding-layer:not([data-staging]) .matrix-panel-status')).toBeEmpty();
  }

  function expectDocumentFits(page: Page, geometry: { document: number[]; body: number[]; scroll: number[] }) {
    expect(geometry).toEqual({ document: [page.viewportSize()!.width, page.viewportSize()!.height],
      body: [page.viewportSize()!.width, page.viewportSize()!.height], scroll: [0, 0] });
  }

  async function documentFits(page: Page) {
    expectDocumentFits(page, await page.evaluate(() => ({
      document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
      body: [document.body.scrollWidth, document.body.scrollHeight], scroll: [scrollX, scrollY],
    })));
  }

  async function polishCapture(page: Page, info: TestInfo, name: string) {
    await page.mouse.move(0, 0);
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    const path = info.outputPath(`${name}.png`);
    await page.screenshot({ path, animations: 'disabled' });
    await info.attach(name, { path, contentType: 'image/png' });
  }

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
  async function start(capturePixels: boolean) {
    timing = new HarnessTiming();
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
    if (isReference) {
      const directory = process.env.LMEX_REFERENCE_MODEL_DIR!;
      referenceSamples = JSON.parse(execFileSync(`${repo}backend/.venv/bin/python`,
        ['-m', 'acceptance.reference', directory], { cwd: repo, encoding: 'utf8' }));
      referenceRoot = mkdtempSync(join(tmpdir(), 'lmex-reference-'));
      command = ['-m', 'llm_model_explorer', '--model-root', dirname(directory),
        '--cache-dir', referenceRoot, '--port', String(ports.backend), '--cors-origin', uiOrigin,
        '--device', process.env.LMEX_REFERENCE_DEVICE ?? 'cpu'];
    }
    timing.spawned = performance.now();
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
    timing.ready = performance.now();
    page.on('console', (message) => { if (message.type() === 'error' || message.type() === 'warning') log += `\nBrowser: ${message.text()}`; });
    page.on('pageerror', (error) => { log += `\nPage: ${error.message}`; });
    await page.addInitScript(installProbe, { capturePixels });
    await page.goto('/');
    await expect(page.getByTestId('backend-url')).toHaveText(backend);
    timing.bodyStarted = performance.now();
  }
  async function stop(capturePixels: boolean) {
    if (testInfo.status === 'skipped') return;
    timing.teardownStarted = performance.now();
    let probe: any;
    try {
      probe = !page.isClosed() ? await page.evaluate(() => (window as any).__acceptance?.metrics() ?? null) : null;
      if (testInfo.status !== testInfo.expectedStatus) {
        await testInfo.attach('resource-state', { body: JSON.stringify(probe), contentType: 'application/json' });
        await testInfo.attach('backend-log', { body: log, contentType: 'text/plain' });
      }
    } finally {
      try { await page.close(); }
      finally {
        if (service?.exitCode === null) {
          service.kill('SIGTERM');
          const timer = setTimeout(() => service.kill('SIGKILL'), 10_000);
          try { await once(service, 'exit'); } finally { clearTimeout(timer); }
        }
        if (referenceRoot) { rmSync(referenceRoot, { recursive: true }); referenceRoot = undefined; }
      }
    }
    await timing.attach(testInfo, log, probe);
    // Setup failures retain their original reference/server diagnostic. Cleanup
    // above still runs; resource assertions require the installed browser probe.
    if (!timing.bodyStarted) return;
    expect(probe?.capturePixels).toBe(capturePixels);
    if (capturePixels) {
      expect(probe?.framebufferReadbacks).toBeGreaterThan(0);
      expect(probe?.pixelQueries).toBeGreaterThan(0);
    } else expect(probe?.framebufferReadbacks).toBe(0);
  }
  return { start, stop, control, metrics, idle, open, complete, pixel, tokenizer, closeSession, embeddingDone, expectDocumentFits, documentFits, polishCapture, embeddingOracle, science, captureScience, get backend() { return backend; }, get referenceSamples() { return referenceSamples; }, get timing() { return timing; } };
}

export const test = base.extend<{ capturePixels: boolean; product: ReturnType<typeof makeProduct> }>({
  capturePixels: [false, { option: true }],
  product: [async ({ page, capturePixels }, use, testInfo) => {
    const product = makeProduct(page, testInfo);
    try { await product.start(capturePixels); await use(product); }
    finally { await product.stop(capturePixels); }
  }, { auto: true }],
});
