/* eslint-disable @typescript-eslint/no-explicit-any -- Test-only JSON evidence. */
import { expect } from '@playwright/test';
import { test, value } from './product-harness';
import { nativeCamera } from '../tests/native-camera';

// Logical transport cases use headless Chromium; native focus/geometry stays headed.
test.use({ headless: false });

test('real ordered embeddings render progressively with exact duplicate rows and linked annotations', async ({ page, context , product }, testInfo) => {
  const { control, metrics, tokenizer, closeSession, embeddingDone, documentFits } = product;
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

for (const [name, low, high] of [
  // TCP and component-renderer owners retain the full domain matrix. These
  // bridges cover asymmetric finite metadata and the distinct null fallback.
  ['science', -2, 6], ['scale.nonfinite', null, null],
] as const) test(`production distribution scale is truthful for ${name}`, async ({ page, product }) => {
  const { control, open, complete, closeSession } = product;
  await control('arm', { kind: 'tensor_distributions' });
  await open(page, `${name}.weight`); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveAttribute('data-state', 'streaming');
  const held = await control();
  expect(held.control.entered).toBe(true);
  expect(held.control.released).toBe(false);
  expect(Object.values(held.artifacts).some((artifact: any) => artifact.manifest.spec.operation === 'tensor_distributions')).toBe(false);
  for (const ruler of await page.locator('.distribution-scale').all()) {
    if (low === null) {
      await expect(ruler).toHaveAttribute('aria-label', /no finite values/);
      await expect(ruler.locator('.distribution-endpoint')).toHaveCount(0);
    } else {
      await expect(ruler).toHaveAttribute('data-minimum', String(low));
      await expect(ruler).toHaveAttribute('data-maximum', String(high));
      expect(await ruler.evaluate(n => parseFloat((n as HTMLElement).style.getPropertyValue('--distribution-zero'))))
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
  await control('release', {});
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  expect(await dialog.textContent()).toBe(stable);
  await page.locator('.matrix-scroll').dispatchEvent('wheel', { deltaY: -100, ctrlKey: true });
  expect(await dialog.textContent()).toBe(stable);
  await page.keyboard.press('Escape');
  await closeSession(page);
});

test('real embedding distribution cancellation preserves completed values and independent statistics', async ({ page, product }) => {
  const { control, tokenizer, closeSession } = product;
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

test('real embedding statistics failure keeps values, histograms and successful tokenization authoritative', async ({ page, product }) => {
  const { control, tokenizer, closeSession } = product;
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

test.describe(() => {
  test.use({ capturePixels: true });
  test('production UI renders exact source pixels before producer completion and releases ownership', async ({ page, context , product }, testInfo) => {
  const { control, metrics, open, complete, pixel, closeSession } = product;
    await page.setViewportSize({ width: 1000, height: 700 });
    await control('arm', { kind: 'logical_tensor' });
    const started = await page.evaluate(() => performance.now());
    await open(page);
    await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'streaming');
    const state = await control();
    expect(state.control.entered).toBe(true);
    expect(state.control.released).toBe(false);
    expect(Object.keys(state.artifacts)).toHaveLength(0);
    const first = await metrics(page), known = await pixel(page, 8, 8);
    expect(first.firstRender).toBeGreaterThan(started);
    expect(known[1]).toBeGreaterThan(known[0]);
    expect(known[1]).toBeGreaterThan(known[2]);
    const releasedAt = await page.evaluate(() => performance.now());
    expect(first.firstRender).toBeLessThan(releasedAt);
    await control('release', {}); await complete(page);
    await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
    await nativeCamera(page);
    const canvas = page.locator('.matrix-scroll canvas'), box = (await canvas.boundingBox())!;
    await expect(canvas).toHaveAttribute('data-origin', '0,0');
    const dpr = await page.evaluate(() => devicePixelRatio);
    const resource = await metrics(page);
    expect(resource.textures).toBe(3);
    expect(resource.gpuBytes).toBe(4 * (576 * 1536 + (576 + 1536) * 100));
    expect(resource.maxUploadBytes).toBeLessThanOrEqual(262144);
    await page.mouse.move(box.x + 8.5 / dpr, box.y + 8.5 / dpr);
    await expect(page.locator('.inspection-readout')).toHaveText(`row 8 · column 8${value(8 * 1536 + 8)}`);
    expect(await metrics(page)).toMatchObject({ uploads: resource.uploads, createdTextures: resource.createdTextures });
    await testInfo.attach('first-pixels', { body: JSON.stringify({ started, firstRender: first.firstRender, releasedAt, resource }), contentType: 'application/json' });
    await closeSession(page);
    const cdp = await context.newCDPSession(page);
    await cdp.send('HeapProfiler.collectGarbage'); await cdp.detach();
    expect((await metrics(page)).arrays).toEqual([]);
  });
});

test('integrated two-card embeddings have independent scientific values and three exact requests', async ({ page, product }, info) => {
  const { metrics, idle, open, complete, closeSession, embeddingDone, embeddingOracle, science, captureScience } = product;
  await captureScience(page); await open(page, 'embedding.parity.weight'); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  await idle(); await nativeCamera(page);
  const tensor = await science(page);
  expect(tensor.transfer.mode).toEqual([1]);
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await embeddingDone(page, 1); await captureScience(page);
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
  // Neither card is used as the numerical oracle for the other.
  for (const card of [tensor, embeddings]) {
    expect(card.values).toEqual(expected.values);
    expect(card.counts).toEqual({ rows: expected.rows, columns: expected.columns });
    expect(card.domains).toEqual(Array.from({ length: 2 }, () => [String(expected.minimum), String(expected.maximum)]));
  }
  expect(embeddings.transfer).toEqual(tensor.transfer);
  await expect(page.locator('.tokenizer-workspace .viewer-panel')).toHaveCount(2);
  const resources = await metrics(page);
  await page.locator('.matrix-scroll').focus();
  await page.locator('.matrix-scroll').press('ArrowDown');
  await expect(page.locator('.inspection-readout')).toHaveText(`row 1 · column 0${value(ids[1]! * 576)}`);
  await expect(page.locator('[data-token-index="1"]')).toHaveAttribute('data-active-token', '');
  expect(await metrics(page)).toMatchObject({ uploads: resources.uploads, createdTextures: resources.createdTextures });
  expect(orderedRequests).toHaveLength(3);
  await info.attach('scientific-parity', { contentType: 'application/json', body: JSON.stringify({ ids, resources }) });
  await closeSession(page);
});
