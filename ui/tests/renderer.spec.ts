import { expect, test } from '@playwright/test';
import type {} from './renderer-harness';
import { readFile } from 'node:fs/promises';

import { color, green, intensity, luminance } from './scalar-oracle';

test.beforeEach(async ({ page }) => {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/renderer.html`);
  await page.waitForFunction(() => Boolean(window.harness));
});

for (const dpr of [1, 2]) {
  test.describe(`device pixel ratio ${dpr}`, () => {
    test.use({ deviceScaleFactor: dpr });

    test('asymmetric matrix orientation, physical pixels, prefix uploads and exact cells', async ({ page }, testInfo) => {
      const result = await page.evaluate(() => {
        const { create, pixels, metrics } = window.harness;
        const r = create([2, 3]);
        r.canvas.style.display = 'block';
        r.setView(50, 50);
        r.setTransfer({ anchors: [-1, 1] });
        r.upload(new Float32Array([-1, -0.5, 0, 0.25]));
        const partial = pixels(r);
        const pending = r.readCell(1, 2);
        const allocations = metrics.allocations;
        r.upload(new Float32Array([0.5, 1]));
        const complete = pixels(r);
        const rect = r.canvas.getBoundingClientRect();
        const answer = { partial, complete, pending, cell: r.readCell(1, 1), view: r.view,
          hit: r.cellAt(1.25 / devicePixelRatio, 1.25 / devicePixelRatio),
          outside: r.cellAt(3 / devicePixelRatio, 0), rect: { width: rect.width, height: rect.height },
          diagnostics: r.diagnostics, allocations, finalAllocations: metrics.allocations, error: r.canvas.getContext('webgl2')!.getError() };
        window.renderer = r;
        return answer;
      });
      expect(result.complete).toEqual([[-1, -0.5, 0].map((v) => green(v)), [0.25, 0.5, 1].map((v) => green(v))]);
      for (const row of result.complete) for (const pixel of row) expect(pixel[1]).toBeGreaterThan(pixel[0]!);
      expect(result.partial[1]![1]).toEqual([46, 61, 76, 255]);
      expect(result.pending?.state).toBe('pending');
      expect(result.cell).toMatchObject({ state: 'finite', row: 1, column: 1, value: 0.5 });
      expect(result.hit).toEqual({ row: 1, column: 1 });
      expect(result.outside).toBeNull();
      expect(result.rect).toEqual({ width: 3 / dpr, height: 2 / dpr });
      expect(result.view).toMatchObject({ width: 3, height: 2, dpr });
      expect(result.diagnostics).toMatchObject({ scalarTextures: 1, scalarBytes: 24, cpuBytes: 24 });
      expect(result.finalAllocations).toBe(result.allocations);
      expect(result.error).toBe(0);
      await testInfo.attach('pixel-and-resource-evidence', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
      expect(await page.screenshot({ clip: { x: 0, y: 0, width: 4, height: 4 }, scale: 'device' }))
        .toMatchSnapshot(`exact-device-pixels-dpr-${dpr}.png`, { maxDiffPixels: 0, threshold: 0 });
      await page.screenshot({ path: testInfo.outputPath(`exact-pixels-dpr-${dpr}.png`), scale: 'device' });
    });

    test('bands preserve first, last and boundary cells in a bounded native scroll viewport', async ({ page }) => {
      const initial = await page.evaluate(() => {
        const host = document.createElement('div');
        host.style.cssText = 'width:120px;height:120px;margin-left:0.25px;margin-top:0.25px';
        document.body.append(host);
        const v = new window.harness.TensorViewport(host, window.harness.descriptor([7, 9]), { textureLimit: 4, framebufferLimit: 4 });
        window.viewport = v;
        v.renderer.setTransfer({ anchors: [0, 62] });
        v.renderer.upload(Float32Array.from({ length: 63 }, (_, i) => i));
        v.refresh();
        return { pixels: window.harness.pixels(v.renderer), diagnostics: v.renderer.diagnostics, view: v.renderer.view };
      });
      expect(initial.diagnostics).toMatchObject({ scalarTextures: 6, scalarBytes: 252, cpuBytes: 252 });
      expect(initial.pixels[0]![0]).toEqual(green(0, 0, 62));
      for (const origin of [[2, 2], [5, 3], [0, 0]]) {
        const result = await page.evaluate(async ([x, y]) => {
          const v = window.viewport;
          v.host.scrollLeft = x! / devicePixelRatio;
          v.host.scrollTop = y! / devicePixelRatio;
          await new Promise(requestAnimationFrame);
          v.refresh();
          const rect = v.canvas.getBoundingClientRect();
          return { pixels: window.harness.pixels(v.renderer), view: v.renderer.view!,
            hit: v.renderer.cellAt(0.25 / devicePixelRatio, 0.25 / devicePixelRatio),
            physical: { x: rect.x * devicePixelRatio, y: rect.y * devicePixelRatio },
            size: [v.canvas.width, v.canvas.height], allocations: v.renderer.diagnostics.totalScalarAllocations };
        }, origin);
        expect(result.view).toMatchObject({ x: origin[0], y: origin[1], scrollWidth: 9 / dpr, scrollHeight: 7 / dpr });
        expect(result.size).toEqual([4, 4]);
        expect(result.allocations).toBe(6);
        expect(result.physical.x).toBeCloseTo(Math.round(result.physical.x), 5);
        expect(result.physical.y).toBeCloseTo(Math.round(result.physical.y), 5);
        expect(result.hit).toEqual({ row: origin[1], column: origin[0] });
        for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) {
          expect(result.pixels[y]![x]![1]).toBeCloseTo(green((origin[1]! + y) * 9 + origin[0]! + x, 0, 62)[1]!, 0);
        }
      }
    });
  });
}

test('vector, empty tensors, rejected descriptors and non-consecutive uploads', async ({ page }) => {
  const result = await page.evaluate(() => {
    const { create, pixels, TensorViewport, descriptor, metrics } = window.harness;
    const r = create([5], { textureLimit: 2 });
    r.upload(new Float32Array([-1, 0, 1]));
    r.setView(50, 50);
    const vector = pixels(r);
    const rejected: string[] = [];
    for (const action of [() => r.upload(new Float32Array([2]), 1), () => r.upload(new Float32Array(3)),
      () => create([2, 2, 2]), () => create([-1]), () => create([Number.MAX_SAFE_INTEGER])]) {
      try { action(); } catch (error) { rejected.push(String(error)); }
    }
    r.dispose();
    const before = metrics.allocations;
    const empties = [[0], [0, 4], [4, 0]].map((shape) => {
      const host = document.createElement('div'); document.body.append(host);
      const v = new TensorViewport(host, descriptor(shape));
      const state = { state: v.renderer.state, text: host.textContent, textures: v.renderer.diagnostics.scalarTextures };
      v.dispose(); return state;
    });
    return { vector, rejected, empties, emptyAllocations: metrics.allocations - before, live: metrics.live.size };
  });
  expect(result.vector).toHaveLength(1);
  expect(result.vector[0]).toHaveLength(5);
  expect(result.vector[0]![1]).toEqual([...color(0.5), 255]);
  expect(result.vector[0]![3]).toEqual([46, 61, 76, 255]);
  expect(result.rejected).toHaveLength(5);
  expect(result.empties).toEqual(Array(3).fill({ state: 'empty', text: 'Empty tensor', textures: 0 }));
  expect(result.emptyAllocations).toBe(0);
  expect(result.live).toBe(0);
});

test('late robust statistics, constant/fallback/extreme values and transfer-only redraws', async ({ page }) => {
  const result = await page.evaluate(() => {
    const { create, pixels, metrics } = window.harness;
    const r = create([10]);
    r.setView(50, 50);
    r.upload(new Float32Array([-3.4028234663852886e38, -1, -0.5, 0, 0.5, 1, 3.4028234663852886e38, NaN, Infinity, -Infinity]));
    const provisional = pixels(r);
    const before = { allocations: metrics.allocations, uploads: metrics.uploads };
    r.setTransfer({ statistics: { minimum: -3.4028234663852886e38, maximum: 3.4028234663852886e38, percentiles: { p01: -1, p99: 1 } } });
    const robust = pixels(r);
    r.setTransfer({ anchors: [-3.4028234663852886e38, 3.4028234663852886e38] });
    const extreme = pixels(r);
    r.setTransfer({ statistics: { minimum: -1, maximum: 1, percentiles: { p01: 0, p99: 0 } } });
    const fallback = pixels(r);
    r.setTransfer({ statistics: { minimum: 0, maximum: 0, percentiles: { p01: 0, p99: 0 } } });
    const constant = pixels(r);
    r.setTransfer({ anchors: [-1, 1], slope: 2 });
    const slope = pixels(r);
    r.setTransfer({ anchors: [1 + 1e-8, 1 + 3e-8] });
    const closeOutside = pixels(r);
    r.setTransfer({ anchors: [1 - 1e-8, 1 + 3e-8] });
    const closeInside = pixels(r);
    const after = { allocations: metrics.allocations, uploads: metrics.uploads };
    r.dispose();
    return { provisional, robust, extreme, fallback, constant, slope, closeOutside, closeInside, before, after };
  });
  expect(result.after).toEqual(result.before);
  expect(result.robust[0]!.slice(0, 7)).toEqual([-1, -1, -0.5, 0, 0.5, 1, 1].map((v) => green(v)));
  expect(result.fallback).toEqual(result.robust);
  expect(result.extreme[0]![0]).toEqual(green(-1));
  expect(result.extreme[0]![6]).toEqual(green(1));
  expect(result.constant[0]!.slice(0, 7)).toEqual(Array(7).fill([...color(0.5), 255]));
  expect(result.robust[0]!.slice(7)).toEqual(Array(3).fill([178, 51, 140, 255]));
  expect(result.provisional[0]![2]).not.toEqual(result.robust[0]![2]);
  expect(result.slope[0]![2]).toEqual(green(-0.5, -1, 1, 2));
  expect(result.closeOutside[0]![5]).toEqual(green(-1));
  expect(result.closeInside[0]![5]).toEqual(green(1, 1 - 1e-8, 1 + 3e-8));
});

for (const retainValues of [true, false]) {
  test(`context loss invalidates storage and explicit reconstruction (retain=${retainValues})`, async ({ page }) => {
    await page.evaluate((retainValues) => {
      const r = window.harness.create([2, 3], { retainValues, textureLimit: 2 });
      window.renderer = r;
      r.upload(new Float32Array([-1, 0, 1, 0.5]));
      r.setView(50, 50);
      r.draw();
      window.loss = r.canvas.getContext('webgl2')!.getExtension('WEBGL_lose_context')!;
      if (!window.loss) throw new Error('Context-loss testing extension unavailable.');
      window.loss.loseContext();
    }, retainValues);
    await page.waitForFunction(() => window.renderer.state === 'lost');
    const lost = await page.evaluate(() => {
      let rejected = false;
      try { window.renderer.draw(); } catch { rejected = true; }
      const result = { rejected, diagnostics: window.renderer.diagnostics };
      window.loss.restoreContext();
      return result;
    });
    expect(lost.rejected).toBe(true);
    expect(lost.diagnostics.scalarTextures).toBe(0);
    await page.waitForFunction(() => window.renderer.state === 'needs-reconstruction');
    const restored = await page.evaluate(() => {
      const r = window.renderer;
      const before = r.diagnostics.scalarUploadCalls;
      r.reconstruct();
      const pixels = window.harness.pixels(r);
      const result = { pixels, prefix: r.populatedPrefix, before, after: r.diagnostics.scalarUploadCalls, generations: r.diagnostics.generations };
      r.dispose();
      return { ...result, disposed: r.diagnostics, live: window.harness.metrics.live.size };
    });
    expect(restored.prefix).toBe(retainValues ? 4 : 0);
    expect(restored.generations).toBe(2);
    if (retainValues) {
      expect(restored.after).toBeGreaterThan(restored.before);
      expect(restored.pixels[0]![1]).toEqual([...color(0.5), 255]);
    } else {
      expect(restored.after).toBe(restored.before);
      expect(restored.pixels[0]![1]).toEqual([46, 61, 76, 255]);
    }
    expect(restored.disposed).toMatchObject({ state: 'disposed', scalarBytes: 0, cpuBytes: 0 });
    expect(restored.live).toBe(0);
  });
}

test('repeated disposal and partial allocation failure release every scalar texture', async ({ page }) => {
  const result = await page.evaluate(() => {
    const { create, metrics } = window.harness;
    for (let i = 0; i < 12; i++) {
      const r = create([5, 7], { textureLimit: 3 });
      r.upload(new Float32Array(35));
      r.setView(4, 4);
      r.draw();
      r.dispose(); r.dispose();
    }
    const before = metrics.live.size;
    metrics.failAllocation = metrics.allocations + 2;
    let failure = '';
    try { create([5, 7], { textureLimit: 3 }); } catch (error) { failure = String(error); }
    return { before, after: metrics.live.size, failure, allocations: metrics.allocations, deletes: metrics.deletes };
  });
  expect(result.before).toBe(0);
  expect(result.after).toBe(0);
  expect(result.failure).toContain('scalar allocation failed');
  expect(result.allocations).toBe(74);
  expect(result.deletes).toBe(74);
});

test('DPR changes explicitly rebuild geometry without scalar allocations or uploads', async ({ page }) => {
  await page.evaluate(() => {
    const host = document.createElement('div'); host.style.cssText = 'width:4px;height:4px'; document.body.append(host);
    window.viewport = new window.harness.TensorViewport(host, window.harness.descriptor([8, 10]));
    window.viewport.renderer.upload(new Float32Array(80));
    window.viewport.refresh();
  });
  const before = await page.evaluate(() => window.viewport.renderer.diagnostics);
  const session = await page.context().newCDPSession(page);
  await session.send('Emulation.setDeviceMetricsOverride', { width: 800, height: 600, deviceScaleFactor: 2, mobile: false });
  await page.waitForFunction(() => window.viewport.renderer.view?.dpr === 2);
  const after = await page.evaluate(() => ({ diagnostics: window.viewport.renderer.diagnostics, view: window.viewport.renderer.view }));
  expect(after.diagnostics.totalScalarAllocations).toBe(before.totalScalarAllocations);
  expect(after.diagnostics.scalarUploadCalls).toBe(before.scalarUploadCalls);
  expect(after.view).toMatchObject({ dpr: 2, width: 8, height: 8, scrollWidth: 5, scrollHeight: 4 });
});

test('consecutive chunks cross both band axes without populating unseen cells', async ({ page }) => {
  const snapshots = await page.evaluate(() => {
    const r = window.harness.create([5, 7], { textureLimit: 3 });
    r.setView(100, 100);
    r.setTransfer({ anchors: [0, 34] });
    const snapshots = [];
    for (const end of [2, 11, 13, 24, 35]) {
      const start = r.populatedPrefix;
      const chunk = Float32Array.from({ length: end - start }, (_, i) => start + i);
      r.upload(chunk);
      chunk.fill(999); // caller reuse cannot change retained exact values or GPU storage
      snapshots.push({ end, pixels: window.harness.pixels(r), exact: r.readCell(0, 0) });
    }
    r.dispose();
    return snapshots;
  });
  for (const { end, pixels, exact } of snapshots) {
    expect(exact).toMatchObject({ value: 0 });
    for (let y = 0; y < 5; y++) for (let x = 0; x < 7; x++) {
      const index = y * 7 + x;
      if (index >= end) expect(pixels[y]![x]).toEqual([46, 61, 76, 255]);
      else expect(pixels[y]![x]![1]).toBeCloseTo(green(index, 0, 34)[1]!, 0);
    }
  }
});

test('actual WebGL2 limits partition oversized dimensions and retain edge pixels', async ({ page }, testInfo) => {
  const result = await page.evaluate(() => {
    const probe = window.harness.create([1]);
    const limits = probe.limits;
    probe.dispose();
    const dimension = limits.textureSize + 3;
    const cases = [[2, dimension], [dimension, 2]].map((shape) => {
      const r = window.harness.create(shape);
      r.upload(Float32Array.from({ length: dimension * 2 }, (_, i) => i % 2 ? 1 : -1));
      r.setTransfer({ anchors: [-1, 1] });
      r.setView(3 / devicePixelRatio, 3 / devicePixelRatio,
        (r.geometry.columns - 3) / devicePixelRatio, (r.geometry.rows - 3) / devicePixelRatio);
      const result = { view: r.view, pixels: window.harness.pixels(r), diagnostics: r.diagnostics };
      r.dispose();
      return result;
    });
    return { limits, dimension, cases };
  });
  expect(result.limits.textureSize).toBeGreaterThanOrEqual(2048);
  for (const entry of result.cases) {
    expect(entry.diagnostics.scalarTextures).toBe(2);
    expect(entry.diagnostics.scalarBytes).toBe(result.dimension * 2 * 4);
    expect(entry.diagnostics.scalarUploadCalls).toBe(2);
    const view = entry.view!;
    const columns = view.scrollWidth; // default test context DPR=1
    for (let y = 0; y < view.height; y++) for (let x = 0; x < view.width; x++) {
      const value = ((view.y + y) * columns + view.x + x) % 2 ? 1 : -1;
      expect(entry.pixels[y]![x]).toEqual(green(value));
    }
  }
  await testInfo.attach('actual-webgl2-limits', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
});

test('unavailable WebGL2 is an explicit error with a visible retry explanation', async ({ page }) => {
  const result = await page.evaluate(() => {
    const host = document.createElement('div'); document.body.append(host);
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = (() => null) as typeof original;
    let error = '';
    try { new window.harness.TensorViewport(host, window.harness.descriptor([3])); }
    catch (reason) { error = String(reason); }
    finally { HTMLCanvasElement.prototype.getContext = original; }
    return { error, text: host.textContent, canvases: host.querySelectorAll('canvas').length };
  });
  expect(result.error).toContain('WebGL2 is unavailable');
  expect(result.text).toContain('retry with a supported browser');
  expect(result.canvases).toBe(0);
});

test('production non-React demonstration accepts progressive data and late statistics', async ({ page }, testInfo) => {
  await page.goto('/renderer-demo.html');
  await expect(page.locator('#progress')).toContainText('98,304 / 786,432');
  await page.getByRole('button', { name: 'Receive next chunk' }).click();
  await expect(page.locator('#progress')).toContainText('196,608 / 786,432');
  await page.getByRole('button', { name: 'Apply p01/p99 statistics' }).click();
  await expect(page.locator('canvas')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('renderer-demo.png'), fullPage: true });
});

test('shared API float32 fixture preserves row-major values, signed zero and nonfinite states', async ({ page }) => {
  const fixtures = JSON.parse(await readFile(new URL('../../api/fixtures/conformance.json', import.meta.url), 'utf8')) as {
    wire_cases: { name: string; expected: { metadata: { shape: number[] }; data_hex: string } }[];
  };
  const fixture = fixtures.wire_cases.find((entry) => entry.name === 'tensor-special-floats')!.expected;
  const result = await page.evaluate(({ metadata, data_hex }) => {
    const bytes = Uint8Array.from(data_hex.match(/../g)!, (hex) => parseInt(hex, 16));
    const data = new DataView(bytes.buffer);
    const values = Float32Array.from({ length: bytes.length / 4 }, (_, i) => data.getFloat32(i * 4, true));
    const r = window.harness.create(metadata.shape);
    r.setView(30, 30);
    r.upload(values.subarray(0, 1));
    r.upload(values.subarray(1));
    const cells = [r.readCell(0, 0), r.readCell(0, 1), r.readCell(0, 2)];
    const negativeZero = Object.is(cells[2] && 'value' in cells[2] ? cells[2].value : null, -0);
    return { cells, negativeZero, pixels: window.harness.pixels(r) };
  }, fixture);
  expect(result.cells[0]).toMatchObject({ value: 1.2345000505447388 });
  expect(result.cells[1]).toMatchObject({ value: -2.5 });
  expect(result.negativeZero).toBe(true);
  expect(result.pixels[1]).toEqual(Array(3).fill([178, 51, 140, 255]));
});

test('green levels are ordered and guides remain salient at both endpoints without uploads', async ({ page }) => {
  const result = await page.evaluate(() => {
    const { create, pixels, metrics } = window.harness;
    const r = create([1, 101], { textureLimit: 16 });
    r.upload(Float32Array.from({ length: 101 }, (_, i) => i / 50 - 1));
    r.setTransfer({ anchors: [-1, 1] }); r.setView(101, 1);
    const neutral = pixels(r);
    const before = { allocations: metrics.allocations, uploads: metrics.uploads };
    r.setSelection({ row: 0, column: 50 });
    const selected = pixels(r);
    r.setSelection(null);
    const cleared = pixels(r);
    const after = { allocations: metrics.allocations, uploads: metrics.uploads };
    r.dispose();
    return { neutral, selected, cleared, before, after };
  });
  expect(result.after).toEqual(result.before);
  expect(result.cleared).toEqual(result.neutral);
  let previous = -1;
  for (let i = 0; i <= 100; i++) {
    const t = intensity(Math.fround(i / 50 - 1));
    const rgb = result.neutral[0]![i]!;
    expect(rgb[1]).toBeGreaterThan(rgb[0]!);
    expect(rgb[1]).toBeGreaterThan(rgb[2]!);
    expect(luminance(rgb)).toBeGreaterThanOrEqual(previous); previous = luminance(rgb);
    const overlay = result.selected[0]![i]!;
    expect(overlay[0]! - overlay[1]!).toBeGreaterThan(30);
    for (const [actual, expected] of [[rgb, color(t)], [overlay, color(t, i === 50 ? 0.9 : 0.65)]]) {
      expected!.forEach((v, j) => expect(Math.abs(actual![j]! - v)).toBeLessThanOrEqual(1));
    }
  }
});

for (const scenario of ['tightly centered', 'outlier-heavy']) test(`robust contrast: ${scenario}`, async ({ page }, testInfo) => {
  const values = Array.from({ length: 101 }, (_, i) => Math.fround(0.01 + (i - 50) * 0.00001));
  if (scenario === 'outlier-heavy') { values[0] = -1e6; values[100] = 1e6; }
  const result = await page.evaluate((values) => {
    const { create, pixels, metrics } = window.harness;
    const r = create([101]); r.setView(101, 1); r.upload(new Float32Array(values));
    const before = { uploads: metrics.uploads, allocations: metrics.allocations };
    const transfer = { statistics: { minimum: values[0]!, maximum: values[100]!,
      percentiles: { p01: values[1]!, p99: values[99]! } } };
    r.setTransfer(transfer);
    const first = pixels(r);
    r.setTransfer(transfer);
    const second = pixels(r);
    const exact = values.map((_, i) => r.readCell(0, i));
    r.dispose();
    return { first, second, exact, before, after: { uploads: metrics.uploads, allocations: metrics.allocations } };
  }, values);
  expect(result.first).toEqual(result.second);
  expect(result.after).toEqual(result.before);
  values.forEach((value, i) => expect(result.exact[i]).toMatchObject({ value }));
  const tones = result.first[0]!;
  // Independent old default: slope 8, neutral linear Y encoded for display.
  const oldCode = (v: number) => Math.round(255 * (1.055 * intensity(v, values[1], values[99], 8) ** (1 / 2.4) - 0.055));
  const oldSpread = oldCode(values[60]!) - oldCode(values[40]!);
  const newSpread = tones[60]![1]! - tones[40]![1]!;
  expect(newSpread).toBeGreaterThan(oldSpread * 1.2);
  expect(luminance(tones[75]!) - luminance(tones[25]!)).toBeGreaterThan(0.5);
  // The sigmoid intentionally compresses tails; every central-half level stays distinct.
  expect(new Set(tones.slice(25, 76).map((p) => p.join(','))).size).toBe(51);
  await testInfo.attach('contrast-comparison', { body: JSON.stringify({ scenario, oldSpread, newSpread,
    p25Luminance: luminance(tones[25]!), p75Luminance: luminance(tones[75]!) }), contentType: 'application/json' });
});

test('fractional zoom samples exact scalar cells across texture bands and preserves the magnifier', async ({ page }, testInfo) => {
  const results = await page.evaluate(() => {
    const { create, pixels } = window.harness;
    const r = create([17, 19], { textureLimit: 8 });
    r.setTransfer({ anchors: [-1, 1] });
    r.upload(Float32Array.from({ length: 17 * 19 - 5 }, (_, i) => i % 2 ? 1 : -1));
    const initial = r.diagnostics;
    const output = [];
    for (const scale of [1, 2, 3.25, 4.9, 23.7]) {
      const view = r.setView(120, 100, 11, 13, devicePixelRatio, scale);
      const frame = pixels(r);
      const cells = frame.map((row, y) => row.map((_, x) => r.cellAt((x + .25) / devicePixelRatio, (y + .25) / devicePixelRatio)));
      const card = document.createElement('canvas'); card.width = card.height = 9;
      r.drawNeighborhood(8, 8, card.getContext('2d')!);
      output.push({ view, frame, cells, neighborhood: Array.from(card.getContext('2d')!.getImageData(0, 0, 9, 9).data) });
    }
    const final = r.diagnostics;
    r.dispose();
    return { output, initial, final };
  });
  for (const { frame, cells, view, neighborhood } of results.output) {
    for (let y = 0; y < frame.length; y++) for (let x = 0; x < frame[y]!.length; x++) {
      const cell = cells[y]![x]!;
      const index = cell.row * 19 + cell.column;
      const expected = index >= 17 * 19 - 5 ? [46, 61, 76, 255] : green(index % 2 ? 1 : -1);
      expect(frame[y]![x], `scale ${view.scaleX}, device pixel ${x}:${y}, cell ${cell.row}:${cell.column}`).toEqual(expected);
    }
    expect(neighborhood).toEqual(results.output[0]!.neighborhood);
  }
  expect(results.final.scalarUploadCalls).toBe(results.initial.scalarUploadCalls);
  expect(results.final.scalarBytes).toBe(results.initial.scalarBytes);
  await testInfo.attach('zoom-geometry', { body: JSON.stringify(results.output.map(({ view }) => view)), contentType: 'application/json' });
});


test('fractional selection guides stay inside the exact cell with one-pixel thickness', async ({ page }) => {
  const results = await page.evaluate(() => {
    const { create, pixels } = window.harness;
    const r = create([32, 32], { textureLimit: 8 });
    r.upload(new Float32Array(32 * 32));
    const initial = r.diagnostics;
    const output = [];
    const dpr = devicePixelRatio;
    for (const scale of [1, 1.01, 1.25, 1.5, 1.99, 3.25, 4.9]) {
      for (const scroll of [0, 1, 7]) {
        const view = r.setView(12, 12, scroll, scroll, dpr, scale);
        const selected = { row: Math.floor(view.y) + 1, column: Math.floor(view.x) + 1 };
        r.setSelection(selected);
        const frame = pixels(r);
        const cells = frame.map((row, y) => row.map((_, x) => r.cellAt((x + .25) / dpr, (y + .25) / dpr)));
        output.push({ view, selected, frame, cells });
      }
    }
    const final = r.diagnostics;
    r.dispose();
    return { output, initial, final };
  });
  for (const { frame, cells, selected, view } of results.output) {
    const rowPixels = new Set<number>(), columnPixels = new Set<number>();
    let intersections = 0;
    for (let y = 0; y < frame.length; y++) for (let x = 0; x < frame[y]!.length; x++) {
      const pixel = frame[y]![x]!, cell = cells[y]![x]!;
      if (pixel[0]! <= pixel[1]!) continue;
      const row = cell.row === selected.row, column = cell.column === selected.column;
      expect(row || column, `scale ${view.scaleX}, origin ${view.x}, pixel ${x}:${y}`).toBe(true);
      if (row && !column) rowPixels.add(y);
      if (column && !row) columnPixels.add(x);
      if (pixel.every((v, i) => Math.abs(v - [...color(.5, .9), 255][i]!) <= 1)) {
        expect(row && column).toBe(true);
        intersections++;
      }
    }
    expect(rowPixels.size).toBe(1);
    expect(columnPixels.size).toBe(1);
    expect(intersections).toBe(1);
    const guideY = [...rowPixels][0]!, guideX = [...columnPixels][0]!;
    for (let y = 0; y < frame.length; y++) for (let x = 0; x < frame[y]!.length; x++) {
      const expected = [...color(.5, y === guideY && x === guideX ? .9 : y === guideY || x === guideX ? .65 : 0), 255];
      frame[y]![x]!.forEach((v, i) => expect(Math.abs(v - expected[i]!)).toBeLessThanOrEqual(1));
    }
  }
  expect(results.final.scalarUploadCalls).toBe(results.initial.scalarUploadCalls);
  expect(results.final.scalarBytes).toBe(results.initial.scalarBytes);
});
