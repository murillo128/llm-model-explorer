/* eslint-disable @typescript-eslint/no-explicit-any -- Exercise the injected test-only observer. */
import { test, expect } from '@playwright/test';
import { installProbe } from '../acceptance/probe';

for (const capturePixels of [false, true]) test(`probe capture ${capturePixels}: real draws retain independent counters and exact evidence`, async ({ page }) => {
  await page.addInitScript(installProbe, { capturePixels });
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/renderer.html`);
  await page.waitForFunction(() => Boolean(window.harness));
  const result = await page.evaluate(() => {
    const host = document.createElement('div'); host.className = 'matrix-scroll'; document.body.append(host);
    const renderer = window.harness.create([2, 3]); host.append(renderer.canvas); renderer.canvas.id = 'observed';
    const probe = (window as any).__acceptance;
    const query = () => { try { return probe.pixel('#observed', 0, 0); } catch (error) { return String(error); } };
    const missing = query();
    renderer.setView(3, 2); renderer.setTransfer({ anchors: [-1, 1] });
    probe.captureScalars = true; renderer.upload(new Float32Array([-1, 0, 1, 1, 0, -1]));
    renderer.draw();
    // Count uploads are separately controlled and remain exact with capture off.
    const gl = renderer.canvas.getContext('webgl2')!;
    host.classList.add('row-distributions'); probe.captureCounts = true;
    const counts = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, counts);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32UI, 2, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 2, 1, gl.RED_INTEGER, gl.UNSIGNED_INT, new Uint32Array([16909060, 4294967295]));
    gl.deleteTexture(counts); host.classList.remove('row-distributions');
    const reader = new ReadableStream().getReader(); reader.releaseLock();
    const first = structuredClone(probe.metrics()), pixel = query();
    renderer.draw(); const second = probe.metrics();
    const texture = renderer.canvas.getContext('webgl2')!;
    const bytes = new Uint8Array(4); texture.readPixels(0, 1, 1, 1, texture.RGBA, texture.UNSIGNED_BYTE, bytes);
    texture.enable(0xDEAD); texture.getError();
    renderer.canvas.dispatchEvent(new Event('webglcontextlost', { bubbles: true, cancelable: true }));
    renderer.dispose();
    return { missing, pixel, actual: [...bytes], first, second, final: probe.metrics(), values: probe.scalarValues, counts: probe.countValues };
  });
  expect(result.first.uploads).toBeGreaterThan(0);
  expect(result.first.firstUpload).toBeGreaterThan(0); expect(result.first.firstRender).toBeGreaterThan(0);
  expect(result.first.createdTextures).toBe(2); expect(result.final.textures).toBe(0);
  expect(result.values).toEqual([-1, 0, 1, 1, 0, -1]);
  expect(result.counts).toEqual({ rows: [16909060, 4294967295], columns: [] });
  expect(result.final.errors).toMatchObject([{ code: 1280 }]); expect(result.final.contextLosses).toBe(1);
  expect(result.first.errors).toEqual([]); expect(result.first.peakReaders).toBe(1); expect(result.first.readers).toBe(0);
  if (capturePixels) {
    expect(result.missing).toContain('Missing pixel snapshot');
    expect(result.pixel).toEqual(result.actual);
    expect(result.first.framebufferReadbacks).toBeGreaterThan(0);
    expect(result.second.framebufferReadbacks).toBe(result.first.framebufferReadbacks + 1);
    expect(result.second.snapshotAllocations).toBe(result.first.snapshotAllocations);
  } else {
    expect(result.missing).toContain('capture is disabled'); expect(result.pixel).toContain('capture is disabled');
    expect(result.second.framebufferReadbacks).toBe(0); expect(result.second.snapshotAllocations).toBe(0);
  }
});

test('probe invalidates resized, DPR-changed, replaced and lost canvas evidence', async ({ page }) => {
  await page.addInitScript(installProbe, { capturePixels: true });
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/renderer.html`);
  await page.waitForFunction(() => Boolean(window.harness));
  const result = await page.evaluate(() => {
    const r = window.harness.create([4, 4]); r.canvas.id = 'observed'; r.setView(4, 4); r.upload(new Float32Array(16)); r.draw();
    const p = (window as any).__acceptance;
    const query = (x = 0, y = 0) => { try { return p.pixel('#observed', x, y); } catch (error) { return String(error); } };
    const before = query();
    const sameWidth = r.canvas.width; r.canvas.width = sameWidth; const sameSize = query(); r.draw(); const redrawn = query();
    r.canvas.setAttribute('width', String(sameWidth)); const attributeReset = query(); r.draw();
    r.canvas.getAttributeNode('height')!.value = String(r.canvas.height); const attributeNodeReset = query(); r.draw();
    const parent = r.canvas.parentElement!; r.canvas.remove();
    r.canvas.setAttribute('height', String(r.canvas.height)); parent.append(r.canvas);
    const detachedReset = query(); r.draw();
    r.canvas.height = 3; const resized = query(); r.setView(4, 3); r.draw(); const afterResize = query();
    const dpr = devicePixelRatio; Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: dpr + 1 });
    const changedDpr = query(); r.setView(4, 3, 0, 0, dpr + 1); r.draw(); const afterDpr = query();
    Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: dpr }); r.setView(4, 3, 0, 0, dpr); r.draw();
    const invalid = [[-1, 0], [4, 0], [0, 3], [.5, 0], [NaN, 0]].map(([x, y]) => query(x, y));
    r.canvas.dispatchEvent(new Event('webglcontextlost', { bubbles: true, cancelable: true })); const lost = query();
    const losses = p.metrics().contextLosses;
    const replacement = document.createElement('canvas'); replacement.id = 'observed'; replacement.width = 4; replacement.height = 3;
    r.canvas.replaceWith(replacement); const replaced = query();
    replacement.remove(); const absent = query(); r.dispose();
    return { before, sameSize, redrawn, attributeReset, attributeNodeReset, detachedReset, resized, afterResize, changedDpr, afterDpr, invalid, lost, losses, replaced, absent };
  });
  expect(result.before).toHaveLength(4); expect(result.redrawn).toEqual(result.before);
  for (const reset of [result.attributeReset, result.attributeNodeReset, result.detachedReset]) expect(reset).toContain('Missing pixel snapshot');
  expect(result.sameSize).toContain('Missing pixel snapshot'); expect(result.resized).toContain('Missing pixel snapshot');
  expect(result.afterResize).toHaveLength(4); expect(result.changedDpr).toContain('dimensions/DPR'); expect(result.afterDpr).toHaveLength(4);
  for (const invalid of result.invalid) expect(invalid).toContain('Pixel coordinates');
  expect(result.lost).toContain('Missing pixel snapshot'); expect(result.losses).toBe(1);
  expect(result.replaced).toContain('Missing pixel snapshot'); expect(result.absent).toContain('requires a canvas');
});
