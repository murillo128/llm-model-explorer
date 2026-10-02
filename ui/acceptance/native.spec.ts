/* eslint-disable @typescript-eslint/no-explicit-any -- Test-only JSON evidence. */
import { expect } from '@playwright/test';
import { test, value } from './product-harness';
import { camera, zoom, drag, panelGeometry, promptViewport } from './usability';
import { nativeCamera } from '../tests/native-camera';
import { revealTensor } from '../tests/tensor-tree-helpers';

// Logical transport cases use headless Chromium; native focus/geometry stays headed.
test.use({ headless: false });

test.describe('production native pane geometry', () => {
  for (const viewport of [{ width: 1000, height: 700 }, { width: 390, height: 640 }]) {
    test(`compact shell, navigation and four overflow modes at ${viewport.width}×${viewport.height}`, { tag: viewport.width === 390 ? '@density' : '@extended' }, async ({ page, product }, testInfo) => {
  const { open, complete, closeSession, expectDocumentFits, timing } = product;
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
          return { dpr: devicePixelRatio, horizontal: m.scrollWidth > m.clientWidth, vertical: m.scrollHeight > m.clientHeight,
            gutter: m.offsetWidth - m.clientWidth, scroll: [m.scrollLeft, m.scrollTop],
            extent: [m.scrollWidth, m.scrollHeight], client: [m.clientWidth, m.clientHeight],
            matrix: rect('.matrix-scroll canvas'), row: rect('.row-distributions canvas'), column: rect('.column-distributions canvas'),
            origins: ['.matrix-scroll canvas', '.row-distributions canvas', '.column-distributions canvas'].map(s => document.querySelector(s)!.getAttribute('data-origin')),
            pageGeometry: { document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
              body: [document.body.scrollWidth, document.body.scrollHeight], scroll: [scrollX, scrollY] } };
        });
        let initial!: Awaited<ReturnType<typeof geometry>>;
        await expect.poll(async () => { initial = await geometry(); return [initial.horizontal, initial.vertical]; }).toEqual([horizontal, vertical]);
        expect(initial.gutter).toBe(0);
        const inventory = page.getByRole('complementary', { name: 'Tensor inventory' });
        await inventory.evaluate(e => { e.scrollTop = e.scrollHeight; });
        expect(await inventory.evaluate(e => e.scrollTop)).toBeGreaterThan(0);
        expect((await geometry()).scroll).toEqual(initial.scroll);
        for (const [x, y] of [[0, 0], [1e9, 0], [0, 1e9], [1e9, 1e9]]) {
          await page.locator('.matrix-scroll').evaluate((e, [x, y]) => { e.scrollLeft = x!; e.scrollTop = y!; }, [x, y]);
          let g!: Awaited<ReturnType<typeof geometry>>;
          await expect.poll(async () => {
            g = await geometry();
            // Independent native scroll/DPR oracle, observed in the same settled
            // frame as all three renderer origins and their rectangles.
            const x = Math.round(g.scroll[0]! * g.dpr), y = Math.round(g.scroll[1]! * g.dpr);
            return g.origins.map((origin, i) => origin === [`${x},${y}`, `0,${y}`, `${x},0`][i]);
          }).toEqual([true, true, true]);
          expect(g.matrix[1]).toBe(g.row[1]); expect(g.matrix[0]).toBe(g.column[0]);
          expect(g.matrix[3]).toBe(g.row[3]); expect(g.matrix[2]).toBe(g.column[2]);
          expect(g.scroll).toEqual([x ? g.extent[0]! - g.client[0]! : 0, y ? g.extent[1]! - g.client[1]! : 0]);
          expectDocumentFits(page, g.pageGeometry); evidence.push({ name, ...g });
        }
        timing.mark(`native pane ${name}`);
      }
      await expect(page.locator('.matrix-panel-header')).not.toContainText('complete');
      await page.screenshot({ path: testInfo.outputPath('compact-tensor.png') });
      await testInfo.attach('measurements', { body: JSON.stringify({ scenario: 'native geometry', viewport,
        limits: await page.evaluate(() => (window as any).__acceptance.limits()), cases: evidence }), contentType: 'application/json' });
      await closeSession(page);
    });
  }
});

test('integrated inventory preferences and metadata preserve streaming panel geometry', async ({ page, product }, testInfo) => {
  const { control, metrics, open, complete, closeSession, documentFits } = product;
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

test('production stale results remain visible and generation-fenced while embedding camera leaves prompt intact', async ({ page, product }, testInfo) => {
  const { control, metrics, tokenizer, closeSession, embeddingDone, documentFits } = product;
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

test('production inspection tolerates DPR change before viewport resize notification', async ({ page, product }) => {
  const { metrics, open, complete, closeSession } = product;
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

test('integrated camera gestures, exact selection, aligned scales and adaptive inspection retain scalar storage', async ({ page, product }, testInfo) => {
  const { metrics, open, complete, closeSession, documentFits, timing } = product;
  await open(page, 'layout.fits.weight'); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  let c = await camera(page, 32, 32);
  expect(c.scaleX).toBeCloseTo(Math.max(1, Math.floor(c.width * c.dpr) / 32));
  expect(c.scaleY).toBeCloseTo(c.scaleX);
  await open(page); await complete(page);
  await expect(page.locator('[data-result=distributions]')).toHaveCount(0);
  await nativeCamera(page);
  expect((await camera(page, 576, 1536)).scaleX).toBe(1);
  timing.mark('camera native lower bound');
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
  timing.mark('camera wheel and pinch');
  await page.getByRole('button', { name: 'Fit width', exact: true }).click();
  await zoom(page, 576, 1536, 7);
  c = await camera(page, 576, 1536);
  await page.mouse.click(c.rect.left + 3.5 * 7, c.rect.top + 3.5 * 7);
  await expect(page.locator('.inspection-readout')).toHaveText(`row 3 · column 3${value(3 * 1536 + 3)}`);
  expect(await page.locator('.distribution-scale').evaluateAll(nodes => nodes.map(n => n.getAttribute('aria-label')))).toEqual(rulers);
  expect(await metrics(page)).toMatchObject({ uploads: resource.uploads, createdTextures: resource.createdTextures, gpuBytes: resource.gpuBytes, errors: [] });
  expect(await canvas!.evaluate(node => node === document.querySelector('.matrix-scroll canvas'))).toBe(true);
  await documentFits(page);
  await testInfo.attach('measurements', { body: JSON.stringify({ fit: c, focalBefore: after, resources: await metrics(page), rulers }), contentType: 'application/json' });
  await closeSession(page);
});
