/* eslint-disable @typescript-eslint/no-explicit-any -- Test-only JSON evidence. */
import { expect } from '@playwright/test';
import { test, value } from './product-harness';
import { nativeCamera } from '../tests/native-camera';
import { revealTensor } from '../tests/tensor-tree-helpers';

// Logical transport cases use headless Chromium; native focus/geometry stays headed.
test.use({ headless: false });

test('local reference Base opens normalization, both MLP orientations and embedding with exact samples', { tag: '@extended' }, async ({ page, product }, testInfo) => {
  const { backend, referenceSamples } = product;
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

for (const family of ['qwen3_5']) test(`real ${family} input embeddings preserve token linkage and recover after unsupported model`, async ({ page, product }) => {
  const { tokenizer, closeSession, embeddingDone } = product;
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
