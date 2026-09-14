import { expect, test } from '@playwright/test';
import type { Route } from '@playwright/test';
import { models, sessionA, sessionB } from '../src/test/shell-fixtures';
import { frame, meta, data } from './embedding-fixtures';

test('production shell model A→B→A never revives another model embedding result', async ({ page }) => {
  const pending: Route[] = [];
  await page.route('**/runtime-config.json', route => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  await page.route('https://backend.example/**', route => {
    const path = new URL(route.request().url()).pathname;
    if (route.request().method() === 'DELETE') return route.fulfill({ status: 204 });
    if (path === '/models') return route.fulfill({ json: { models: models.map(model => ({ ...model, tokenizer_available: true })) } });
    if (path === '/sessions') return route.fulfill({ status: 201, json: route.request().postDataJSON().model_id === models[0]!.id ? sessionA : sessionB });
    if (path.endsWith('/tensors')) return route.fulfill({ json: { coverage: 'complete', diagnostics: [], tensors: [] } });
    if (path.endsWith('/tokenize')) return route.fulfill({ json: { ...route.request().postDataJSON(), tokens: [
      { index: 0, id: path.includes(sessionA.id) ? 2 : 1, token: '<bos>', decoded: '', special: true },
    ] } });
    if (path.endsWith('/embeddings')) { pending.push(route); return; }
    return route.abort();
  });
  const complete = async (index: number, ids: number[], width: number) => pending[index]!.fulfill({
    contentType: 'application/vnd.llm-model-explorer.stream', headers: { 'Access-Control-Expose-Headers': 'X-Operation-Id', 'X-Operation-Id': `01234567-89ab-cdef-0123-${String(index).padStart(12, '0')}` },
    body: Buffer.from([...meta(ids, width), ...data(ids, width), ...frame(4)]),
  });
  await page.goto('/'); await page.getByRole('combobox').selectOption(models[0]!.id);
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await expect.poll(() => pending.length).toBe(1);
  await page.getByRole('combobox').selectOption(models[1]!.id);
  await expect.poll(() => pending.length).toBe(2);
  expect(pending[1]!.request().postDataJSON()).toEqual({ token_ids: [1] });
  await complete(1, [1], 9);
  await expect(page.getByText('[1 × 9] · float32')).toBeVisible();
  await complete(0, [2], 3);
  await expect(page.getByText('[1 × 9] · float32')).toBeVisible();
  await expect(page.locator('[data-token-index="0"]')).toHaveText('1');
  await page.getByRole('combobox').selectOption(models[0]!.id);
  await expect.poll(() => pending.length).toBe(3);
  await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  await complete(2, [2], 3);
  await expect(page.getByText('[1 × 3] · float32')).toBeVisible();
  await expect(page.locator('[data-token-index="0"]')).toHaveText('2');
});
