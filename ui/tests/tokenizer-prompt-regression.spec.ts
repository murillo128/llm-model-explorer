import { expect, test } from '@playwright/test';
import { frame, meta, data } from './embedding-fixtures';
import { models, sessionA } from '../src/test/shell-fixtures';

// Recorded at pinned integration base 8166d94, before issue #44 source edits.
// Tokenizer source at that base is identical to planning baseline a65025c.
test('frozen prompt pixels and geometry', async ({ page }) => {
  await page.route('**/runtime-config.json', route => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  await page.route('https://backend.example/**', route => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/models') return route.fulfill({ json: { models } });
    if (path === '/sessions') return route.fulfill({ status: 201, json: sessionA });
    if (path.endsWith('/tensors')) return route.fulfill({ json: { tensors: [] } });
    if (path.endsWith('/tokenize')) {
      const { text } = route.request().postDataJSON() as { text: string };
      return route.fulfill({ json: { text, add_special_tokens: true, tokens: text ? [
        { index: 0, id: 1, token: '<bos>', decoded: '', special: true },
        { index: 1, id: 10, token: 'A', decoded: 'A', special: false, start: 0, end: 1 },
        { index: 2, id: 20, token: '😀', decoded: '😀', special: false, start: 1, end: 2 },
        { index: 3, id: 21, token: '😀', decoded: '😀', special: false, start: 1, end: 2 },
        { index: 4, id: 10, token: ' A', decoded: ' A', special: false, start: 2, end: 4 },
      ] : [] } });
    }
    if (path.endsWith('/embeddings')) {
      const { token_ids } = route.request().postDataJSON() as { token_ids: number[] };
      return route.fulfill({ contentType: 'application/vnd.llm-model-explorer.stream',
        headers: { 'Access-Control-Expose-Headers': 'X-Operation-Id', 'X-Operation-Id': '01234567-89ab-cdef-0123-456789abcdef' },
        body: Buffer.from([...meta(token_ids), ...data(token_ids), ...frame(4)]) });
    }
    return route.fulfill({ status: 422, json: { code: 'unsupported_representation', message: 'Fixture has no embedding table' } });
  });
  await page.goto('/');
  await page.getByRole('combobox').selectOption(models[0]!.id);
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  const prompt = page.getByRole('region', { name: 'Live prompt tokenization' });
  await expect(prompt.getByRole('status')).toContainText('0 tokens');
  await expect(prompt).toHaveScreenshot('empty-prompt.png');
  await page.getByRole('textbox', { name: 'Prompt' }).fill('A😀 A');
  await expect(prompt.getByRole('status')).toContainText('5 tokens');
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).focus();
  await expect(page.getByText('5 token rows · 7 hidden dimensions')).toBeVisible();
  await expect(prompt).toHaveScreenshot('annotated-prompt.png');
  expect((await page.locator('.tokenizer-editor').boundingBox())!.height).toBe(260);
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight && document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
