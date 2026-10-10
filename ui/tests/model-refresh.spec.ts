import { expect, test } from '@playwright/test';
import { models, sessionA } from '../src/test/shell-fixtures';

// Controller/unit tests own revision races. This browser bridge owns mounted
// CodeMirror, native selection/history/composition and current-source replay.
test('same-model refresh preserves the native editor and pending edits through pinning', async ({ page }) => {
  await page.addInitScript(() => {
    const sources: EventTarget[] = [];
    let subscriptions = 0;
    Object.defineProperty(window, 'modelSubscriptions', { get: () => subscriptions });
    class Source extends EventTarget {
      constructor() { super(); sources.push(this); subscriptions++; }
      close() { sources.splice(sources.indexOf(this), 1); }
    }
    Object.defineProperty(window, 'EventSource', { value: Source });
    Object.defineProperty(window, 'emitModelState', { value: (revision: string) => {
      const epoch = '11111111-1111-4111-8111-111111111111';
      sources.at(-1)!.dispatchEvent(new MessageEvent('model-state', { lastEventId: `${epoch}:1`, data: JSON.stringify({
        epoch, sequence: 1, model_id: 'lab/alpha', status: 'present', model_revision: revision,
      }) }));
    } });
  });
  const requests: { session: string; text: string; add_special_tokens: boolean }[] = [];
  let pins = 0, release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  const next = { ...sessionA, id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', model_revision: 'B' };
  await page.route('**/runtime-config.json', route => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  await page.route('https://backend.example/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (path === '/models') return route.fulfill({ json: { models, diagnostics: [] } });
    if (path === '/sessions') {
      pins++;
      if (pins === 2) await pending;
      return route.fulfill({ status: 201, json: pins === 1 ? sessionA : next });
    }
    if (request.method() === 'DELETE') return route.fulfill({ status: 204 });
    if (path.endsWith('/tensors')) return route.fulfill({ json: { coverage: 'complete', diagnostics: [], tensors: [] } });
    if (path.endsWith('/tokenize')) {
      const body = request.postDataJSON() as { text: string; add_special_tokens: boolean };
      requests.push({ session: path.split('/')[2]!, ...body });
      return route.fulfill({ json: { ...body, tokens: [] } });
    }
    return route.fulfill({ status: 422, json: { code: 'unsupported_representation', message: 'Fixture capability unavailable' } });
  });
  await page.goto('/');
  await page.getByRole('combobox').selectOption(sessionA.model_id);
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  const editor = page.getByRole('textbox', { name: 'Prompt' });
  await editor.fill('alpha');
  await expect.poll(() => requests.at(-1)?.text).toBe('alpha');
  await editor.evaluate(node => { node.dataset.owner = 'original'; });
  await editor.press('End'); await editor.press('Shift+ArrowLeft');
  expect(await page.evaluate(() => getSelection()?.toString())).toBe('a');
  const emit = (revision: string) => page.evaluate(value => {
    (window as unknown as { emitModelState: (value: string) => void }).emitModelState(value);
  }, revision);
  await emit('B');
  await expect(page.getByRole('contentinfo')).toContainText('Updating model');
  await expect(editor).toHaveAttribute('data-owner', 'original');
  expect(await page.evaluate(() => getSelection()?.toString())).toBe('a');
  await editor.press('End'); await editor.press('!');
  await editor.dispatchEvent('compositionstart', { data: '' });
  release();
  await expect.poll(() => page.evaluate(() => (window as unknown as { modelSubscriptions: number }).modelSubscriptions)).toBe(2);
  await emit('B');
  await expect(page.getByText('Model updated.', { exact: true })).toBeVisible();
  await expect(editor).toHaveAttribute('data-owner', 'original');
  await expect(page.getByText(/Composing text/)).toBeVisible();
  expect(requests.filter(request => request.session === next.id)).toEqual([]);
  await editor.dispatchEvent('compositionend', { data: '' });
  await expect.poll(() => requests.at(-1)).toEqual({ session: next.id, text: 'alpha!', add_special_tokens: true });
  await editor.press('Control+z');
  await expect.poll(() => requests.at(-1)?.text).toBe('alpha');
  await expect(editor).toHaveAttribute('data-owner', 'original');
  expect(pins).toBe(2);
});
