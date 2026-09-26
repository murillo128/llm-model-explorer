import { expect, test } from '@playwright/test';
import type { BrowserContext, Page } from '@playwright/test';
import { models, sessionA, tensors } from '../src/test/shell-fixtures';

async function fixture(context: BrowserContext) {
  const sessions = new Map<string, { id: string; model_id: string }>();
  const deletes: string[] = [];
  const gets: string[] = [];
  let serial = 0;
  await context.route('**/runtime-config.json', (route) => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  await context.route('https://backend.example/**', async (route) => {
    const request = route.request(); const path = new URL(request.url()).pathname;
    if (path === '/models') return route.fulfill({ json: { models, diagnostics: [] } });
    if (path === '/sessions' && request.method() === 'POST') {
      const session = { id: `aaaaaaaa-aaaa-4aaa-8aaa-${String(++serial).padStart(12, '0')}`, model_id: request.postDataJSON().model_id as string };
      sessions.set(session.id, session);
      return route.fulfill({ status: 201, json: session });
    }
    const id = path.split('/')[2]!;
    if (request.method() === 'DELETE') {
      deletes.push(id); sessions.delete(id); return route.fulfill({ status: 204 });
    }
    if (!sessions.has(id)) return route.fulfill({ status: 404, json: { code: 'session_not_found', message: 'Session not found' } });
    if (path.endsWith('/tensors')) return route.fulfill({ json: { tensors, coverage: 'complete', diagnostics: [] } });
    gets.push(id); return route.fulfill({ json: sessions.get(id) });
  });
  return { sessions, deletes, gets };
}
async function choose(page: Page, id = models[0]!.id) {
  await expect(page.getByRole('combobox')).toBeEnabled();
  await page.getByRole('combobox').selectOption(id);
  await expect(page.getByText('Session active.')).toBeVisible();
}

test('browser tabs create, recover, expire and close independent sessions', async ({ page, context }) => {
  const backend = await fixture(context);
  await page.goto('/'); await choose(page);
  const first = [...backend.sessions.keys()][0]!;
  await page.reload(); await expect(page.getByText('Session active.')).toBeVisible();
  expect(backend.gets).toContain(first);
  const tab = await context.newPage();
  await tab.goto('/');
  await expect(tab.getByRole('combobox', { name: 'Model' })).toHaveValue('');
  await choose(tab, models[1]!.id);
  const second = [...backend.sessions.keys()][1]!;
  await page.getByRole('button', { name: 'Session options', exact: true }).click();
  await page.getByRole('button', { name: 'Close session' }).click();
  await expect(page.getByText('Session closed.')).toBeVisible();
  expect(backend.deletes).toEqual([first]); expect(backend.sessions.has(second)).toBe(true);
  await tab.reload(); await expect(tab.getByText('Session active.')).toBeVisible();
  backend.sessions.delete(second);
  await tab.reload();
  await expect(tab.getByText(/Session expired/)).toBeVisible();
  await choose(tab);
  await expect(tab.getByText('Session active.')).toBeVisible();
  await tab.close();
});

test('keyboard operates logical disclosures, duplicate leaves, unsupported ranks and both tools', async ({ page, context }, testInfo) => {
  await fixture(context); await page.goto('/'); await choose(page);
  const leftBranch = page.locator('summary').filter({ hasText: /^left$/ });
  await leftBranch.focus();
  expect(await leftBranch.evaluate((element) => getComputedStyle(element).outlineStyle)).toBe('solid');
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: /left.weight/ })).not.toBeVisible();
  await page.keyboard.press('Enter'); await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: /left.weight/ })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'left › weight' })).toBeVisible();
  const right = page.getByRole('button', { name: /right.weight/ });
  await right.focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'right › weight' })).toBeVisible();
  await page.getByRole('button', { name: /cube/ }).focus(); await page.keyboard.press('Enter');
  await expect(page.getByText(/This rank-3 tensor/)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('session-navigation.png'), fullPage: true });
  await testInfo.attach('session navigation', { path: testInfo.outputPath('session-navigation.png'), contentType: 'image/png' });
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'Tokenizer Explorer', exact: true })).toHaveAttribute('aria-current', 'page');
  await expect(page.getByText('Session active.')).toBeVisible();
});

test('backend URL changes on refresh never recover the previous backend session', async ({ page, context }) => {
  await fixture(context); await page.goto('/'); await choose(page);
  await context.unroute('**/runtime-config.json');
  await context.route('**/runtime-config.json', (route) => route.fulfill({ json: { backend_base_url: 'https://second.example' } }));
  const requests: string[] = [];
  await context.route('https://second.example/**', (route) => {
    requests.push(route.request().url()); return route.fulfill({ json: { models: [], diagnostics: [] } });
  });
  await page.reload();
  await expect(page.getByText('No models available on this backend.')).toBeVisible();
  expect(requests).toEqual(['https://second.example/models']);
  await expect(page.getByRole('combobox', { name: 'Model' })).toHaveValue('');
});

test('invalid and unreachable backend responses expose no private error details', async ({ page, context }) => {
  await fixture(context);
  await context.route('https://backend.example/models', (route) => route.fulfill({ json: { models: [{ ...models[0], local_path: '/srv/private/model' }], diagnostics: [] } }));
  await page.goto('/'); await expect(page.getByRole('alert')).toContainText('invalid response');
  await expect(page.locator('body')).not.toContainText('/srv/private');
  await context.unroute('https://backend.example/models');
  await context.route('https://backend.example/models', (route) => route.abort('connectionrefused'));
  await page.getByRole('button', { name: 'Refresh models' }).click();
  await expect(page.getByRole('alert')).toContainText('unreachable');
});

test('missing session record is visibly expired after reload', async ({ page, context }) => {
  await fixture(context); await page.goto('/');
  await expect(page.getByRole('combobox')).toBeEnabled();
  await page.evaluate((id) => sessionStorage.setItem('llm-model-explorer:session:https://backend.example', id), sessionA.id);
  await page.reload(); await expect(page.getByText(/Session expired/)).toBeVisible();
});
