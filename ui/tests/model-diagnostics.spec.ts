import { expect, test } from '@playwright/test';
import type { BrowserContext, Page } from '@playwright/test';
import type { components } from '../src/api/generated/types';
import { contractInventory, contractResponse } from './architecture-fixtures';
type Response = components['schemas']['ArchitectureResponse'];
const precise = 'Invalid model-owned architecture: node "projection" references missing port "output".';
function authored(many = false): Response {
  const response = structuredClone(contractResponse);
  response.graph.scope = 'model_defined'; response.graph.coverage = 'partial';
  response.graph.nodes.push({ id: 'checkpoint', parent_id: 'root', label: 'Trained checkpoint', kind: 'input',
    ports: [{ id: 'in', direction: 'input', label: 'ambiguous', shape: null }], parameter_ids: [], references: [], attributes: [], provenance: [] });
  const root = response.graph.nodes.find((n) => n.id === 'root')!;
  if (root.kind === 'group') root.children.push('checkpoint');
  const diagnostic = { code: 'author_declared_partial', message: 'Only the trained checkpoint is described.', node_id: 'linear0' };
  response.diagnostics = [diagnostic]; response.graph.diagnostics = [diagnostic, { ...diagnostic, node_id: 'linear1' }];
  if (many) response.graph.diagnostics.push(...Array.from({ length: 30 }, (_, i) => ({ code: `detail_${i}`, message: `Finding ${i}: ${'Complete explanatory evidence. '.repeat(30)}`, node_id: 'checkpoint' })));
  return response;
}
async function backend(context: BrowserContext, initial: Response) {
  let response = initial;
  const requests: string[] = [];
  await context.route('**/runtime-config.json', (r) => r.fulfill({ json: { backend_base_url: 'https://diagnostics.example' } }));
  await context.route('https://diagnostics.example/**', (r) => {
    const path = new URL(r.request().url()).pathname; requests.push(path);
    if (path === '/models') return r.fulfill({ json: { models: [contractResponse.model_id, 'visual-model'].map((id) => ({ id, display_name: id, architectures: [], tokenizer_available: id === contractResponse.model_id })) } });
    if (path === '/sessions') return r.fulfill({ status: 201, json: { id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', model_id: r.request().postDataJSON().model_id } });
    if (path.endsWith('/tensors')) return r.fulfill({ json: contractInventory });
    if (path.endsWith('/architecture')) return r.fulfill({ json: response });
    return r.fulfill({ status: 204 });
  });
  return { requests, replace: (next: Response) => { response = next; } };
}
async function open(page: Page) {
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(contractResponse.model_id);
  await expect(page.getByText('Session active.')).toBeVisible();
}
async function information(page: Page) {
  await page.getByRole('button', { name: 'Session options', exact: true }).click();
  return page.getByRole('region', { name: 'Diagnostics', exact: true });
}
async function graph(page: Page) {
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('aria-busy', 'false');
}
test('card notices dismiss and rediscover without graph, camera, layout or request side effects', async ({ page, context }, testInfo) => {
  const api = await backend(context, authored()); await open(page);
  const before = await information(page);
  await expect(before).toContainText('Architecture diagnostics have not been retrieved.');
  expect(api.requests.some((r) => r.endsWith('/architecture'))).toBe(false);
  await page.keyboard.press('Escape'); await graph(page);
  const canvas = page.getByLabel('Architecture graph', { exact: true }), band = page.getByRole('region', { name: 'Architecture notices' });
  await expect(band).toContainText('Warning');
  expect(await canvas.locator('.architecture-controls').evaluate((header) => header.nextElementSibling?.className)).toBe('diagnostic-band');
  await expect(page.locator('.workspace-notices')).not.toContainText('checkpoint');
  const element = await page.locator('.react-flow').elementHandle();
  const camera = await page.locator('.react-flow__viewport').getAttribute('style');
  const layout = await canvas.getAttribute('data-layout-count'), requests = [...api.requests];
  await page.screenshot({ path: testInfo.outputPath('compact-notices.png') });
  await band.getByText('ⓘ Model-supplied', { exact: true }).click();
  await expect(band.getByRole('note')).toContainText('equivalence to model code is not verified');
  await band.getByText('View details', { exact: true }).click();
  await expect(band.locator('.diagnostic-list li')).toHaveCount(3);
  await expect(band).toContainText('Trained checkpoint · checkpoint');
  await expect(band).toContainText('Interface mapping is ambiguous; the original component and connections remain visible.');
  await page.screenshot({ path: testInfo.outputPath('diagnostics.png') });
  await band.getByRole('button', { name: 'Dismiss architecture notices' }).click();
  await expect(canvas).toBeFocused();
  await expect(band.getByRole('status')).toHaveCount(0);
  const details = await information(page);
  await expect(details.locator('.diagnostic-list li')).toHaveCount(3);
  await expect(details).toContainText('3 warnings · 0 errors');
  await expect(details).toContainText(contractResponse.graph.graph_id);
  await expect(details).toContainText('Partial architecture coverage · model defined');
  await expect(details).toContainText('Interface mapping is ambiguous');
  await page.screenshot({ path: testInfo.outputPath('model-information.png') });
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Session options', exact: true })).toBeFocused();
  expect(await element!.evaluate((el) => el === document.querySelector('.react-flow'))).toBe(true);
  await expect(canvas).toHaveAttribute('data-layout-count', layout!);
  expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
  expect(api.requests).toEqual(requests);
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click(); await graph(page);
  await expect(band.getByRole('status')).toHaveCount(0);
  // A real graph replacement makes its findings visible again.
  const replacement = authored(); if (replacement.status === 'available') replacement.graph.graph_id = 'replacement';
  api.replace(replacement);
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click(); await graph(page);
  await expect(band.getByRole('status')).toContainText('Warning');
  expect(api.requests.some((r) => /tokenize|\/data|statistics|distribution/.test(r))).toBe(false);
});
test('blocking safe validation finding remains discoverable after dismissal without a rendered graph', async ({ page, context }) => {
  await backend(context, { status: 'unavailable', model_id: contractResponse.model_id, reason: 'analysis_failed', requires_restart: true,
    diagnostics: [{ code: 'invalid_model_definition', message: precise }] });
  await open(page); await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const band = page.getByRole('region', { name: 'Architecture notices' });
  await expect(band.getByRole('status')).toContainText('Error'); await expect(band).toContainText(precise);
  await band.getByRole('button', { name: 'Dismiss architecture notices' }).click();
  await expect(page.getByLabel('Architecture capability')).toBeFocused();
  await expect(page.getByText('Architecture preparation failed for this model.', { exact: true })).toBeVisible();
  await expect(page.getByText('Restart the backend to prepare this model again.')).toBeVisible();
  const details = await information(page); await expect(details).toContainText(precise); await expect(details).toContainText('0 warnings · 1 errors');
  await expect(page.locator('.react-flow')).toHaveCount(0);
});
test('many long findings stay bounded with complete accessible details and clear on model replacement', async ({ page, context }) => {
  const api = await backend(context, authored(true)); await open(page); await graph(page);
  const band = page.getByRole('region', { name: 'Architecture notices' });
  await band.getByText('View details', { exact: true }).click();
  await expect(band.locator('.diagnostic-list li')).toHaveCount(33);
  expect(await band.evaluate((el) => el.scrollHeight > el.clientHeight)).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true);
  const box = (await band.boundingBox())!, card = (await page.getByLabel('Architecture graph', { exact: true }).boundingBox())!;
  expect(box.height).toBeLessThanOrEqual(card.height * 0.33);
  api.replace({ status: 'unavailable', model_id: 'visual-model', reason: 'unsupported_architecture', requires_restart: false, diagnostics: [] });
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('visual-model');
  await expect(page.getByLabel('Architecture capability')).toBeVisible();
  const details = await information(page); await expect(details).not.toContainText('checkpoint'); await expect(details).not.toContainText('Model-supplied');
});
