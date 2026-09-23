import { selectComponent, graphAction } from './architecture-controls';
import { expect, test } from '@playwright/test';
import type { BrowserContext, Route } from '@playwright/test';
import { contractInventory, contractResponse, referenceFixture } from './architecture-fixtures';

async function backend(context: BrowserContext) {
  const visual = referenceFixture('vjepa2');
  const requests: string[] = [];
  await context.route('**/runtime-config.json', (r) => r.fulfill({ json: { backend_base_url: 'https://architecture.example' } }));
  await context.route('https://architecture.example/**', (r) => {
    const path = new URL(r.request().url()).pathname; requests.push(path);
    if (path === '/models') return r.fulfill({ json: { models: [
      { id: contractResponse.model_id, display_name: 'Contract', architectures: [], tokenizer_available: true },
      { id: visual.model_id, display_name: 'Visual', architectures: [], tokenizer_available: false },
    ] } });
    if (path === '/sessions') return r.fulfill({ status: 201, json: {
      id: r.request().postDataJSON().model_id === visual.model_id ? 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb' : 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
      model_id: r.request().postDataJSON().model_id,
    } });
    const isVisual = path.includes('bbbbbbbb');
    if (path.endsWith('/tensors')) return r.fulfill({ json: isVisual ? { tensors: [], coverage: 'partial', diagnostics: [{ code: 'partial', message: 'Fixture has no native weights.' }] } : contractInventory });
    if (path.endsWith('/architecture')) return r.fulfill({ json: isVisual ? visual : contractResponse });
    return r.fulfill({ status: 404, json: { code: 'session_not_found', message: 'Session not found' } });
  });
  return requests;
}
test('built shell retrieves on demand, supports V-JEPA without tokenization and restores view state', async ({ page, context }) => {
  const requests = await backend(context);
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(contractResponse.model_id);
  await expect(page.getByText('Session active.')).toBeVisible();
  expect(requests.filter((p) => p.endsWith('/architecture'))).toHaveLength(0);
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const graph = page.getByLabel('Architecture graph', { exact: true });
  await expect(graph).toHaveAttribute('data-visible-nodes', '3');
  const received = [...requests];
  const canvasElement = await page.locator('.react-flow').elementHandle();
  await page.getByRole('searchbox', { name: 'Search components', exact: true }).focus();
  await selectComponent(page, 'linear1');
  await page.getByRole('button', { name: 'Collapse browser', exact: true }).click();
  await page.getByRole('button', { name: 'Expand browser', exact: true }).click();
  await page.getByRole('searchbox', { name: 'Search components', exact: true }).focus();
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'View options', exact: true }).click();
  await page.keyboard.press('Escape');
  expect(requests).toEqual(received);
  expect(await canvasElement!.evaluate((element) => element === document.querySelector('.react-flow'))).toBe(true);
  await graphAction(page, 'Show all operations');
  await expect(graph).toHaveAttribute('data-visible-nodes', '6');
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  await expect(page.getByRole('button', { name: /linear.weight/ })).toBeVisible();
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await expect(graph).toHaveAttribute('data-visible-nodes', '6');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(referenceFixture('vjepa2').model_id);
  await expect(graph).toHaveAttribute('data-graph-id', 'fixture-vjepa2');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveCount(0);
  expect(requests.some((p) => p.endsWith('/tokenize'))).toBe(false);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true);
});
test('malformed architecture responses stay local and partial inventories remain explicit', async ({ page, context }) => {
  await backend(context);
  await context.route('**/architecture', (r) => r.fulfill({ json: { ...contractResponse, graph: { ...contractResponse.graph, nodes: [] } } }));
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(contractResponse.model_id);
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await expect(page.getByLabel('Architecture capability').getByRole('alert')).toContainText('Invalid getArchitecture response');
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  await expect(page.getByRole('button', { name: /linear.weight/ })).toBeVisible();
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(referenceFixture('vjepa2').model_id);
  await expect(page.getByText(/Partial tensor inventory:/)).toBeVisible();
  await expect(page.getByText('Fixture has no native weights.')).toBeVisible();
});

test('a delayed architecture response cannot replace a newer model/session', async ({ page, context }) => {
  const requests = await backend(context);
  let old: Route | undefined;
  await context.route('https://architecture.example/sessions/aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa/architecture', (route) => { old = route; });
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(contractResponse.model_id);
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await expect.poll(() => Boolean(old)).toBe(true);
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(referenceFixture('vjepa2').model_id);
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await expect(canvas).toHaveAttribute('data-graph-id', 'fixture-vjepa2');
  await old!.fulfill({ json: contractResponse });
  await expect(canvas).toHaveAttribute('data-graph-id', 'fixture-vjepa2');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveCount(0);
  expect(requests.some((request) => request.endsWith('/tokenize'))).toBe(false);
});

test('architecture navigation leaves model and session controls usable at 280 pixels', async ({ page, context }) => {
  await backend(context); await page.setViewportSize({ width: 280, height: 400 }); await page.goto('/');
  const model = page.getByRole('combobox', { name: 'Model', exact: true });
  const refresh = page.getByRole('button', { name: 'Refresh models', exact: true });
  await expect(model).toBeEnabled();
  const a = (await model.boundingBox())!, b = (await refresh.boundingBox())!;
  expect(a.width).toBeGreaterThanOrEqual(36); expect(a.x + a.width).toBeLessThanOrEqual(b.x);
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await model.selectOption(contractResponse.model_id);
  const graph = page.getByLabel('Architecture graph', { exact: true });
  // This small graph body needs the readable collapsed-model fallback.
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await expect(graph).toHaveAttribute('data-visible-nodes', '1');
  await expect(graph).toHaveAttribute('data-source-node-ids', '["root"]');
  await expect(graph).toHaveAttribute('data-node-count', '6');
  await graphAction(page, 'Show all operations');
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await expect(graph).toHaveAttribute('data-visible-nodes', '6');
});
