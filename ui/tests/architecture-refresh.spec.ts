import { expect, test } from '@playwright/test';
import fixture from './fixtures/architecture-refresh.json' with { type: 'json' };
import { findComponent } from './architecture-controls';
import { frame } from './embedding-fixtures';

const sessionA = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', sessionB = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const model = 'navigation/model';
const oldNode = fixture.before.nodes.find((n) => n.operation === 'linear')!;
const newNode = fixture.after.nodes.find((n) => n.operation === 'linear')!;
const tensors = [
  { id: 'actual-weight-id', name: 'encoder.proj.weight', shape: [4, 3], rank: 2, numel: 12, path: ['encoder', 'proj', 'weight'], storage_dtype: 'F32', logical_dtype: 'float32' },
  { id: 'actual-bias-id', name: 'encoder.proj.bias', shape: [4], rank: 1, numel: 4, path: ['encoder', 'proj', 'bias'], storage_dtype: 'F32', logical_dtype: 'float32' },
];

test('real importer revisions restore scope, normalized camera, Back and a fresh weight consumer', async ({ page }) => {
  await page.addInitScript(() => {
    const sources: EventTarget[] = []; let subscriptions = 0;
    class Source extends EventTarget {
      constructor() { super(); sources.push(this); subscriptions++; }
      close() { const index = sources.indexOf(this); if (index >= 0) sources.splice(index, 1); }
    }
    Object.defineProperty(window, 'EventSource', { value: Source });
    Object.defineProperty(window, 'modelSubscriptions', { get: () => subscriptions });
    Object.defineProperty(window, 'emitModelState', { value: (revision: string) => {
      const epoch = '11111111-1111-4111-8111-111111111111';
      sources.at(-1)!.dispatchEvent(new MessageEvent('model-state', { lastEventId: `${epoch}:1`, data: JSON.stringify({
        epoch, sequence: 1, model_id: 'navigation/model', model_revision: revision, status: 'present' }) }));
    } });
  });
  let pins = 0, release!: () => void;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  const numeric: string[] = [];
  await page.route('**/runtime-config.json', (route) => route.fulfill({ json: { backend_base_url: 'https://refresh.example' } }));
  await page.route('https://refresh.example/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/models') return route.fulfill({ json: { models: [{ id: model, display_name: 'Navigation fixture', architectures: [], tokenizer_available: false }], diagnostics: [] } });
    if (path === '/sessions') {
      pins++;
      if (pins > 1) await pending;
      return route.fulfill({ status: 201, json: { id: pins === 1 ? sessionA : sessionB, model_id: model, model_revision: pins === 1 ? 'A' : 'B' } });
    }
    if (route.request().method() === 'DELETE') return route.fulfill({ status: 204 });
    if (path.endsWith('/tensors')) return route.fulfill({ json: { tensors, coverage: 'complete', diagnostics: [] } });
    if (path.endsWith('/architecture')) return route.fulfill({ json: { status: 'available', model_id: model, diagnostics: [], graph: path.includes(sessionA) ? fixture.before : fixture.after } });
    if (path.includes('/tensors/') && path.endsWith('/data')) {
      numeric.push(path);
      const tensor = tensors.find((t) => path.includes(t.id))!;
      const metadata = { kind: 'tensor', tensor_id: tensor.id, name: tensor.name, shape: tensor.shape,
        dtype: 'float32', byte_order: 'little', layout: 'c', byte_length: tensor.numel * 4 };
      const values = new Float32Array(tensor.numel).fill(path.includes(sessionA) ? 1 : 9);
      return route.fulfill({ headers: { 'Access-Control-Expose-Headers': 'X-Operation-Id', 'Content-Type': 'application/vnd.llm-model-explorer.stream', 'X-Operation-Id': path.includes(sessionA) ? sessionA : sessionB },
        body: Buffer.from([...frame(1, new TextEncoder().encode(JSON.stringify(metadata))), ...frame(2, new Uint8Array(values.buffer)), ...frame(4)]) });
    }
    return route.fulfill({ status: 422, json: { code: 'unsupported_representation', message: 'Optional fixture analysis unavailable.' } });
  });
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(model);
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const graph = page.getByLabel('Architecture graph', { exact: true });
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await findComponent(page, oldNode.parent_id!);
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await findComponent(page, oldNode.id);
  const box = page.locator(`.react-flow__node[data-id="${oldNode.id}"]`);
  await expect(box).toBeVisible();
  const camera = () => page.locator('.react-flow__viewport').evaluate((node) => new DOMMatrixReadOnly(getComputedStyle(node).transform).a);
  const zoom = await camera();
  const pane = page.getByRole('separator', { name: 'Resize architecture browser' });
  await pane.press('ArrowRight'); const width = await pane.getAttribute('aria-valuenow');
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  await page.getByLabel('Inspect parameter').selectOption(fixture.before.parameters.find((p) => p.name === 'encoder.proj.bias')!.id);
  await expect.poll(() => numeric.some((path) => path.includes(sessionA) && path.includes('actual-bias-id'))).toBe(true);
  const emit = (revision: string) => page.evaluate((value) => {
    (window as unknown as { emitModelState: (revision: string) => void }).emitModelState(value);
  }, revision);
  await emit('B');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByRole('contentinfo')).toContainText('Updating model');
  await expect(page.locator(`[data-node-id="${oldNode.id}"]`)).toHaveCount(0);
  release();
  await expect.poll(() => pins).toBe(2);
  // The newly pinned session must be confirmed by its own subscription.
  await expect.poll(() => page.evaluate(() => (window as unknown as { modelSubscriptions: number }).modelSubscriptions)).toBe(2);
  await emit('B');
  await expect(graph).toHaveAttribute('data-graph-id', fixture.after.graph_id);
  await expect(graph).toHaveAttribute('data-scope-id', newNode.parent_id!);
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByRole('dialog')).toHaveAccessibleName('Renamed projection');
  await expect(page.getByRole('dialog')).toContainText('Updated projection description after a model edit.');
  await expect(page.getByLabel('Inspect parameter')).toHaveValue(fixture.after.parameters.find((p) => p.name === 'encoder.proj.bias')!.id);
  await expect.poll(() => numeric.some((path) => path.includes(sessionB) && path.includes('actual-bias-id'))).toBe(true);
  await expect(page.getByRole('dialog').locator('.matrix-scroll canvas').first()).toBeVisible();
  await expect(page.getByRole('dialog').locator('[data-result="tensor"]')).toHaveCount(0);
  await page.getByRole('button', { name: 'Close inspection' }).click();
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', newNode.id);
  expect(await camera()).toBeCloseTo(zoom, 3);
  expect(await pane.getAttribute('aria-valuenow')).toBe(width);
  const bounds = await page.locator(`.react-flow__node[data-id="${newNode.id}"]`).boundingBox();
  const viewport = await page.locator('.architecture-flow').boundingBox();
  expect(bounds!.x + bounds!.width).toBeGreaterThan(viewport!.x);
  expect(bounds!.x).toBeLessThan(viewport!.x + viewport!.width);
  expect(bounds!.y + bounds!.height).toBeGreaterThan(viewport!.y);
  expect(bounds!.y).toBeLessThan(viewport!.y + viewport!.height);
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(graph).toHaveAttribute('aria-busy', 'false');
  await expect(graph).toHaveAttribute('data-scope-id', '');
  await expect(page.getByRole('button', { name: 'Back', exact: true })).toHaveCount(0);
});
