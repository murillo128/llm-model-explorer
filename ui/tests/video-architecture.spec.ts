import { test, expect } from '@playwright/test';
import { findComponent, graphAction } from './architecture-controls';
import { videoFixture, videoStream, videoValues } from './video-architecture-fixture';

test('video ports, conditional experts and noninitial weights retain distinct identities', async ({ page, context }, info) => {
  const fixture = videoFixture(), requests: string[] = [];
  await context.route('**/runtime-config.json', route => route.fulfill({ json: { backend_base_url: 'https://video.example' } }));
  await context.route('https://video.example/**', route => {
    const path = new URL(route.request().url()).pathname; requests.push(path);
    if (path === '/models') return route.fulfill({ json: { models: [{ id: 'fixture/video', display_name: 'Video fixture', architectures: ['WanVideoInspection'], tokenizer_available: false }], diagnostics: [] } });
    if (path === '/sessions') return route.fulfill({ status: 201, json: { id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', model_id: 'fixture/video', model_revision: 'snapshot_A' } });
    if (path.endsWith('/tensors')) return route.fulfill({ json: fixture.inventory });
    if (path.endsWith('/architecture')) return route.fulfill({ json: fixture.response });
    const match = /\/tensors\/([^/]+)\/(data|statistics|distributions)$/.exec(path);
    if (match) return route.fulfill({ contentType: 'application/vnd.llm-model-explorer.stream',
      headers: { 'X-Operation-Id': 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb', 'Access-Control-Expose-Headers': 'X-Operation-Id' },
      body: videoStream(match[1]!, fixture.inventory.tensors.find(t => t.id === match[1])!.name, match[2]!) });
    return route.fulfill({ status: 404, json: { code: 'unsupported_representation', message: 'Not part of this UI fixture.' } });
  });
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('fixture/video');
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await expect(canvas).toHaveAttribute('data-graph-id', 'synthetic-video');
  await findComponent(page, 'pipeline'); await graphAction(page, 'Inspect selected');
  await expect(page.getByRole('dialog')).toContainText('required initial RGB image');
  await expect(page.getByRole('dialog')).toContainText('RGB video frames');
  await expect(page.getByRole('dialog')).toContainText('video generation');
  await page.keyboard.press('Escape');
  for (const role of ['high_noise', 'low_noise']) {
    const id = `${role}-block-1-q`;
    await findComponent(page, id); await expect(canvas).toHaveAttribute('aria-busy', 'false');
    await graphAction(page, 'Inspect selected');
    await page.getByLabel('Inspect parameter', { exact: true }).selectOption(`${id}-parameter`);
    await expect(page.getByRole('dialog')).toContainText(`video.${role}.blocks.1.self_attn.q.weight`);
    await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
    await expect(page.getByLabel('Result status', { exact: true })).toBeEmpty();
    await page.locator('.matrix-scroll').focus();
    const coordinate = page.locator('.inspection-readout span').first();
    await expect(coordinate).toHaveText(/^row \d+ · column \d+$/);
    const point = /^row (\d+) · column (\d+)$/.exec((await coordinate.textContent())!)!;
    await expect.poll(async () => Number(await page.locator('.inspection-readout span').last().textContent()))
      .toBe(videoValues(id)[Number(point[1]) * 2 + Number(point[2])]);
    await page.screenshot({ path: info.outputPath(`${role}-instance-1.png`) });
    await page.keyboard.press('Escape');
    expect(requests.some(p => p.endsWith(`/tensors/${id}/data`))).toBe(true);
  }
  const camera = await page.locator('.react-flow__viewport').getAttribute('style');
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await expect(page.locator('.react-flow__viewport')).toHaveAttribute('style', camera!);
  await expect(page.getByRole('button', { name: 'Generate', exact: true })).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
});
