import { test, expect } from '@playwright/test';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { findComponent, graphAction } from '../tests/architecture-controls';
import type { Graph } from '../src/architecture-explorer/graph';

test('complete local video reference preserves semantic ports and native weight inspection', { tag: '@extended' }, async ({ page }, info) => {
  const selected = process.env.LMEX_VIDEO_REFERENCE_PACKAGE;
  test.skip(!selected, 'Complete video package absent; reference acceptance remains pending');
  test.setTimeout(600_000);
  const directory = resolve(selected!);
  const config = JSON.parse(readFileSync(join(directory, 'config.json'), 'utf8'));
  const repo = fileURLToPath(new URL('../../', import.meta.url));
  const scratch = mkdtempSync(join(tmpdir(), 'video-browser-'));
  const component = Number(process.env.UI_TEST_PORT ?? 4173);
  const port = process.env.UI_TEST_PORT ? component + 3 : 8765;
  const origin = `http://127.0.0.1:${process.env.UI_TEST_PORT ? component + 2 : 4175}`;
  const backend = `http://127.0.0.1:${port}`;
  let log = '';
  const service = spawn(join(repo, 'backend/.venv/bin/python'), ['-m', 'llm_model_explorer',
    '--model-root', dirname(directory), '--cache-dir', join(scratch, 'cache'),
    '--port', String(port), '--cors-origin', origin], { cwd: repo,
    env: { ...process.env, HF_HUB_OFFLINE: '1', TOKENIZERS_PARALLELISM: 'false' }, stdio: ['ignore', 'pipe', 'pipe'] });
  service.stdout.on('data', data => { log += data; }); service.stderr.on('data', data => { log += data; });
  try {
    await expect.poll(async () => {
      if (service.exitCode !== null) throw new Error(log);
      try { return (await fetch(`${backend}/models`)).status; } catch { return 0; }
    }, { timeout: 300_000 }).toBe(200);
    const models = (await (await fetch(`${backend}/models`)).json()).models as { id: string }[];
    const model = models.find(m => m.id.startsWith(config._name_or_path))!;
    await page.goto('/');
    await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(model.id);
    const received = page.waitForResponse(r => r.url().endsWith('/architecture') && r.status() === 200);
    await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
    const graph = (await (await received).json()).graph as Graph;
    expect(graph.scope).toBe('video_generation'); expect(graph.coverage).toBe('complete');
    const canvas = page.getByLabel('Architecture graph', { exact: true });
    await expect(canvas).toHaveAttribute('aria-busy', 'false');
    await page.screenshot({ path: info.outputPath('video-overview.png') });
    const root = graph.nodes.find(n => !n.parent_id)!;
    await findComponent(page, root.id); await graphAction(page, 'Inspect selected');
    await expect(page.getByRole('dialog')).toContainText('RGB video frames');
    if (config.video_inspection.modalities.includes('image_to_video')) {
      await expect(page.getByRole('dialog')).toContainText('initial RGB image');
    }
    await expect(page.getByRole('dialog')).not.toContainText('Model-supplied');
    await page.keyboard.press('Escape');
    const ltx = config.video_inspection.family === 'ltx';
    const role = config.video_inspection.profile.includes('a14b') ? 'high_noise' : 'denoiser';
    const parameter = graph.parameters.find(p => p.name === (ltx
      ? 'video.denoiser.transformer_blocks.1.scale_shift_table'
      : `video.${role}.blocks.1.self_attn.q.weight`))!;
    expect(parameter.inspection.status).toBe('available');
    const owner = graph.nodes.find(n => n.parameter_ids.includes(parameter.id))!;
    await findComponent(page, owner.id); await expect(canvas).toHaveAttribute('aria-busy', 'false');
    await page.screenshot({ path: info.outputPath('video-noninitial-component.png') });
    await graphAction(page, 'Inspect selected');
    await page.getByLabel('Inspect parameter', { exact: true }).selectOption(parameter.id);
    await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
    await expect(page.getByLabel('Result status', { exact: true })).toBeEmpty({ timeout: 180_000 });
    await page.screenshot({ path: info.outputPath('video-weight.png') });
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.keyboard.press('Escape');
    await info.attach('native-reference-graph', { body: JSON.stringify({ model: model.id, graph: graph.graph_id,
      scope: graph.scope, nodes: graph.nodes.length, parameters: graph.parameters.length, inspected: parameter.name }), contentType: 'application/json' });
  } finally {
    await info.attach('backend-log', { body: log, contentType: 'text/plain' });
    await page.close();
    if (service.exitCode === null) {
      service.kill('SIGTERM');
      const timer = setTimeout(() => service.kill('SIGKILL'), 10_000);
      try { await once(service, 'exit'); } finally { clearTimeout(timer); }
    }
    rmSync(scratch, { recursive: true });
  }
});
