import { test, expect } from '@playwright/test';
import type { Graph } from '../src/architecture-explorer/graph';
import { findComponent } from '../tests/architecture-controls';
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const python = `${repo}backend/.venv/bin/python`;
let service: ReturnType<typeof spawn> | undefined;
let temporary = '';
let backend = '';
let serviceLog = '';

test.beforeEach(async ({ page }, info) => {
  info.setTimeout(600_000);
  temporary = mkdtempSync(join(tmpdir(), 'clm-browser-'));
  const reference = process.env.LMEX_CLM_REFERENCE_MODEL_ROOT;
  const modelRoot = reference ?? join(temporary, 'models');
  const env = { ...process.env, HF_HUB_OFFLINE: '1', TOKENIZERS_PARALLELISM: 'false',
    OMP_NUM_THREADS: '2', PYTHONPATH: `${repo}backend/src:${repo}backend/tests:${repo}` };
  if (!reference) execFileSync(python, ['-c',
    'from pathlib import Path; import sys; from clm_fixtures import fixture, exporter; ' +
    'root=Path(sys.argv[1]); encoder,head,_=fixture(root); ' +
    'exporter().export_package(encoder,head,root/"clm",encoder_revision="1"*40,head_revision="2"*40)',
    modelRoot], { cwd: repo, env });
  const componentPort = Number(process.env.UI_TEST_PORT ?? 4173);
  const offset = info.project.name === 'dpr2' ? 4 : 2;
  const port = process.env.UI_TEST_PORT ? componentPort + offset + 1 :
    info.project.name === 'dpr2' ? 8767 : 8765;
  backend = `http://127.0.0.1:${port}`;
  serviceLog = '';
  service = spawn(python, ['-m', 'llm_model_explorer', '--model-root', modelRoot,
    '--cache-dir', join(temporary, 'cache'), '--port', String(port),
    '--cors-origin', info.project.use.baseURL!], { cwd: repo, env, stdio: ['ignore', 'pipe', 'pipe'] });
  service.stdout!.on('data', (data) => { serviceLog += data; });
  service.stderr!.on('data', (data) => { serviceLog += data; });
  await expect.poll(async () => {
    if (service?.exitCode !== null) throw new Error(serviceLog);
    try { return (await fetch(`${backend}/models`)).status; } catch { return 0; }
  }, { timeout: 300_000 }).toBe(200);
  await page.goto('/');
});

test.afterEach(async ({ page }, info) => {
  await page.close();
  if (service?.exitCode === null) {
    service.kill('SIGTERM');
    const timer = setTimeout(() => service?.kill('SIGKILL'), 15_000);
    try { await once(service, 'exit'); } finally { clearTimeout(timer); }
  }
  if (info.status !== info.expectedStatus) await info.attach('backend-log', {
    body: serviceLog, contentType: 'text/plain',
  });
  if (temporary) rmSync(temporary, { recursive: true });
});

test('CLM package expands a configured head and opens its native projection weight', async ({ page }, info) => {
  const catalogue = await (await fetch(`${backend}/models`)).json() as { models: { id: string }[] };
  const modelId = catalogue.models.find((m) => m.id.startsWith('Contrastive-LM/CLM-v0.1-8B@'))!.id;
  const architecture = page.waitForResponse((r) => r.url().startsWith(backend) &&
    r.url().endsWith('/architecture') && r.status() === 200);
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(modelId);
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const body = await (await architecture).json() as { status: string; graph: Graph };
  expect(body.status).toBe('available');
  expect(body.graph.scope).toBe('model_defined');
  expect(body.graph.coverage).toBe('complete');
  const group = body.graph.nodes.find((n) => n.label === 'Action head')!;
  await findComponent(page, group.id);
  await page.getByRole('button', { name: 'Expand Action head', exact: true }).click();
  const parameter = body.graph.parameters.find((p) => p.name === 'clm.action_head.out.weight')!;
  const node = body.graph.nodes.find((n) => n.parameter_ids.includes(parameter.id))!;
  await findComponent(page, node.id);
  const card = page.locator(`.react-flow__node[data-id=${JSON.stringify(node.id)}]`);
  await card.locator('.architecture-info').click();
  expect(parameter.inspection.status).toBe('available');
  if (parameter.inspection.status !== 'available') throw new Error('Missing head matrix');
  const response = page.waitForResponse((r) => r.url().includes(
    `/tensors/${parameter.inspection.status === 'available' ? parameter.inspection.tensor_id : ''}/data`,
  ) && r.request().method() === 'GET');
  await page.getByLabel('Inspect parameter', { exact: true }).selectOption(parameter.id);
  expect((await response).status()).toBe(200);
  await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.screenshot({ path: info.outputPath('clm-head-inspection.png') });
  await info.attach('clm-summary', { body: JSON.stringify({
    reference: Boolean(process.env.LMEX_CLM_REFERENCE_MODEL_ROOT), modelId,
    graphId: body.graph.graph_id, group: group.label, tensor: parameter.name,
    shape: parameter.logical_shape,
  }), contentType: 'application/json' });
});
