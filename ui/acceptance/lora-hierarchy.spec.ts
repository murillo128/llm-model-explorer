import { test, expect } from '@playwright/test';
import { spawn, execFileSync } from 'node:child_process';
import { once } from 'node:events';
import { linkSync, mkdirSync, mkdtempSync, readdirSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Graph } from '../src/architecture-explorer/graph';
import { findComponent, graphAction } from '../tests/architecture-controls';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const python = `${repo}backend/.venv/bin/python`;
const sourceRule = 'Semantic source key in the reviewed packaged description';

// Use the existing deterministic backend composition fixture through a real
// production service. The optional local pair runs exactly the same consumer path.
function prepareModels(root: string) {
  const sourceRoot = process.env.LMEX_LORA_REFERENCE_MODEL_ROOT;
  if (!sourceRoot) {
    execFileSync(python, ['-c',
      'import sys; from pathlib import Path; sys.path.insert(0, "backend/tests"); ' +
      'from test_lora_architecture import make_smollm2_lora; make_smollm2_lora(Path(sys.argv[1]))',
      root], { cwd: repo });
    return;
  }
  const sources = readdirSync(sourceRoot).map((name) => join(sourceRoot, name));
  const file = (directory: string, name: string) => {
    try { return statSync(join(directory, name)).isFile(); } catch { return false; }
  };
  const base = sources.find((path) => path.endsWith('/smollm2-135m-bf16') && file(path, 'config.json'));
  const adapter = sources.find((path) => path.endsWith('/smollm2-135m-smoltalk-lora') && file(path, 'adapter_config.json'));
  if (!base || !adapter) throw new Error('The supplied local SmolLM2 base/LoRA reference pair is incomplete');
  for (const source of [base, adapter]) {
    const target = join(root, source.split('/').at(-1)!); mkdirSync(target);
    for (const name of readdirSync(source)) if (file(source, name)) linkSync(join(source, name), join(target, name));
  }
}

test('adapted projections expand through ordinary groups with truthful cards and A/B inspection', async ({ page }, info) => {
  info.setTimeout(180_000);
  const port = Number(process.env.UI_TEST_PORT ?? 4173);
  const isolated = Boolean(process.env.UI_TEST_PORT);
  const backendPort = info.project.name === 'dpr2' ? (isolated ? port + 5 : 8767) : (isolated ? port + 3 : 8765);
  const backend = `http://127.0.0.1:${backendPort}`;
  const origin = info.project.use.baseURL!;
  const root = mkdtempSync(join(tmpdir(), 'lmex-lora-hierarchy-'));
  let service: ReturnType<typeof spawn> | undefined;
  let log = '';
  try {
    const modelRoot = join(root, 'models'); mkdirSync(modelRoot); prepareModels(modelRoot);
    service = spawn(python, ['-m', 'llm_model_explorer', '--model-root', modelRoot,
      '--cache-dir', join(root, 'cache'), '--port', String(backendPort), '--cors-origin', origin], {
      cwd: repo, env: { ...process.env, HF_HUB_OFFLINE: '1', TOKENIZERS_PARALLELISM: 'false' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    service.stdout!.on('data', (data) => { log += data; });
    service.stderr!.on('data', (data) => { log += data; });
    await expect.poll(async () => {
      if (service?.exitCode !== null) throw new Error(log);
      try { return (await fetch(`${backend}/models`)).status; } catch { return 0; }
    }, { timeout: 90_000 }).toBe(200);
    const catalogue = await (await fetch(`${backend}/models`)).json() as { models: { id: string }[] };
    const composite = catalogue.models.find((model) => model.id.includes('+peft-lora:'))!;
    expect(composite).toBeTruthy();
    await page.goto('/');
    await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(composite.id);
    const response = page.waitForResponse((candidate) => candidate.url().startsWith(backend) && candidate.url().endsWith('/architecture') && candidate.status() === 200);
    await graphAction(page, 'Architecture Explorer');
    const body = await (await response).json() as { status: string; graph: Graph };
    expect(body.status).toBe('available');
    const graph = body.graph;
    const nodes = new Map(graph.nodes.map((node) => [node.provenance.find((p) => p.rule === sourceRule)?.source, node]));
    const canvas = page.getByLabel('Architecture graph', { exact: true });
    const ready = () => expect(canvas).toHaveAttribute('aria-busy', 'false');
    const card = (id: string) => page.locator(`.react-flow__node[data-id=${JSON.stringify(id)}]`);
    const target = 'model.layers.0.self_attn.q_proj';
    const group = nodes.get(target)!;
    expect(group.kind).toBe('group');
    if (group.kind !== 'group') throw new Error('The adapted projection is not expandable');
    const children = ['.base', '.lora_A', '.lora_B', '.lora_scale', '.lora_add'].map((suffix) => nodes.get(target + suffix)!);
    expect(group.children).toEqual(children.map((node) => node.id));
    await findComponent(page, group.id); await ready();
    await expect(card(group.id)).toBeVisible();
    await expect(card(group.id).locator('.architecture-tensor-row')).toHaveCount(0);
    for (const child of children) await expect(card(child.id)).toHaveCount(0);
    await page.screenshot({ path: info.outputPath('collapsed-projection.png') });
    await graphAction(page, 'Toggle selected group'); await ready();
    await graphAction(page, 'Fit view');
    for (const child of children) await expect(card(child.id)).toBeAttached();
    await graphAction(page, 'Explore component'); await ready();
    await expect(canvas).toHaveAttribute('data-scope-id', group.id);
    await graphAction(page, 'Fit view');
    await page.screenshot({ path: info.outputPath('expanded-projection.png') });
    await graphAction(page, 'Back'); await ready();
    const scale = nodes.get(`${target}.lora_scale`)!;
    await findComponent(page, scale.id); await ready();
    await expect(card(scale.id).locator('.architecture-node-type > .architecture-summary-text')).toHaveAccessibleName('out = factor * x');
    await expect(card(scale.id).getByLabel('factor = 2', { exact: true })).toBeVisible();
    for (const name of ['x', 'out']) await expect(card(scale.id).locator(`[data-port-id="${name}"].architecture-port`)).toBeVisible();
    for (const [suffix, formula] of [
      ['q_heads', 'out = reshape(x, ...)'], ['q_transpose', 'out = transpose(x, ...)'],
      ['softmax', 'out = softmax(x, axis=axis)'],
    ]) {
      const node = nodes.get(`model.layers.0.self_attn.${suffix}`)!;
      await findComponent(page, node.id); await ready();
      await expect(card(node.id).locator('.architecture-node-type > .architecture-summary-text')).toHaveAccessibleName(formula!);
      if (suffix === 'softmax') await expect(card(node.id).getByLabel('axis = -1', { exact: true })).toBeVisible();
    }
    const inspected: string[] = [];
    for (const factor of ['A', 'B']) {
      const node = nodes.get(`${target}.lora_${factor}`)!;
      const parameter = graph.parameters.find((p) => p.id === node.parameter_ids[0])!;
      expect(parameter.inspection.status).toBe('available');
      if (parameter.inspection.status !== 'available') throw new Error('Missing adapter inspection');
      await findComponent(page, node.id); await ready();
      const stream = page.waitForResponse((candidate) => candidate.url().endsWith(`/tensors/${parameter.inspection.status === 'available' ? parameter.inspection.tensor_id : ''}/data`) && candidate.status() === 200);
      await card(node.id).getByRole('button', { name: `Inspect matrix ${parameter.name}`, exact: true }).click();
      await stream;
      await expect(page.getByLabel('Inspect parameter', { exact: true })).toHaveValue(parameter.id);
      await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
      inspected.push(parameter.name);
      await page.keyboard.press('Escape'); await expect(page.getByRole('dialog')).toHaveCount(0);
    }
    await info.attach('lora-hierarchy-summary', { contentType: 'application/json', body: JSON.stringify({
      reference: Boolean(process.env.LMEX_LORA_REFERENCE_MODEL_ROOT), modelId: composite.id,
      graph: graph.graph_id, nodes: graph.nodes.length, edges: graph.edges.length,
      children: children.map((node) => node.id), inspected,
    }) });
  } finally {
    await page.close();
    if (service?.exitCode === null) {
      service.kill('SIGTERM');
      const timer = setTimeout(() => service?.kill('SIGKILL'), 10_000);
      try { await once(service, 'exit'); } finally { clearTimeout(timer); }
    }
    await info.attach('backend-log', { body: log, contentType: 'text/plain' });
    rmSync(root, { recursive: true });
  }
});
