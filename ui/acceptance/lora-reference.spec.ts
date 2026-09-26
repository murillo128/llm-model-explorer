import { test, expect } from '@playwright/test';
import type { Graph } from '../src/architecture-explorer/graph';
import { findComponent } from '../tests/architecture-controls';
import { installProbe } from './probe';
import { installArchitectureProbe } from './architecture-probe';
import { spawn } from 'node:child_process';
import { linkSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const python = `${repo}backend/.venv/bin/python`;
const componentPort = Number(process.env.UI_TEST_PORT ?? 4173);
const isolatedPorts = Boolean(process.env.UI_TEST_PORT);

function acceptancePorts(project: string) {
  if (project === 'dpr2') return {
    ui: isolatedPorts ? componentPort + 4 : 4177,
    backend: isolatedPorts ? componentPort + 5 : 8767,
  };
  return {
    ui: isolatedPorts ? componentPort + 2 : 4175,
    backend: isolatedPorts ? componentPort + 3 : 8765,
  };
}

const sourceRule = 'Semantic source key in the reviewed packaged description';
const targetPattern = /^__peft__\..*\.model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.weight$/;
let backend = '';
let service: ReturnType<typeof spawn> | undefined;
let cacheRoot: string | undefined;
let serviceLog = '';

test.use({ headless: true });

function isolatedCompositionRoot(sourceRoot: string, destination: string) {
  mkdirSync(destination, { recursive: true });
  const models = readdirSync(sourceRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => join(sourceRoot, entry.name));
  const native = models.find((directory) => {
    const config = JSON.parse(readFileSync(join(directory, 'config.json'), 'utf8'));
    return config.model_type === 'llama' && !config.quantization_config &&
      statSync(join(directory, 'model.safetensors')).isFile();
  });
  const quantized = models.find((directory) => {
    const config = JSON.parse(readFileSync(join(directory, 'config.json'), 'utf8'));
    return config.model_type === 'llama' && config.quantization_config?.quant_method === 'bitsandbytes';
  });
  const adapter = models.find((directory) => {
    try { return statSync(join(directory, 'adapter_config.json')).isFile(); } catch { return false; }
  });
  if (!native || !quantized || !adapter) throw new Error('Pinned SmolLM2 base/adapter set is incomplete');
  for (const source of [native, quantized, adapter]) {
    const target = join(destination, source.split('/').at(-1)!);
    mkdirSync(target);
    for (const name of readdirSync(source)) {
      const original = join(source, name);
      if (statSync(original).isFile()) linkSync(original, join(target, name));
    }
  }
  return destination;
}

function sourceKey(node: Graph['nodes'][number]) {
  return node.provenance.find((record) => record.rule === sourceRule)?.source ?? node.label;
}

function shape(parameter: Graph['parameters'][number]) {
  return parameter.logical_shape?.map((dimension) =>
    dimension.kind === 'constant' ? dimension.value : -1,
  ) ?? null;
}

function hasEdge(graph: Graph, from: string, sourcePort: string, to: string, targetPort: string) {
  return graph.edges.some((edge) =>
    edge.source.node_id === from && edge.source.port_id === sourcePort &&
    edge.target.node_id === to && edge.target.port_id === targetPort,
  );
}

test.beforeEach(async ({ page }, info) => {
  const sourceRoot = process.env.LMEX_LORA_REFERENCE_MODEL_ROOT;
  test.skip(!sourceRoot, 'LMEX_LORA_REFERENCE_MODEL_ROOT not supplied; local SmolLM2 LoRA pair not tested');
  info.setTimeout(600_000);
  serviceLog = '';
  const ports = acceptancePorts(info.project.name);
  backend = `http://127.0.0.1:${ports.backend}`;
  const origin = `http://127.0.0.1:${ports.ui}`;
  cacheRoot = mkdtempSync(join(tmpdir(), 'lmex-smollm2-lora-ui-'));
  const modelRoot = isolatedCompositionRoot(sourceRoot!, join(cacheRoot, 'models'));
  service = spawn(python, [
    '-m', 'llm_model_explorer', '--model-root', modelRoot!,
    '--cache-dir', join(cacheRoot, 'cache'), '--port', String(ports.backend), '--cors-origin', origin,
  ], {
    cwd: repo,
    env: { ...process.env, HF_HUB_OFFLINE: '1', TOKENIZERS_PARALLELISM: 'false' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  service.stdout!.on('data', (data) => { serviceLog += data; });
  service.stderr!.on('data', (data) => { serviceLog += data; });
  await expect.poll(async () => {
    if (service?.exitCode !== null) throw new Error(serviceLog);
    try { return (await fetch(`${backend}/models`)).status; } catch { return 0; }
  }, { timeout: 300_000 }).toBe(200);
  await page.addInitScript(installProbe);
  await page.addInitScript(installArchitectureProbe);
  await page.goto('/');
});

test.afterEach(async ({ page }, info) => {
  if (info.status !== 'skipped') {
    await page.close();
    if (service?.exitCode === null) {
      service.kill('SIGTERM');
      const timer = setTimeout(() => service?.kill('SIGKILL'), 10_000);
      try { await once(service, 'exit'); } finally { clearTimeout(timer); }
    }
    if (info.status !== info.expectedStatus) {
      await info.attach('backend-log', { body: serviceLog, contentType: 'text/plain' });
    }
  }
  if (cacheRoot) { rmSync(cacheRoot, { recursive: true }); cacheRoot = undefined; }
});

test('SmolLM2 LoRA reference graph navigates and inspects actual A/B weights', async ({ page }, info) => {
  const catalogueResponse = await fetch(`${backend}/models`);
  expect(catalogueResponse.ok).toBe(true);
  const catalogue = await catalogueResponse.json() as {
    models: { id: string }[];
    diagnostics: unknown[];
  };
  expect(catalogue.diagnostics).toEqual([]);
  const composites = catalogue.models.filter((model) =>
    model.id.includes('+peft-lora:') && model.id.endsWith(':smollm2-135m-smoltalk-lora'),
  );
  expect(composites).toHaveLength(2);
  const modelId = composites.find((model) => model.id.startsWith('SmolLM2-135M+'))!.id;
  const architectureResponse = page.waitForResponse((response) =>
    response.url().startsWith(backend) && response.url().endsWith('/architecture') &&
    response.request().method() === 'GET' && response.status() === 200,
  );
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(modelId);
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const response = await architectureResponse;
  const body = await response.json() as { status: string; model_id: string; graph: Graph };
  expect(body.status).toBe('available');
  expect(body.model_id).toBe(modelId);
  const graph = body.graph;
  expect(graph.coverage).toBe('complete');
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute(
    'data-graph-id', graph.graph_id,
  );

  const factors = new Map<string, Partial<Record<'A' | 'B', Graph['parameters'][number]>>>();
  for (const parameter of graph.parameters) {
    const match = targetPattern.exec(parameter.name);
    if (!match) continue;
    const [, layer, projection, factor] = match;
    const target = `${layer}.${projection}`;
    const pair = factors.get(target) ?? {};
    pair[factor as 'A' | 'B'] = parameter;
    factors.set(target, pair);
  }
  expect(factors.size).toBe(60);
  expect([...factors.values()].every((pair) => pair.A && pair.B)).toBe(true);
  const nodes = new Map(graph.nodes.map((node) => [sourceKey(node), node]));
  const inspect = [
    factors.get('0.q_proj')!.A!, factors.get('0.q_proj')!.B!,
    factors.get('0.v_proj')!.A!, factors.get('0.v_proj')!.B!,
  ];
  const inspected: string[] = [];
  const requests: string[] = [];
  page.on('request', (request) => {
    if (request.url().startsWith(backend)) requests.push(new URL(request.url()).pathname);
  });

  for (const parameter of inspect) {
    expect(parameter.binding).toBe('native');
    expect(parameter.inspection.status).toBe('available');
    if (parameter.inspection.status !== 'available') throw new Error('Expected an inspectable LoRA factor');
    expect(parameter.storage[0]?.role).toBe('adapter_factor');
    expect(shape(parameter)).toEqual(parameter.name.includes('lora_A') ? [8, 576] :
      parameter.name.includes('q_proj') ? [576, 8] : [192, 8]);
    const node = graph.nodes.find((candidate) => candidate.parameter_ids.includes(parameter.id))!;
    const key = sourceKey(node);
    expect(key).toMatch(/\.lora_[AB]$/);
    const module = key.replace(/\.lora_[AB]$/, '');
    expect(nodes.get(module)?.operation).toBe('linear');
    const scale = nodes.get(`${module}.lora_scale`)!;
    const residual = nodes.get(`${module}.lora_add`)!;
    expect(scale.operation).toBe('scale');
    expect(residual.operation).toBe('add');
    expect(scale.attributes.find((attribute) => attribute.name === 'factor')?.value).toBe(2);
    const factorA = nodes.get(`${module}.lora_A`)!;
    const factorB = nodes.get(`${module}.lora_B`)!;
    expect(hasEdge(graph, factorA.id, 'out', factorB.id, 'x')).toBe(true);
    expect(hasEdge(graph, factorB.id, 'out', scale.id, 'x')).toBe(true);
    expect(hasEdge(graph, nodes.get(module)!.id, 'out', residual.id, 'base')).toBe(true);
    expect(hasEdge(graph, scale.id, 'out', residual.id, 'adapter')).toBe(true);

    await findComponent(page, node.id);
    const card = page.locator(`.react-flow__node[data-id=${JSON.stringify(node.id)}]`);
    await card.locator('.architecture-node-label').dblclick();
    await expect(page.getByLabel('Inspect parameter', { exact: true })).toBeVisible();
    const tensorPath = `/tensors/${parameter.inspection.tensor_id}/data`;
    const tensorResponse = page.waitForResponse(
      (candidate) => candidate.url().includes(tensorPath) && candidate.request().method() === 'GET',
      { timeout: 15_000 },
    );
    await page.getByLabel('Inspect parameter', { exact: true }).selectOption(parameter.id);
    expect((await tensorResponse).status()).toBe(200);
    await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
    await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('aria-busy', 'false');
    if (inspected.length === 0) {
      const path = info.outputPath('smollm2-lora-factor-inspection.png');
      await page.screenshot({ path });
      await info.attach('actual-lora-factor-inspection', { path, contentType: 'image/png' });
    }
    inspected.push(parameter.name);
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toHaveCount(0);
  }

  expect(inspected).toHaveLength(4);
  expect(requests.filter((path) => path.includes('/tensors/') && path.endsWith('/data'))).toHaveLength(4);
  await info.attach('actual-lora-reference-summary', {
    body: JSON.stringify({
      modelId,
      graphCoverage: graph.coverage,
      graphNodes: graph.nodes.length,
      graphEdges: graph.edges.length,
      adapterTargets: factors.size,
      inspectedFactors: inspected,
      tensorRequests: requests.filter((path) => path.includes('/tensors/') && path.endsWith('/data')),
      viewport: page.viewportSize(),
      browser: page.context().browser()!.version(),
    }),
    contentType: 'application/json',
  });
});

test('SmolLM2 QLoRA reference graph selects NF4 base and actual adapter weights', async ({ page }, info) => {
  const response = await fetch(`${backend}/models`);
  expect(response.ok).toBe(true);
  const catalogue = await response.json() as { models: { id: string }[]; diagnostics: unknown[] };
  expect(catalogue.diagnostics).toEqual([]);
  const modelId = catalogue.models.find((model) =>
    model.id.startsWith('HuggingFaceTB/SmolLM2-135M@bnb-nf4-dq+') &&
    model.id.endsWith(`:${'smollm2-135m-smoltalk-lora'}`),
  )!.id;
  const architectureResponse = page.waitForResponse((candidate) =>
    candidate.url().startsWith(backend) && candidate.url().endsWith('/architecture') &&
    candidate.request().method() === 'GET' && candidate.status() === 200,
  );
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(modelId);
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const body = await (await architectureResponse).json() as { status: string; graph: Graph };
  expect(body.status).toBe('available');
  const graph = body.graph;
  expect(graph.coverage).toBe('complete');
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute(
    'data-graph-id', graph.graph_id,
  );
  const baseWeight = graph.parameters.find((parameter) =>
    parameter.name === 'model.layers.0.self_attn.q_proj.weight',
  )!;
  expect(baseWeight.binding).toBe('quantized');
  expect(baseWeight.inspection.status).toBe('available');
  const factors = graph.parameters.filter((parameter) =>
    targetPattern.test(parameter.name),
  );
  expect(factors).toHaveLength(120);

  const nodes = new Map(graph.nodes.map((node) => [sourceKey(node), node]));
  const baseNode = nodes.get('model.layers.0.self_attn.q_proj')!;
  await findComponent(page, baseNode.id);
  const baseCard = page.locator(`.react-flow__node[data-id=${JSON.stringify(baseNode.id)}]`);
  await baseCard.locator('.architecture-node-label').dblclick();
  await expect(page.getByLabel('Inspect parameter', { exact: true })).toBeVisible();
  const basePath = `/tensors/${baseWeight.inspection.status === 'available' ? baseWeight.inspection.tensor_id : ''}/data`;
  const baseStream = page.waitForResponse((candidate) => candidate.url().includes(basePath) && candidate.status() === 200);
  await page.getByLabel('Inspect parameter', { exact: true }).selectOption(baseWeight.id);
  await baseStream;
  await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
  await page.keyboard.press('Escape');

  const factor = graph.parameters.find((parameter) =>
    parameter.name.endsWith('.model.layers.0.self_attn.q_proj.lora_A.weight'),
  )!;
  expect(factor.binding).toBe('native');
  expect(factor.inspection.status).toBe('available');
  if (factor.inspection.status !== 'available') throw new Error('Expected an inspectable QLoRA factor');
  const factorNode = graph.nodes.find((node) => node.parameter_ids.includes(factor.id))!;
  await findComponent(page, factorNode.id);
  await page.locator(`.react-flow__node[data-id=${JSON.stringify(factorNode.id)}] .architecture-node-label`).dblclick();
  await expect(page.getByLabel('Inspect parameter', { exact: true })).toBeVisible();
  const factorPath = `/tensors/${factor.inspection.tensor_id}/data`;
  const factorStream = page.waitForResponse((candidate) => candidate.url().includes(factorPath) && candidate.status() === 200);
  await page.getByLabel('Inspect parameter', { exact: true }).selectOption(factor.id);
  await factorStream;
  await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
  await info.attach('qlora-browser-summary', {
    body: JSON.stringify({
      modelId,
      coverage: graph.coverage,
      graphNodes: graph.nodes.length,
      graphEdges: graph.edges.length,
      targetBranches: 60,
      inspectedQuantizedBase: baseWeight.name,
      inspectedAdapterFactor: factor.name,
      browser: page.context().browser()!.version(),
      viewport: page.viewportSize(),
    }),
    contentType: 'application/json',
  });
});
