import { openShared, findComponent, graphAction, graphPreference, viewOptions } from '../tests/architecture-controls';
import { revealTensor } from '../tests/tensor-tree-helpers';
/* eslint-disable @typescript-eslint/no-explicit-any -- Native test-only observations and evidence. */
import { test, expect } from '@playwright/test';
import type { Page, TestInfo } from '@playwright/test';
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import type { Graph } from '../src/architecture-explorer/graph';
import { expandCompactGraph } from '../src/api/compact-architecture';
import { projectGraph } from '../src/architecture-explorer/projection';
import { assertTraceability, assertInterfaceCoverage } from '../tests/architecture-invariants';
import { installProbe } from './probe';
import { HarnessTiming } from './harness-timing';

let timing: HarnessTiming;
import { installArchitectureProbe } from './architecture-probe';
import { nativeCamera } from '../tests/native-camera';
import { fanoutPoint } from '../tests/architecture-pointer';

// Independently expected component set for the four reviewed producer fixtures.
// Their passive declarations are verified by the conservation oracle, not by
// importing the runtime classification as the expected result.
const componentsOf = (graph: Graph) => graph.nodes.filter((node) => !['input', 'output'].includes(node.kind) &&
  !(node.kind === 'context' && !node.ports.length && node.references.some((r) => r.kind === 'tokenizer')));
const visibleCount = (graph: Graph) => {
  const nodes = componentsOf(graph), roots = nodes.filter((n) => !n.parent_id);
  return nodes.length + (roots.length === 1 && roots[0]!.kind === 'group' ? 0 : 1);
};

const repo = fileURLToPath(new URL('../../', import.meta.url));
const diagnosticCases = JSON.parse(readFileSync(join(repo, 'api/fixtures/model-defined-diagnostics.json'), 'utf8')) as { name: string; code: string; message: string }[];
const python = `${repo}backend/.venv/bin/python`;
let service: ReturnType<typeof spawn>;
let root: string;
let backend: string;
let modelId: string;
let log: string;
let observed: string[];
let fixtureModelIds: Record<string, string> = {};

function fixtureModelId(family: string, models: { id: string }[]) {
  return models.find((candidate) => candidate.id === family || candidate.id.startsWith(`${family}@`))?.id;
}

function browserProcessMemory() {
  const samples = [];
  for (const entry of readdirSync('/proc')) {
    if (!/^\d+$/.test(entry)) continue;
    try {
      const command = readFileSync(`/proc/${entry}/comm`, 'utf8').trim();
      if (!/^(chrome|chromium)/i.test(command)) continue;
      const status = readFileSync(`/proc/${entry}/status`, 'utf8');
      const match = /VmRSS:\s+(\d+)/.exec(status);
      if (match) samples.push(Number(match[1]));
    } catch { /* The process exited while observations were being sampled. */ }
  }
  return {
    process_count: samples.length,
    aggregate_rss_kib: samples.reduce((sum, sample) => sum + sample, 0),
    largest_process_rss_kib: samples.length ? Math.max(...samples) : 0,
  };
}

// Scalar oracle for the physical synthetic words in architecture_fixtures.py.
// It does not use application decoding, renderer readout, or uploaded GPU values.
function packedValue(family: string, row: number, column: number, inputs: number) {
  if (family === 'qwen3') {
    const group = (Math.floor(column / 3) + Math.floor(column / 128)) % (inputs / 128);
    return ((3 * column + 5 * row) % 16 - ((group + row) % 14 + 1)) * ((group + row % 7 + 1) / 8);
  }
  expect(family).toBe('qwen35');
  const code = (column + row * 3) % 16;
  const exponent = (code >> 1) & 3, mantissa = code & 1;
  const magnitude = exponent === 0 ? mantissa / 2 : (1 + mantissa / 2) * 2 ** (exponent - 1);
  const scalar = code & 8 ? -magnitude : magnitude;
  return scalar * [0.5, 1, 2, 3][(row + Math.floor(column / 16)) % 4]! * 0.5;
}

async function metrics(page: Page) { return page.evaluate(() => (window as any).__acceptance.metrics()); }
async function graphObservation(page: Page, collect = false) {
  const client = await page.context().newCDPSession(page);
  if (collect) await client.send('HeapProfiler.collectGarbage');
  const heap = await client.send('Runtime.getHeapUsage');
  await client.detach();
  return { heap, collected: collect, ...await page.evaluate(() => window.__architectureProbe()) };
}
async function recordGraph(page: Page, info: TestInfo, label: string, graph: Graph) {
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await info.attach(label, { body: JSON.stringify({ sourceNodes: graph.nodes.length, sourceEdges: graph.edges.length,
    viewport: page.viewportSize(), dpr: await page.evaluate(() => devicePixelRatio), ...await graphObservation(page) }), contentType: 'application/json' });
  const viewport = page.viewportSize()!;
  const widths = info.title.startsWith('complete local reference') && info.project.name === 'dpr1' ? [1178, viewport.width] : [viewport.width];
  for (const width of widths) {
    await page.setViewportSize({ ...viewport, width });
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    await expect(canvas).toHaveAttribute('aria-busy', 'false');
    await expect(page.getByRole('alert')).toHaveCount(0);
    const path = info.outputPath(`${label}-${width}.png`);
    await page.screenshot({ path }); await info.attach(`${label}-${width}-capture`, { path, contentType: 'image/png' });
  }
}
async function control(path: string, body?: object) {
  const response = await fetch(`${backend}/__test/${path}`, body ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  } : undefined);
  expect(response.ok).toBeTruthy(); return response.json();
}
async function openParameter(page: Page, graph: Graph, parameter: Graph['parameters'][number], isolated = false) {
  const node = graph.nodes.find((n) => n.parameter_ids.includes(parameter.id))!;
  await findComponent(page, node.id);
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByRole('alert')).toHaveCount(0);
  const card = page.locator(`.react-flow__node[data-id=${JSON.stringify(node.id)}]`);
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  const camera = await page.locator('.react-flow__viewport').getAttribute('style');
  const layoutCount = await canvas.getAttribute('data-layout-count'), scope = await canvas.getAttribute('data-scope-id');
  const beforeSelection = observed.length;
  for (const target of ['.architecture-node-label', '.architecture-node-heading']) {
    await card.locator(target).click(target === '.architecture-node-heading' ? { position: { x: 2, y: (await card.locator(target).boundingBox())!.height - 2 } } : {});
    await expect(card.locator('.architecture-node')).toHaveAttribute('data-selected', 'true');
    await expect(page.getByRole('dialog')).toHaveCount(0);
    expect(await canvas.getAttribute('data-layout-count')).toBe(layoutCount);
    expect(await canvas.getAttribute('data-scope-id')).toBe(scope);
    expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
  }
  expect(observed.slice(beforeSelection)).toEqual([]);
  if (isolated && await page.getByLabel('Architecture graph', { exact: true }).getAttribute('data-scope-id') !== node.id) {
    const requests = observed.length;
    await card.locator('.architecture-navigate').click();
    await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('data-scope-id', node.id);
    await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('aria-busy', 'false');
    expect(observed.slice(requests)).toEqual([]);
  }
  let trigger = card.locator(`[data-parameter-id=${JSON.stringify(parameter.id)}] .architecture-matrix-action`);
  if (parameter.inspection.status === 'unavailable') {
    await expect(trigger).toBeDisabled();
    await expect(trigger).toHaveAccessibleDescription(`${parameter.inspection.reason.replaceAll('_', ' ')}: ${parameter.inspection.message}`);
    await trigger.focus(); await page.keyboard.press('Enter');
    await expect(page.getByRole('dialog')).toHaveCount(0);
    expect(observed.slice(beforeSelection)).toEqual([]);
    // Unavailable weights keep descriptor inspection through the ordinary modal.
    trigger = card.locator('.architecture-info');
    await trigger.click();
    await page.getByLabel('Inspect parameter', { exact: true }).selectOption(parameter.id);
  } else await trigger.click();
  await expect(page.getByLabel('Inspect parameter', { exact: true })).toHaveValue(parameter.id);
  return trigger;
}

function inspectionLayer(family: string, reference: boolean) {
  if (!reference) return 1;
  return ({ deepseek_v2: 26, glm4_moe_lite: 46, kimi_linear: 26 } as Record<string, number>)[family] ?? 1;
}
async function inspectActualMoeWeightAtWidth(
  page: Page,
  info: TestInfo,
  graph: Graph,
  family: string,
  parameter: Graph['parameters'][number],
  width: number,
) {
  expect(parameter.inspection.status).toBe('available');
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await page.setViewportSize({ width, height: 1000 });
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
  const layoutCountBeforeFind = await canvas.getAttribute('data-layout-count');
  const trigger = await openParameter(page, graph, parameter);
  await expect(canvas).toHaveAttribute('data-layout-count', layoutCountBeforeFind!);
  await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
  await expect(page.getByRole('dialog')).toContainText(parameter.name);
  const node = graph.nodes.find((item) => item.parameter_ids.includes(parameter.id))!;
  await expect(page.locator(`.react-flow__node[data-id=${JSON.stringify(node.id)}]`)).toBeInViewport();
  await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
  await nativeCamera(page);
  await page.mouse.move(0, 0);
  const scroller = page.locator('.matrix-scroll');
  await scroller.focus();
  const coordinate = page.locator('.inspection-readout span').first();
  const scalar = page.locator('.inspection-readout span').last();
  await expect(coordinate).toHaveText(/^row \d+ · column \d+$/);
  const match = /^row (\d+) · column (\d+)$/.exec((await coordinate.textContent())!)!;
  const row = Number(match[1]), column = Number(match[2]);
  const expectedValue = JSON.parse(execFileSync(python, [
    '-m', 'acceptance.architecture_reference', family,
    ...(parameter.binding === 'quantized' ? ['--packed'] : []), '--tensor', parameter.name,
    '--row', String(row), '--column', String(column),
  ], { cwd: repo, encoding: 'utf8' })).samples[0].value;
  await expect.poll(async () => Number(await scalar.textContent())).toBe(expectedValue);
  const tensor = parameter.inspection.status === 'available' ? parameter.inspection.tensor_id : '';
  expect(observed.some((path) => path.endsWith(`/tensors/${tensor}/data`))).toBe(true);
  const targetKind = parameter.name.includes('.experts.') ? 'expert' : 'router';
  await info.attach(`issue-178-${family}-focused-${targetKind}-weight-${width}`, {
    body: JSON.stringify({ family, graph: graph.graph_id, parameter: parameter.name, tensor, viewport: page.viewportSize(),
      logicalNodes: graph.nodes.length, logicalEdges: graph.edges.length,
      visibleNodes: Number(await canvas.getAttribute('data-visible-nodes')),
      layoutCountBeforeFind, layoutCountAfterFind: await canvas.getAttribute('data-layout-count'),
      layoutMs: Number(await canvas.getAttribute('data-layout-ms')), row, column, value: expectedValue,
      browserProcessMemory: browserProcessMemory() }),
    contentType: 'application/json',
  });
  await page.keyboard.press('Escape'); await released(page);
  await expect(trigger).toBeFocused();
  await expect(page.getByRole('alert')).toHaveCount(0);
}
async function released(page: Page) {
  await expect.poll(async () => { const m = await metrics(page); return [m.textures, m.readers]; }).toEqual([0, 0]);
}
async function selectGraph(page: Page): Promise<Graph> {
  const response = page.waitForResponse((r) => r.url().startsWith(backend) &&
    r.url().endsWith('/architecture') && r.request().method() === 'GET' && r.status() === 200);
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(modelId);
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  // The bounded browser reader can release the DevTools response body. Read the
  // same immutable prepared graph over real HTTP and require the UI to match it.
  const prepared = await fetch((await response).url());
  expect(prepared.ok).toBe(true);
  const body = await prepared.json();
  expect(body.status).toBe('available');
  expect(body.model_id).toBe(modelId);
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('data-graph-id', body.graph.graph_id);
  return body.graph;
}

async function inspectComponents(page: Page, graph: Graph, info: TestInfo, family: string, reference: boolean) {
  const role = (node: Graph['nodes'][number], value: string) => node.attributes.some((a) => a.name === 'semantic_role' && a.value === value);
  const layerRepetitions = graph.repetitions.filter((repetition) =>
    repetition.instances.some((instance) => !instance.variant.startsWith('routed_')),
  );
  const layerInstances = layerRepetitions.flatMap((repetition) =>
    repetition.instances.filter((instance) => !instance.variant.startsWith('routed_')),
  );
  const layerNodeIds = new Set(layerInstances.map((instance) => instance.node_id));
  const components = graph.nodes.filter((node) => node.kind === 'group' &&
    layerNodeIds.has(node.parent_id ?? '') && (role(node, 'attention') || role(node, 'mlp')));
  expect(components).toHaveLength(2 * layerInstances.length);
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  const ready = () => expect(canvas).toHaveAttribute('aria-busy', 'false');
  const capture = async (name: string) => {
    if (info.project.name !== 'dpr1') return;
    const viewport = page.viewportSize()!;
    for (const width of reference && family === 'qwen35' ? [1178, viewport.width] : [viewport.width]) {
      await page.setViewportSize({ ...viewport, width });
      await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      const path = info.outputPath(`${reference ? 'reference' : 'fixture'}-${family}-${name}-${width}.png`);
      await page.screenshot({ path });
      await info.attach(`components-${name}-${width}`, { path, contentType: 'image/png' });
    }
  };
  const nodeBox = (id: string) => page.locator(`.react-flow__node[data-id=${JSON.stringify(id)}]`);
  const horizontal = async (before: string, after: string) => {
    // Expanded source groups can be wider than the viewport. Fit the current
    // focus before comparing both ends; React Flow culls offscreen node DOM.
    await page.getByRole('button', { name: 'Fit view', exact: true }).click();
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    await expect(nodeBox(before)).toBeAttached(); await expect(nodeBox(after)).toBeAttached();
    const a = await nodeBox(before).boundingBox(), b = await nodeBox(after).boundingBox();
    expect(a).toBeTruthy(); expect(b).toBeTruthy();
    expect(b!.x).toBeGreaterThanOrEqual(a!.x + a!.width - 1);
  };
  for (const repetition of layerRepetitions) {
    const instances = family === 'qwen35'
      ? repetition.instances.filter((instance, index, all) => !instance.variant.startsWith('routed_') &&
        all.findIndex((other) => other.variant === instance.variant) === index)
      : repetition.instances.filter((instance) => !instance.variant.startsWith('routed_')).slice(0, 1);
    for (const instance of instances) {
      const attention = components.find((n) => n.parent_id === instance.node_id && role(n, 'attention'))!;
      const mlp = components.find((n) => n.parent_id === instance.node_id && role(n, 'mlp'))!;
      await findComponent(page, instance.node_id);
      await graphAction(page, 'Focus layer'); await ready();
      await horizontal(attention.id, mlp.id);
      await expect(page.locator('.architecture-node[data-presentation="mlp"]')).toHaveCount(0);
      const label = `${graph.repetitions.indexOf(repetition)}-${instance.index}`;
      await capture(`${label}-compact`);
      await findComponent(page, attention.id);
      await graphAction(page, 'Toggle selected group'); await ready();
      await page.getByRole('button', { name: 'Center selected', exact: true }).click(); await ready();
      const projections = graph.nodes.filter((n) => n.parent_id === attention.id &&
        (role(n, 'query_projection') || role(n, 'query_gate_projection') || role(n, 'mask_padding_states')));
      const output = graph.nodes.find((n) => n.parent_id === attention.id && role(n, 'output_projection'));
      if (projections.length && output) await horizontal(projections[0]!.id, output.id);
      await capture(`${label}-attention`);
      await graphAction(page, 'Focus MLP'); await ready();
      await expect(page.getByText(mlp.label, { exact: true }).first()).toBeVisible();
      await expect(page.getByText('MLP (derived)', { exact: true })).toHaveCount(0);
      const up = graph.nodes.find((n) => n.parent_id === mlp.id && role(n, 'up_projection'))!;
      const down = graph.nodes.find((n) => n.parent_id === mlp.id && role(n, 'down_projection'))!;
      await horizontal(up.id, down.id);
      await capture(`${label}-mlp`);
    }
  }
}

async function inspectIsolation(page: Page, graph: Graph, info: TestInfo, family: string, reference: boolean) {
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  const ready = async () => {
    await expect(canvas).toHaveAttribute('aria-busy', 'false');
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  };
  const role = (node: Graph['nodes'][number], name: string) => node.attributes.some((a) => a.name === 'semantic_role' && a.value === name);
  const layerRepetition = graph.repetitions.find((repetition) =>
    repetition.instances.some((instance) => !instance.variant.startsWith('routed_')),
  )!;
  const owner = layerRepetition.instances.find((instance) => !instance.variant.startsWith('routed_'))!.node_id;
  const components = graph.nodes.filter((node) => node.parent_id === owner && (role(node, 'attention') || role(node, 'mlp')));
  const widths = reference && ['qwen3', 'vjepa2'].includes(family) && info.project.name === 'dpr1' ? [1178, 1440] : [1440];
  for (const width of widths) {
    await page.setViewportSize({ width, height: 1000 });
    for (const component of components) {
      const name = role(component, 'attention') ? 'attention' : 'mlp';
      await findComponent(page, component.id); await ready();
      const requests = observed.length;
      await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready();
      await expect(canvas).toHaveAttribute('data-scope-id', component.id);
      const root = page.locator(`.react-flow__node[data-id=${JSON.stringify(component.id)}]`);
      await expect(root).toBeAttached();
      const camera = await page.locator('.react-flow__viewport').getAttribute('style');
      const count = await canvas.getAttribute('data-layout-count');
      if (reference && info.project.name === 'dpr1' && ['qwen3', 'vjepa2'].includes(family)) {
        const path = info.outputPath(`isolated-${family}-${name}-${width}-initial.png`);
        await page.screenshot({ path }); await info.attach(`isolated-${name}-${width}-initial`, { path, contentType: 'image/png' });
      }
      // Explicit Fit includes every context endpoint. Sample real pointer hits
      // only after culling has exposed the complete generated scope geometry.
      await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready();
      const connections = page.locator('.architecture-connection');
      const targets = await connections.evaluateAll((elements) => elements.map((element) => ({
        source: element.getAttribute('data-source-node')!, port: element.getAttribute('data-source-port')!,
        target: element.getAttribute('data-target-node')!, targetPort: element.getAttribute('data-target-port')!,
        id: element.getAttribute('data-edge-id')!,
      })));
      // The external producer may first enter the expanded component's exact
      // boundary port; its internal branches still belong to that same signal.
      const fanout = targets.filter((item) => item.source.startsWith('external:')).map((input) => ({
        input, branches: targets.filter((other) => input.target === component.id
          ? other.source === input.target && other.port === input.targetPort
          : other.source === input.source && other.port === input.port),
      })).find((item) => item.branches.length > 1);
      if (family === 'qwen3') expect(fanout, 'Dense Attention/MLP inputs retain their genuine shared fan-out').toBeTruthy();
      if (fanout) {
        const source = fanout.branches[0]!;
        const branches = page.locator(`.architecture-connection[data-source-node=${JSON.stringify(source.source)}][data-source-port=${JSON.stringify(source.port)}]`);
        const branchList = await branches.all();
        const expected = [...new Set([fanout.input.id, ...fanout.branches.map((item) => item.id)])].sort();
        const port = page.locator(`.architecture-port[data-node-id=${JSON.stringify(fanout.input.source)}][data-port-id=${JSON.stringify(fanout.input.port)}]`);
        await port.locator('.architecture-port-dot').hover();
        await expect.poll(() => page.locator('.architecture-connection[data-emphasized="true"]').evaluateAll((elements) => elements.map((e) => e.getAttribute('data-edge-id')!).sort())).toEqual(expected);
        if (reference && info.project.name === 'dpr1' && ['qwen3', 'vjepa2'].includes(family)) {
          const path = info.outputPath(`isolated-${family}-${name}-${width}-boundary.png`);
          await page.screenshot({ path }); await info.attach(`isolated-${name}-${width}-boundary`, { path, contentType: 'image/png' });
        }
        const trunk = await fanoutPoint(page, branchList, branchList);
        await page.mouse.move(trunk.x, trunk.y);
        await expect.poll(() => page.locator('.architecture-connection[data-emphasized="true"]').evaluateAll((elements) => elements.map((e) => e.getAttribute('data-edge-id')!).sort())).toEqual(expected);
      }
      if (reference && info.project.name === 'dpr1' && ['qwen3', 'vjepa2'].includes(family)) {
        const path = info.outputPath(`isolated-${family}-${name}-${width}.png`);
        await page.screenshot({ path }); await info.attach(`isolated-${name}-${width}`, { path, contentType: 'image/png' });
      }
      expect(await canvas.getAttribute('data-layout-count')).toBe(count);
      expect(observed.slice(requests)).toEqual([]);
      await info.attach(`isolated-${family}-${name}-${width}-geometry`, { body: JSON.stringify({
        width, graph: graph.graph_id, scope: component.id, sourceNodes: graph.nodes.length,
        visibleNodes: Number(await canvas.getAttribute('data-visible-nodes')), visibleEdges: Number(await canvas.getAttribute('data-visible-edges')),
        layoutMs: Number(await canvas.getAttribute('data-layout-ms')), initialCamera: camera,
        component: await root.boundingBox(), boundaryFanout: fanout?.branches.length ?? 0,
        resources: await graphObservation(page),
      }), contentType: 'application/json' });
      await page.mouse.move(0, 0);
      await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready();
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
}

test.beforeEach(async ({ page }, info) => {
  timing = new HarnessTiming();
  const family = /\[(\w+)\]/.exec(info.title)![1]!;
  const reference = info.title.startsWith('complete local reference');
  if (!reference && ['deepseek_v2', 'glm4_moe_lite'].includes(family)) {
    test.skip(true, `${family} has no deterministic local checkpoint fixture; actual-reference coverage owns this family`);
  }
  if (!reference && family === 'kimi_linear') {
    test.skip(true, 'The tiny Kimi fixture covers graph topology; actionable packed tensors require the pinned local reference');
  }
  if (reference) test.setTimeout(600_000);
  const supplied = process.env.LMEX_ARCHITECTURE_REFERENCES;
  const hasReference = supplied && JSON.parse(readFileSync(supplied, 'utf8'))[family];
  if (reference && process.env.LMEX_REQUIRE_ARCHITECTURE_REFERENCES === '1') expect(hasReference, `Required local ${family} missing`).toBeTruthy();
  test.skip(reference && !hasReference, `Complete local ${family} absent; actual-reference acceptance pending`);
  const component = Number(process.env.UI_TEST_PORT ?? 4173);
  const dpr2 = info.project.name === 'dpr2';
  const port = process.env.UI_TEST_PORT ? component + (dpr2 ? 5 : 3) : dpr2 ? 8767 : 8765;
  const origin = `http://127.0.0.1:${process.env.UI_TEST_PORT ? component + (dpr2 ? 4 : 2) : dpr2 ? 4177 : 4175}`;
  backend = `http://127.0.0.1:${port}`; log = ''; observed = [];
  root = mkdtempSync(join(tmpdir(), 'lmex-architecture-browser-'));
  let args: string[];
  if (reference) {
    const selected = JSON.parse(execFileSync(python, [
      '-m', 'acceptance.architecture_reference', family,
      '--link-root', join(root, 'model-root'),
    ], { cwd: repo, encoding: 'utf8' }));
    modelId = selected.model_id;
    expect(selected.model_root).toBeTruthy();
    await info.attach('actual-reference-inventory', { body: JSON.stringify(selected.report), contentType: 'application/json' });
    args = ['-m', 'llm_model_explorer', '--model-root', selected.model_root, '--cache-dir', join(root, 'cache'), '--port', String(port), '--cors-origin', origin];
  } else {
    const fixtureStarted = performance.now();
    if (family === 'kimi_linear') {
      execFileSync(python, ['-m', 'acceptance.kimi_linear_fixture', join(root, 'models')], { cwd: repo });
    } else {
      execFileSync(python, ['-m', 'acceptance.architecture_fixtures', join(root, 'models'), ...(family === 'templates' ? ['--templates'] : [])], { cwd: repo });
    }
    if (info.title.includes('shows the precise model-owned load failure')) {
      writeFileSync(join(root, 'models', family, 'architecture.json'), '{"nodes":1,"nodes":2}');
    }
    timing.fixtureGenerationMs = performance.now() - fixtureStarted;
    modelId = family;
    args = ['-m', 'acceptance.server', '--root', root, '--port', String(port), '--origin', origin,
      ...(family === 'kimi_linear' ? ['--kimi-architecture-fixture'] : [])];
  }
  timing.spawned = performance.now();
  service = spawn(python, args, { cwd: repo, env: { ...process.env, HF_HUB_OFFLINE: '1', TOKENIZERS_PARALLELISM: 'false' }, stdio: ['ignore', 'pipe', 'pipe'] });
  service.stdout!.on('data', (data) => { log += data; }); service.stderr!.on('data', (data) => { log += data; });
  await expect.poll(async () => {
    if (service.exitCode !== null) throw new Error(log);
    try { return (await fetch(`${backend}/models`)).status; } catch { return 0; }
  }, { timeout: reference ? 300_000 : 30_000 }).toBe(200);
  timing.ready = performance.now();
  if (!reference) {
    const response = await fetch(`${backend}/models`);
    const models = (await response.json()).models as { id: string }[];
    fixtureModelIds = Object.fromEntries(
      ['smollm2', 'qwen3', 'qwen35', 'vjepa2', 'kimi_linear']
        .map((name) => [name, fixtureModelId(name, models)] as const)
        .filter((entry): entry is readonly [string, string] => entry[1] !== undefined),
    );
    // Template-only fixture runs intentionally advertise a different set of
    // models. Resolve an ID only when this test's family is present.
    modelId = fixtureModelIds[family] ?? family;
  }
  page.on('request', (r) => { if (r.url().startsWith(backend)) observed.push(new URL(r.url()).pathname); });
  await page.addInitScript(installProbe, { capturePixels: false });
  await page.addInitScript(installArchitectureProbe);
  await page.goto('/');
  timing.bodyStarted = performance.now();
});

test.afterEach(async ({ page }, info) => {
  if (info.status === 'skipped') return;
  timing.teardownStarted = performance.now();
  const probe = !page.isClosed() ? await page.evaluate(() => (window as any).__acceptance?.metrics() ?? null) : null;
  await info.attach('backend-log', { body: log ?? '', contentType: 'text/plain' });
  await page.close();
  if (service?.exitCode === null) {
    service.kill('SIGTERM'); const timer = setTimeout(() => service.kill('SIGKILL'), 10_000);
    try { await once(service, 'exit'); } finally { clearTimeout(timer); }
  }
  if (root) rmSync(root, { recursive: true });
  await timing.attach(info, log, probe);
  expect(probe).toMatchObject({ capturePixels: false, framebufferReadbacks: 0 });
});

test('deterministic production [smollm2] shows the precise model-owned load failure', async ({ page }) => {
  const finding = diagnosticCases.find((item) => item.name === 'duplicate_json_key')!;
  const response = page.waitForResponse((r) => r.url().startsWith(backend) &&
    r.url().endsWith('/architecture') && r.request().method() === 'GET');
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(modelId);
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const body = await (await response).json();
  expect(body).toMatchObject({ status: 'unavailable', reason: 'analysis_failed',
    diagnostics: [{ code: finding.code, message: finding.message }] });
  await expect(page.getByLabel('Architecture capability', { exact: true })
    .getByRole('status').getByText('Architecture preparation failed for this model.', { exact: true })).toBeVisible();
  await expect(page.locator('.architecture-capability-state').getByText(finding.message)).toBeVisible();
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveCount(0);
});

test('deterministic production [smollm2] port labels keep cable clearance and share terminal emphasis', async ({ page }, info) => {
  const graph = await selectGraph(page);
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  const ready = () => expect(canvas).toHaveAttribute('aria-busy', 'false');
  await ready();
  const root = page.locator('.react-flow__node').first();
  await root.locator('.architecture-expand').click();
  await ready();
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  const label = page.locator('.architecture-port-label[data-raised="true"]').first();
  await expect(label).toBeVisible();
  const port = label.locator('xpath=..');
  const highlighted = () => page.locator('.architecture-connection[data-emphasized="true"]')
    .evaluateAll((elements) => elements.map((element) => element.getAttribute('data-edge-id')!).sort());
  const inspectGeometry = async () => label.evaluate((element) => {
    const button = element.closest('button')!;
    const metric = JSON.parse(element.getAttribute('data-layout-bounds')!) as { x: number; y: number; width: number; height: number; clearance: number };
    const zoom = new DOMMatrix(getComputedStyle(document.querySelector('.react-flow__viewport')!).transform).a;
    const rect = element.getBoundingClientRect(), terminal = button.getBoundingClientRect();
    const cx = terminal.left + terminal.width / 2, cy = terminal.top + terminal.height / 2;
    return { actual: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
      expected: { x: cx + (metric.x - Number(button.dataset.absoluteX)) * zoom,
        y: cy + (metric.y - Number(button.dataset.absoluteY)) * zoom,
        width: metric.width * zoom, height: metric.height * zoom },
      gap: (cy - rect.bottom) / zoom, clearance: metric.clearance };
  });
  const assertGeometry = async () => {
    const { actual, expected, gap, clearance } = await inspectGeometry();
    for (const key of ['x', 'y', 'width', 'height'] as const) expect(actual[key]).toBeCloseTo(expected[key], 0);
    expect(gap).toBeGreaterThanOrEqual(clearance - 0.1);
  };
  await assertGeometry();
  if (info.project.name === 'dpr1') {
    const bounds = await port.boundingBox();
    expect(bounds).toBeTruthy();
    const clip = { x: bounds!.x - 4, y: bounds!.y - 28, width: 100, height: 64 };
    const overlap = info.outputPath('port-label-overlap-baseline.png');
    await label.evaluate((element) => {
      const span = element as HTMLElement, button = element.closest('button')!;
      span.dataset.savedTop = span.style.top;
      span.style.top = `${button.clientHeight / 2 - 7}px`;
    });
    await page.screenshot({ path: overlap, clip });
    await info.attach('port-label-overlap-baseline', { path: overlap, contentType: 'image/png' });
    await label.evaluate((element) => {
      const span = element as HTMLElement;
      span.style.top = span.dataset.savedTop!;
      delete span.dataset.savedTop;
    });
    const clear = info.outputPath('port-label-clearance.png');
    await page.screenshot({ path: clear, clip });
    await info.attach('port-label-clearance', { path: clear, contentType: 'image/png' });
    await assertGeometry();
  }
  const before = info.outputPath('port-label-before.png');
  await page.screenshot({ path: before });
  await info.attach('port-label-before', { path: before, contentType: 'image/png' });
  const count = await canvas.getAttribute('data-layout-count');
  const camera = await page.locator('.react-flow__viewport').getAttribute('style');
  const requests = observed.length;
  const numeric = await metrics(page);
  await label.hover();
  await expect.poll(highlighted).not.toEqual([]);
  const labelEdges = await highlighted();
  await expect(port).toHaveAttribute('data-emphasized', 'true');
  await port.locator('.architecture-port-dot').hover();
  await expect.poll(highlighted).toEqual(labelEdges);
  if (labelEdges.length === 1) {
    const line = page.locator(`.architecture-connection[data-edge-id=${JSON.stringify(labelEdges[0])}]`);
    const point = await fanoutPoint(page, [line], [line]);
    await page.mouse.move(point.x, point.y);
    await expect.poll(highlighted).toEqual(labelEdges);
  }
  await label.click();
  await page.mouse.move(0, 0);
  await expect.poll(highlighted).toEqual(labelEdges);
  const otherLabel = page.locator('.architecture-port-label[data-raised="true"]').nth(1);
  await otherLabel.hover();
  await expect.poll(highlighted).not.toEqual(labelEdges);
  await page.mouse.move(0, 0);
  await expect.poll(highlighted).toEqual(labelEdges);
  await port.focus();
  await expect.poll(highlighted).toEqual(labelEdges);
  await expect(label).toHaveAttribute('data-emphasized', 'true');
  const after = info.outputPath('port-label-emphasis.png');
  await page.screenshot({ path: after });
  await info.attach('port-label-emphasis', { path: after, contentType: 'image/png' });
  expect(await canvas.getAttribute('data-layout-count')).toBe(count);
  expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
  expect(observed.slice(requests)).toEqual([]);
  expect((await metrics(page)).createdTextures).toBe(numeric.createdTextures);
  await graphPreference(page, 'Show dimensions', true);
  await ready();
  await assertGeometry();
  await page.setViewportSize({ width: 1178, height: 900 });
  await assertGeometry();
  await page.setViewportSize({ width: 1440, height: 1000 });
  const instance = graph.repetitions[0]!.instances.at(-1)!;
  expect(instance.index).toBeGreaterThan(0);
  await findComponent(page, instance.node_id);
  await ready();
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await ready();
  await expect(canvas).toHaveAttribute('data-scope-id', instance.node_id);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  const isolated = page.locator(`.architecture-port[data-node-id=${JSON.stringify(instance.node_id)}] .architecture-port-label[data-raised="true"]`).first();
  await expect(isolated).toBeVisible();
  const isolatedCount = await canvas.getAttribute('data-layout-count');
  const isolatedCamera = await page.locator('.react-flow__viewport').getAttribute('style');
  const isolatedRequests = observed.length;
  await isolated.hover();
  await expect.poll(highlighted).not.toEqual([]);
  const isolatedEdges = await highlighted();
  await isolated.locator('xpath=..').focus();
  await expect.poll(highlighted).toEqual(isolatedEdges);
  expect(await canvas.getAttribute('data-layout-count')).toBe(isolatedCount);
  expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(isolatedCamera);
  expect(observed.slice(isolatedRequests)).toEqual([]);
});

for (const reference of [false, true]) for (const family of [
  'smollm2', 'qwen3', 'qwen35', 'vjepa2', 'deepseek_v2', 'glm4_moe_lite', 'kimi_linear',
]) {
  test(`${reference ? 'complete local reference' : 'deterministic production'} [${family}] full graph, concrete bindings and logical weight modal`, async ({ page }, info) => {
    test.setTimeout(reference && family === 'kimi_linear' ? 900_000 : reference ? 600_000 : 90_000);
    const graph = expandCompactGraph(await selectGraph(page));
    expect(graph.coverage).toBe('complete');
    const actualMoeReference = reference && ['deepseek_v2', 'glm4_moe_lite', 'kimi_linear'].includes(family);
    // Apply the common source oracle to actual backend graphs, including dense
    // Qwen/Llama differences and both visual stacks (not UI stress templates).
    assertTraceability(graph, projectGraph(graph, { expanded: [], exhaustive: true }), true);
    if (reference && family === 'smollm2') {
      const vector = graph.parameters.find((parameter) => parameter.binding === 'native' &&
        parameter.inspection.status === 'available' && parameter.logical_shape?.length === 1 &&
        parameter.name.includes('.1.'))!;
      await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
      await expect(page.getByRole('region', { name: 'Tensor Explorer workspace', exact: true })).toBeVisible();
      const tensorId = vector.inspection.status === 'available' ? vector.inspection.tensor_id : '';
      const tensorResponse = page.waitForResponse((response) =>
        response.url().includes(`/tensors/${tensorId}/data`) && response.status() === 200,
      );
      await (await revealTensor(page.getByRole('button', {
        includeHidden: true,
        name: new RegExp(vector.name.replaceAll('.', '\\.'), 'i'),
      }))).click();
      await tensorResponse;
      await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
      await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
      const prompt = page.getByRole('textbox', { name: 'Prompt', exact: true });
      const text = 'real SmolLM2 tokenizer ✓';
      const tokenized = page.waitForResponse((response) =>
        response.url().endsWith('/tokenize') && response.status() === 200 &&
        response.request().postDataJSON()?.text === text,
      );
      await prompt.fill(text);
      const result = await (await tokenized).json() as { text: string; tokens: { id: number }[] };
      expect(result.text).toBe(text);
      expect(result.tokens.length).toBeGreaterThan(0);
      await expect(page.locator('.tokenizer-status')).toContainText(`${result.tokens.length} tokens`);
      await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
      await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('data-graph-id', graph.graph_id);
    }
    for (const repetition of graph.repetitions) for (const instance of [repetition.instances[0]!, repetition.instances.at(-1)!]) {
      const expanded = [instance.node_id];
      for (let node = graph.nodes.find((item) => item.id === instance.node_id); node?.parent_id;
        node = graph.nodes.find((item) => item.id === node!.parent_id)) expanded.push(node.parent_id);
      assertTraceability(graph, projectGraph(graph, { expanded }));
    }
    const canvas = page.getByLabel('Architecture graph', { exact: true });
    await recordGraph(page, info, 'compact-graph', graph);
    await inspectComponents(page, graph, info, family, reference);
    await inspectIsolation(page, graph, info, family, reference);
    await viewOptions(page);
    await expect(page.getByLabel('Show dimensions')).not.toBeChecked();
    await page.keyboard.press('Escape');
    await page.getByRole('searchbox', { name: 'Search components', exact: true }).fill(' ');
    await graphAction(page, 'Show all operations');
    const ids = await page.getByRole('tree', { name: 'Model components', exact: true }).locator('[data-node-id]').evaluateAll((options) => options.map((o) => (o as HTMLElement).dataset.nodeId));
    expect(new Set(ids)).toEqual(new Set(componentsOf(graph).map((n) => n.id)));
    await page.keyboard.press('Escape');
    await graphAction(page, 'Show all operations');
    await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
    await recordGraph(page, info, 'exhaustive-graph', graph);
    // Visible routes compose boundary forwarding. Every original edge remains
    // traceable even though one route may represent several source segments.
    expect(JSON.parse((await canvas.getAttribute('data-source-node-ids'))!)).toEqual(componentsOf(graph).map((n) => n.id));
    expect(JSON.parse((await canvas.getAttribute('data-represented-edge-ids'))!)).toEqual(graph.edges.map((e) => e.id));
    assertInterfaceCoverage(graph, projectGraph(graph, { expanded: [], exhaustive: true }));
    const layerRepetitions = graph.repetitions.filter((repetition) =>
      repetition.instances.some((instance) => !instance.variant.startsWith('routed_')),
    );
    // The actual-reference contract searches from the exhaustive model scope
    // straight to an expert weight; walking every layer first changes that scope.
    if (!actualMoeReference) {
      for (const repetition of layerRepetitions) {
        const last = repetition.instances.filter((instance) => !instance.variant.startsWith('routed_')).at(-1)!;
        await findComponent(page, last.node_id);
        await expect(page.getByRole('combobox', { name: /Expand instance of/ }).locator('option:checked')).toContainText(new RegExp(`instance ${last.index}`, 'i'));
      }
    }
    if (!actualMoeReference) {
      await graphPreference(page, 'Show dimensions', true);
      await viewOptions(page);
      await expect(page.getByLabel('Show dimensions')).toBeChecked();
      await page.keyboard.press('Escape');
    }
    const client = await page.context().newCDPSession(page);
    const browserMemory = browserProcessMemory();
    expect(browserMemory.process_count).toBeGreaterThan(0);
    await info.attach('production-architecture-layout', { body: JSON.stringify({ kind: reference ? 'actual checkpoint' : 'synthetic checkpoint', family,
      nodes: graph.nodes.length, edges: graph.edges.length, layoutMs: Number(await canvas.getAttribute('data-layout-ms')),
      heapSnapshot: await client.send('Runtime.getHeapUsage'), browserProcessMemory: browserMemory,
      viewport: page.viewportSize(), browser: page.context().browser()!.version() }), contentType: 'application/json' });
    const matrixCanvas = page.locator('.matrix-scroll canvas');
    const scroller = page.locator('.matrix-scroll');
    const coordinate = page.locator('.inspection-readout span').first();
    const scalar = page.locator('.inspection-readout span').last();
    const size = (p: Graph['parameters'][number]) => p.logical_shape!.reduce((total, d) => total * (d.kind === 'constant' ? d.value : 1), 1);
    if (actualMoeReference) {
      const gateSuffix = family === 'kimi_linear' ? '.block_sparse_moe.gate.weight' : '.mlp.gate.weight';
      const parameter = graph.parameters.find((p) => p.binding === 'native' && p.inspection.status === 'available' &&
        p.logical_shape?.length === 2 && p.name.includes(`.layers.${inspectionLayer(family, true)}.`) && p.name.endsWith(gateSuffix))!;
      expect(parameter).toBeTruthy();
      await inspectActualMoeWeightAtWidth(page, info, graph, family, parameter, 1440);
      await inspectActualMoeWeightAtWidth(page, info, graph, family, parameter, 1178);
      const expertName = family === 'deepseek_v2'
        ? 'model.layers.26.mlp.experts.0.gate_proj.weight'
        : family === 'glm4_moe_lite'
          ? 'model.layers.46.mlp.experts.0.gate_proj.weight'
          : 'model.layers.26.block_sparse_moe.experts.255.w1.weight';
      const expert = graph.parameters.find((p) => p.name === expertName)!;
      expect(expert).toBeTruthy();
      expect(expert.binding).toBe('quantized');
      expect(expert.inspection.status).toBe('available');
      await inspectActualMoeWeightAtWidth(page, info, graph, family, expert, 1440);
    } else {
      // Later-layer vector proves the modal uses concrete bindings, never layer-zero fallback.
      const layer = inspectionLayer(family, reference);
      const layerPath = family === 'vjepa2' ? `.layer.${layer}.` : `.layers.${layer}.`;
      const parameter = graph.parameters.find((p) => p.binding === 'native' && p.inspection.status === 'available' && p.logical_shape?.length === 1 && p.name.includes(layerPath))!;
      expect(parameter).toBeTruthy();
      await page.evaluate(() => { (window as any).__acceptance.captureScalars = true; });
      const parameterTrigger = await openParameter(page, graph, parameter, true);
      await expect(matrixCanvas).toBeVisible();
      await expect(page.locator('[data-result=tensor]')).toHaveCount(0);
      expect(observed.some((p) => p.includes(`/tensors/${parameter.inspection.status === 'available' ? parameter.inspection.tensor_id : ''}/data`))).toBe(true);
      const uploaded = await page.evaluate(() => (window as any).__acceptance.scalarValues as number[]);
      if (!reference) expect(uploaded).toEqual(Array.from({ length: (parameter.logical_shape![0] as { value: number }).value }, (_, i) => (i % 29 - 14) / 8));
      else {
        const oracle = JSON.parse(execFileSync(python, ['-m', 'acceptance.architecture_reference', family, '--tensor', parameter.name], { cwd: repo, encoding: 'utf8' }));
        expect(uploaded).toHaveLength(oracle.shape[0]);
        for (const sample of oracle.samples) expect(uploaded[sample.offset]).toBe(sample.value);
      }
      const camera = await page.locator('.react-flow__viewport').getAttribute('style');
      await page.keyboard.press('Escape'); await released(page);
      await expect(parameterTrigger).toBeFocused();
      expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
      await page.getByRole('button', { name: 'Back', exact: true }).click();
      await expect(canvas).toHaveAttribute('aria-busy', 'false');
    }
    if (!actualMoeReference) {
      // Inspect a native matrix too. Observe progressive first-cell values without
      // retaining a second copy of a potentially large reference embedding table.
      const matrices = graph.parameters.filter((p) => p.binding === 'native' && p.inspection.status === 'available' && p.logical_shape?.length === 2);
      const matrix = matrices.sort((a, b) => size(a) - size(b))[0]!;
      await page.evaluate(() => { (window as any).__acceptance.captureScalars = false; });
      await openParameter(page, graph, matrix);
      await expect(matrixCanvas).toBeVisible();
      await nativeCamera(page);
      // The accepted camera may retain a nonzero/fractional logical origin.
      // Validate the displayed coordinate and an adjacent keyboard-selected cell,
      // independently of the renderer's value lookup. Do not impose a camera reset.
      await page.mouse.move(0, 0);
      await scroller.focus();
      await expect(coordinate).toHaveText(/^row \d+ · column \d+$/);
      const match = /^row (\d+) · column (\d+)$/.exec((await coordinate.textContent())!)!;
      const row = Number(match[1]), column = Number(match[2]);
      const columns = (matrix.logical_shape![1] as { value: number }).value;
      for (const selectedColumn of [column, Math.min(column + 1, columns - 1)]) {
        if (selectedColumn !== column) await scroller.press('ArrowRight');
        const expectedValue = reference ? JSON.parse(execFileSync(python,
          ['-m', 'acceptance.architecture_reference', family, '--tensor', matrix.name, '--row', String(row), '--column', String(selectedColumn)],
          { cwd: repo, encoding: 'utf8' })).samples[0].value : ((row * columns + selectedColumn) % 29 - 14) / 8;
        await expect(coordinate).toHaveText(`row ${row} · column ${selectedColumn}`);
        // Software WebGL can delay a browser read beyond the fixture deadline
        // while a complete reference embedding table is uploading and drawing.
        await expect.poll(async () => Number(await scalar.textContent()), {
          timeout: reference ? 90_000 : 15_000,
        }).toBe(expectedValue);
      }
      await page.keyboard.press('Escape'); await released(page);
    }
    if (!actualMoeReference && ['qwen3', 'qwen35', 'deepseek_v2', 'glm4_moe_lite', 'kimi_linear'].includes(family)) {
      const packedCandidates = graph.parameters.filter((p) => p.binding === 'quantized' && p.inspection.status === 'available' &&
        (['deepseek_v2', 'glm4_moe_lite', 'kimi_linear'].includes(family)
          ? p.name.includes('.experts.') && p.name.endsWith(family === 'kimi_linear' ? 'w1.weight' : 'gate_proj.weight')
          : p.name.includes('.1.')));
      const packed = actualMoeReference
        ? packedCandidates.filter((p) => p.name.includes(`.layers.${inspectionLayer(family, true)}.`)).at(-1)!
        : packedCandidates.sort((a, b) => size(a) - size(b))[0]!;
      expect(packed).toBeTruthy();
      const packedRequests = observed.length;
      const packedViewport = page.viewportSize();
      const packedLayoutCount = await canvas.getAttribute('data-layout-count');
      await openParameter(page, graph, packed);
      expect(await canvas.getAttribute('data-layout-count')).toBe(packedLayoutCount);
      await expect(matrixCanvas).toBeVisible();
      await nativeCamera(page);
      await page.mouse.move(0, 0);
      await scroller.focus();
      await expect(coordinate).toHaveText(/^row \d+ · column \d+$/);
      const packedMatch = /^row (\d+) · column (\d+)$/.exec((await coordinate.textContent())!)!;
      const packedRow = Number(packedMatch[1]), packedColumn = Number(packedMatch[2]);
      const packedColumns = (packed.logical_shape![1] as { value: number }).value;
      const packedSamples = [];
      for (const selectedColumn of [packedColumn, Math.min(packedColumn + 1, packedColumns - 1)]) {
        if (selectedColumn !== packedColumn) await scroller.press('ArrowRight');
        await expect(coordinate).toHaveText(`row ${packedRow} · column ${selectedColumn}`);
        const expected = reference ? JSON.parse(execFileSync(python,
          ['-m', 'acceptance.architecture_reference', family, '--tensor', packed.name, '--packed',
            '--row', String(packedRow), '--column', String(selectedColumn)],
          { cwd: repo, encoding: 'utf8' })).samples[0].value : packedValue(family, packedRow, selectedColumn, packedColumns);
        await expect.poll(async () => Number(await scalar.textContent())).toBe(expected);
        packedSamples.push({ row: packedRow, column: selectedColumn, value: expected });
      }
      const tensor = packed.inspection.status === 'available' ? packed.inspection.tensor_id : '';
      expect(observed.slice(packedRequests).filter((path) => path.endsWith('/data'))).toEqual([
        expect.stringMatching(new RegExp(`/tensors/${tensor}/data$`)),
      ]);
      await info.attach('decoded-concrete-binding', { body: JSON.stringify({ kind: reference ? 'actual checkpoint' : 'synthetic checkpoint',
        parameter: packed.name, tensor, shape: packed.logical_shape, samples: packedSamples,
        oracle: reference
          ? ['deepseek_v2', 'glm4_moe_lite', 'kimi_linear'].includes(family)
            ? family === 'deepseek_v2'
              ? 'bitsandbytes 0.50.2 NF4 codebook and nested double-quant scale oracle at pinned source revision 8336490; scalar Safetensors reads'
              : 'compressed-tensors pack-quantized W4A16 scalar oracle from pinned source revision 4a69662; scalar Safetensors reads'
            : 'Independent scalar_reference from check_quantized_reference.py; bounded physical reads'
          : 'Authored fixture scalar formula' }),
      contentType: 'application/json' });
      if (reference && ['deepseek_v2', 'glm4_moe_lite', 'kimi_linear'].includes(family)) {
        await info.attach(`issue-178-${family}-focused-weight-${packedViewport?.width}`, {
          body: JSON.stringify({
            family,
            graph: graph.graph_id,
            parameter: packed.name,
            viewport: packedViewport,
            logicalNodes: graph.nodes.length,
            logicalEdges: graph.edges.length,
            visibleNodes: Number(await canvas.getAttribute('data-visible-nodes')),
            layoutCountBeforeFind: packedLayoutCount,
            layoutCountAfterFind: await canvas.getAttribute('data-layout-count'),
            layoutMs: Number(await canvas.getAttribute('data-layout-ms')),
            browserProcessMemory: browserProcessMemory(),
          }),
          contentType: 'application/json',
        });
      }
      await page.keyboard.press('Escape'); await released(page);
      await expect(canvas).toHaveAttribute('aria-busy', 'false');
      if (reference && ['deepseek_v2', 'glm4_moe_lite', 'kimi_linear'].includes(family)) {
        await page.setViewportSize({ width: 1178, height: 1000 });
        await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
        await expect(canvas).toHaveAttribute('aria-busy', 'false');
        await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
        const layoutCountBeforeFind = await canvas.getAttribute('data-layout-count');
        const trigger = await openParameter(page, graph, packed);
        await expect(canvas).toHaveAttribute('data-layout-count', layoutCountBeforeFind!);
        await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
        await expect(page.getByRole('dialog')).toContainText(packed.name);
        const node = graph.nodes.find((item) => item.parameter_ids.includes(packed.id))!;
        await expect(page.locator(`.react-flow__node[data-id=${JSON.stringify(node.id)}]`)).toBeInViewport();
        await info.attach(`issue-178-${family}-focused-weight-1178`, {
          body: JSON.stringify({
            family,
            graph: graph.graph_id,
            parameter: packed.name,
            viewport: page.viewportSize(),
            logicalNodes: graph.nodes.length,
            logicalEdges: graph.edges.length,
            visibleNodes: Number(await canvas.getAttribute('data-visible-nodes')),
            layoutCountBeforeFind,
            layoutCountAfterFind: await canvas.getAttribute('data-layout-count'),
            layoutMs: Number(await canvas.getAttribute('data-layout-ms')),
            browserProcessMemory: browserProcessMemory(),
          }),
          contentType: 'application/json',
        });
        await page.keyboard.press('Escape'); await released(page);
        await expect(trigger).toBeFocused();
        await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
        expect(JSON.parse((await canvas.getAttribute('data-source-node-ids'))!)).toEqual(componentsOf(graph).map((n) => n.id));
        expect(JSON.parse((await canvas.getAttribute('data-represented-edge-ids'))!)).toEqual(graph.edges.map((e) => e.id));
      }
    }
    const unavailable = graph.parameters.find((p) => p.inspection.status === 'unavailable');
    if (unavailable) {
      const requests = observed.length;
      await openParameter(page, graph, unavailable);
      await expect(page.getByRole('dialog').getByRole('status').first()).toBeVisible();
      await expect(page.locator('.matrix-scroll canvas')).toHaveCount(0);
      expect(observed.slice(requests).some((p) => p.endsWith('/data'))).toBe(false);
      await page.keyboard.press('Escape');
    }
    await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
    await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
    await expect(canvas).toHaveAttribute('aria-busy', 'false', { timeout: 90_000 });
    await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
    await viewOptions(page);
    if (!actualMoeReference) await expect(page.getByLabel('Show dimensions')).toBeChecked();
    await page.keyboard.press('Escape');
    if (family === 'vjepa2') expect(observed.some((p) => p.endsWith('/tokenize'))).toBe(false);
    await graphAction(page, 'Collapse all');
    await expect(canvas).toHaveAttribute('data-visible-nodes', '1');
    for (const node of graph.nodes.filter((n) => ['input', 'output'].includes(n.kind))) {
      await expect(page.locator(`.architecture-browser-row[data-node-id=${JSON.stringify(node.id)}]`)).toHaveCount(0);
    }
    await page.getByRole('button', { name: 'Center selected', exact: true }).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true);
  });
}

test('complete local reference [kimi_linear] issue 178 full expansion and focused router weight', async ({ page }, info) => {
  test.setTimeout(600_000);
  const graph = expandCompactGraph(await selectGraph(page));
  expect(graph.coverage).toBe('complete');
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await graphAction(page, 'Show all operations');
  await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
  expect(JSON.parse((await canvas.getAttribute('data-source-node-ids'))!)).toEqual(componentsOf(graph).map((node) => node.id));
  expect(JSON.parse((await canvas.getAttribute('data-represented-edge-ids'))!)).toEqual(graph.edges.map((edge) => edge.id));
  assertInterfaceCoverage(graph, projectGraph(graph, { expanded: [], exhaustive: true }));
  await recordGraph(page, info, 'issue-178-kimi-full-graph', graph);
  const parameter = graph.parameters.find((p) => p.binding === 'native' && p.inspection.status === 'available' &&
    p.logical_shape?.length === 2 && p.name.includes('.layers.26.') && p.name.endsWith('.block_sparse_moe.gate.weight'))!;
  expect(parameter).toBeTruthy();
  await inspectActualMoeWeightAtWidth(page, info, graph, 'kimi_linear', parameter, 1440);
  await inspectActualMoeWeightAtWidth(page, info, graph, 'kimi_linear', parameter, 1178);
  const expert = graph.parameters.find((p) => p.name === 'model.layers.26.block_sparse_moe.experts.255.w1.weight')!;
  expect(expert).toBeTruthy();
  expect(expert.binding).toBe('quantized');
  expect(expert.inspection.status).toBe('available');
  await inspectActualMoeWeightAtWidth(page, info, graph, 'kimi_linear', expert, 1440);
});

test('deterministic production [kimi_linear] complete repeated experts remain reachable after Find', async ({ page }, info) => {
  test.setTimeout(300_000);
  const graph = expandCompactGraph(await selectGraph(page));
  expect(graph.coverage).toBe('complete');

  const experts = graph.parameters.filter((parameter) =>
    /^model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\.weight$/.test(parameter.name));
  expect(experts).toHaveLength(26 * 256 * 3);
  expect(new Set(experts.map((parameter) => parameter.name)).size).toBe(experts.length);

  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await graphAction(page, 'Show all operations');
  const ids = await page.getByRole('tree', { name: 'Model components', exact: true })
    .locator('[data-node-id]').evaluateAll((rows) => rows.map((row) => (row as HTMLElement).dataset.nodeId));
  expect(new Set(ids)).toEqual(new Set(componentsOf(graph).map((node) => node.id)));
  assertInterfaceCoverage(graph, projectGraph(graph, { expanded: [], exhaustive: true }));
  await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
  expect(JSON.parse((await canvas.getAttribute('data-source-node-ids'))!)).toEqual(componentsOf(graph).map((node) => node.id));
  expect(JSON.parse((await canvas.getAttribute('data-represented-edge-ids'))!)).toEqual(graph.edges.map((edge) => edge.id));
  await recordGraph(page, info, 'kimi-linear-full-graph', graph);

  const parameter = experts.at(-1)!;
  const target = graph.nodes.find((node) => node.parameter_ids.includes(parameter.id))!;
  const layoutCount = await canvas.getAttribute('data-layout-count');
  await findComponent(page, target.id);
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await expect(canvas).toHaveAttribute('data-layout-count', layoutCount!);
  await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
  expect(JSON.parse((await canvas.getAttribute('data-source-node-ids'))!)).toEqual(componentsOf(graph).map((node) => node.id));
  expect(JSON.parse((await canvas.getAttribute('data-represented-edge-ids'))!)).toEqual(graph.edges.map((edge) => edge.id));
  const card = page.locator(`.react-flow__node[data-id=${JSON.stringify(target.id)}]`);
  await expect(card).toBeInViewport();
  await card.getByRole('button', { name: `Inspect ${target.label}` }).click();
  await expect(page.getByRole('dialog')).toContainText(parameter.name);
});

test('complete local reference [qwen3] exhaustive global detail remains reachable at readable scale', async ({ page }, info) => {
  const graph = await selectGraph(page);
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await graphAction(page, 'Show all operations');
  await expect(canvas).toHaveAttribute('data-visible-nodes', String(visibleCount(graph)));
  const mlp = graph.nodes.find((node) => node.kind === 'group' && node.attributes.some((a) => a.name === 'semantic_role' && a.value === 'mlp'))!;
  const operation = graph.nodes.find((node) => node.parent_id === mlp.id && node.kind === 'operation')!;
  const requests = observed.length;
  await findComponent(page, operation.id);
  await page.getByRole('button', { name: 'Center selected', exact: true }).click();
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await expect(page.locator(`.react-flow__node[data-id=${JSON.stringify(operation.id)}]`)).toBeInViewport();
  expect(JSON.parse((await canvas.getAttribute('data-source-node-ids'))!)).toEqual(componentsOf(graph).map((node) => node.id));
  expect(JSON.parse((await canvas.getAttribute('data-represented-edge-ids'))!)).toEqual(graph.edges.map((edge) => edge.id));
  expect(observed.slice(requests)).toEqual([]);
  await recordGraph(page, info, 'exhaustive-global-detail', graph);
});

test('deterministic production [smollm2] close during progressive data cancels and releases modal resources', async ({ page }) => {
  const graph = await selectGraph(page);
  const parameter = graph.parameters.find((p) => p.binding === 'native' && p.inspection.status === 'available' && p.logical_shape?.length === 2)!;
  for (let cycle = 0; cycle < 3; cycle++) {
    await page.evaluate(() => { const p = (window as any).__acceptance; p.captureScalars = true; p.scalarValues.length = 0; });
    await control('arm', { kind: 'logical_tensor' });
    await openParameter(page, graph, parameter, true);
    await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'streaming');
    await expect.poll(async () => (await metrics(page)).firstRender).toBeGreaterThan(0);
    const values = await page.evaluate(() => (window as any).__acceptance.scalarValues as number[]);
    expect(values.length).toBeGreaterThan(0);
    expect(values).toEqual(values.map((_, i) => (i % 29 - 14) / 8));
    await page.keyboard.press('Escape');
    await released(page);
    await expect.poll(async () => { const s = await control('state'); return [s.operations, s.readers, s.consumers, s.flights, s.tasks, s.temporary]; }).toEqual([0, 0, 0, 0, 0, 0]);
    await control('release', {});
    await expect(page.getByRole('dialog')).toHaveCount(0);
  }
});

// Lifetime and delayed-session regressions preserved from issue #123 at 78a7d3a6d427f28dd42420da6ef2b41fb5d5a2f2.
test('deterministic production [smollm2] repeated nested return releases obsolete layouts', async ({ page }, info) => {
  const graph = await selectGraph(page);
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  const ready = () => expect(canvas).toHaveAttribute('aria-busy', 'false');
  const layer = graph.repetitions[0]!.instances.at(-1)!.node_id;
  const attention = graph.nodes.find((node) => node.parent_id === layer && node.attributes.some(
    (attribute) => attribute.name === 'semantic_role' && attribute.value === 'attention'))!;
  await findComponent(page, layer); await ready();
  const camera = await page.locator('.react-flow__viewport').getAttribute('style');
  const records = await canvas.getAttribute('data-source-node-ids');
  await page.evaluate(() => { (window as any).__returnCanvas = new WeakRef(document.querySelector('.react-flow')!); });
  const requests = observed.length;
  const samples = [];
  for (let cycle = 0; cycle < 8; cycle++) {
    await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready();
    await expect(canvas).toHaveAttribute('data-scope-id', layer);
    await findComponent(page, attention.id); await ready();
    const layerCamera = await page.locator('.react-flow__viewport').getAttribute('style');
    await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready();
    await expect(canvas).toHaveAttribute('data-scope-id', attention.id);
    await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready();
    expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(layerCamera);
    await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready();
    await expect(canvas).toHaveAttribute('data-scope-id', '');
    expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
    expect(await canvas.getAttribute('data-source-node-ids')).toBe(records);
    await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', layer);
    expect(await page.evaluate(() => (window as any).__returnCanvas.deref() === document.querySelector('.react-flow'))).toBe(true);
    await expect.poll(() => page.evaluate(() => window.__architectureProbe().active)).toBe(0);
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    const observation = await graphObservation(page, true);
    expect(observation.observedGraphs).toBe(1);
    expect(observation.retainedGraphs).toBe(1);
    expect(observation.peak).toBeLessThanOrEqual(2);
    samples.push({ cycle, ...observation });
    await released(page);
  }
  expect(observed.slice(requests)).toEqual([]);
  await info.attach('repeated-return-resources', { body: JSON.stringify({ sourceNodes: graph.nodes.length,
    sourceEdges: graph.edges.length, samples, memory: 'Main-thread CDP JS heap after explicit GC; not browser RSS or GPU memory.' }), contentType: 'application/json' });
  const path = info.outputPath('nested-return.png');
  await page.screenshot({ path }); await info.attach('nested-return', { path, contentType: 'image/png' });

  // Compare identical returned views after two complete warm-up cycles. This
  // checks retained object growth, not a machine-dependent heap-byte limit.
  expect(samples.at(-1)!.retainedLayouts).toBeLessThanOrEqual(samples[1]!.retainedLayouts);
});

test('deterministic production [smollm2] isolated session replacement rejects a late actual response', async ({ page }) => {
  let release!: () => void, arrived!: () => void;
  const waiting = new Promise<void>((resolve) => { arrived = resolve; });
  const held = new Promise<void>((resolve) => { release = resolve; });
  let holdNext = false;
  // Register interception before layout workers start; enabling CDP network
  // interception after terminated workers can stall this Chromium harness.
  await page.route('**/architecture', async (route) => {
    if (!holdNext || route.request().method() !== 'GET') { await route.continue(); return; }
    holdNext = false;
    const response = await route.fetch();
    arrived(); await held;
    await route.fulfill({ response });
  });
  const graph = await selectGraph(page);
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  const ready = () => expect(canvas).toHaveAttribute('aria-busy', 'false');
  await findComponent(page, graph.repetitions[0]!.instances.at(-1)!.node_id); await ready();
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready();
  holdNext = true;
  try {
    await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(fixtureModelIds.qwen3!);
    await waiting;
    await expect(canvas).toHaveCount(0);
    const replacement = page.waitForResponse((response) => response.request().method() === 'GET' &&
      response.url().endsWith('/architecture') && response.status() === 200);
    await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('vjepa2');
    const current = await (await fetch((await replacement).url())).json();
    await expect(canvas).toHaveAttribute('data-graph-id', current.graph.graph_id); await ready();
    release();
    await page.unrouteAll({ behavior: 'wait' });
    await expect(canvas).toHaveAttribute('data-graph-id', current.graph.graph_id);
    await expect(canvas).toHaveAttribute('data-scope-id', '');
    await expect(page.getByLabel('Graph selection', { exact: true })).toHaveCount(0);
    await expect(page.getByRole('dialog')).toHaveCount(0);
    expect(JSON.parse((await canvas.getAttribute('data-source-node-ids'))!).every(
      (id: string) => current.graph.nodes.some((node: { id: string }) => node.id === id))).toBe(true);
    expect(observed.some((request) => request.endsWith('/tokenize'))).toBe(false);
  } finally { release(); }
});

test.describe('Extended mounted lifetime without persistent element handles', () => {
  test('deterministic production [smollm2] extended nested returns stay bounded and explorer teardown releases layouts', async ({ page }, info) => {
    const graph = await selectGraph(page);
    const canvas = page.getByLabel('Architecture graph', { exact: true });
    const ready = async () => {
      await expect(canvas).toHaveAttribute('aria-busy', 'false');
      // Layout arrival precedes the camera's animation-frame commit. Match the
      // component harness before capturing the context that Back must restore.
      await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    };
    const layer = graph.repetitions[0]!.instances.at(-1)!.node_id;
    const attention = graph.nodes.find((node) => node.parent_id === layer && node.attributes.some(
      (attribute) => attribute.name === 'semantic_role' && attribute.value === 'attention'))!;
    await findComponent(page, layer); await ready();
    const camera = await page.locator('.react-flow__viewport').getAttribute('style');
    const records = await canvas.getAttribute('data-source-node-ids');
    await page.evaluate(() => { (window as any).__returnCanvas = new WeakRef(document.querySelector('.react-flow')!); });
    const requests = observed.length;
    const samples = [];
    for (let cycle = 0; cycle < 16; cycle++) {
      await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready();
      await expect(canvas).toHaveAttribute('data-scope-id', layer);
      await findComponent(page, attention.id); await ready();
      const layerCamera = await page.locator('.react-flow__viewport').getAttribute('style');
      await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready();
      await expect(canvas).toHaveAttribute('data-scope-id', attention.id);
      await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready();
      expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(layerCamera);
      await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready();
      await expect(canvas).toHaveAttribute('data-scope-id', '');
      expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
      expect(await canvas.getAttribute('data-source-node-ids')).toBe(records);
      await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', layer);
      expect(await page.evaluate(() => (window as any).__returnCanvas.deref() === document.querySelector('.react-flow'))).toBe(true);
      await expect.poll(() => page.evaluate(() => window.__architectureProbe().active)).toBe(0);
      await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      const observation = await graphObservation(page, true);
      expect(observation.observedGraphs).toBe(1);
      expect(observation.retainedGraphs).toBe(1);
      expect(observation.peak).toBeLessThanOrEqual(2);
      samples.push({ cycle, ...observation });
      await released(page);
    }
    expect(observed.slice(requests)).toEqual([]);
    await info.attach('repeated-return-resources', { body: JSON.stringify({ sourceNodes: graph.nodes.length,
      sourceEdges: graph.edges.length, samples, memory: 'Main-thread CDP JS heap after explicit GC; not browser RSS or GPU memory.' }), contentType: 'application/json' });
    // A second warmed window must not merely defer the original linear growth.
    await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
    await expect(canvas).toHaveCount(0);
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    const teardown = await graphObservation(page, true);
    await info.attach('explorer-teardown', { body: JSON.stringify(teardown), contentType: 'application/json' });
    expect(teardown.active).toBe(0);
    expect(teardown.retainedLayouts).toBe(0);
    expect(teardown.retainedGraphs).toBe(0);
    expect(await page.evaluate(() => (window as any).__returnCanvas.deref())).toBeUndefined();
    expect(samples[7]!.retainedLayouts).toBeLessThanOrEqual(samples[1]!.retainedLayouts);
    expect(samples.at(-1)!.retainedLayouts).toBeLessThanOrEqual(samples[7]!.retainedLayouts);
  });
});

test('shared structure production [templates] neutral mode, distinct instance weights and cancellation preserve one canvas', async ({ page }, info) => {
  const graph = await selectGraph(page), canvas = page.getByLabel('Architecture graph', { exact: true });
  const template = graph.templates!.find((t) => t.component_role === 'attention')!;
  expect(template.instances).toHaveLength(2);
  const [first, second] = template.instances;
  const query = (instance: NonNullable<Graph['templates']>[number]['instances'][number]) => graph.nodes.find((n) =>
    instance.nodes.some((m) => m.node_id === n.id) && n.attributes.some((a) => a.name === 'semantic_role' && a.value === 'query_projection'))!;
  const q0 = query(first!), q1 = query(second!);
  const p0 = graph.parameters.find((p) => p.id === q0.parameter_ids[0])!, p1 = graph.parameters.find((p) => p.id === q1.parameter_ids[0])!;
  expect(p0.name).toBe('model.layers.0.self_attn.q_proj.weight'); expect(p1.name).toBe('model.layers.1.self_attn.q_proj.weight');
  await openShared(page, template.id);
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  const before = observed.length;
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  await expect(page.getByRole('dialog')).toContainText('No instance selected');
  await expect(page.getByLabel('Inspect parameter', { exact: true })).toHaveCount(0);
  expect(observed.slice(before)).toEqual([]);
  await page.keyboard.press('Escape');
  await page.getByLabel('Shared structure instance', { exact: true }).selectOption(first!.node_id);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click();
  await page.locator(`.react-flow__node[data-id=${JSON.stringify(q0.id)}] .architecture-node-label`).click();
  await page.evaluate(() => { const p = (window as any).__acceptance; p.captureScalars = true; p.scalarValues.length = 0; });
  await control('arm', { kind: 'logical_tensor' });
  await page.getByRole('button', { name: `Inspect matrix ${p0.name}`, exact: true }).click();
  await expect(page.getByLabel('Inspect parameter', { exact: true })).toHaveValue(p0.id);
  await expect(page.locator('[data-result=tensor]')).toHaveAttribute('data-state', 'streaming');
  await expect.poll(async () => (await control('state')).control.entered).toBe(true);
  await expect.poll(async () => (await metrics(page)).firstRender).toBeGreaterThan(0);
  expect(await page.evaluate(() => (window as any).__acceptance.scalarValues)).toEqual(Array.from({ length: 144 }, (_, i) => 100 + (i % 29 - 14) / 8));
  const camera = await page.locator('.react-flow__viewport').getAttribute('style');
  const layoutCount = await canvas.getAttribute('data-layout-count');
  // Drive the same instance-change event with a modal open to exercise its
  // defensive dismissal/lifetime fence; ordinary pointer use closes the modal first.
  await page.getByLabel('Shared structure instance', { exact: true }).evaluate((element, id) => {
    (element as HTMLSelectElement).value = id; element.dispatchEvent(new Event('change', { bubbles: true }));
  }, second!.node_id);
  await expect(page.getByRole('dialog')).toHaveCount(0); await released(page);
  await expect.poll(async () => { const s = await control('state'); return [s.operations, s.readers, s.consumers, s.flights, s.tasks, s.temporary]; }).toEqual([0, 0, 0, 0, 0, 0]);
  await control('release', {});
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', q1.id);
  expect(await canvas.getAttribute('data-layout-count')).toBe(layoutCount);
  expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
  await page.evaluate(() => { (window as any).__acceptance.scalarValues.length = 0; });
  const matrixTrigger = page.getByRole('button', { name: `Inspect matrix ${p1.name}`, exact: true });
  await matrixTrigger.focus(); await page.keyboard.press('Enter');
  await expect(page.getByLabel('Inspect parameter', { exact: true })).toHaveValue(p1.id);
  await expect(page.getByRole('dialog')).toContainText('Module: model.layers.1.self_attn.q_proj');
  await page.getByText('Concrete instance interface connections', { exact: true }).click();
  const memberIds = new Set(second!.nodes.map((n) => n.node_id));
  const external = graph.edges.filter((e) => memberIds.has(e.source.node_id) !== memberIds.has(e.target.node_id));
  for (const edge of external) await expect(page.getByRole('dialog')).toContainText(edge.id);
  await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
  await expect(page.locator('[data-result=tensor]')).toHaveCount(0);
  expect(await page.evaluate(() => (window as any).__acceptance.scalarValues)).toEqual(Array.from({ length: 144 }, (_, i) => 200 + (i % 29 - 14) / 8));
  for (const parameter of [p0, p1]) {
    expect(parameter.inspection.status).toBe('available');
    if (parameter.inspection.status === 'available') {
      const tensorId = parameter.inspection.tensor_id;
      expect(observed.some((path) => path.endsWith(`/tensors/${tensorId}/data`))).toBe(true);
    }
  }
  await page.keyboard.press('Escape'); await released(page);
  const requests = observed.length;
  const samples = [];
  for (let cycle = 0; cycle < 8; cycle++) {
    await page.getByLabel('Shared structure instance', { exact: true }).selectOption(cycle % 2 ? first!.node_id : second!.node_id);
    samples.push(await graphObservation(page, true));
  }
  expect(observed.slice(requests)).toEqual([]);
  await info.attach('shared-instance-resources', { body: JSON.stringify(samples), contentType: 'application/json' });
  expect(samples.every((sample) => sample.active === 0 && sample.retainedLayouts <= 1 && sample.retainedGraphs <= 2)).toBe(true);
  expect(await canvas.getAttribute('data-layout-count')).toBe(layoutCount);
  expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
  await recordGraph(page, info, 'shared-instance-weight-cancelled', graph);
});
