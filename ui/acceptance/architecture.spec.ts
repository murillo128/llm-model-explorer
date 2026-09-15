import { findComponent, graphAction, graphPreference, viewOptions } from '../tests/architecture-controls';
/* eslint-disable @typescript-eslint/no-explicit-any -- Native test-only observations and evidence. */
import { test, expect } from '@playwright/test';
import type { Page, TestInfo } from '@playwright/test';
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import type { Graph } from '../src/architecture-explorer/graph';
import { projectGraph } from '../src/architecture-explorer/projection';
import { assertTraceability } from '../tests/architecture-invariants';
import { installProbe } from './probe';
import { nativeCamera } from '../tests/native-camera';
import { fanoutPoint } from '../tests/architecture-pointer';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const python = `${repo}backend/.venv/bin/python`;
let service: ReturnType<typeof spawn>;
let root: string;
let backend: string;
let modelId: string;
let log: string;
let observed: string[];

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
  if (isolated && await page.getByLabel('Architecture graph', { exact: true }).getAttribute('data-scope-id') !== node.id) {
    const requests = observed.length;
    await page.getByRole('button', { name: 'Explore component', exact: true }).click();
    await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('data-scope-id', node.id);
    await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('aria-busy', 'false');
    expect(observed.slice(requests)).toEqual([]);
  }
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  await page.getByLabel('Inspect parameter', { exact: true }).selectOption(parameter.id);
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
  const components = graph.nodes.filter((node) => node.kind === 'group' && (role(node, 'attention') || role(node, 'mlp')));
  expect(components).toHaveLength(2 * graph.repetitions.reduce((n, r) => n + r.instances.length, 0));
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  const ready = () => expect(canvas).toHaveAttribute('aria-busy', 'false');
  const capture = async (name: string) => {
    if (info.project.name !== 'dpr1') return;
    const path = info.outputPath(`${reference ? 'reference' : 'fixture'}-${family}-${name}.png`);
    await page.screenshot({ path });
    await info.attach(`components-${name}`, { path, contentType: 'image/png' });
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
  for (const repetition of graph.repetitions) {
    const instances = family === 'qwen35'
      ? repetition.instances.filter((instance, index, all) => all.findIndex((other) => other.variant === instance.variant) === index)
      : repetition.instances.slice(0, 1);
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
      await expect(page.getByRole('button', { name: 'MLP', exact: true })).toBeVisible();
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
  const owner = graph.repetitions[0]!.instances[0]!.node_id;
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
      const connections = page.locator('.architecture-connection[data-source-node^="external:"]');
      const targets = await connections.evaluateAll((elements) => elements.map((element) => ({
        source: element.getAttribute('data-source-node')!, port: element.getAttribute('data-source-port')!,
        id: element.getAttribute('data-edge-id')!,
      })));
      const fanout = targets.find((item) => targets.filter((other) => item.source === other.source && item.port === other.port).length > 1);
      if (fanout) {
        // A fitted scope contains its external context as well as its internal body.
        await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready();
        const branches = page.locator(`.architecture-connection[data-source-node=${JSON.stringify(fanout.source)}][data-source-port=${JSON.stringify(fanout.port)}]`);
        const branchList = await branches.all();
        const expected = (await branches.evaluateAll((elements) => elements.map((element) => element.getAttribute('data-edge-id')!))).sort();
        const port = page.locator(`.architecture-port[data-node-id=${JSON.stringify(fanout.source)}][data-port-id=${JSON.stringify(fanout.port)}]`);
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
        component: await root.boundingBox(), boundaryFanout: fanout ? targets.filter((item) => item.source === fanout.source).length : 0,
      }), contentType: 'application/json' });
      await page.mouse.move(0, 0);
      await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready();
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
}

test.beforeEach(async ({ page }, info) => {
  const family = /\[(\w+)\]/.exec(info.title)![1]!;
  const reference = info.title.startsWith('complete local reference');
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
    const selected = JSON.parse(execFileSync(python, ['-m', 'acceptance.architecture_reference', family], { cwd: repo, encoding: 'utf8' }));
    modelId = selected.model_id;
    await info.attach('actual-reference-inventory', { body: JSON.stringify(selected.report), contentType: 'application/json' });
    args = ['-m', 'llm_model_explorer', '--model-root', dirname(selected.directory), '--cache-dir', join(root, 'cache'), '--port', String(port), '--cors-origin', origin];
  } else {
    execFileSync(python, ['-m', 'acceptance.architecture_fixtures', join(root, 'models')], { cwd: repo });
    modelId = family;
    args = ['-m', 'acceptance.server', '--root', root, '--port', String(port), '--origin', origin];
  }
  service = spawn(python, args, { cwd: repo, env: { ...process.env, HF_HUB_OFFLINE: '1', TOKENIZERS_PARALLELISM: 'false' }, stdio: ['ignore', 'pipe', 'pipe'] });
  service.stdout!.on('data', (data) => { log += data; }); service.stderr!.on('data', (data) => { log += data; });
  await expect.poll(async () => {
    if (service.exitCode !== null) throw new Error(log);
    try { return (await fetch(`${backend}/models`)).status; } catch { return 0; }
  }, { timeout: reference ? 300_000 : 30_000 }).toBe(200);
  page.on('request', (r) => { if (r.url().startsWith(backend)) observed.push(new URL(r.url()).pathname); });
  await page.addInitScript(installProbe);
  await page.addInitScript(() => { (window as any).__acceptance.capture = false; });
  await page.goto('/');
});

test.afterEach(async ({ page }, info) => {
  if (info.status === 'skipped') return;
  await info.attach('backend-log', { body: log ?? '', contentType: 'text/plain' });
  await page.close();
  if (service?.exitCode === null) {
    service.kill('SIGTERM'); const timer = setTimeout(() => service.kill('SIGKILL'), 10_000);
    try { await once(service, 'exit'); } finally { clearTimeout(timer); }
  }
  if (root) rmSync(root, { recursive: true });
});

for (const reference of [false, true]) for (const family of ['smollm2', 'qwen3', 'qwen35', 'vjepa2']) {
  test(`${reference ? 'complete local reference' : 'deterministic production'} [${family}] full graph, concrete bindings and logical weight modal`, async ({ page }, info) => {
    test.setTimeout(reference ? 600_000 : 90_000);
    const graph = await selectGraph(page);
    expect(graph.coverage).toBe('complete');
    // Apply the common source oracle to actual backend graphs, including dense
    // Qwen/Llama differences and both visual stacks (not UI stress templates).
    assertTraceability(graph, projectGraph(graph, { expanded: [], exhaustive: true }), true);
    for (const repetition of graph.repetitions) for (const instance of [repetition.instances[0]!, repetition.instances.at(-1)!]) {
      const expanded = [instance.node_id];
      for (let node = graph.nodes.find((item) => item.id === instance.node_id); node?.parent_id;
        node = graph.nodes.find((item) => item.id === node!.parent_id)) expanded.push(node.parent_id);
      assertTraceability(graph, projectGraph(graph, { expanded }));
    }
    const canvas = page.getByLabel('Architecture graph', { exact: true });
    await inspectComponents(page, graph, info, family, reference);
    await inspectIsolation(page, graph, info, family, reference);
    await viewOptions(page);
    await expect(page.getByLabel('Show dimensions')).not.toBeChecked();
    await page.keyboard.press('Escape');
    await page.getByRole('button', { name: 'Find component', exact: true }).click();
    const ids = await page.getByRole('listbox', { name: 'Components', exact: true }).getByRole('option').evaluateAll((options) => options.map((o) => (o as HTMLElement).dataset.nodeId));
    expect(ids).toEqual(graph.nodes.map((n) => n.id));
    await page.keyboard.press('Escape');
    await graphAction(page, 'Show all operations');
    await expect(canvas).toHaveAttribute('data-visible-nodes', String(graph.nodes.length));
    // Visible routes compose boundary forwarding. Every original edge remains
    // traceable even though one route may represent several source segments.
    expect(JSON.parse((await canvas.getAttribute('data-source-node-ids'))!)).toEqual(graph.nodes.map((n) => n.id));
    expect(JSON.parse((await canvas.getAttribute('data-represented-edge-ids'))!)).toEqual(graph.edges.map((e) => e.id));
    const roots = new Set(graph.nodes.filter((n) => !n.parent_id).map((n) => n.id));
    for (const repetition of graph.repetitions) {
      const last = repetition.instances.at(-1)!;
      await findComponent(page, last.node_id);
      await expect(page.getByRole('combobox', { name: /Expand instance of/ }).locator('option:checked')).toContainText(new RegExp(`instance ${last.index}`, 'i'));
    }
    await graphPreference(page, 'Show dimensions', true);
    await viewOptions(page);
    await expect(page.getByLabel('Show dimensions')).toBeChecked();
    await page.keyboard.press('Escape');
    const client = await page.context().newCDPSession(page);
    await info.attach('production-architecture-layout', { body: JSON.stringify({ kind: reference ? 'actual checkpoint' : 'synthetic checkpoint', family,
      nodes: graph.nodes.length, edges: graph.edges.length, layoutMs: Number(await canvas.getAttribute('data-layout-ms')),
      heapSnapshot: await client.send('Runtime.getHeapUsage'), viewport: page.viewportSize(), browser: page.context().browser()!.version() }), contentType: 'application/json' });
    // Later-layer vector proves the modal uses concrete bindings, never layer-zero fallback.
    const parameter = graph.parameters.find((p) => p.binding === 'native' && p.inspection.status === 'available' && p.logical_shape?.length === 1 && p.name.includes('.1.'))!;
    expect(parameter).toBeTruthy();
    await page.evaluate(() => { (window as any).__acceptance.captureScalars = true; });
    await openParameter(page, graph, parameter, true);
    await expect(page.locator('.matrix-scroll canvas')).toBeVisible();
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
    await expect(page.getByRole('button', { name: 'Inspect selected', exact: true })).toBeFocused();
    expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
    await page.getByRole('button', { name: 'Back', exact: true }).click();
    await expect(canvas).toHaveAttribute('aria-busy', 'false');
    // Inspect a native matrix too. Observe progressive first-cell values without
    // retaining a second copy of a potentially large reference embedding table.
    const matrices = graph.parameters.filter((p) => p.binding === 'native' && p.inspection.status === 'available' && p.logical_shape?.length === 2);
    const size = (p: typeof parameter) => p.logical_shape!.reduce((total, d) => total * (d.kind === 'constant' ? d.value : 1), 1);
    const matrix = matrices.sort((a, b) => size(a) - size(b))[0]!;
    await page.evaluate(() => { (window as any).__acceptance.captureScalars = false; });
    await openParameter(page, graph, matrix);
    const matrixCanvas = page.locator('.matrix-scroll canvas');
    await expect(matrixCanvas).toBeVisible();
    await nativeCamera(page);
    // The accepted camera may retain a nonzero/fractional logical origin.
    // Validate the displayed coordinate and an adjacent keyboard-selected cell,
    // independently of the renderer's value lookup. Do not impose a camera reset.
    await page.mouse.move(0, 0);
    const scroller = page.locator('.matrix-scroll');
    await scroller.focus();
    const coordinate = page.locator('.inspection-readout span').first();
    const scalar = page.locator('.inspection-readout span').last();
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
    if (!reference && ['qwen3', 'qwen35'].includes(family)) {
      const packed = graph.parameters.filter((p) => p.binding === 'quantized' && p.inspection.status === 'available')
        .sort((a, b) => size(a) - size(b))[0]!;
      expect(packed).toBeTruthy();
      await openParameter(page, graph, packed);
      await expect(matrixCanvas).toBeVisible();
      await nativeCamera(page);
      await page.mouse.move(0, 0);
      await scroller.focus();
      await expect(coordinate).toHaveText(/^row \d+ · column \d+$/);
      const packedMatch = /^row (\d+) · column (\d+)$/.exec((await coordinate.textContent())!)!;
      const packedRow = Number(packedMatch[1]), packedColumn = Number(packedMatch[2]);
      const packedColumns = (packed.logical_shape![1] as { value: number }).value;
      for (const selectedColumn of [packedColumn, Math.min(packedColumn + 1, packedColumns - 1)]) {
        if (selectedColumn !== packedColumn) await scroller.press('ArrowRight');
        await expect(coordinate).toHaveText(`row ${packedRow} · column ${selectedColumn}`);
        await expect.poll(async () => Number(await scalar.textContent())).toBe(
          packedValue(family, packedRow, selectedColumn, packedColumns));
      }
      expect(observed.some((path) => path.includes(`/tensors/${packed.inspection.status === 'available' ? packed.inspection.tensor_id : ''}/data`))).toBe(true);
      await page.keyboard.press('Escape'); await released(page);
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
    await expect(canvas).toHaveAttribute('data-visible-nodes', String(graph.nodes.length));
    await viewOptions(page);
    await expect(page.getByLabel('Show dimensions')).toBeChecked();
    await page.keyboard.press('Escape');
    if (family === 'vjepa2') expect(observed.some((p) => p.endsWith('/tokenize'))).toBe(false);
    await graphAction(page, 'Collapse all');
    await expect(canvas).toHaveAttribute('data-visible-nodes', String(roots.size));
    expect(JSON.parse((await canvas.getAttribute('data-represented-edge-ids'))!)).toEqual(
      graph.edges.filter((edge) => roots.has(edge.source.node_id) && roots.has(edge.target.node_id)).map((edge) => edge.id));
    await page.getByRole('button', { name: 'Center selected', exact: true }).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true);
  });
}

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
