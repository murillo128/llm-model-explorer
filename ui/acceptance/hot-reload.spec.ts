/* eslint-disable @typescript-eslint/no-explicit-any -- Existing native resource probe has a dynamic observation API. */
import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Graph } from '../src/architecture-explorer/graph';
import type { Session } from '../src/app/session-controller';
import { findComponent } from '../tests/architecture-controls';
import { revealTensor } from '../tests/tensor-tree-helpers';
import { installProbe } from './probe';
import { installArchitectureProbe } from './architecture-probe';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const python = join(repo, 'backend/.venv/bin/python');
let root: string, backend: string, log: string;
let service: ReturnType<typeof spawn>;
const graphCanvas = (page: Page) => page.getByLabel('Architecture graph', { exact: true });
const modelPicker = (page: Page) => page.getByRole('combobox', { name: 'Model', exact: true });
const nodeBox = (page: Page, id: string) => page.locator(`.react-flow__node[data-id="${id}"]`);

function atomic(path: string, data: string) {
  writeFileSync(`${path}.tmp`, data); renameSync(`${path}.tmp`, path);
}
function editDefinition(change: (definition: any) => void, model = 'a') {
  const path = join(root, 'models', model, 'architecture.json');
  const definition = JSON.parse(readFileSync(path, 'utf8')); change(definition);
  atomic(path, JSON.stringify(definition));
}
function changeWeights(value: number, model = 'a') {
  execFileSync(python, ['-m', 'acceptance.hot_reload_fixtures', join(root, 'models', model), '--value', String(value)], { cwd: repo });
}
async function control(path = 'state', post = false) {
  const response = await fetch(`${backend}/__test/${path}`, { method: post ? 'POST' : 'GET' });
  expect(response.ok).toBe(true); return response.json();
}
async function session(page: Page): Promise<Session> {
  const id = await page.evaluate(url => sessionStorage.getItem(`llm-model-explorer:session:${url}`), backend);
  expect(id).toBeTruthy();
  const response = await fetch(`${backend}/sessions/${id}`); expect(response.ok).toBe(true);
  return response.json();
}
async function replaced(page: Page, old: Session): Promise<Session> {
  await expect.poll(() => page.evaluate(url => sessionStorage.getItem(`llm-model-explorer:session:${url}`), backend)).not.toBe(old.id);
  await expect.poll(() => page.evaluate(url => sessionStorage.getItem(`llm-model-explorer:session:${url}`), backend)).toBeTruthy();
  const next = await session(page); expect(next.model_revision).not.toBe(old.model_revision);
  expect(next.model_id).toBe(old.model_id);
  await expect.poll(async () => (await fetch(`${backend}/sessions/${old.id}`)).status).toBe(404);
  return next;
}
async function graph(page: Page, current?: Session): Promise<Graph> {
  const pinned = current ?? await session(page);
  const response = await fetch(`${backend}/sessions/${pinned.id}/architecture`);
  expect(response.ok).toBe(true);
  const body = await response.json(); expect(body.status).toBe('available');
  await expect(graphCanvas(page)).toHaveAttribute('data-graph-id', body.graph.graph_id);
  await expect(graphCanvas(page)).toHaveAttribute('aria-busy', 'false');
  return body.graph;
}
function stageOne(graph: Graph) { return graph.repetitions[0]!.instances.find(instance => instance.index === 1)!.node_id; }
function component(graph: Graph) { return graph.nodes.find(node => node.parent_id === stageOne(graph) && node.kind === 'group')!; }
function projection(graph: Graph) { return graph.nodes.find(node => node.parent_id === component(graph).id && node.operation === 'linear')!; }
async function openGraph(page: Page) {
  await modelPicker(page).selectOption('reload/a');
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  return graph(page);
}
async function camera(page: Page) {
  return page.locator('.react-flow__viewport').evaluate(node => {
    const m = new DOMMatrixReadOnly(getComputedStyle(node).transform); return { x: m.e, y: m.f, zoom: m.a };
  });
}
async function anchor(page: Page, id: string) {
  const box = await nodeBox(page, id).boundingBox(), view = await page.locator('.architecture-flow').boundingBox();
  expect(box).toBeTruthy(); expect(view).toBeTruthy();
  return { x: (box!.x + box!.width / 2 - view!.x) / view!.width, y: (box!.y + box!.height / 2 - view!.y) / view!.height };
}
async function close(page: Page) {
  await page.getByRole('button', { name: 'Session options', exact: true }).click();
  await page.getByRole('button', { name: 'Close session', exact: true }).click();
}
async function observe(page: Page) {
  await page.addInitScript(installProbe, { capturePixels: false });
  await page.addInitScript(installArchitectureProbe);
  await page.addInitScript(() => {
    const Native = window.EventSource;
    const states: any[] = [];
    // Passive native transport observation: no fake messages or restoration.
    window.EventSource = class extends Native {
      constructor(url: string | URL, options?: EventSourceInit) {
        super(url, options);
        this.addEventListener('model-state', event => states.push({ url: this.url, ...JSON.parse(event.data) }));
      }
    };
    (window as any).__reloadEvidence = { document: crypto.randomUUID(), states };
  });
}
async function states(page: Page) { return page.evaluate(() => (window as any).__reloadEvidence.states as any[]); }
async function ready(page: Page) {
  await observe(page); await page.goto('/');
  await expect(page.getByTestId('backend-url')).toHaveText(backend);
}

test.beforeEach(async ({ page, baseURL }) => {
  root = mkdtempSync(join(tmpdir(), 'lmex-hot-reload-')); log = '';
  const port = Number(new URL(baseURL!).port) === 4175 ? 8765 : Number(new URL(baseURL!).port) + 1;
  backend = `http://127.0.0.1:${port}`;
  execFileSync(python, ['-m', 'acceptance.hot_reload_fixtures', join(root, 'models')], { cwd: repo });
  service = spawn(python, ['-m', 'acceptance.server', '--root', root, '--port', String(port), '--origin', baseURL!], {
    cwd: repo, env: { ...process.env, HF_HUB_OFFLINE: '1', TOKENIZERS_PARALLELISM: 'false' }, stdio: ['ignore', 'pipe', 'pipe'],
  });
  service.stdout!.on('data', data => { log += data; }); service.stderr!.on('data', data => { log += data; });
  await expect.poll(async () => {
    if (service.exitCode !== null) throw new Error(log);
    try { return (await fetch(`${backend}/models`)).status; } catch { return 0; }
  }, { timeout: 30_000 }).toBe(200);
  await ready(page);
});
test.afterEach(async ({ page }, info) => {
  try {
    if (info.status !== info.expectedStatus) await info.attach('backend-log', { body: log, contentType: 'text/plain' });
    await page.close();
  } finally {
    if (service?.exitCode === null) {
      service.kill('SIGTERM'); const timer = setTimeout(() => service.kill('SIGKILL'), 10_000);
      try { await once(service, 'exit'); } finally { clearTimeout(timer); }
    }
    if (root) rmSync(root, { recursive: true, force: true });
  }
});

test('filesystem graph revisions preserve nonzero scope, camera, bindings and conservative recovery', async ({ page }, info) => {
  const before = await openGraph(page), old = await session(page);
  const process = await control(), document = await page.evaluate(() => (window as any).__reloadEvidence.document);
  await expect.poll(() => states(page)).not.toHaveLength(0);
  expect((await states(page))[0]).toMatchObject({ model_id: 'reload/a', model_revision: old.model_revision, status: 'present' });
  expect(new URL((await states(page))[0].url).searchParams.get('model_id')).toBe('reload/a');
  await findComponent(page, stageOne(before));
  await page.getByRole('button', { name: 'Explore component', exact: true }).click();
  await findComponent(page, component(before).id);
  await page.getByRole('button', { name: 'Toggle selected group', exact: true }).click();
  await findComponent(page, projection(before).id);
  const flow = await page.locator('.architecture-flow').boundingBox();
  await page.mouse.move(flow!.x + flow!.width / 2, flow!.y + flow!.height / 2);
  const initialCamera = await camera(page);
  await page.mouse.wheel(0, -140);
  await expect.poll(async () => (await camera(page)).zoom).toBeGreaterThan(initialCamera.zoom);
  // Drag the canvas background, away from the selected card.
  await page.mouse.move(flow!.x + 20, flow!.y + 30); await page.mouse.down();
  await page.mouse.move(flow!.x + 65, flow!.y + 55, { steps: 5 }); await page.mouse.up();
  const oldCamera = await camera(page), oldAnchor = await anchor(page, projection(before).id);
  const numeric: string[] = [];
  page.on('request', request => { if (request.url().endsWith('/data')) numeric.push(request.url()); });
  const controlSession = await (await fetch(`${backend}/sessions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id: 'reload/a' }) })).json() as Session;
  editDefinition(definition => {
    const node = definition.nodes.find((node: any) => node.id === 'projection1');
    node.label = 'Revised projection'; node.formula = 'out = linear(x, W, b)';
  });
  const next = await replaced(page, old), after = await graph(page, next);
  expect(after.graph_id).not.toBe(before.graph_id);
  for (const kind of ['nodes', 'edges', 'parameters', 'repetitions'] as const) {
    expect(after[kind].some(record => before[kind].some(old => old.id === record.id))).toBe(false);
  }
  await expect(graphCanvas(page)).toHaveAttribute('data-scope-id', stageOne(after));
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', projection(after).id);
  await expect(nodeBox(page, projection(after).id)).toContainText('Revised projection');
  await expect(nodeBox(page, projection(after).id)).toContainText('out = linear(x, W, b)');
  const newAnchor = await anchor(page, projection(after).id);
  expect((await camera(page)).zoom).toBeCloseTo(oldCamera.zoom, 3);
  expect(Math.abs(newAnchor.x - oldAnchor.x)).toBeLessThan(0.08);
  expect(Math.abs(newAnchor.y - oldAnchor.y)).toBeLessThan(0.08);
  const stale = await fetch(`${backend}/sessions/${controlSession.id}/tensors`);
  expect(stale.status).toBe(409); expect(await stale.json()).toMatchObject({ code: 'model_content_changed' });
  await fetch(`${backend}/sessions/${controlSession.id}`, { method: 'DELETE' });
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  const weight = after.parameters.find(p => p.name === 'encoder.proj.weight')!;
  await page.getByLabel('Inspect parameter', { exact: true }).selectOption(weight.id);
  await expect(page.getByRole('dialog').locator('.matrix-scroll canvas').first()).toBeVisible();
  expect(weight.inspection.status).toBe('available');
  await expect.poll(() => numeric.some(url => url.includes(`/sessions/${next.id}/tensors/`))).toBe(true);
  const numericCount = numeric.length, stableCamera = await camera(page), eventCount = (await states(page)).length;
  editDefinition(d => { d.name = 'Other model edited'; }, 'b');
  // Wait for independent B observation; no fixed sleep can prove A's no-op.
  const response = await fetch(`${backend}/models/events?model_id=reload%2Fb`);
  const reader = response.body!.getReader(); let frames = '';
  while (!frames.includes('event: model-state')) frames += new TextDecoder().decode((await reader.read()).value);
  await reader.cancel();
  expect(await session(page)).toEqual(next); expect((await states(page)).length).toBe(eventCount);
  expect(await camera(page)).toEqual(stableCamera); expect(numeric).toHaveLength(numericCount);
  await page.getByRole('button', { name: 'Close inspection', exact: true }).click();
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(graphCanvas(page)).toHaveAttribute('data-scope-id', '');
  await findComponent(page, projection(after).id);
  // Remove the selected author identity but retain its parent and a same-label
  // replacement. Label matching or selecting stage zero would both be wrong.
  editDefinition(d => {
    d.nodes.find((n: any) => n.id === 'projection1').id = 'replacement1';
    d.nodes.find((n: any) => n.id === 'encoder1').children = ['replacement1', 'activation1'];
    for (const edge of d.edges) for (const end of ['source', 'target']) if (edge[end].node_id === 'projection1') edge[end].node_id = 'replacement1';
  });
  const fallbackSession = await replaced(page, next), fallback = await graph(page, fallbackSession);
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', component(fallback).id);
  await expect(page.getByText(/nearest available component/)).toBeVisible();
  const config = join(root, 'models/a/config.json'), saved = readFileSync(config, 'utf8');
  atomic(config, '{');
  await expect(page.getByRole('contentinfo')).toContainText('Waiting for valid model');
  await expect(graphCanvas(page)).toHaveCount(0);
  await expect(modelPicker(page)).toHaveValue('reload/a');
  await expect.poll(async () => (await control()).sessions).toEqual([]);
  atomic(config, saved);
  const recovered = await replaced(page, fallbackSession); await graph(page, recovered);
  const definition = join(root, 'models/a/architecture.json'), valid = readFileSync(definition, 'utf8');
  atomic(definition, '{');
  const malformed = await replaced(page, recovered);
  await expect(graphCanvas(page)).toHaveCount(0);
  const failedGraph = await (await fetch(`${backend}/sessions/${malformed.id}/architecture`)).json();
  expect(failedGraph.status).toBe('unavailable');
  const inventory = await (await fetch(`${backend}/sessions/${malformed.id}/tensors`)).json();
  const tensor = inventory.tensors.find((tensor: any) => tensor.name === 'encoder.proj.weight');
  const stream = await fetch(`${backend}/sessions/${malformed.id}/tensors/${tensor.id}/data`, { method: 'POST' });
  expect(stream.status).toBe(200); expect((await stream.arrayBuffer()).byteLength).toBeGreaterThan(48);
  atomic(definition, valid);
  const repaired = await replaced(page, malformed); await graph(page, repaired);
  const state = await control(); expect(state.pid).toBe(process.pid); expect(state.observer.epoch).toBe(process.observer.epoch);
  expect(await page.evaluate(() => (window as any).__reloadEvidence.document)).toBe(document);
  await expect.poll(async () => (await control()).sessions).toEqual([repaired.id]);
  await expect.poll(async () => (await control()).observer.subscribers).toBe(1);
  const cdp = await page.context().newCDPSession(page); await cdp.send('HeapProfiler.collectGarbage'); await cdp.detach();
  const resources = await page.evaluate(() => ({ graph: window.__architectureProbe(), numeric: (window as any).__acceptance.metrics() }));
  expect(resources.graph.retainedGraphs).toBeLessThanOrEqual(1); expect(resources.graph.active).toBeLessThanOrEqual(1);
  expect([resources.numeric.textures, resources.numeric.readers]).toEqual([0, 0]);
  await info.attach('hot-reload-identities', { contentType: 'application/json', body: JSON.stringify({ old, next, graph: [before.graph_id, after.graph_id], oldCamera, oldAnchor, newAnchor, resources, revisions: (await states(page)).map(state => state.model_revision) }) });
  await close(page);
  await expect.poll(async () => { const s = await control(); return [s.sessions.length, s.observer.subscribers, s.observer.running]; }).toEqual([0, 0, false]);
});

test('native SSE reconnect catches a missed revision once and latest tab intent releases abandoned sessions', async ({ page, browser, baseURL }, info) => {
  const secondContext = await browser.newContext({ baseURL: baseURL! });
  const second = await secondContext.newPage();
  const pins: Session[] = [];
  page.on('response', async response => {
    if (response.url() === `${backend}/sessions` && response.status() === 201) pins.push(await response.json());
  });
  try {
    await modelPicker(page).selectOption('reload/a');
    await expect.poll(() => pins.length).toBe(1);
    const old = await session(page);
    await ready(second); await modelPicker(second).selectOption('reload/b');
    await expect.poll(() => second.evaluate(url => sessionStorage.getItem(`llm-model-explorer:session:${url}`), backend)).toBeTruthy();
    const unaffected = await session(second); expect(unaffected.id).not.toBe(old.id);
    await expect.poll(async () => (await control()).observer.subscribers).toBe(2);
    const observer = (await control()).observer;
    let reconnect = false;
    await page.route('**/models/events?*', route => reconnect ? route.continue() : route.abort('connectionfailed'));
    await control('disconnect-model-events', true);
    await expect(page.getByRole('contentinfo')).toContainText('Live updates reconnecting');
    expect(await session(page)).toEqual(old);
    editDefinition(d => { d.name = 'Changed during disconnect'; });
    reconnect = true;
    const next = await replaced(page, old);
    await expect.poll(() => pins.length).toBe(2);
    await expect.poll(async () => (await states(page)).at(-1)?.model_revision).toBe(next.model_revision);
    expect(await session(second)).toEqual(unaffected);
    expect((await control()).observer.epoch).toBe(observer.epoch);
    await expect.poll(async () => (await control()).observer.subscribers).toBe(2);
    await page.unroute('**/models/events?*');

    // Hold the real POST response after backend pinning. The browser must clean
    // up its actual returned session when a newer explicit model choice wins.
    let candidate: Session | undefined, release!: () => void;
    const barrier = new Promise<void>(resolve => { release = resolve; });
    await page.route(`${backend}/sessions`, async route => {
      const response = await route.fetch(); candidate = await response.json();
      await barrier; await route.fulfill({ response });
    }, { times: 1 });
    editDefinition(d => { d.name = 'Delayed replacement'; });
    await expect.poll(() => candidate?.id).toBeTruthy();
    await expect(page.getByRole('contentinfo')).toContainText('Updating model');
    expect(pins).toHaveLength(2); // Reconnect made exactly one replacement.
    await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
    await modelPicker(page).selectOption('reload/b');
    await expect.poll(async () => {
      const id = await page.evaluate(url => sessionStorage.getItem(`llm-model-explorer:session:${url}`), backend);
      if (!id) return null;
      return (await (await fetch(`${backend}/sessions/${id}`)).json()).model_id;
    }).toBe('reload/b');
    release();
    await expect.poll(async () => (await fetch(`${backend}/sessions/${candidate!.id}`)).status).toBe(404);
    const chosen = await session(page);
    await expect(page.getByRole('textbox', { name: 'Prompt', exact: true })).toBeVisible();
    expect(await session(second)).toEqual(unaffected);
    await expect.poll(async () => (await control()).sessions.sort()).toEqual([chosen.id, unaffected.id].sort());
    // Close must also supersede a pinned but not yet adopted refresh candidate.
    let abandoned: Session | undefined, finish!: () => void;
    const closeBarrier = new Promise<void>(resolve => { finish = resolve; });
    await page.route(`${backend}/sessions`, async route => {
      const response = await route.fetch(); abandoned = await response.json();
      await closeBarrier; await route.fulfill({ response });
    }, { times: 1 });
    editDefinition(d => { d.name = 'Close pending replacement'; }, 'b');
    await expect.poll(() => abandoned?.id).toBeTruthy();
    await close(page); finish();
    await expect.poll(async () => (await fetch(`${backend}/sessions/${abandoned!.id}`)).status).toBe(404);
    // The second context independently refreshes B, but its new session is not
    // owned/deleted by the closing first context.
    const secondNext = await replaced(second, unaffected);
    await expect.poll(async () => (await control()).sessions).toEqual([secondNext.id]);
    await expect.poll(async () => (await control()).observer.subscribers).toBe(1);
    await expect(modelPicker(page)).toHaveValue('');
    await info.attach('transport-and-intent', { contentType: 'application/json', body: JSON.stringify({ old, next, unaffected, chosen, secondNext, candidate, abandoned, observer: (await control()).observer }) });
    await close(second);
    await expect.poll(async () => { const state = await control(); return [state.sessions.length, state.observer.subscribers, state.observer.running]; }).toEqual([0, 0, false]);
  } finally { await secondContext.close(); }
});

test('real tensor and tokenizer refresh retain selection and live edits while replacing numerical content', async ({ page }, info) => {
  const dataRequests: string[] = [], tokens: { session: string; result: any }[] = [];
  page.on('request', request => { if (request.url().endsWith('/data')) dataRequests.push(request.url()); });
  page.on('response', async response => {
    if (response.url().endsWith('/tokenize') && response.ok()) tokens.push({ session: response.url().split('/sessions/')[1]!.split('/')[0]!, result: await response.json() });
  });
  await modelPicker(page).selectOption('reload/a');
  await page.evaluate(() => { (window as any).__acceptance.captureScalars = true; });
  await (await revealTensor(page.getByRole('button', { includeHidden: true, name: /encoder\.proj\.weight/ }))).click();
  await expect(page.locator('.matrix-scroll canvas').first()).toBeVisible();
  await expect.poll(() => page.evaluate(() => (window as any).__acceptance.scalarValues.slice(0, 12))).toEqual(Array(12).fill(1));
  const old = await session(page), oldData = dataRequests.at(-1)!;
  await page.evaluate(() => { (window as any).__acceptance.scalarValues.length = 0; });
  changeWeights(9);
  const next = await replaced(page, old);
  await expect.poll(() => dataRequests.some(url => url.includes(`/sessions/${next.id}/`))).toBe(true);
  expect(dataRequests.at(-1)!.split('/tensors/')[1]).toBe(oldData.split('/tensors/')[1]);
  await expect.poll(() => page.evaluate(() => (window as any).__acceptance.scalarValues.slice(0, 12))).toEqual(Array(12).fill(9));
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  const editor = page.getByRole('textbox', { name: 'Prompt', exact: true });
  await editor.fill('alpha');
  await expect.poll(() => tokens.at(-1)?.result.tokens.map((token: any) => token.id)).toEqual([1, 2]);
  await expect(page.locator('.input-embeddings [data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
  await editor.evaluate(node => { node.dataset.owner = 'original'; });
  await editor.press('End'); await editor.press('Shift+ArrowLeft');
  expect(await page.evaluate(() => getSelection()?.toString())).toBe('a');
  let pinned: Session | undefined, release!: () => void;
  const barrier = new Promise<void>(resolve => { release = resolve; });
  await page.route(`${backend}/sessions`, async route => {
    const response = await route.fetch(); pinned = await response.json();
    await barrier; await route.fulfill({ response });
  }, { times: 1 });
  // Real tokenizer vocabulary changes: the result cannot pass by reusing IDs.
  const tokenizerPath = join(root, 'models/a/tokenizer.json');
  const tokenizer = JSON.parse(readFileSync(tokenizerPath, 'utf8'));
  tokenizer.model.vocab.alpha = 3; tokenizer.model.vocab.beta = 2;
  atomic(tokenizerPath, JSON.stringify(tokenizer));
  changeWeights(17);
  await expect.poll(() => pinned?.id).toBeTruthy();
  await expect(page.getByRole('contentinfo')).toContainText('Updating model');
  await expect(editor).toHaveAttribute('data-owner', 'original');
  expect(await page.evaluate(() => getSelection()?.toString())).toBe('a');
  await editor.press('End'); await editor.press(' '); await editor.pressSequentially('beta');
  await editor.press('Shift+ArrowLeft');
  const selection = await page.evaluate(() => getSelection()?.toString());
  await page.evaluate(() => { (window as any).__acceptance.scalarValues.length = 0; });
  release();
  const refreshed = await replaced(page, next);
  await expect.poll(() => tokens.at(-1)?.session).toBe(refreshed.id);
  expect(tokens.at(-1)!.result).toMatchObject({ text: 'alpha beta', add_special_tokens: true });
  expect(tokens.at(-1)!.result.tokens.map((token: any) => token.id)).toEqual([1, 3, 2]);
  await expect(editor).toHaveAttribute('data-owner', 'original');
  expect(await page.evaluate(() => getSelection()?.toString())).toBe(selection);
  await expect(page.locator('.input-embeddings [data-embeddings]')).toHaveAttribute('data-embeddings', 'current');
  const expected = [21, 22, 23, 24, 29, 30, 31, 32, 25, 26, 27, 28];
  await expect.poll(() => page.evaluate(() => (window as any).__acceptance.scalarValues.slice(0, 12))).toEqual(expected);
  await info.attach('numeric-refresh', { contentType: 'application/json', body: JSON.stringify({ old, next, refreshed, tensor: oldData.split('/tensors/')[1], tokenIds: [1, 3, 2], embeddingValues: expected }) });
  await close(page);
  await expect.poll(async () => { const state = await control(); return [state.sessions.length, state.observer.subscribers, state.operations, state.consumers, state.readers, state.tasks]; }).toEqual([0, 0, 0, 0, 0, 0]);
  await expect.poll(() => page.evaluate(() => { const m = (window as any).__acceptance.metrics(); return [m.textures, m.readers]; })).toEqual([0, 0]);
});
