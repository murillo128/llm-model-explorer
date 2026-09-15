#!/usr/bin/env node
/** Reproduce automatic layout from an explicitly supplied local graph.
 * The report contains bounded coordinate samples and geometry checks; it never
 * copies the supplied graph, tensor values, attributes, or parameter records.
 * node ui/scripts/architecture-layout-evidence.mjs --graph /local/graph.json --out /local/report.json
 */
import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const flags = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  const flag = process.argv[i], value = process.argv[i + 1];
  if (!['--graph', '--out'].includes(flag) || !value) throw new Error('Usage: node ui/scripts/architecture-layout-evidence.mjs --graph /local/graph.json [--out /local/report.json]');
  flags.set(flag, value);
}
if (!flags.has('--graph')) throw new Error('An explicit --graph local JSON file is required. No reference data is bundled.');
const bytes = await readFile(resolve(flags.get('--graph')));
const graph = JSON.parse(bytes.toString('utf8'));
const uiRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root: uiRoot, logLevel: 'silent', server: { middlewareMode: true } });
try {
  const { layoutGraph, groupHeaderHeight, layerGap } = await server.ssrLoadModule('/src/architecture-explorer/auto-layout.ts');
  const { projectGraph, endpointKey } = await server.ssrLoadModule('/src/architecture-explorer/projection.ts');
  const { validateSchema } = await server.ssrLoadModule('/src/api/validation.ts');
  validateSchema('ArchitectureGraph', graph);
  const records = new Map(graph.nodes.map((node) => [node.id, node]));
  const roots = graph.nodes.filter((node) => node.kind === 'group' && !node.parent_id).map((node) => node.id);
  const repetition = graph.repetitions[0];
  const instances = repetition?.instances ?? [];
  const first = instances[0], other = instances.find((instance) => instance.variant !== first?.variant);
  const scenes = [{ name: 'overview', options: { expanded: roots } }];
  function expandedThrough(id) {
    const expanded = new Set(roots);
    let node = records.get(id);
    while (node) { if (node.kind === 'group') expanded.add(node.id); node = records.get(node.parent_id); }
    return [...expanded];
  }
  function belongs(id, scope) {
    let node = records.get(id);
    while (node) { if (node.id === scope) return true; node = records.get(node.parent_id); }
    return false;
  }
  if (instances.length) {
    scenes.push({ name: 'window', options: { expanded: roots, repetitions: { [repetition.id]: { start: 0, count: Math.min(4, instances.length) } } } });
    for (const [name, instance] of [['first', first], ['different-variant', other], ['last', instances.at(-1)]]) {
      if (!instance) continue;
      const options = { expanded: expandedThrough(instance.node_id) };
      scenes.push({ name: `layer-${name}`, instance: { index: instance.index, variant: instance.variant }, options });
      if (name === 'last') continue;
      const expanded = new Set(options.expanded);
      for (const node of graph.nodes) if (node.kind === 'group' && belongs(node.id, instance.node_id)) expanded.add(node.id);
      const derived = projectGraph(graph, { expanded: [...expanded] }).nodes.filter((node) => node.presentation === 'mlp');
      for (const node of derived) if (node.sourceIds.some((id) => belongs(id, instance.node_id))) expanded.add(node.id);
      scenes.push({ name: `nested-${name}`, instance: { index: instance.index, variant: instance.variant }, options: { expanded: [...expanded] } });
    }
  }
  scenes.push({ name: 'overview-dimensions', options: { expanded: roots, dimensions: true } });
  scenes.push({ name: 'exhaustive', options: { expanded: [], exhaustive: true } });

  const round = (value) => Math.round(value * 100) / 100;
  const overlap = (a, b) => Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x) > 0.001 &&
    Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y) > 0.001;
  function inspect(layout) {
    const nodes = new Map(layout.projection.nodes.map((node) => [node.id, node]));
    const boxes = new Map(layout.boxes.map((box) => [box.id, box]));
    const ports = new Map(layout.ports.map((port) => [endpointKey({ node_id: port.nodeId, port_id: port.portId }), port]));
    const edges = new Map(layout.projection.edges.map((edge) => [edge.id, edge]));
    const violations = [], lineBuckets = new Map(), dependencies = new Map();
    const counters = { checkedRoutes: layout.routes.length, checkedSerialSiblingDependencies: 0, cyclicSiblingDependencies: 0,
      endpointMismatches: 0, unrelatedNodeOrHeaderIntersections: 0, distinctSignalsSharingSegments: 0, labelNodeIntersections: 0, labelOverlaps: 0 };
    const issue = (kind, details) => { counters[kind]++; if (violations.length < 8) violations.push({ kind, ...details }); };
    function chain(id) {
      const result = [id];
      while (nodes.get(id)?.parentId) { id = nodes.get(id).parentId; result.push(id); }
      return [...result, null];
    }
    for (const route of layout.routes) {
      const edge = edges.get(route.id), source = ports.get(endpointKey(edge.source)), target = ports.get(endpointKey(edge.target));
      const ends = route.sections.flatMap((section) => [section[0], section.at(-1)]);
      for (const [role, port] of [['source', source], ['target', target]]) {
        if (!port || !ends.some((point) => Math.abs(point.x - port.absoluteX) < 0.001 && Math.abs(point.y - port.absoluteY) < 0.001)) issue('endpointMismatches', { route: route.id, role });
      }
      const sourceChain = chain(edge.source.node_id), targetChain = chain(edge.target.node_id);
      const common = sourceChain.find((id) => targetChain.includes(id));
      const a = sourceChain[sourceChain.indexOf(common) - 1], b = targetChain[targetChain.indexOf(common) - 1];
      if (a && b && a !== b) {
        const scope = dependencies.get(common) ?? new Map(); scope.set(JSON.stringify([a, b]), [a, b]); dependencies.set(common, scope);
      }
      for (const section of route.sections) for (let i = 1; i < section.length; i++) {
        const a = section[i - 1], b = section[i];
        if (a.x === b.x && a.y === b.y) continue;
        for (const box of layout.boxes) {
          if (box.id === edge.source.node_id || box.id === edge.target.node_id) continue;
          const bounds = { x: box.absoluteX, y: box.absoluteY, width: box.width, height: nodes.get(box.id).expanded ? groupHeaderHeight : box.height };
          // Strict interiors keep legitimate border contact and boundary ports.
          if (Math.min(a.x, b.x) < bounds.x + bounds.width - 0.001 && Math.max(a.x, b.x) > bounds.x + 0.001 &&
              Math.min(a.y, b.y) < bounds.y + bounds.height - 0.001 && Math.max(a.y, b.y) > bounds.y + 0.001) issue('unrelatedNodeOrHeaderIntersections', { route: route.id, node: box.id });
        }
        const axis = a.x === b.x ? 'y' : 'x', fixed = axis === 'x' ? a.y : a.x;
        const key = `${axis}:${fixed}`, intervals = lineBuckets.get(key) ?? [];
        intervals.push({ from: Math.min(a[axis], b[axis]), to: Math.max(a[axis], b[axis]), signal: endpointKey(edge.source) });
        lineBuckets.set(key, intervals);
      }
    }
    for (const intervals of lineBuckets.values()) for (let i = 0; i < intervals.length; i++) for (let j = i + 1; j < intervals.length; j++) {
      const a = intervals[i], b = intervals[j];
      if (a.signal !== b.signal && Math.min(a.to, b.to) - Math.max(a.from, b.from) > 0.001) issue('distinctSignalsSharingSegments', { signals: [a.signal, b.signal] });
    }
    const labels = layout.routes.flatMap((route) => route.labels ?? []);
    for (const label of labels) for (const box of layout.boxes) if (overlap(label,
      { x: box.absoluteX, y: box.absoluteY, width: box.width, height: nodes.get(box.id).expanded ? groupHeaderHeight : box.height })) issue('labelNodeIntersections', { node: box.id });
    for (let i = 0; i < labels.length; i++) for (let j = i + 1; j < labels.length; j++) if (overlap(labels[i], labels[j])) issue('labelOverlaps', {});
    const scopeSamples = [];
    for (const [scopeId, pairs] of dependencies) {
      const adjacency = new Map();
      for (const [a, b] of pairs.values()) adjacency.set(a, [...(adjacency.get(a) ?? []), b]);
      function reaches(start, target, seen = new Set()) {
        if (start === target) return true;
        if (seen.has(start)) return false;
        seen.add(start); return (adjacency.get(start) ?? []).some((id) => reaches(id, target, seen));
      }
      const samples = [];
      for (const [a, b] of pairs.values()) {
        if (reaches(b, a)) { counters.cyclicSiblingDependencies++; continue; }
        counters.checkedSerialSiblingDependencies++;
        const from = boxes.get(a), to = boxes.get(b), gap = to.absoluteX - from.absoluteX - from.width;
        if (gap < layerGap - 0.001) violations.push({ kind: 'serialSiblingProgression', scope: scopeId, from: a, to: b, gap });
        if (samples.length < 4) samples.push({ from: a, to: b, gap: round(gap) });
      }
      if (scopeSamples.length < 6) {
        const ids = new Set(samples.flatMap((pair) => [pair.from, pair.to]));
        scopeSamples.push({ scope: scopeId, label: scopeId ? nodes.get(scopeId).label : 'Global graph', dependencies: samples,
          boxes: [...ids].map((id) => {
            const box = boxes.get(id); return { id, label: nodes.get(id).label, x: round(box.absoluteX), y: round(box.absoluteY), width: round(box.width), height: round(box.height) };
          }) });
      }
    }
    return { pass: !violations.length, ...counters, violations: violations.slice(0, 8), coordinateSamples: scopeSamples };
  }
  const report = { input: { file: basename(flags.get('--graph')), sha256: createHash('sha256').update(bytes).digest('hex'), graphId: graph.graph_id,
    sourceNodes: graph.nodes.length, sourceEdges: graph.edges.length, instances: graph.repetitions.reduce((n, repetition) => n + repetition.instances.length, 0) },
  environment: { node: process.version, platform: process.platform, architecture: process.arch },
  method: 'Source-preserving projection and ELK automatic horizontal layout. No supplied coordinates, node dragging, or checkpoint-specific placement.',
  viewport: 'Geometry is independent of viewport width. Browser evidence separately exercises resize, camera retention, and context-selected repetition windows at 1178 and 1440 CSS pixels.',
  scenarios: [] };
  for (const scene of scenes) {
    const layout = await layoutGraph(graph, scene.options);
    report.scenarios.push({ name: scene.name, ...(scene.instance ? { instance: scene.instance } : {}),
      visibleNodes: layout.boxes.length, visibleConnections: layout.routes.length, representedSourceEdges: layout.edgeIds.length,
      hiddenSourceEdges: layout.projection.hiddenEdgeIds.length, filteredSourceEdges: layout.projection.filteredEdgeIds.length,
      bounds: { width: layout.width, height: layout.height }, milliseconds: round(layout.milliseconds), ...inspect(layout) });
  }
  const output = JSON.stringify(report, null, 2) + '\n';
  if (flags.has('--out')) { const path = resolve(flags.get('--out')); await mkdir(dirname(path), { recursive: true }); await writeFile(path, output); }
  else process.stdout.write(output);
  if (report.scenarios.some((scene) => !scene.pass)) process.exitCode = 1;
} finally { await server.close(); }
