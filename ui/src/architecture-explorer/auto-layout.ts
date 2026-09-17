import ELK from 'elkjs/lib/elk-api.js';
import elkWorkerUrl from 'elkjs/lib/elk-worker.min.js?url';
import type { ELK as ElkEngine, ElkNode, ElkExtendedEdge, LayoutOptions } from 'elkjs/lib/elk-api';
import type { Box, Graph, Layout, Point, PortPosition, Route } from './graph';
import { formatShape } from './graph';
import { cardMetrics, cardSummary } from './card-summary';
import { endpointKey, projectGraph } from './projection';
import type { ProjectionOptions } from './projection';

export const groupHeaderHeight = 64;
export const layerGap = 40;
export const nodeGap = 28;
const rootId = 'layout-root';

const scopeOptions: LayoutOptions = {
  'elk.algorithm': 'layered',
  'elk.direction': 'RIGHT',
  'elk.edgeRouting': 'ORTHOGONAL',
  'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
  'elk.layered.mergeEdges': 'false',
  'elk.layered.mergeHierarchyEdges': 'false',
  'elk.layered.spacing.nodeNodeBetweenLayers': String(layerGap),
  'elk.layered.spacing.edgeNodeBetweenLayers': '24',
  'elk.spacing.nodeNode': String(nodeGap),
  'elk.spacing.edgeNode': '20',
  'elk.spacing.edgeEdge': '16',
  'elk.layered.spacing.edgeEdgeBetweenLayers': '16',
  'elk.randomSeed': '1',
};

/** Layout has no knowledge of hidden source descendants. Source endpoints and
 * composed paths stay in the projection; ELK owns presentation geometry only.
 * A caller runs this in a terminable worker to bound otherwise synchronous
 * native-transpiled layout work. */
export async function layoutGraph(graph: Graph, options: ProjectionOptions, signal?: AbortSignal): Promise<Layout> {
  const started = performance.now();
  if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
  const projection = projectGraph(graph, options);
  const elkNodes = new Map<string, ElkNode>();
  const sourceIds = new Map<string, string>();
  const portIds = new Map<string, { nodeId: string; portId: string; side: 'left' | 'right' }>();
  const endpoints = new Map<string, string>();
  const parameters = new Map(graph.parameters.map((p) => [p.id, p]));
  const annotated = new Set([...graph.repetitions.flatMap((r) => r.instances.map((i) => i.node_id)),
    ...graph.diagnostics.flatMap((d) => d.node_id ? [d.node_id] : [])]);
  for (const [index, node] of projection.nodes.entries()) {
    const metrics = cardMetrics(node, cardSummary(node.record, parameters), Boolean(options.dimensions), annotated.has(node.id));
    const id = `node-${index}`;
    sourceIds.set(id, node.id);
    const rows = { input: 0, output: 0 };
    const ports = node.ports.map((port, p) => {
      const portId = `port-${index}-${p}`;
      const side = port.direction === 'input' ? 'left' : 'right';
      const row = rows[port.direction]++;
      portIds.set(portId, { nodeId: node.id, portId: port.id, side });
      endpoints.set(endpointKey({ node_id: node.id, port_id: port.id }), portId);
      return { id: portId, x: side === 'left' ? 0 : metrics.width,
        y: metrics.portStart + row * metrics.portGap, width: 0, height: 0,
        layoutOptions: { 'elk.port.side': side === 'left' ? 'WEST' : 'EAST', 'elk.port.index': String(p) } };
    });
    const height = metrics.height;
    const gutter = (direction: 'input' | 'output') => Math.max(24, 20 + metrics.portLabelWidth[direction]);
    elkNodes.set(node.id, {
      id, width: metrics.width, height, ports,
      ...(node.expanded ? { children: [] } : {}),
      layoutOptions: { ...scopeOptions,
        'elk.portConstraints': node.expanded ? 'FIXED_SIDE' : 'FIXED_POS',
        'elk.spacing.portPort': String(metrics.portGap),
        'elk.padding': `[top=${height + 16},left=${gutter('input')},bottom=24,right=${gutter('output')}]`,
        'elk.spacing.portsSurrounding': `[top=${height},left=0,bottom=16,right=0]`,
      },
    });
  }
  const root: ElkNode = { id: rootId, children: [], edges: [],
    layoutOptions: { ...scopeOptions, 'elk.padding': '[top=24,left=24,bottom=24,right=24]' } };
  // Portless context is an independent explanatory component, never a dataflow
  // dependency. Pack these components in a separate row before the computation;
  // tokenizer references come first, preserving their source-grounded role.
  const contexts = projection.nodes.filter((node) => !node.parentId && node.kind === 'context' && !node.ports.length)
    .sort((a, b) => Number(b.record?.references.some((r) => r.kind === 'tokenizer')) - Number(a.record?.references.some((r) => r.kind === 'tokenizer')));
  const contextIds = new Set(contexts.map((node) => node.id));
  for (const node of projection.nodes) {
    if (contextIds.has(node.id)) continue;
    const parent = node.parentId ? elkNodes.get(node.parentId) : root;
    if (!parent) throw new Error('Visible containment could not be laid out. Collapse groups and retry.');
    (parent.children ??= []).push(elkNodes.get(node.id)!);
  }
  const edgeIds = new Map<string, string>();
  const labelLines = new Map<string, string[]>();
  const portShapes = new Map(projection.nodes.flatMap((node) => node.ports.map((port) =>
    [endpointKey({ node_id: node.id, port_id: port.id }), port.shape] as const)));
  for (const [index, edge] of projection.edges.entries()) {
    const source = endpoints.get(endpointKey(edge.source)), target = endpoints.get(endpointKey(edge.target));
    if (!source || !target) throw new Error('A visible connection has no exact port. Collapse groups and retry.');
    const id = `edge-${index}`;
    edgeIds.set(id, edge.id);
    const layoutEdge: ElkExtendedEdge = { id, sources: [source], targets: [target] };
    if (options.dimensions) {
      const sourceShape = formatShape(portShapes.get(endpointKey(edge.source))!);
      const targetShape = formatShape(portShapes.get(endpointKey(edge.target))!);
      const bound = (line: string) => line.length > 80 ? `${line.slice(0, 79)}…` : line;
      const lines = sourceShape === targetShape ? [bound(sourceShape)] : [bound(`source ${sourceShape}`), bound(`target ${targetShape}`)];
      labelLines.set(id, lines);
      layoutEdge.labels = [{ id: `${id}-dimensions`, text: lines.join('\n'),
        width: Math.max(...lines.map((line) => line.length)) * 7 + 12, height: lines.length * 16 + 8,
        layoutOptions: { 'elk.edgeLabels.placement': 'CENTER' } }];
    }
    root.edges!.push(layoutEdge);
  }
  // ELK's bundled entry detects worker globals and installs its own handler,
  // making it unsuitable inside our projection worker. Use the official API
  // and worker asset there; Node-only geometry tests use its bundled adapter.
  let engine: ElkEngine;
  const nodeAdapter = import.meta.env.SSR || import.meta.env.MODE === 'test';
  if (nodeAdapter) engine = new (await import('elkjs/lib/elk.bundled.js')).default({ algorithms: ['layered'] });
  else engine = new ELK({ algorithms: ['layered'], workerFactory: () => new Worker(elkWorkerUrl) });
  let laidOut: ElkNode;
  const abort = () => { if (!nodeAdapter) engine.terminateWorker(); };
  signal?.addEventListener('abort', abort, { once: true });
  try { laidOut = await engine.layout(root); }
  finally { signal?.removeEventListener('abort', abort); if (!nodeAdapter) engine.terminateWorker(); }
  if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
  if (contexts.length) {
    const gap = nodeGap, contextWidth = Math.max(...contexts.map((n) => elkNodes.get(n.id)!.width!)),
      contextHeight = Math.max(...contexts.map((n) => elkNodes.get(n.id)!.height!));
    const columns = Math.max(1, Math.floor(((laidOut.width ?? 0) - 48 + gap) / (contextWidth + gap)));
    const rows = Math.ceil(contexts.length / columns), offset = rows * (contextHeight + gap);
    for (const child of laidOut.children ?? []) child.y = (child.y ?? 0) + offset;
    // Root-owned edge coordinates share the same translation; descendant-owned
    // edges move with their containers when converted to absolute coordinates.
    for (const edge of laidOut.edges ?? []) if (!edge.container || edge.container === rootId) {
      for (const section of edge.sections ?? []) for (const point of [section.startPoint, ...(section.bendPoints ?? []), section.endPoint]) point.y += offset;
      for (const point of edge.junctionPoints ?? []) point.y += offset;
      for (const label of edge.labels ?? []) label.y = (label.y ?? 0) + offset;
    }
    contexts.forEach((node, index) => {
      const elk = elkNodes.get(node.id)!;
      elk.x = 24 + index % columns * (contextWidth + gap);
      elk.y = 24 + Math.floor(index / columns) * (contextHeight + gap);
      laidOut.children!.push(elk);
    });
    laidOut.width = Math.max(laidOut.width ?? 0, 48 + Math.min(columns, contexts.length) * (contextWidth + gap) - gap);
    laidOut.height = (laidOut.height ?? 0) + offset;
  }
  const boxes: Box[] = [], ports: PortPosition[] = [], routes: Route[] = [];
  const origins = new Map<string, Point>([[rootId, { x: 0, y: 0 }]]);
  const routed: { edge: ElkExtendedEdge; parent: string }[] = [];
  function collect(node: ElkNode, parent: ElkNode | undefined) {
    const parentOrigin = parent ? origins.get(parent.id)! : { x: 0, y: 0 };
    const absoluteX = parentOrigin.x + (node.x ?? 0), absoluteY = parentOrigin.y + (node.y ?? 0);
    origins.set(node.id, { x: absoluteX, y: absoluteY });
    const id = sourceIds.get(node.id);
    if (id) {
      const parentId = parent && sourceIds.get(parent.id);
      boxes.push({ id, ...(parentId ? { parentId } : {}), x: node.x ?? 0, y: node.y ?? 0,
        absoluteX, absoluteY, width: node.width!, height: node.height! });
      for (const port of node.ports ?? []) {
        const identity = portIds.get(port.id)!;
        const x = (port.x ?? 0) + (port.width ?? 0) / 2, y = (port.y ?? 0) + (port.height ?? 0) / 2;
        ports.push({ ...identity, x, y, absoluteX: absoluteX + x, absoluteY: absoluteY + y });
      }
    }
    for (const edge of node.edges ?? []) routed.push({ edge, parent: node.id });
    for (const child of node.children ?? []) collect(child, node);
  }
  collect(laidOut, undefined);
  for (const { edge, parent } of routed) {
    const id = edgeIds.get(edge.id);
    if (!id) continue;
    const origin = origins.get(edge.container ?? parent);
    if (!origin || !edge.sections?.length) throw new Error('A connection could not be routed. Collapse groups and retry.');
    const absolute = (point: Point) => ({ x: origin.x + point.x, y: origin.y + point.y });
    routes.push({ id, sections: edge.sections.map((section) =>
      [section.startPoint, ...(section.bendPoints ?? []), section.endPoint].map(absolute)),
    junctions: (edge.junctionPoints ?? []).map(absolute),
    labels: (edge.labels ?? []).map((label) => ({ x: origin.x + (label.x ?? 0), y: origin.y + (label.y ?? 0),
      width: label.width ?? 0, height: label.height ?? 0, lines: labelLines.get(edge.id) ?? [] })) });
  }
  if (routes.length !== projection.edges.length) throw new Error('Layout did not route every connection. Collapse groups and retry.');
  // Hierarchical ELK layouts may reorder compound ports even under FIXED_ORDER.
  // Keep its spaced slots and child geometry, assign those slots in source order,
  // and reconnect only incident route ends through the existing boundary gutter.
  const moved = new Map<string, { from: Point; to: Point }>();
  const byEndpoint = new Map(ports.map((p) => [endpointKey({ node_id: p.nodeId, port_id: p.portId }), p]));
  for (const node of projection.nodes) {
    if (!node.expanded || !node.ports.some((p) => p.interfaces?.length)) continue;
    for (const side of ['left', 'right'] as const) {
      const positions = node.ports.map((p) => byEndpoint.get(endpointKey({ node_id: node.id, port_id: p.id }))).filter((p): p is PortPosition => p?.side === side);
      const slots = positions.map((p) => p.y).sort((a, b) => a - b);
      positions.forEach((port, i) => {
        const delta = slots[i]! - port.y;
        if (!delta) return;
        const from = { x: port.absoluteX, y: port.absoluteY };
        port.y += delta; port.absoluteY += delta;
        moved.set(endpointKey({ node_id: node.id, port_id: port.portId }), { from, to: { x: port.absoluteX, y: port.absoluteY } });
      });
    }
  }
  const near = (a: Point, b: Point) => Math.abs(a.x - b.x) < 0.001 && Math.abs(a.y - b.y) < 0.001;
  const byRoute = new Map(routes.map((r) => [r.id, r]));
  for (const edge of projection.edges) {
    const route = byRoute.get(edge.id)!;
    for (const endpoint of [edge.source, edge.target]) {
      const move = moved.get(endpointKey(endpoint));
      if (!move) continue;
      route.sections = route.sections.map((section) => {
        const reverse = near(section.at(-1)!, move.from);
        const points = reverse ? [...section].reverse() : [...section];
        if (!near(points[0]!, move.from)) return section;
        const next = points[1];
        points[0] = move.to;
        if (next && next.x !== move.to.x && next.y !== move.to.y) points.splice(1, 0, { x: next.x, y: move.to.y });
        return reverse ? points.reverse() : points;
      });
    }
  }
  const represented = new Set([...projection.edges.flatMap((edge) => edge.originalEdgeIds), ...(projection.boundaryPaths ?? []).flat().map((e) => e.id)]);
  return { boxes, ports, routes, projection, edgeIds: graph.edges.filter((edge) => represented.has(edge.id)).map((edge) => edge.id),
    width: laidOut.width ?? 0, height: laidOut.height ?? 0, milliseconds: performance.now() - started };
}
