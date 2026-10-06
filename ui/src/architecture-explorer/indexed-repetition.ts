import type { Graph, GraphNode } from './graph';
import type { Endpoint, ProjectedEdge, ProjectedNode, Projection, ProjectionOptions } from './projection';
import { endpointKey, projectGraph } from './projection';
import { commonNode, templateGraph } from './shared-structure';
import type { Template } from './shared-structure';

type Edge = Graph['edges'][number];
export interface IndexedStack {
  repetition: Graph['repetitions'][number];
  template: Template;
  instances: Template['instances'];
  input: string; output: string;
  invariantPorts: Set<string>; invariantParameters: Set<string>;
  entry: Edge[]; transitions: Edge[][]; exits: Edge[][];
  invariants: { port: string; paths: Edge[][] }[];
  sides: { port: string; direction: 'input' | 'output'; paths: Edge[][] }[];
}

/** Maximal consecutive runs, preserving every real source index and variant. */
export function depthRanges(graph: Graph, repetition: Graph['repetitions'][number]) {
  const families = new Map(graph.templates?.filter((t) => t.component_role === 'layer')
    .flatMap((t) => t.instances.map((i) => [i.node_id, t.id] as const)));
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  if (!repetition.instances.some((i) => families.has(i.node_id)) &&
    !(new Set(repetition.instances.map((i) => i.variant)).size > 1 && repetition.instances.every((i) =>
      nodes.get(i.node_id)?.attributes.some((a) => a.name === 'semantic_role' && a.value === 'layer')))) return [repetition];
  if (repetition.bodies?.length) {
    const result: Graph['repetitions'] = [];
    let cursor = 0;
    const ordinary = (end: number) => {
      if (end > cursor) result.push(...depthRanges(graph, { ...repetition, bodies: [],
        id: `${repetition.id}:${cursor}:${end - 1}`, instances: repetition.instances.slice(cursor, end) }));
    };
    for (const body of repetition.bodies) {
      ordinary(body.start);
      const end = body.start + body.width * body.count;
      result.push({ ...repetition, id: body.start === 0 && end === repetition.instances.length ? repetition.id : `${repetition.id}:${body.start}:${end - 1}`,
        instances: repetition.instances.slice(body.start, end), bodies: [{ ...body, start: 0 }] });
      cursor = end;
    }
    ordinary(repetition.instances.length);
    return result;
  }
  const ranges: Graph['repetitions'] = [];
  for (const instance of repetition.instances) {
    const previous = ranges.at(-1)?.instances.at(-1);
    if (previous && instance.index === previous.index + 1 && instance.variant === previous.variant &&
      families.has(instance.node_id) && families.get(instance.node_id) === families.get(previous.node_id)) {
      ranges.at(-1)!.instances.push(instance);
    } else ranges.push({ ...repetition, instances: [instance] });
  }
  return ranges.length === 1 ? [repetition] : ranges.map((range) => ({ ...range,
    id: `${repetition.id}:${range.instances[0]!.index}:${range.instances.at(-1)!.index}` }));
}

export const stackKey = (node: ProjectedNode, graph: Graph) => {
  const repetition = graph.repetitions.find((r) => r.id === node.repetitionId);
  return repetition && depthRanges(graph, repetition).find((r) => r.instances.length === node.instances?.length &&
    r.instances[0]?.node_id === node.instances?.[0]?.node_id)?.id;
};

/** Correspondence proves interiors, never the wiring between them. Traverse only
 * external group forwarding; each transition remains its own ordered source path. */
export function indexedStacks(graph: Graph): Map<string, IndexedStack> {
  const result = new Map<string, IndexedStack>();
  if (!graph.templates?.some((t) => t.component_role === 'layer')) return result;
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  const parameters = new Map(graph.parameters.map((p) => [p.id, p]));
  const terminal = (id: string) => {
    const seen = new Set<string>();
    let parameter = parameters.get(id);
    while (parameter?.binding === 'alias' && !seen.has(parameter.id)) {
      seen.add(parameter.id); parameter = parameters.get(parameter.alias_of);
    }
    return parameter?.id;
  };
  const incoming = new Map<string, Edge[]>(), outgoing = new Map<string, Edge[]>();
  for (const e of graph.edges) {
    const s = endpointKey(e.source), t = endpointKey(e.target);
    outgoing.set(s, [...outgoing.get(s) ?? [], e]); incoming.set(t, [...incoming.get(t) ?? [], e]);
  }
  const families = new Map(graph.templates?.filter((t) => t.component_role === 'layer')
    .flatMap((t) => t.instances.map((i) => [i.node_id, t] as const)));
  for (const repetition of graph.repetitions.flatMap((r) => depthRanges(graph, r))) {
    const roots = repetition.instances.map((i) => i.node_id), rootSet = new Set(roots);
    const template = families.get(roots[0]!);
    if (!template || roots.length < 2 || roots.some((id) => families.get(id) !== template) ||
      new Set(repetition.instances.map((i) => i.variant)).size !== 1 ||
      repetition.instances.some((i, at) => i.index !== repetition.instances[0]!.index + at)) continue;
    const parent = nodes.get(repetition.parent_id);
    if (parent?.kind !== 'group') continue;
    const offset = parent.children.indexOf(roots[0]!);
    if (!roots.every((id, at) => parent.children[offset + at] === id)) continue;
    const byRoot = new Map(template.instances.map((i) => [i.node_id, i]));
    const instances = roots.map((id) => byRoot.get(id)!);
    const members = new Set(instances.flatMap((i) => i.nodes.map((n) => n.node_id)));
    // External dependencies must cross the declared layer interface. A per-layer
    // state interface cannot be represented as an invariant shared input.
    if (graph.edges.some((e) =>
      members.has(e.source.node_id) !== members.has(e.target.node_id) &&
      !(members.has(e.source.node_id) ? rootSet.has(e.source.node_id) : rootSet.has(e.target.node_id)))) continue;
    function pathTo(target: Endpoint): Edge[] | undefined {
      const path: Edge[] = [], seen = new Set<string>();
      let current = target;
      for (;;) {
        const previous = (incoming.get(endpointKey(current)) ?? []).filter((e) =>
          !(e.target.node_id === target.node_id && members.has(e.source.node_id) && !rootSet.has(e.source.node_id)));
        if (previous.length !== 1 || seen.has(previous[0]!.id)) return undefined;
        const edge = previous[0]!; path.unshift(edge); seen.add(edge.id);
        current = edge.source;
        if (rootSet.has(current.node_id) || nodes.get(current.node_id)?.kind !== 'group' || !incoming.has(endpointKey(current))) return path;
        if (path.length > graph.edges.length) return undefined;
      }
    }
    const first = nodes.get(roots[0]!)!;
    const outputs = first.ports.filter((p) => p.direction === 'output');
    const inputs = first.ports.filter((p) => p.direction === 'input');
    const paths = new Map(inputs.map((p) => [p.id, roots.map((id) => pathTo({ node_id: id, port_id: p.id }))]));
    const unused = new Set(inputs.filter((p) => roots.every((id) =>
      !incoming.has(endpointKey(ep(id, p.id))) && !outgoing.has(endpointKey(ep(id, p.id))))).map((p) => p.id));
    if ([...paths].some(([port, ps]) => !unused.has(port) && ps.some((p) => !p))) continue;
    // Exactly one data output/input pair carries the serial activation. State
    // outputs need explicit per-index correspondence, never a guessed invariant.
    const serial = outputs.flatMap((out) => inputs.filter((p) => !unused.has(p.id) && paths.get(p.id)!.every((path, at) =>
      path!.every((e) => e.kind === 'data') && (at === 0 ? !members.has(path![0]!.source.node_id)
        : path![0]!.source.node_id === roots[at - 1] && path![0]!.source.port_id === out.id)))
      .map((p) => ({ input: p.id, output: out.id })));
    if (serial.length !== 1) continue;
    const { input, output } = serial[0]!;
    const sides: IndexedStack['sides'] = [];
    let sideInvalid = false;
    for (const side of repetition.side_ports ?? []) {
      if (side.port_id === input || side.port_id === output) { sideInvalid = true; break; }
      const bound = repetition.instances.map((i) => side.bindings.find((b) => b.index === i.index));
      if (bound.some((b) => !b) || new Set(bound.map((b) => endpointKey(b!.endpoint))).size !== roots.length) { sideInvalid = true; break; }
      const sidePaths = roots.map((root) => (side.direction === 'input'
        ? incoming.get(endpointKey(ep(root, side.port_id))) : outgoing.get(endpointKey(ep(root, side.port_id))))?.filter((e) =>
        !members.has((side.direction === 'input' ? e.source : e.target).node_id)) ?? []);
      if (sidePaths.some((path, at) => path.length !== 1 || path[0]!.kind !== 'state' ||
        endpointKey(side.direction === 'input' ? path[0]!.source : path[0]!.target) !== endpointKey(bound[at]!.endpoint))) { sideInvalid = true; break; }
      sides.push({ port: side.port_id, direction: side.direction, paths: sidePaths.map((path) => [path[0]!]) });
    }
    if (sideInvalid || outputs.some((p) => p.id !== output && !sides.some((s) => s.port === p.id && s.direction === 'output') &&
      roots.some((root) => (outgoing.get(endpointKey(ep(root, p.id))) ?? []).some((e) => !members.has(e.target.node_id))))) continue;
    const invariants = inputs.filter((p) => p.id !== input && !unused.has(p.id) && !sides.some((s) => s.port === p.id))
      .map((p) => ({ port: p.id, paths: paths.get(p.id)! as Edge[][] }));
    if (invariants.some((p) => p.paths.some((path) => members.has(path[0]!.source.node_id) ||
      endpointKey(path[0]!.source) !== endpointKey(p.paths[0]![0]!.source) ||
      path.some((e) => e.kind !== p.paths[0]![0]!.kind)))) continue;
    const serialPaths = paths.get(input)! as Edge[][];
    // Follow all branches, including an extra consumer after an external
    // forwarding segment. Checking only the first outgoing edge misses bypasses.
    let invalid = false, visits = 0;
    const pathsFrom = (root: string): Edge[][] => {
      const found: Edge[][] = [];
      const queue = (outgoing.get(endpointKey(ep(root, output))) ?? []).map((e) => [e]);
      while (queue.length) {
        if (++visits > Math.max(100_000, graph.edges.length * 128)) { invalid = true; break; }
        const path = queue.pop()!, end = path.at(-1)!.target;
        if (path.some((e) => e.kind !== 'data') || members.has(end.node_id) && !rootSet.has(end.node_id)) { invalid = true; break; }
        const next = !rootSet.has(end.node_id) && nodes.get(end.node_id)?.kind === 'group' ? outgoing.get(endpointKey(end)) : undefined;
        if (!next?.length) { found.push(path); continue; }
        for (const e of next) {
          if (path.some((p) => p.id === e.id)) { invalid = true; break; }
          queue.push([...path, e]);
        }
        if (invalid) break;
      }
      return found;
    };
    let exits: Edge[][] = [];
    for (const [at, root] of roots.entries()) {
      const paths = pathsFrom(root);
      if (at === roots.length - 1) {
        exits = paths;
        invalid ||= !exits.length || exits.some((p) => members.has(p.at(-1)!.target.node_id));
      } else {
        const expected = serialPaths[at + 1]!;
        invalid ||= paths.length !== 1 || paths[0]!.length !== expected.length || paths[0]!.some((e, i) => e.id !== expected[i]!.id);
      }
      if (invalid) break;
    }
    if (invalid) continue;
    const invariantPorts = new Set<string>([...unused].map((port) => endpointKey(ep(roots[0]!, port))));
    const pending = invariants.map((p) => ep(roots[0]!, p.port));
    while (pending.length) {
      const endpoint = pending.pop()!, key = endpointKey(endpoint);
      if (invariantPorts.has(key)) continue;
      invariantPorts.add(key);
      if (nodes.get(endpoint.node_id)?.kind === 'group')
        for (const e of outgoing.get(key) ?? []) if (members.has(e.target.node_id)) pending.push(e.target);
    }
    const mappedParameters = instances.map((i) => new Map(i.parameters.map((p) => [p.role, terminal(p.parameter_id)])));
    const invariantParameters = new Set(instances[0]!.parameters.filter((p) =>
      mappedParameters.every((m) => m.get(p.role) === terminal(p.parameter_id))).map((p) => p.role));
    result.set(repetition.id, { repetition, template, instances, input, output, invariantPorts, invariantParameters, entry: serialPaths[0]!, transitions: serialPaths.slice(1), exits, invariants, sides });
  }
  return result;
}

export const depthIndex = (base: number, width: number, offset: number) => `${base ? `${base}+` : ''}${width}j${offset ? `+${offset}` : ''}`;

export const indexedNodeId = (repetitionId: string, nodeId: string) => `indexed:${repetitionId}:${nodeId}`;
const ep = (node_id: string, port_id: string): Endpoint => ({ node_id, port_id });

/** Bounded token substitution for declared port/parameter symbols. Existing
 * mathematical subscripts and constants remain verbatim; source formulas do too. */
function indexedRecord(node: GraphNode, stack: IndexedStack, root: boolean, expression = 'i'): GraphNode {
  const invariant = new Set(node.ports.filter((p) => stack.invariantPorts.has(endpointKey(ep(node.id, p.id)))).map((p) => p.id));
  const symbols = new Set(node.ports.filter((p) => !invariant.has(p.id)).flatMap((p) => [p.id, p.label]));
  const own = new Set([...node.parameter_ids, ...node.references.flatMap((r) => r.kind === 'parameter' ? [r.parameter_id] : [])]);
  for (const p of stack.instances[0]!.parameters) if (own.has(p.parameter_id)) {
    if (!stack.invariantParameters.has(p.role)) symbols.add(p.role.split('.').at(-1)!);
  }
  const alias = (text: string) => text.replace(/\b[A-Za-z_]\w*\b(?!\[)/g, (word) => symbols.has(word) ? `${word}[${expression}]` : word);
  const localSymbols = new Map((stack.instances[0]!.symbols ?? []).map((s) => [s.name, `${s.role.replace(/[^A-Za-z0-9_]/g, '_')}[${expression}]`]));
  const shape = (value: GraphNode['ports'][number]['shape']) => value?.map((d) => d.kind === 'symbol'
    ? { ...d, name: localSymbols.get(d.name) ?? d.name } : d.kind === 'expression'
      ? { ...d, text: d.text.replace(/\b[A-Za-z_]\w*\b/g, (name) => localSymbols.get(name) ?? name),
        symbols: d.symbols.map((name) => localSymbols.get(name) ?? name) } : d) ?? null;
  const common = commonNode(node, node.label, root ? `Layer[${expression}]` : node.label);
  return { ...common, attributes: common.attributes.map((a) => root && ['sequence_index', 'layer_index'].includes(a.name) &&
    a.value === stack.repetition.instances[0]!.index ? { ...a, value: expression } : a), ports: node.ports.map((p) => ({ ...p, label: alias(p.label), shape: shape(p.shape) })),
    ...(node.formula ? { formula: alias(node.formula) } : {}),
    ...(root ? { formula: `${node.ports.filter((p) => p.direction === 'output').map((p) => `${p.label}[${expression}]`).join(', ')} = layer[${expression}](${node.ports.filter((p) => p.direction === 'input').map((p) => invariant.has(p.id) ? p.label : `${p.label}[${expression}]`).join(', ')})` } : {}) };
}

/** Actions and projection must agree on whether the current interface filter
 * leaves every required handle available. Otherwise expansion uses a window. */
export function indexedBoundaryPorts(outer: ProjectedNode, stack: IndexedStack | undefined) {
  if (!stack || outer.presentation !== 'repetition') return;
  const roots = new Set(stack.instances.map((i) => i.node_id));
  const portFor = (port: string, direction: 'input' | 'output') => outer.ports.find((p) => p.direction === direction && p.endpoints.some((e) =>
    roots.has(e.node_id) && e.port_id === port));
  const input = portFor(stack.input, 'input'), output = portFor(stack.output, 'output');
  if (!input || !output) return;
  return { input, output, portFor };
}

export function appendIndexedLayer(graph: Graph, base: Projection, options: ProjectionOptions,
  stack: IndexedStack, ownerId: string, expression = 'i', rootLabel = `${stack.repetition.label.replace(/\s*layers?\s*/gi, ' ').trim()} Layer[i]`) {
  const anchor = stack.instances[0]!;
  const id = (source: string) => indexedNodeId(stack.repetition.id, source);
  const parameters = new Map(graph.parameters.map((p) => [p.id, p]));
  const sourceEdges = new Map(graph.edges.map((e) => [e.id, e]));
  const anchorEdgeRoles = new Map(anchor.edges.map((e) => [e.edge_id, e.role]));
  const instanceEdges = stack.instances.map((instance) => new Map(instance.edges.map((e) => [e.role, sourceEdges.get(e.edge_id)!])));
  const scoped = templateGraph(graph, anchor);
  const expanded = anchor.nodes.filter((n) => options.expanded.includes(id(n.node_id))).map((n) => n.node_id);
  // A nested expert/component range keeps its own disclosure inside this
  // neutral occurrence. Opening it reveals its bounded concrete role set.
  for (const repetition of scoped.repetitions) if (options.expanded.includes(id(`repeat:${repetition.id}:0:${repetition.instances.length - 1}`)))
    expanded.push(...repetition.instances.map((i) => i.node_id));
  const local = projectGraph(scoped, { expanded,
    scope: anchor.node_id, showUnused: options.showUnused === true, deriveMlp: false });
  for (const n of local.nodes) {
    const source = n.record;
    if (!source) {
      base.nodes.push({ ...n, id: id(n.id), parentId: n.parentId ? id(n.parentId) : ownerId,
        sourceIds: [], nestedRepetition: true,
        ports: n.ports.map((p) => ({ ...p, endpoints: [], interfaces: [] })) });
      continue;
    }
    const role = anchor.nodes.find((m) => m.node_id === source.id)!.role;
    const record = indexedRecord(source, stack, source.id === anchor.node_id, expression);
    const own = new Set([...source.parameter_ids, ...source.references.flatMap((r) => r.kind === 'parameter' ? [r.parameter_id] : [])]);
    const symbolicParameters = anchor.parameters.filter((p) => own.has(p.parameter_id)).map((p) => ({
      role: p.role, label: p.role.split('.').at(-1)! + (stack.invariantParameters.has(p.role) ? '' : `[${expression}]`),
      shape: parameters.get(p.parameter_id)!.logical_shape,
    }));
    base.nodes.push({ ...n, id: id(n.id), parentId: n.parentId ? id(n.parentId) : ownerId, record,
      label: source.id === anchor.node_id ? rootLabel : record.label,
      sourceIds: [], presentation: 'shared', symbolicParameters, shared: { templateId: stack.template.id, anchorId: anchor.node_id, nodeRole: role },
      ports: n.ports.filter((p) => options.showUnused || source.id !== anchor.node_id ||
        !base.unusedInputs.some((v) => v.node_id === anchor.node_id && v.port_id === p.id)).map((p) => ({ ...p, label: record.ports.find((v) => v.id === p.id)?.label ?? p.label,
        endpoints: [], interfaces: [], templatePort: { kind: 'template-port', templateId: stack.template.id, nodeRole: role,
          portRole: anchor.ports.find((v) => v.node_id === source.id && v.port_id === p.id)!.role } })) });
  }
  for (const e of local.edges) {
    const roles = e.paths.map((path) => path.map((p) => anchorEdgeRoles.get(p.id)!));
    const paths = instanceEdges.flatMap((edges) => roles.map((path) => path.map((role) => edges.get(role)!)));
    base.edges.push({ ...e, id: id(e.id), source: ep(id(e.source.node_id), e.source.port_id), target: ep(id(e.target.node_id), e.target.port_id), paths,
      originalEdgeIds: [...new Set(paths.flat().map((e) => e.id))] });
  }
  return id(anchor.node_id);
}

/** Compose two existing projections; the source graph stays concrete and immutable. */
export function projectIndexedRepetitions(graph: Graph, base: Projection, options: ProjectionOptions): Projection {
  if (options.exhaustive || options.stateScope) return base;
  const eligible = indexedStacks(graph);
  for (const outer of [...base.nodes]) {
    if (outer.presentation !== 'repetition' || !options.expanded.includes(outer.id)) continue;
    const stack = eligible.get(stackKey(outer, graph) ?? '');
    const boundary = indexedBoundaryPorts(outer, stack);
    if (!stack || !boundary) continue;
    const { input, output, portFor } = boundary;
    const repetition = stack.repetition;
    const anchor = stack.instances[0]!;
    const id = (source: string) => indexedNodeId(repetition.id, source);
    outer.expanded = true;
    appendIndexedLayer(graph, base, options, stack, outer.id);
    function relationship(kind: NonNullable<ProjectedEdge['relationship']>['kind'], source: Endpoint, target: Endpoint, paths: Edge[][], port: string) {
      base.edges.push({ id: `${outer.id}:${kind}:${port}`, source, target, kind: paths[0]![0]!.kind, paths,
        originalEdgeIds: [...new Set(paths.flat().map((e) => e.id))], relationship: { owner: 'repetition', kind, repetitionId: repetition.id, templateId: stack!.template.id,
          portRole: anchor.ports.find((p) => p.node_id === anchor.node_id && p.port_id === port)!.role,
          instances: repetition.instances.map((i) => ({ nodeId: i.node_id, index: i.index })) } });
    }
    input.label = `${input.label}[${repetition.instances[0]!.index}]`;
    output.label = `${output.label}[${repetition.instances.at(-1)!.index}]`;
    relationship('entry', ep(outer.id, input.id), ep(id(anchor.node_id), stack.input), [stack.entry], stack.input);
    if (stack.transitions.length) relationship('return', ep(id(anchor.node_id), stack.output), ep(id(anchor.node_id), stack.input), stack.transitions, stack.input);
    relationship('exit', ep(id(anchor.node_id), stack.output), ep(outer.id, output.id), stack.exits, stack.output);
    for (const side of stack.sides) {
      const port = portFor(side.port, side.direction);
      if (port) {
        port.label = `${port.label}[i]`;
        const endpoints = [ep(outer.id, port.id), ep(id(anchor.node_id), side.port)];
        relationship(side.direction === 'input' ? 'side-input' : 'side-output',
          endpoints[side.direction === 'input' ? 0 : 1]!, endpoints[side.direction === 'input' ? 1 : 0]!, side.paths, side.port);
      }
    }
    for (const invariant of stack.invariants) {
      const port = portFor(invariant.port, 'input');
      if (port) relationship('invariant', ep(outer.id, port.id), ep(id(anchor.node_id), invariant.port), invariant.paths, invariant.port);
    }
  }
  const represented = new Set(base.edges.flatMap((e) => e.originalEdgeIds));
  return { ...base, hiddenEdgeIds: base.hiddenEdgeIds.filter((id) => !represented.has(id)), filteredEdgeIds: base.filteredEdgeIds.filter((id) => !represented.has(id)) };
}
