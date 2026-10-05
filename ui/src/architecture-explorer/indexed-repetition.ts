import type { Graph, GraphNode } from './graph';
import type { Endpoint, ProjectedEdge, ProjectedNode, Projection, ProjectionOptions } from './projection';
import { endpointKey, projectGraph } from './projection';
import { commonNode, templateGraph } from './shared-structure';
import type { Template } from './shared-structure';

type Edge = Graph['edges'][number];
export interface IndexedStack {
  template: Template;
  instances: Template['instances'];
  input: string; output: string;
  invariantPorts: Set<string>; invariantParameters: Set<string>;
  entry: Edge[]; transitions: Edge[][]; exits: Edge[][];
  invariants: { port: string; paths: Edge[][] }[];
}

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
  for (const repetition of graph.repetitions) {
    const roots = repetition.instances.map((i) => i.node_id), rootSet = new Set(roots);
    const template = families.get(roots[0]!);
    if (!template || !roots.length || roots.some((id) => families.get(id) !== template) ||
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
    if ([...members].some((id) => nodes.get(id)?.kind === 'state') || graph.edges.some((e) =>
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
        if (edge.kind === 'state') return undefined;
        current = edge.source;
        if (rootSet.has(current.node_id) || nodes.get(current.node_id)?.kind !== 'group' || !incoming.has(endpointKey(current))) return path;
        if (path.length > graph.edges.length) return undefined;
      }
    }
    const first = nodes.get(roots[0]!)!;
    const outputs = first.ports.filter((p) => p.direction === 'output');
    if (outputs.length !== 1) continue;
    const output = outputs[0]!.id;
    const inputs = first.ports.filter((p) => p.direction === 'input');
    const paths = new Map(inputs.map((p) => [p.id, roots.map((id) => pathTo({ node_id: id, port_id: p.id }))]));
    if ([...paths.values()].some((ps) => ps.some((p) => !p))) continue;
    const serial = inputs.filter((p) => paths.get(p.id)!.every((path, at) => path!.every((e) => e.kind === 'data') && (at === 0
      ? !members.has(path![0]!.source.node_id)
      : path![0]!.source.node_id === roots[at - 1] && path![0]!.source.port_id === output)));
    if (serial.length !== 1) continue;
    const input = serial[0]!.id;
    const invariants = inputs.filter((p) => p.id !== input).map((p) => ({ port: p.id, paths: paths.get(p.id)! as Edge[][] }));
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
    const invariantPorts = new Set<string>();
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
    result.set(repetition.id, { template, instances, input, output, invariantPorts, invariantParameters, entry: serialPaths[0]!, transitions: serialPaths.slice(1), exits, invariants });
  }
  return result;
}

export const indexedNodeId = (repetitionId: string, nodeId: string) => `indexed:${repetitionId}:${nodeId}`;
const ep = (node_id: string, port_id: string): Endpoint => ({ node_id, port_id });

/** Bounded token substitution for declared port/parameter symbols. Existing
 * mathematical subscripts and constants remain verbatim; source formulas do too. */
function indexedRecord(node: GraphNode, stack: IndexedStack, root: boolean): GraphNode {
  const invariant = new Set(node.ports.filter((p) => stack.invariantPorts.has(endpointKey(ep(node.id, p.id)))).map((p) => p.id));
  const symbols = new Set(node.ports.filter((p) => !invariant.has(p.id)).flatMap((p) => [p.id, p.label]));
  const own = new Set([...node.parameter_ids, ...node.references.flatMap((r) => r.kind === 'parameter' ? [r.parameter_id] : [])]);
  for (const p of stack.instances[0]!.parameters) if (own.has(p.parameter_id)) {
    if (!stack.invariantParameters.has(p.role)) symbols.add(p.role.split('.').at(-1)!);
  }
  const alias = (text: string) => text.replace(/\b[A-Za-z_]\w*\b(?!\[)/g, (word) => symbols.has(word) ? `${word}[i]` : word);
  const common = commonNode(node, node.label, root ? 'Layer[i]' : node.label);
  return { ...common, ports: node.ports.map((p) => ({ ...p, label: alias(p.label) })),
    ...(node.formula ? { formula: alias(node.formula) } : {}),
    ...(root ? { formula: `${node.ports.find((p) => p.id === stack.output)!.label}[i] = layer[i](${node.ports.filter((p) => p.direction === 'input').map((p) => invariant.has(p.id) ? p.label : `${p.label}[i]`).join(', ')})` } : {}) };
}

/** Actions and projection must agree on whether the current interface filter
 * leaves every required handle available. Otherwise expansion uses a window. */
export function indexedBoundaryPorts(outer: ProjectedNode, stack: IndexedStack | undefined) {
  if (!stack || outer.presentation !== 'repetition') return;
  const roots = new Set(stack.instances.map((i) => i.node_id));
  const portFor = (port: string, direction: 'input' | 'output') => outer.ports.find((p) => p.direction === direction && p.endpoints.some((e) =>
    roots.has(e.node_id) && e.port_id === port));
  const input = portFor(stack.input, 'input'), output = portFor(stack.output, 'output');
  if (!input || !output || stack.invariants.some((p) => !portFor(p.port, 'input'))) return;
  return { input, output, portFor };
}

/** Compose two existing projections; the source graph stays concrete and immutable. */
export function projectIndexedRepetitions(graph: Graph, base: Projection, options: ProjectionOptions): Projection {
  if (options.exhaustive || options.stateScope) return base;
  const eligible = indexedStacks(graph);
  const parameters = new Map(graph.parameters.map((p) => [p.id, p]));
  const sourceEdges = new Map(graph.edges.map((e) => [e.id, e]));
  for (const outer of [...base.nodes]) {
    if (outer.presentation !== 'repetition' || !options.expanded.includes(outer.id)) continue;
    const stack = eligible.get(outer.repetitionId!);
    const boundary = indexedBoundaryPorts(outer, stack);
    if (!stack || !boundary) continue;
    const { input, output, portFor } = boundary;
    const repetition = graph.repetitions.find((r) => r.id === outer.repetitionId)!;
    const anchor = stack.instances[0]!;
    const id = (source: string) => indexedNodeId(repetition.id, source);
    const anchorEdgeRoles = new Map(anchor.edges.map((e) => [e.edge_id, e.role]));
    const instanceEdges = stack.instances.map((instance) => new Map(instance.edges.map((e) => [e.role, sourceEdges.get(e.edge_id)!])));
    const local = projectGraph(templateGraph(graph, anchor), { expanded: anchor.nodes.filter((n) => options.expanded.includes(id(n.node_id))).map((n) => n.node_id),
      scope: anchor.node_id, showUnused: true, deriveMlp: false });
    outer.expanded = true;
    for (const n of local.nodes) {
      const source = n.record!;
      const role = anchor.nodes.find((m) => m.node_id === source.id)!.role;
      const record = indexedRecord(source, stack, source.id === anchor.node_id);
      const own = new Set([...source.parameter_ids, ...source.references.flatMap((r) => r.kind === 'parameter' ? [r.parameter_id] : [])]);
      const symbolicParameters = anchor.parameters.filter((p) => own.has(p.parameter_id)).map((p) => ({
        role: p.role, label: p.role.split('.').at(-1)! + (stack.invariantParameters.has(p.role) ? '' : '[i]'),
        shape: parameters.get(p.parameter_id)!.logical_shape,
      }));
      base.nodes.push({ ...n, id: id(n.id), parentId: n.parentId ? id(n.parentId) : outer.id, record,
        label: source.id === anchor.node_id ? `${repetition.label.replace(/\s*layers?\s*/gi, ' ').trim()} Layer[i]` : record.label,
        sourceIds: [], presentation: 'shared', symbolicParameters, shared: { templateId: stack.template.id, anchorId: anchor.node_id, nodeRole: role },
        ports: n.ports.map((p) => ({ ...p, label: record.ports.find((v) => v.id === p.id)?.label ?? p.label,
          endpoints: [], interfaces: [], templatePort: { kind: 'template-port', templateId: stack.template.id, nodeRole: role,
            portRole: anchor.ports.find((v) => v.node_id === source.id && v.port_id === p.id)!.role } })) });
    }
    for (const e of local.edges) {
      const roles = e.paths.map((path) => path.map((p) => anchorEdgeRoles.get(p.id)!));
      const paths = instanceEdges.flatMap((edges) => roles.map((path) => path.map((role) => edges.get(role)!)));
      base.edges.push({ ...e, id: id(e.id), source: ep(id(e.source.node_id), e.source.port_id), target: ep(id(e.target.node_id), e.target.port_id), paths,
        originalEdgeIds: [...new Set(paths.flat().map((e) => e.id))] });
    }
    function relationship(kind: NonNullable<ProjectedEdge['relationship']>['kind'], source: Endpoint, target: Endpoint, paths: Edge[][], port: string) {
      base.edges.push({ id: `${outer.id}:${kind}:${port}`, source, target, kind: paths[0]![0]!.kind, paths,
        originalEdgeIds: [...new Set(paths.flat().map((e) => e.id))], relationship: { kind, repetitionId: repetition.id, templateId: stack!.template.id,
          portRole: anchor.ports.find((p) => p.node_id === anchor.node_id && p.port_id === port)!.role,
          instances: repetition.instances.map((i) => ({ nodeId: i.node_id, index: i.index })) } });
    }
    input.label = `${input.label}[${repetition.instances[0]!.index}]`;
    output.label = `${output.label}[${repetition.instances.at(-1)!.index}]`;
    relationship('entry', ep(outer.id, input.id), ep(id(anchor.node_id), stack.input), [stack.entry], stack.input);
    if (stack.transitions.length) relationship('return', ep(id(anchor.node_id), stack.output), ep(id(anchor.node_id), stack.input), stack.transitions, stack.input);
    relationship('exit', ep(id(anchor.node_id), stack.output), ep(outer.id, output.id), stack.exits, stack.output);
    for (const invariant of stack.invariants) {
      const port = portFor(invariant.port, 'input');
      if (port) relationship('invariant', ep(outer.id, port.id), ep(id(anchor.node_id), invariant.port), invariant.paths, invariant.port);
    }
  }
  const represented = new Set(base.edges.flatMap((e) => e.originalEdgeIds));
  return { ...base, hiddenEdgeIds: base.hiddenEdgeIds.filter((id) => !represented.has(id)), filteredEdgeIds: base.filteredEdgeIds.filter((id) => !represented.has(id)) };
}
