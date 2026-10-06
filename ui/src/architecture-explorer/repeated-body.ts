import type { Graph } from './graph';
import type { Endpoint, ProjectedEdge, ProjectedNode, Projection, ProjectionOptions } from './projection';
import { endpointKey } from './projection';
import { appendIndexedLayer, indexedNodeId, type IndexedStack } from './indexed-repetition';

type Repetition = Graph['repetitions'][number];
type Body = NonNullable<Repetition['bodies']>[number];
type Edge = Graph['edges'][number];
type IndexScope = Extract<NonNullable<ProjectedEdge['relationship']>, { owner: 'repetition' }>['indexScope'];
const ep = (node_id: string, port_id: string): Endpoint => ({ node_id, port_id });
export const bodyId = (rep: Repetition, body: Body) => `body:${rep.id}:${body.start}`;
export const bodyRangeId = (rep: Repetition, body: Body, start: number) => `${bodyId(rep, body)}:slot:${start}`;
export function bodyProjectionIds(graph: Graph) {
  return graph.repetitions.flatMap((rep) => (rep.bodies ?? []).flatMap((body) => [bodyId(rep, body),
    ...body.ranges.flatMap((range) => {
      const family = graph.templates?.find((t) => t.id === body.slots[range.start]);
      const anchor = family?.instances.find((i) => i.node_id === rep.instances[body.start + range.start]?.node_id);
      const scope = bodyRangeId(rep, body, range.start);
      const members = new Set(anchor?.nodes.map((n) => n.node_id));
      return [scope, ...(anchor?.nodes.map((n) => indexedNodeId(scope, n.node_id)) ?? []),
        ...graph.repetitions.filter((r) => members.has(r.parent_id)).map((r) => indexedNodeId(scope, `repeat:${r.id}:0:${r.instances.length - 1}`))];
    })]));
}

/** Consume the producer's verified intervals. No period or equivalence discovery
 * occurs here. All numeric navigation remains on the original Shared instances. */
export function projectRepeatedBodies(graph: Graph, base: Projection, options: ProjectionOptions): Projection {
  if (options.exhaustive || options.stateScope) return base;
  const records = new Map(graph.nodes.map((n) => [n.id, n]));
  const incoming = new Map<string, Edge[]>(), outgoing = new Map<string, Edge[]>();
  for (const e of graph.edges) {
    incoming.set(endpointKey(e.target), [...incoming.get(endpointKey(e.target)) ?? [], e]);
    outgoing.set(endpointKey(e.source), [...outgoing.get(endpointKey(e.source)) ?? [], e]);
  }
  for (const rep of graph.repetitions) for (const body of rep.bodies ?? []) {
    if (options.repetitions?.[rep.id]) continue;
    const covered = rep.instances.slice(body.start, body.start + body.width * body.count);
    const outer = base.nodes.find((n) => n.presentation === 'repetition' && n.repetitionId === rep.id &&
      n.instances?.length === covered.length && n.instances[0]?.node_id === covered[0]?.node_id);
    if (!outer || !options.expanded.includes(outer.id)) continue;
    const roots = new Set(covered.map((i) => i.node_id));
    const families = body.slots.map((id) => graph.templates!.find((t) => t.id === id)!);
    const members = new Set(families.flatMap((t) => t.instances.filter((i) => roots.has(i.node_id)).flatMap((i) => i.nodes.map((n) => n.node_id))));
    const pathTo = (target: Endpoint) => {
      const path: Edge[] = [], seen = new Set<string>();
      let current = target;
      for (;;) {
        const edges = (incoming.get(endpointKey(current)) ?? []).filter((e) =>
          !(e.target.node_id === target.node_id && members.has(e.source.node_id) && !roots.has(e.source.node_id)));
        if (edges.length !== 1 || seen.has(edges[0]!.id)) throw new Error('Repeated body source path changed.');
        const edge = edges[0]!; path.unshift(edge); seen.add(edge.id); current = edge.source;
        if (roots.has(current.node_id) || records.get(current.node_id)?.kind !== 'group' || !incoming.has(endpointKey(current))) return path;
      }
    };
    const paths = covered.map((i) => pathTo(ep(i.node_id, body.input_port)));
    const serialPort = (port: string, direction: 'input' | 'output') => outer.ports.find((p) => p.direction === direction &&
      p.endpoints.some((e) => roots.has(e.node_id) && e.port_id === port));
    const input = serialPort(body.input_port, 'input'), output = serialPort(body.output_port, 'output');
    if (!input || !output) continue;
    outer.expanded = true;
    outer.summary = `${body.count} blocks × ${body.width} layers · j = 0..${body.count - 1}`;
    input.label = `${body.input_port}[${covered[0]!.index}]`;
    output.label = `${body.output_port}[${covered.at(-1)!.index}]`;
    const id = bodyId(rep, body);
    const origin = covered[0]!.index;
    const expression = (offset: number, inner = false) => `${origin ? `${origin}+` : ''}${body.width}j${inner ? (offset ? `+${offset}+k` : '+k') : offset ? `+${offset}` : ''}`;
    const shell: ProjectedNode = { id, parentId: outer.id, kind: 'group', label: 'Hybrid block[j]', sourceIds: [],
      expanded: true, presentation: 'repetition', nestedRepetition: true, componentCount: body.ranges.length,
      ports: outer.ports.map((p) => ({ ...p, endpoints: [...p.endpoints], label: p === input ? `${body.input_port}[${expression(0)}]`
        : p === output ? `${body.output_port}[${expression(body.width - 1)}]` : p.label })),
      summary: `${body.width} layers per block` };
    base.nodes.push(shell);
    let sequence = 0;
    const relationship = (scope: string, kind: NonNullable<ProjectedEdge['relationship']>['kind'], source: Endpoint, target: Endpoint, paths: Edge[][], templateId: string, indexScope?: IndexScope, represented = covered) => {
      base.edges.push({ id: `${scope}:${kind}:${sequence++}`, source, target, paths, kind: paths[0]?.[0]?.kind ?? 'data',
        originalEdgeIds: [...new Set(paths.flat().map((e) => e.id))], relationship: { owner: 'repetition', kind,
          repetitionId: rep.id, scopeId: scope, templateId, portRole: body.input_port, ...(indexScope ? { indexScope } : {}),
          instances: represented.map((i) => ({ nodeId: i.node_id, index: i.index })) } });
    };
    relationship(outer.id, 'entry', ep(outer.id, input.id), ep(id, input.id), [paths[0]!], families[0]!.id);
    relationship(outer.id, 'return', ep(id, output.id), ep(id, input.id), paths.filter((_, i) => i > 0 && i % body.width === 0), families[0]!.id,
      { variable: 'j', base: origin, width: body.width, offset: body.width - 1, count: body.count, input: body.input_port, output: body.output_port });
    // The exterior projection already owns final forwarding. This segment is
    // a presentation boundary continuation, not another source computation.
    relationship(outer.id, 'exit', ep(id, output.id), ep(outer.id, output.id), [], families.at(-1)!.id);
    for (const port of outer.ports) if (port !== input && port !== output)
      relationship(outer.id, port.direction === 'input' ? 'invariant' : 'side-output',
        port.direction === 'input' ? ep(outer.id, port.id) : ep(id, port.id),
        port.direction === 'input' ? ep(id, port.id) : ep(outer.id, port.id), [], families[0]!.id);
    const rangeNodes: { input: Endpoint; output: Endpoint }[] = [];
    for (const range of body.ranges) {
      const scope = bodyRangeId(rep, body, range.start), template = families[range.start]!;
      const selected = covered.filter((_, at) => at % body.width >= range.start && at % body.width < range.start + range.count);
      const instances = selected.map((i) => template.instances.find((v) => v.node_id === i.node_id)!);
      const anchor = instances[0]!, record = records.get(anchor.node_id)!;
      const invariantPorts = new Set<string>();
      const side = new Set((rep.side_ports ?? []).map((p) => p.port_id));
      const pending = record.ports.filter((p) => p.direction === 'input' && p.id !== body.input_port && !side.has(p.id)).map((p) => ep(record.id, p.id));
      const closure = new Set(anchor.nodes.map((n) => n.node_id));
      while (pending.length) {
        const endpoint = pending.pop()!, key = endpointKey(endpoint);
        if (invariantPorts.has(key)) continue;
        invariantPorts.add(key);
        if (records.get(endpoint.node_id)?.kind === 'group') for (const edge of outgoing.get(key) ?? [])
          if (closure.has(edge.target.node_id)) pending.push(edge.target);
      }
      const parameters = new Map(graph.parameters.map((p) => [p.id, p]));
      const terminal = (id: string): string => {
        let parameter = parameters.get(id);
        const seen = new Set<string>();
        while (parameter?.binding === 'alias' && !seen.has(parameter.id)) {
          seen.add(parameter.id); parameter = parameters.get(parameter.alias_of);
        }
        return parameter?.id ?? id;
      };
      const invariantParameters = new Set(anchor.parameters.filter((p) => instances.every((i) =>
        terminal(i.parameters.find((q) => q.role === p.role)!.parameter_id) === terminal(p.parameter_id))).map((p) => p.role));
      const stack: IndexedStack = { repetition: { ...rep, id: scope, instances: selected, bodies: [] }, template, instances,
        input: body.input_port, output: body.output_port, invariantPorts, invariantParameters, entry: [], transitions: [], exits: [], invariants: [], sides: [] };
      const symbol = expression(range.start, range.count > 1);
      const variant = selected[0]!.variant.replaceAll('_', ' ').replace(/ attention$/, '');
      const name = variant.charAt(0).toUpperCase() + variant.slice(1);
      const layerNoun = /\bdecoder\b/i.test(rep.label) ? 'decoder' : 'layer';
      const singleName = selected[0]!.variant.replaceAll('_', ' ');
      const singleLabel = singleName.charAt(0).toUpperCase() + singleName.slice(1);
      const ports = record.ports.filter((p) => options.showUnused || !base.unusedInputs.some((u) => u.node_id === record.id && u.port_id === p.id))
        .map((p) => ({ ...p, endpoints: selected.map((i) => ep(i.node_id, p.id)), label: invariantPorts.has(endpointKey(ep(record.id, p.id))) ? p.label : `${p.label}[${symbol}]` }));
      let root: string;
      if (range.count > 1) {
        const container: ProjectedNode = { id: scope, parentId: id, kind: 'group', label: `${name} ${layerNoun} ×${range.count}`,
          sourceIds: [], ports, expanded: options.expanded.includes(scope), presentation: 'repetition', nestedRepetition: true,
          repetitionId: rep.id, instances: selected, summary: `k = 0..${range.count - 1}` };
        base.nodes.push(container);
        root = scope;
        if (container.expanded) {
          const layer = appendIndexedLayer(graph, base, options, stack, scope, symbol, `${name} layer[${symbol}]`);
          relationship(scope, 'entry', ep(scope, body.input_port), ep(layer, body.input_port), [], template.id);
          relationship(scope, 'return', ep(layer, body.output_port), ep(layer, body.input_port), paths.filter((_, at) =>
            at % body.width > range.start && at % body.width < range.start + range.count), template.id,
            { variable: 'k', base: origin, width: body.width, offset: range.start, count: range.count, input: body.input_port, output: body.output_port }, selected);
          relationship(scope, 'exit', ep(layer, body.output_port), ep(scope, body.output_port), [], template.id);
          for (const port of ports) if (![body.input_port, body.output_port].includes(port.id))
            relationship(scope, port.direction === 'input' ? 'invariant' : 'side-output',
              port.direction === 'input' ? ep(scope, port.id) : ep(layer, port.id),
              port.direction === 'input' ? ep(layer, port.id) : ep(scope, port.id), [], template.id);
        }
      } else root = appendIndexedLayer(graph, base, options, stack, id, symbol, `${singleLabel} layer[${symbol}]`);
      rangeNodes.push({ input: ep(root, body.input_port), output: ep(root, body.output_port) });
      for (const port of ports) if (![body.input_port, body.output_port].includes(port.id)) {
        for (const boundary of shell.ports.filter((p) => p.endpoints.some((e) => selected.some((i) => i.node_id === e.node_id) && e.port_id === port.id))) {
          const dependency = graph.edges.filter((e) => port.direction === 'input'
            ? boundary.endpoints.some((p) => endpointKey(p) === endpointKey(e.target)) && e.target.port_id === port.id && !members.has(e.source.node_id)
            : boundary.endpoints.some((p) => endpointKey(p) === endpointKey(e.source)) && e.source.port_id === port.id && !members.has(e.target.node_id));
          relationship(scope, side.has(port.id) ? port.direction === 'input' ? 'side-input' : 'side-output' : 'invariant',
            port.direction === 'input' ? ep(id, boundary.id) : ep(root, port.id),
            port.direction === 'input' ? ep(root, port.id) : ep(id, boundary.id), dependency.map((e) => [e]), template.id);
        }
      }
    }
    relationship(id, 'entry', ep(id, input.id), rangeNodes[0]!.input, [], families[0]!.id);
    for (let at = 1; at < rangeNodes.length; at++) {
      const transitions = paths.filter((_, i) => i % body.width === body.ranges[at]!.start);
      base.edges.push({ id: `${id}:connection:${at}`, source: rangeNodes[at - 1]!.output, target: rangeNodes[at]!.input,
        kind: 'data', paths: transitions, originalEdgeIds: transitions.flat().map((e) => e.id) });
    }
    relationship(id, 'exit', rangeNodes.at(-1)!.output, ep(id, output.id), [], families.at(-1)!.id);
  }
  const represented = new Set(base.edges.flatMap((e) => e.originalEdgeIds));
  return { ...base, hiddenEdgeIds: base.hiddenEdgeIds.filter((id) => !represented.has(id)), filteredEdgeIds: base.filteredEdgeIds.filter((id) => !represented.has(id)) };
}
