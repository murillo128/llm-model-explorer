import type { components } from './generated/types';
import { requireProtocol as require } from './errors';

type S = components['schemas'];
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
  return JSON.stringify(value ?? null);
}
function mapped<T extends { role: string }>(records: T[], target: (record: T) => string) {
  const result = new Map(records.map((record) => [target(record), record.role]));
  require(result.size === records.length && new Set(records.map((r) => r.role)).size === records.length,
    'Duplicate template role or target');
  return result;
}
function sameKeys(map: Map<string, string>, expected: Set<string>) {
  require(map.size === expected.size && [...expected].every((key) => map.has(key)), 'Template mapping coverage');
}
const endpoint = (node: string, port: string) => JSON.stringify([node, port]);

/** Optional metadata is checked independently of canvas projection and numeric actions. */
export function validateTemplates(graph: S['ArchitectureGraph']) {
  if (!graph.templates?.length) return;
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  const parameters = new Map(graph.parameters.map((p) => [p.id, p]));
  const edges = new Map(graph.edges.map((e) => [e.id, e]));
  const ids = [...graph.nodes, ...graph.parameters, ...graph.edges, ...graph.repetitions, ...graph.templates].map((r) => r.id);
  require(new Set(ids).size === ids.length, 'Duplicate template identity');
  const incident = new Map(graph.nodes.map((n) => [n.id, new Set<string>()]));
  for (const e of graph.edges) {
    incident.get(e.source.node_id)!.add(e.id); incident.get(e.target.node_id)!.add(e.id);
  }
  const repeated = new Map<string, [string, number]>();
  for (const repetition of graph.repetitions) repetition.instances.forEach((i, position) => repeated.set(i.node_id, [repetition.id, position]));
  const order = new Map<string, [string, number]>();
  const pending: [string, [string, number] | undefined][] = graph.nodes.filter((n) => !n.parent_id).map((n) => [n.id, undefined]);
  while (pending.length) {
    const [id, inherited] = pending.pop()!;
    const node = nodes.get(id)!;
    const owner = repeated.get(id) ?? inherited;
    if (owner) order.set(id, owner);
    if (node.kind === 'group') node.children.forEach((child, position) => {
      order.set(child, owner ?? [id, position]); pending.push([child, owner]);
    });
  }
  const aliases = new Map<string, string>();
  for (const parameter of graph.parameters) {
    let target = parameter;
    const path: string[] = [];
    while (!aliases.has(target.id) && target.binding === 'alias') {
      path.push(target.id); target = parameters.get(target.alias_of)!;
    }
    const terminal = aliases.get(target.id) ?? target.id;
    aliases.set(parameter.id, terminal); path.forEach((id) => aliases.set(id, terminal));
  }
  const diagnosed = new Set(graph.diagnostics.flatMap((d) => d.node_id ? [d.node_id] : []));
  const seen = new Set<string>();
  for (const template of graph.templates) {
    require(template.provenance.some((p) => p.kind === 'description' && p.revision), 'Reviewed template provenance');
    let scope: string | undefined, previous = -1, baseline: string | undefined;
    for (const instance of template.instances) {
      require(nodes.get(instance.node_id)?.kind === 'group' && !seen.has(instance.node_id), 'Template component identity');
      seen.add(instance.node_id);
      const position = order.get(instance.node_id);
      require(position && (!scope || scope === position[0]) && position[1] > previous, 'Template source order/scope');
      [scope, previous] = position;
      const members = new Set<string>(), pending = [instance.node_id];
      while (pending.length) {
        const id = pending.pop()!;
        require(!members.has(id) && !diagnosed.has(id), 'Unverified template member');
        members.add(id);
        const node = nodes.get(id)!;
        if (node.kind === 'group') pending.push(...node.children);
      }
      const nr = mapped(instance.nodes, (m) => m.node_id);
      const pr = mapped(instance.ports, (m) => endpoint(m.node_id, m.port_id));
      const er = mapped(instance.edges, (m) => m.edge_id);
      const wr = mapped(instance.parameters, (m) => m.parameter_id);
      sameKeys(nr, members);
      sameKeys(pr, new Set([...members].flatMap((id) => nodes.get(id)!.ports.map((p) => endpoint(id, p.id)))));
      const expectedEdges = new Set<string>(), expectedParameters = new Set<string>();
      for (const id of members) {
        for (const eid of incident.get(id)!) {
          const edge = edges.get(eid)!;
          if (members.has(edge.source.node_id) && members.has(edge.target.node_id)) expectedEdges.add(eid);
        }
        const node = nodes.get(id)!;
        node.parameter_ids.forEach((pid) => expectedParameters.add(pid));
        for (const ref of node.references) if (ref.kind === 'parameter') expectedParameters.add(ref.parameter_id);
      }
      sameKeys(er, expectedEdges); sameKeys(wr, expectedParameters);
      function shape(dims: S['ArchitectureShape']) {
        require(dims && dims.every((d) => d.kind !== 'unknown'), 'Unknown template shape');
        return dims;
      }
      const normal: Record<string, unknown> = { root: nr.get(instance.node_id) };
      normal.nodes = Object.fromEntries([...members].map((id) => {
        const node = nodes.get(id)!;
        const attributes = new Map(node.attributes.map((a) => [a.name, a.value]));
        require(attributes.size === node.attributes.length && [...attributes.values()].every((v) => v !== null &&
          !(Array.isArray(v) && v.includes(null))), 'Unknown or duplicate template attribute');
        require(node.kind === 'group' || node.operation, 'Unknown template operation');
        if (id === instance.node_id) require(attributes.get('semantic_role') === template.component_role, 'Template component role');
        return [nr.get(id), [node.kind, node.operation, node.formula, node.description, nr.get(node.parent_id ?? ''),
          node.kind === 'group' ? node.children.map((c) => nr.get(c)) : [], node.ports.map((p) => pr.get(endpoint(id, p.id))),
          node.parameter_ids.map((p) => wr.get(p)), node.references.flatMap((r) => r.kind === 'parameter' ? [wr.get(r.parameter_id)] : []),
          Object.fromEntries(attributes)]];
      }));
      normal.ports = Object.fromEntries([...members].flatMap((id) => nodes.get(id)!.ports.map((p) =>
        [pr.get(endpoint(id, p.id)), [nr.get(id), p.id, p.direction, shape(p.shape)]])));
      normal.edges = Object.fromEntries([...er].map(([id, role]) => {
        const edge = edges.get(id)!;
        return [role, [pr.get(endpoint(edge.source.node_id, edge.source.port_id)),
          pr.get(endpoint(edge.target.node_id, edge.target.port_id)), edge.kind]];
      }));
      const aliasRoles = new Map<string, string[]>();
      for (const [id, role] of wr) {
        const terminal = aliases.get(id)!;
        aliasRoles.set(terminal, [...(aliasRoles.get(terminal) ?? []), role]);
      }
      normal.parameters = Object.fromEntries([...wr].map(([id, role]) => [role,
        [shape(parameters.get(id)!.logical_shape), aliasRoles.get(aliases.get(id)!)!.sort()]]));
      const signature = canonical(normal);
      require(baseline === undefined || signature === baseline, 'Incompatible template computation');
      baseline = signature;
    }
  }
}
