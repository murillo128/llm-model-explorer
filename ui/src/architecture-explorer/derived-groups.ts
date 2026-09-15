import type { Graph } from './graph';

export interface DerivedGroup { id: string; parentId: string; label: string; sourceIds: string[] }

/** A reversible presentation grouping, supported by BOTH parameter ownership and
 * exact topology. Labels never establish computational connectivity. */
export function deriveMlpGroups(graph: Graph): DerivedGroup[] {
  const parameters = new Map(graph.parameters.map((p) => [p.id, p]));
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  const outgoing = new Map<string, typeof graph.edges>();
  const incoming = new Map<string, typeof graph.edges>();
  for (const edge of graph.edges) {
    outgoing.set(edge.source.node_id, [...(outgoing.get(edge.source.node_id) ?? []), edge]);
    incoming.set(edge.target.node_id, [...(incoming.get(edge.target.node_id) ?? []), edge]);
  }
  const families = new Map<string, Map<string, string>>();
  for (const node of graph.nodes) {
    if (node.kind !== 'operation' || node.operation !== 'linear' || !node.parent_id) continue;
    for (const id of node.parameter_ids) {
      const name = parameters.get(id)?.name;
      const match = name?.match(/^(.*\.mlp)\.(gate_proj|up_proj|down_proj)\.weight$/);
      if (!match) continue;
      const key = JSON.stringify([node.parent_id, match[1]]);
      const family = families.get(key) ?? new Map();
      // Ambiguous bindings do not establish a unique operation.
      family.set(match[2]!, family.has(match[2]!) ? '' : node.id);
      families.set(key, family);
    }
  }
  const result: DerivedGroup[] = [];
  for (const [key, family] of families) {
    const gate = family.get('gate_proj'), up = family.get('up_proj'), down = family.get('down_proj');
    if (!gate || !up || !down) continue;
    const gOut = outgoing.get(gate) ?? [], uOut = outgoing.get(up) ?? [];
    if (gOut.length !== 1 || uOut.length !== 1) continue;
    const silu = nodes.get(gOut[0]!.target.node_id), multiply = nodes.get(uOut[0]!.target.node_id);
    const [parentId] = JSON.parse(key) as [string, string];
    if (silu?.operation !== 'silu' || multiply?.operation !== 'multiply' ||
      silu.parent_id !== parentId || multiply.parent_id !== parentId) continue;
    const sOut = outgoing.get(silu.id) ?? [], mOut = outgoing.get(multiply.id) ?? [];
    if (sOut.length !== 1 || sOut[0]!.target.node_id !== multiply.id ||
      mOut.length !== 1 || mOut[0]!.target.node_id !== down ||
      (incoming.get(multiply.id)?.length !== 2) ||
      sOut[0]!.target.port_id === uOut[0]!.target.port_id) continue;
    const gIn = incoming.get(gate) ?? [], uIn = incoming.get(up) ?? [];
    if (gIn.length !== 1 || uIn.length !== 1 || JSON.stringify(gIn[0]!.source) !== JSON.stringify(uIn[0]!.source)) continue;
    if ((incoming.get(silu.id)?.length !== 1) || (incoming.get(down)?.length !== 1)) continue;
    const sourceIds = [gate, up, silu.id, multiply.id, down];
    if (sourceIds.some((id) => nodes.get(id)?.parent_id !== parentId)) continue;
    if (sourceIds.flatMap((id) => [...(outgoing.get(id) ?? []), ...(incoming.get(id) ?? [])]).some((e) => e.kind !== 'data')) continue;
    result.push({ id: `mlp:${gate}`, parentId, label: 'MLP', sourceIds });
  }
  return result;
}
