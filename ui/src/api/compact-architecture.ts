import type { components } from './generated/types';
import { requireProtocol as require } from './errors';

type S = components['schemas'];
type Graph = S['ArchitectureGraph'];

function replace(value: unknown, ids: Map<string, string>, before: string, after: string): unknown {
  if (typeof value === 'string') return ids.get(value) ?? value.replaceAll(before, after);
  if (Array.isArray(value)) return value.map((item) => replace(item, ids, before, after));
  if (value && typeof value === 'object') return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key, replace(item, ids, before, after)]),
  );
  return value;
}

/** Reconstruct exact source records from the verified definition and explicit instance maps. */
export function expandCompactGraph(graph: Graph): Graph {
  if (!graph.compact_components?.length) return graph;
  const nodes = [...graph.nodes], edges = [...graph.edges];
  const parameters = new Map(graph.parameters.map((p) => [p.id, p]));
  const repetitions = new Map(graph.repetitions.map((r) => [r.id, r]));
  const symbolNames = new Set(graph.symbols.map((s) => s.name));
  const seen = new Set([...graph.nodes, ...graph.edges, ...graph.parameters, ...graph.repetitions].map((r) => r.id));
  const prefixes = new Set<string>();
  for (const family of graph.compact_components) {
    require(!seen.has(family.id), 'Duplicate compact family identity');
    seen.add(family.id);
    const repetition = repetitions.get(family.repetition_id);
    require(repetition && family.nodes[0]?.kind === 'group', 'Unknown compact repetition or root');
    require(repetition.instances.every((item) => ['routed_expert', 'routed_swiglu'].includes(item.variant)), 'Unverified compact variant');
    require(JSON.stringify(repetition.instances.map((i) => [i.node_id, i.index])) ===
      JSON.stringify(family.instances.map((i) => [i.node_id, i.index])), 'Compact repetition mismatch');
    const prototypeNames = family.parameter_ids.map((id) => {
      const parameter = parameters.get(id);
      require(parameter?.name.startsWith(family.base_prefix + '.'), 'Compact prototype binding');
      return parameter!.name;
    });
    require(family.symbols.every((name) => symbolNames.has(name)), 'Unknown compact symbol');
    for (const instance of family.instances) {
      require(!prefixes.has(instance.prefix), 'Duplicate compact source prefix');
      prefixes.add(instance.prefix);
      require(instance.node_ids.length === family.nodes.length && instance.edge_ids.length === family.edges.length &&
        instance.parameter_ids.length === family.parameter_ids.length && instance.symbols.length === family.symbols.length &&
        instance.node_ids[0] === instance.node_id, 'Compact mapping length');
      require(instance.symbols.every((name) => symbolNames.has(name)), 'Unknown compact instance symbol');
      for (let i = 0; i < prototypeNames.length; i++) {
        const parameter = parameters.get(instance.parameter_ids[i]!);
        require(parameter?.name === prototypeNames[i]!.replaceAll(family.base_prefix, instance.prefix),
          'Compact expert binding is missing or swapped');
      }
      const ids = new Map<string, string>();
      family.nodes.forEach((node, i) => ids.set(node.id, instance.node_ids[i]!));
      family.edges.forEach((edge, i) => ids.set(edge.id, instance.edge_ids[i]!));
      family.parameter_ids.forEach((id, i) => ids.set(id, instance.parameter_ids[i]!));
      family.symbols.forEach((name, i) => ids.set(name, instance.symbols[i]!));
      const materializedNodes = family.nodes.map((node) => replace(node, ids, family.base_prefix, instance.prefix) as S['ArchitectureNode']);
      const materializedEdges = family.edges.map((edge) => replace(edge, ids, family.base_prefix, instance.prefix) as S['ArchitectureEdge']);
      const root = materializedNodes[0]!;
      require(root.kind === 'group' && root.parent_id === repetition.parent_id, 'Compact parent mismatch');
      root.label = instance.label;
      for (const attribute of root.attributes) if (attribute.name === 'expert_index') attribute.value = instance.index;
      for (const record of [...materializedNodes, ...materializedEdges]) {
        require(!seen.has(record.id), 'Duplicate reconstructed identity');
        seen.add(record.id);
      }
      nodes.push(...materializedNodes);
      edges.push(...materializedEdges);
    }
  }
  // The source graph remains immutable; projection receives concrete records.
  const result = { ...graph, nodes, edges };
  delete result.compact_components;
  return result;
}
