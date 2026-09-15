import type { Graph } from './graph';
import { deriveMlpGroups } from './derived-groups';

/** A scope is membership in the immutable source graph, never a copied graph.
 * Presentation IDs are accepted only for the existing justified MLP grouping. */
export function componentScope(graph: Graph, id: string) {
  const records = new Map(graph.nodes.map((node) => [node.id, node]));
  const source = records.get(id);
  const derived = source ? undefined : deriveMlpGroups(graph).find((group) => group.id === id);
  if ((!source || !['group', 'operation'].includes(source.kind)) && !derived) {
    throw new Error('This component is no longer available. Return to the model view.');
  }
  const roots = source ? [source.id] : derived!.sourceIds;
  const members = new Set<string>();
  const queue = [...roots];
  while (queue.length) {
    const member = queue.pop()!;
    if (members.has(member)) continue;
    const node = records.get(member);
    if (!node) throw new Error('The component contains an unresolved source reference.');
    members.add(member);
    if (node.kind === 'group') queue.push(...node.children);
  }
  return { id, roots, members, derived };
}
