import type { Graph, GraphView } from './graph';
import type { ProjectedNode } from './projection';
import { cardExpandable } from './card-actions';

/** Shared canvas/navigator actions. Neither selection nor hiding a descendant reveals it. */
export function selectComponent(view: GraphView, id: string | null, selectionMode: GraphView['selectionMode'] = 'source') {
  view.update({ selected: id, boundary: undefined, selectionMode, edge: null, ...(view.browser.selectedFamily ? { browser: { ...view.browser, selectedFamily: null } } : {}) });
}

export function toggleComponent(view: GraphView, graph: Graph, node: ProjectedNode, windowSize: number) {
  if (!cardExpandable(node)) return;
  if (node.presentation === 'model') {
    view.update({ modelCollapsed: node.expanded, exhaustive: false, expansionAnchor: node.id }); return;
  }
  const expanded = new Set(view.exhaustive ? [...view.expanded, ...graph.nodes.filter((n) => n.kind === 'group').map((n) => n.id)] : view.expanded);
  // Materialize exhaustive detail before contracting one component. Otherwise
  // the old compact preferences would also hide unrelated operations/interfaces.
  const detail = view.exhaustive ? { deriveMlp: false, showUnused: true, showContext: true, stateScope: undefined } : {};
  if (node.repetitionId) {
    const repetition = graph.repetitions.find((r) => r.id === node.repetitionId);
    const start = repetition?.instances.findIndex((i) => i.node_id === node.instances?.[0]?.node_id) ?? -1;
    if (!repetition || start < 0) return;
    let parent = graph.nodes.find((n) => n.id === repetition.parent_id);
    while (parent) { expanded.add(parent.id); parent = graph.nodes.find((n) => n.id === parent?.parent_id); }
    // Reveal the exact range window without replacing its identity with a layer
    // or erasing independently expanded instances and their nested choices.
    view.update({ ...detail, expansionAnchor: node.id, expanded: [...expanded], exhaustive: false,
      repetitions: { ...view.repetitions, [repetition.id]: { start, count: windowSize } },
      focus: repetition.parent_id, activeStack: repetition.id, stateScope: undefined });
    return;
  }
  if (expanded.has(node.id)) expanded.delete(node.id); else expanded.add(node.id);
  view.update({ ...detail, expansionAnchor: node.id, expanded: [...expanded], exhaustive: false });
}
