import type { Graph, GraphNode, GraphView } from './graph';
import { displayLabel } from './presentation';
import { remapNode } from './shared-structure';
import { componentScope } from './scope';
import { interfaceIndex } from './interfaces';

export function componentLabel(node: GraphNode, graph: Graph) {
  return displayLabel({ id: node.id, kind: node.kind, label: node.label, record: node, sourceIds: [node.id], ports: [], expanded: false }, graph);
}

/** Ordered containment and public provenance only; never infer a model hierarchy. */
export function browserIndex(graph: Graph) {
  const interfaces = interfaceIndex(graph);
  const records = new Map(graph.nodes.map((node) => [node.id, node]));
  const parameters = new Map(graph.parameters.map((parameter) => [parameter.id, parameter.name]));
  const entries: { node: GraphNode; label: string; path: string; search: string; depth: number }[] = [];
  const instances = new Map(graph.repetitions.flatMap((repetition) => repetition.instances.map((instance) => [instance.node_id, { repetition, instance }] as const)));
  type Instance = { repetition: Graph['repetitions'][number]; instance: Graph['repetitions'][number]['instances'][number] };
  const pending: { node: GraphNode; parents: string[]; inherited: Instance | undefined }[] = graph.nodes.filter((node) => !node.parent_id)
    .reverse().map((node) => ({ node, parents: [], inherited: undefined }));
  while (pending.length) {
    const { node, parents, inherited } = pending.pop()!;
    const label = componentLabel(node, graph), info = instances.get(node.id) ?? inherited;
    const context = info ? `${info.repetition.label} / Instance ${info.instance.index} · ${info.instance.variant.replaceAll('_', ' ')}` : '';
    const path = [...parents, label].join(' / ');
    const sources = node.provenance.filter((p) => p.kind === 'description' &&
      p.rule === 'Semantic source key in the reviewed packaged description').map((p) => p.source);
    const resources = node.references.flatMap((reference) => reference.kind === 'parameter'
      ? [parameters.get(reference.parameter_id) ?? reference.parameter_id]
      : reference.kind === 'module' ? [reference.name] : []);
    const weights = node.parameter_ids.map((id) => parameters.get(id) ?? id);
    if (interfaces.eligible(node.id)) entries.push({ node, label, depth: parents.length, path: `${path}${context ? ` · ${context}` : ''}`,
      search: [path, context, node.label, node.id, ...sources, ...resources, ...weights, ...interfaces.interfaces.filter((item) => item.owner.id === node.id).map((item) => interfaceSearch(item.node))].join(' ').toLocaleLowerCase() });
    if (node.kind === 'group') for (const id of [...node.children].reverse()) {
      const child = records.get(id); if (child) pending.push({ node: child, parents: [...parents, label], inherited: info });
    }
  }
  return entries;
}

/** Concrete shared instances use the same anchor expansion, through verified roles. */
export function browserExpansionId(graph: Graph, view: GraphView, id: string) {
  const template = graph.templates?.find((t) => t.id === view.shared?.templateId);
  const anchor = template?.instances.find((i) => i.node_id === view.shared?.anchorId);
  const concrete = template?.instances.find((i) => i.node_id === view.shared?.instanceId);
  return anchor && concrete ? remapNode(id, concrete, anchor) ?? id : id;
}

/** Render predicates must not retain a Canvas render's obsolete layout context. */
export function browserPredicates(graph: Graph, view: GraphView) {
  const members = view.scope ? componentScope(graph, view.scope).members : undefined;
  const instance = graph.templates?.find((family) => family.id === view.shared?.templateId)
    ?.instances.find((item) => item.node_id === view.shared?.instanceId);
  const instanceIds = new Set(instance?.nodes.map((node) => node.node_id));
  const inside = (id: string) => !view.scope || (view.shared ? instanceIds.has(id) : members?.has(id) || view.scope === id);
  const expanded = (id: string) => {
    const state = inside(id) ? view : view.globalView ?? view;
    return state.exhaustive || state.expanded.includes(browserExpansionId(graph, view, id));
  };
  return { inside, expanded };
}

export const interfaceSearch = (node: GraphNode) => [node.label, node.id, ...node.ports.map((p) => p.label),
  ...node.provenance.filter((p) => p.kind === 'description').map((p) => p.source)].join(' ').toLocaleLowerCase();
