import type { Graph, GraphNode, GraphView, Layout } from './graph';
import { snapshotView } from './scope-navigation';
import { endpointKey } from './projection';

export type Template = NonNullable<Graph['templates']>[number];
export type TemplateInstance = Template['instances'][number];
export interface SharedStructure { templateId: string; anchorId: string; instanceId: string | null }

export function sharedContext(graph: Graph, state: SharedStructure) {
  const template = graph.templates?.find((t) => t.id === state.templateId);
  const anchor = template?.instances.find((i) => i.node_id === state.anchorId);
  const instance = state.instanceId ? template?.instances.find((i) => i.node_id === state.instanceId) : null;
  if (!template || !anchor || state.instanceId && !instance) throw new Error('Shared structure correspondence changed. Return to the model and choose a component again.');
  return { template, anchor, instance: instance ?? null };
}

/** A bounded component view, sharing immutable source records. Never another full graph. */
export function templateGraph(graph: Graph, instance: TemplateInstance): Graph {
  const members = new Set(instance.nodes.map((m) => m.node_id));
  const edges = new Set(instance.edges.map((m) => m.edge_id));
  const parameters = new Set(instance.parameters.map((m) => m.parameter_id));
  const nodes = graph.nodes.filter((n) => members.has(n.id)).map((node) => {
    if (node.id !== instance.node_id) return node;
    const root = { ...node }; delete root.parent_id; return root;
  });
  if (nodes.length !== members.size) throw new Error('Shared structure source records are no longer available.');
  return { ...graph, nodes, edges: graph.edges.filter((e) => edges.has(e.id)),
    parameters: graph.parameters.filter((p) => parameters.has(p.id)), repetitions: [], diagnostics: [], templates: [] };
}

export function remapNode(id: string | null, from: TemplateInstance, to: TemplateInstance): string | null {
  const role = from.nodes.find((m) => m.node_id === id)?.role;
  return role ? to.nodes.find((m) => m.role === role)?.node_id ?? null : null;
}

export function commonNode(node: GraphNode, role: string, label?: string): GraphNode {
  return { ...node, label: label ?? role.replaceAll('_', ' ').replaceAll('.', ' / '), parameter_ids: [], references: [],
    ports: node.ports.map((p) => ({ ...p, label: p.id })), provenance: [],
    attributes: node.attributes.map((a) => ({ ...a, provenance: [] })) };
}

/** Geometry and presentation identities stay fixed; only exact source references change.
 * Always bind from the original layout, so switching cannot retain prior generations. */
export function bindTemplateLayout(layout: Layout, graph: Graph, template: Template,
  from: TemplateInstance, chosen: TemplateInstance | null): Layout {
  const to = chosen ?? from;
  const records = new Map(graph.nodes.map((n) => [n.id, n]));
  const sourceEdges = new Map(graph.edges.map((e) => [e.id, e]));
  const nodeTargets = new Map(to.nodes.map((m) => [m.role, m.node_id]));
  const portTargets = new Map(to.ports.map((m) => [m.role, { node_id: m.node_id, port_id: m.port_id }]));
  const edgeTargets = new Map(to.edges.map((m) => [m.role, m.edge_id]));
  const nodes = new Map(from.nodes.map((m) => [m.node_id, nodeTargets.get(m.role)!]));
  const roles = new Map(from.nodes.map((m) => [m.node_id, m.role]));
  const ports = new Map(from.ports.map((m) => [endpointKey(m), portTargets.get(m.role)!]));
  const edges = new Map(from.edges.map((m) => [m.edge_id, edgeTargets.get(m.role)!]));
  const mappedEdge = (id: string) => {
    const target = edges.get(id);
    if (!target || !sourceEdges.has(target)) throw new Error('Shared structure edge correspondence is no longer available.');
    return target;
  };
  const projection = { ...layout.projection,
    nodes: layout.projection.nodes.map((node) => {
      const target = nodes.get(node.id), record = target ? records.get(target) : undefined;
      if (!record) throw new Error('Shared structure node correspondence is no longer available.');
      const common = commonNode(record, roles.get(node.id)!, node.id === from.node_id ? template.label : undefined);
      return { ...node, label: chosen ? record.label : common.label, record: chosen ? record : common,
        sourceIds: chosen ? node.sourceIds.map((id) => nodes.get(id)!) : [],
        ports: node.ports.map((p) => ({ ...p, label: chosen ? record.ports.find((port) => port.id === p.id)!.label : p.id,
          endpoints: chosen ? p.endpoints.map((e) => {
            const target = ports.get(endpointKey(e));
            if (!target) throw new Error('Shared structure port correspondence is no longer available.');
            return target;
          }) : [] })) };
    }),
    edges: layout.projection.edges.map((edge) => ({ ...edge,
      originalEdgeIds: chosen ? edge.originalEdgeIds.map(mappedEdge) : [],
      paths: edge.paths.map((path) => path.map((e) => sourceEdges.get(mappedEdge(e.id))!)) })),
    hiddenEdgeIds: layout.projection.hiddenEdgeIds.map(mappedEdge),
    filteredEdgeIds: layout.projection.filteredEdgeIds.map(mappedEdge),
    unusedInputs: layout.projection.unusedInputs.map((e) => ports.get(endpointKey(e))!),
  };
  return { ...layout, projection, edgeIds: chosen ? layout.edgeIds.map(mappedEdge) : [] };
}

export function enterSharedStructure(view: GraphView, template: Template,
  instanceId: string | null, viewport = view.viewport) {
  const anchor = template.instances.find((i) => i.node_id === instanceId) ?? template.instances[0]!;
  const previous = snapshotView(view, viewport);
  if (!view.scope) view.globalView = previous;
  view.history = [...view.history, previous].slice(-16);
  view.update({ selectionMode: instanceId ? 'source' : 'structure', shared: { templateId: template.id, anchorId: anchor.node_id, instanceId },
    scope: anchor.node_id, expanded: anchor.nodes.map((m) => m.node_id), repetitions: {},
    selected: instanceId && anchor.nodes.some((n) => n.node_id === view.selected) ? view.selected : anchor.node_id,
    edge: null, focus: anchor.node_id, activeStack: null, exhaustive: false, deriveMlp: false, showUnused: true,
    stateScope: undefined, viewport: undefined });
}
