import type { Graph } from '../src/architecture-explorer/graph';
import { makeExplicitFixture } from './architecture-explicit-fixture';

/** Independently authored full-attention correspondence at source layers 0 and 2.
 * The fixture's explicit member list is the oracle; no producer/matcher is used. */
export function makeTemplateFixture(): Graph {
  const graph = makeExplicitFixture({ count: 3, variants: ['full_attention', 'linear_attention', 'full_attention'] });
  const provenance = [{ kind: 'description' as const, source: 'authored-template-fixture', revision: '1' }];
  const roles = ['root', 'Q', 'K', 'V', 'rope-Q', 'rope-K', 'core', 'output'];
  graph.templates = [{ id: 'shared-full-attention', label: 'Full attention', component_role: 'attention', revision: '1', provenance,
    instances: [0, 2].map((index) => {
      const node_id = `layer-${index}.attention`;
      const nodes = roles.map((role) => ({ role, node_id: role === 'root' ? node_id : `${node_id}.${role}` }));
      const byId = new Map(nodes.map((n) => [n.node_id, n.role]));
      return { node_id, nodes,
        ports: nodes.flatMap((n) => graph.nodes.find((r) => r.id === n.node_id)!.ports.map((p) => ({ role: `${n.role}.${p.id}`, node_id: n.node_id, port_id: p.id }))),
        edges: graph.edges.filter((e) => byId.has(e.source.node_id) && byId.has(e.target.node_id)).map((e) => ({
          role: `${byId.get(e.source.node_id)}.${e.source.port_id}--${byId.get(e.target.node_id)}.${e.target.port_id}`, edge_id: e.id })),
        parameters: ['Q', 'K', 'V', 'output'].map((role) => ({ role: `${role}.weight`, parameter_id: graph.nodes.find((n) => n.id === `${node_id}.${role}`)!.parameter_ids[0]! })),
      };
    }) }];
  return graph;
}

/** Authored browser fixture with the accepted V-JEPA stack counts and four
 * independently verified families. It exercises presentation, not checkpoint support. */
export function makeVjepaBrowserFixture(): Graph {
  const variants = Array.from({ length: 24 }, () => 'full_attention' as const);
  const graph = makeExplicitFixture({ count: 24, variants, secondStack: 12 });
  const records = new Map(graph.nodes.map((node) => [node.id, node]));
  const provenance = [{ kind: 'description' as const, source: 'authored-vjepa-browser-fixture', revision: '1' }];
  function family(prefix: 'encoder' | 'predictor', count: number, componentRole: 'attention' | 'mlp') {
    const rootId = (index: number) => `${prefix}.layer-${index}.${componentRole}`;
    const memberIds = (root: string) => {
      const result: string[] = [], pending = [root];
      while (pending.length) {
        const id = pending.shift()!, node = records.get(id);
        if (!node) continue;
        result.push(id);
        if (node.kind === 'group') pending.push(...node.children);
      }
      return result;
    };
    return { id: `shared-${prefix}-${componentRole}`, label: `${prefix === 'encoder' ? 'Encoder' : 'Predictor'} ${componentRole === 'attention' ? 'attention' : 'MLP'}`,
      component_role: componentRole, revision: '1', provenance,
      instances: Array.from({ length: count }, (_, index) => {
        const root = rootId(index), ids = memberIds(root), members = new Set(ids);
        const layerPrefix = `${prefix}.layer-${index}.`;
        const role = (id: string) => id === root ? 'root' : `node.${id.startsWith(layerPrefix) ? id.slice(layerPrefix.length) : id}`;
        return { node_id: root,
          nodes: ids.map((node_id) => ({ role: role(node_id), node_id })),
          ports: ids.flatMap((node_id) => records.get(node_id)!.ports.map((port) => ({ role: `${role(node_id)}.port.${port.id}`, node_id, port_id: port.id }))),
          edges: graph.edges.filter((edge) => members.has(edge.source.node_id) && members.has(edge.target.node_id))
            .map((edge, position) => ({ role: `edge.${position}`, edge_id: edge.id })),
          parameters: ids.flatMap((node_id) => records.get(node_id)!.parameter_ids
            .map((parameter_id, position) => ({ role: `${role(node_id)}.parameter.${position}`, parameter_id }))),
        };
      }) };
  }
  graph.graph_id = 'authored-vjepa-browser';
  graph.templates = [family('encoder', 24, 'attention'), family('encoder', 24, 'mlp'),
    family('predictor', 12, 'attention'), family('predictor', 12, 'mlp')];
  return graph;
}
