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
