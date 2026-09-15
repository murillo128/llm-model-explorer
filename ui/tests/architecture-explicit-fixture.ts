import type { Graph } from '../src/architecture-explorer/graph';
import { makeProjectionFixture, type ProjectionFixtureOptions } from './architecture-projection-fixture';

/** Authored transparent module boundaries; no production grouping helper is used. */
export function makeExplicitFixture(options: ProjectionFixtureOptions = {}): Graph {
  const graph = makeProjectionFixture({ count: 4, ...options });
  graph.graph_id += '-explicit';
  const layers = graph.nodes.filter((node) => node.kind === 'group' && node.id.match(/layer-\d+$/));
  for (const layer of layers) {
    if (layer.kind !== 'group') continue;
    const children = ['gate', 'up', 'silu', 'multiply', 'down'].map((suffix) => `${layer.id}.${suffix}`);
    const id = `${layer.id}.mlp`;
    const gateInput = graph.edges.find((edge) => edge.target.node_id === children[0])!;
    const downOutput = graph.edges.find((edge) => edge.source.node_id === children[4])!;
    const incoming = { ...gateInput.source }, outgoing = { ...downOutput.source };
    const gate = graph.nodes.find((node) => node.id === children[0])!;
    const down = graph.nodes.find((node) => node.id === children[4])!;
    const provenance = [{ kind: 'description' as const, source: 'authored-component-fixture' }];
    graph.nodes.push({ id, kind: 'group', label: 'MLP', parent_id: layer.id, children,
      ports: [structuredClone(gate.ports.find((port) => port.id === 'x')!), structuredClone(down.ports.find((port) => port.id === 'out')!)],
      attributes: [{ name: 'semantic_role', value: 'mlp', provenance }], parameter_ids: [], provenance,
      references: [{ kind: 'module', name: `${layer.references.find((ref) => ref.kind === 'module')!.name}.mlp` }],
    });
    layer.children = layer.children.flatMap((child) => child === children[0] ? [id] : children.includes(child) ? [] : [child]);
    for (const child of graph.nodes) if (children.includes(child.id)) child.parent_id = id;
    for (const edge of graph.edges) {
      if (edge.target.node_id === children[0] || edge.target.node_id === children[1]) edge.source = { node_id: id, port_id: 'x' };
    }
    downOutput.source = { node_id: id, port_id: 'out' };
    graph.edges.push(
      { id: `${id}-input`, source: incoming, target: { node_id: id, port_id: 'x' }, kind: 'data', provenance },
      { id: `${id}-output`, source: outgoing, target: { node_id: id, port_id: 'out' }, kind: 'data', provenance },
    );
    const attention = graph.nodes.find((node) => node.id === `${layer.id}.attention`)!;
    attention.attributes.push({ name: 'semantic_role', value: 'attention', provenance });
  }
  return graph;
}
