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

/** Three independently specified serial layers with three distinct shared signals. */
export function makeIndexedFixture(count: number = 3): Graph {
  const indices = Array.from({ length: count }, (_, i) => i);
  const shape = [{ kind: 'constant' as const, value: 4 }];
  const ports = (names: string[]) => names.map((id) => ({ id, label: id, direction: id === 'out' ? 'output' as const : 'input' as const, shape }));
  const provenance = [{ kind: 'description' as const, source: 'indexed-fixture', revision: '1' }];
  const graph: Graph = { graph_id: 'indexed-fixture', scope: 'language_model', coverage: 'complete', symbols: [], diagnostics: [],
    nodes: [{ id: 'model', kind: 'group', label: 'Model', ports: [], children: ['input', 'cos', 'sin', 'mask', ...indices.map((i) => `layer-${i}`), 'result'], parameter_ids: [], references: [], attributes: [], provenance }],
    edges: [], parameters: [], repetitions: [{ id: 'layers', parent_id: 'model', label: 'Decoder layers', instances: indices.map((index) => ({ index, node_id: `layer-${index}`, variant: 'dense' })) }] };
  const edge = (id: string, source: string, sourcePort: string, target: string, targetPort: string) => graph.edges.push({ id, source: { node_id: source, port_id: sourcePort }, target: { node_id: target, port_id: targetPort }, kind: 'data', provenance });
  for (const id of ['input', 'cos', 'sin', 'mask']) graph.nodes.push({ id, parent_id: 'model', kind: 'input', label: id, operation: 'symbolic', ports: ports(['out']), parameter_ids: [], references: [], attributes: [], provenance });
  graph.nodes.push({ id: 'result', parent_id: 'model', kind: 'output', label: 'Result', operation: 'result', ports: ports(['x']), parameter_ids: [], references: [], attributes: [], provenance });
  for (const index of indices) {
    const id = `layer-${index}`;
    graph.nodes.push({ id, parent_id: 'model', kind: 'group', label: `Layer ${index}`, ports: ports(['x', 'cos', 'sin', 'mask', 'out']), children: [`${id}.op`], parameter_ids: [], references: [], attributes: [{ name: 'semantic_role', value: 'layer', provenance }], provenance });
    graph.nodes.push({ id: `${id}.op`, parent_id: id, kind: 'operation', label: 'Transform', operation: 'transform', formula: 'out = transform(x, cos, sin, mask; weight)', ports: ports(['x', 'cos', 'sin', 'mask', 'out']), parameter_ids: [`weight-${index}`], references: [], attributes: [], provenance });
    graph.parameters.push({ id: `weight-${index}`, name: `layers.${index}.weight`, logical_shape: shape, binding: 'native', storage: [{ name: `layers.${index}.weight`, dtype: 'float32', shape: [4] }], inspection: { status: 'available', tensor_id: `tensor-${index}` }, provenance });
    for (const port of ['x', 'cos', 'sin', 'mask']) edge(`${id}.${port}`, id, port, `${id}.op`, port);
    edge(`${id}.out`, `${id}.op`, 'out', id, 'out');
    for (const port of ['cos', 'sin', 'mask']) edge(`${port}-${index}`, port, 'out', id, port);
  }
  edge('entry', 'input', 'out', 'layer-0', 'x');
  for (let i = 0; i < count - 1; i++) edge(`next-${i}`, `layer-${i}`, 'out', `layer-${i + 1}`, 'x');
  edge('exit', `layer-${count - 1}`, 'out', 'result', 'x');
  graph.templates = count === 1 ? [] : [{ id: 'whole-layer', label: 'Decoder Layer', component_role: 'layer', revision: '1', provenance,
    instances: indices.map((index) => ({ node_id: `layer-${index}`,
      nodes: [{ role: 'root', node_id: `layer-${index}` }, { role: 'op', node_id: `layer-${index}.op` }],
      ports: ['root', 'op'].flatMap((role) => ['x', 'cos', 'sin', 'mask', 'out'].map((port_id) => ({ role: `${role}.${port_id}`, node_id: `layer-${index}${role === 'op' ? '.op' : ''}`, port_id }))),
      edges: ['x', 'cos', 'sin', 'mask', 'out'].map((role) => ({ role, edge_id: `layer-${index}.${role}` })),
      parameters: [{ role: 'weight', parameter_id: `weight-${index}` }],
    })) }];
  return graph;
}
