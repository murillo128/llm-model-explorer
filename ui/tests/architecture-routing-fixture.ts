import type { Graph } from '../src/architecture-explorer/graph';

/** Independent two-signal context case. Source order deliberately opposes the
 * consumer's sin/mask order; the interface must stay source-preserving. */
export function rotaryContextFixture(): Graph {
  const port = (id: string, direction: 'input' | 'output') => ({ id, label: id, direction,
    shape: [{ kind: 'constant' as const, value: 16 }] });
  const common = { parameter_ids: [], references: [], attributes: [], provenance: [] };
  return { graph_id: 'rotary-context-routing', scope: 'language_model', coverage: 'partial', symbols: [], parameters: [], repetitions: [], diagnostics: [],
    nodes: [
      { ...common, id: 'mask', kind: 'context', label: 'From causal mask', ports: [port('out', 'output')] },
      { ...common, id: 'sin', kind: 'context', label: 'From rotary sin', ports: [port('out', 'output')] },
      { ...common, id: 'component', kind: 'group', label: 'Component', children: ['use'], ports: [port('sin', 'input'), port('causal mask', 'input')] },
      { ...common, id: 'use', kind: 'operation', operation: 'attention', label: 'Use rotary context', parent_id: 'component',
        ports: [port('sin', 'input'), port('causal mask', 'input')] },
    ], edges: [
      { id: 'sin-enter', source: { node_id: 'sin', port_id: 'out' }, target: { node_id: 'component', port_id: 'sin' }, kind: 'context', provenance: [] },
      { id: 'mask-enter', source: { node_id: 'mask', port_id: 'out' }, target: { node_id: 'component', port_id: 'causal mask' }, kind: 'context', provenance: [] },
      { id: 'sin-use', source: { node_id: 'component', port_id: 'sin' }, target: { node_id: 'use', port_id: 'sin' }, kind: 'context', provenance: [] },
      { id: 'mask-use', source: { node_id: 'component', port_id: 'causal mask' }, target: { node_id: 'use', port_id: 'causal mask' }, kind: 'context', provenance: [] },
    ] };
}
