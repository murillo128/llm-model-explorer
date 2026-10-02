import type { Graph } from '../src/architecture-explorer/graph';

/** Exact mixed-destination topology from #289, without checkpoint data. */
export function connectorFixture(simple = false, nested = false, unused = false): Graph {
  const common = { parameter_ids: [], references: [], attributes: [], provenance: [] };
  const port = (id: string, direction: 'input' | 'output') => ({ id, label: id, direction, shape: [] });
  const inputs = simple ? ['positions', 'mask', 'current_mask', 'state'] :
    ['decide_idx', 'state', 'question', 'positions', 'mask', 'current_mask', 'opt_idx'];
  const boundary = nested ? 'inner' : 'model';
  const nodes: Graph['nodes'] = [
    { ...common, id: 'model', kind: 'group', label: 'Model', children: nested ? ['inner'] : simple ? ['language', 'concat'] : ['concat', 'language', 'decision', 'options', 'pointer'],
      ports: [...inputs.map((id) => port(id, 'input')), ...(unused ? [port('unused', 'input')] : []), port('result', 'output')] },
    ...(nested ? [{ ...common, id: 'inner', parent_id: 'model', kind: 'group' as const, label: 'Inner', children: ['language', 'concat'],
      ports: [...inputs.map((id) => port(id, 'input')), port('result', 'output')] }] : []),
    { ...common, id: 'concat', parent_id: boundary, kind: 'operation', operation: 'concat', label: 'Shared state plus isolated questions',
      ports: [port('state', 'input'), ...(!simple ? [port('question', 'input')] : []), port('out', 'output')] },
    { ...common, id: 'language', parent_id: boundary, kind: 'group', label: 'Language', children: ['language.body'],
      ports: [...(simple ? [] : [port('tokens', 'input')]), ...['positions', 'mask', 'current_mask'].map((id) => port(id, 'input')), port('out', 'output')] },
    ...(!simple ? ['decision', 'options', 'pointer'].map((id) => ({ ...common, id, parent_id: boundary,
      kind: 'operation' as const, operation: 'select', label: id,
      ports: [...(id === 'pointer' ? ['decide', 'options'] : ['x', 'index']).map((p) => port(p, 'input')), port('out', 'output')] })) : []),
  ];
  const language = nodes.find((n) => n.id === 'language')!;
  nodes.push({ ...common, id: 'language.body', parent_id: 'language', kind: 'operation', operation: 'identity', label: 'Language computation', ports: structuredClone(language.ports) });
  const links = simple ? [
    [boundary, 'positions', 'language', 'positions'], [boundary, 'mask', 'language', 'mask'],
    [boundary, 'current_mask', 'language', 'current_mask'], [boundary, 'state', 'concat', 'state'], ['language', 'out', boundary, 'result'],
  ] : [
    [boundary, 'state', 'concat', 'state'], [boundary, 'question', 'concat', 'question'], ['concat', 'out', 'language', 'tokens'],
    ...['positions', 'mask', 'current_mask'].map((p) => [boundary, p, 'language', p]),
    ['language', 'out', 'decision', 'x'], ['language', 'out', 'options', 'x'],
    [boundary, 'decide_idx', 'decision', 'index'], [boundary, 'opt_idx', 'options', 'index'],
    ['decision', 'out', 'pointer', 'decide'], ['options', 'out', 'pointer', 'options'], ['pointer', 'out', boundary, 'result'],
  ];
  if (nested) {
    links.push(...inputs.map((p) => ['model', p, 'inner', p]), ['inner', 'result', 'model', 'result']);
    const model = nodes.find((n) => n.id === 'model')!;
    if (model.kind !== 'group') throw new Error('Expected model group');
    for (const p of inputs) {
      const id = `input-${p}`;
      nodes.push({ ...common, id, parent_id: 'model', kind: 'input', label: p, ports: [port('out', 'output')] });
      model.children.push(id); links.push([id, 'out', 'model', p]);
    }
    nodes.push({ ...common, id: 'result-declaration', parent_id: 'model', kind: 'output', label: 'result', ports: [port('in', 'input')] });
    model.children.push('result-declaration'); links.push(['model', 'result', 'result-declaration', 'in']);
  }
  links.push(...language.ports.map((p) => p.direction === 'input' ? ['language', p.id, 'language.body', p.id] : ['language.body', p.id, 'language', p.id]));
  if (!simple) {
    const index = nodes.findIndex((n) => n.id === 'pointer'), pointer = nodes[index]!;
    nodes[index] = { ...pointer, kind: 'group', children: ['pointer.body'] };
    nodes.push({ ...common, id: 'pointer.body', parent_id: 'pointer', kind: 'operation', operation: 'select', label: 'Pointer computation', ports: structuredClone(pointer.ports) });
    links.push(...pointer.ports.map((p) => p.direction === 'input' ? ['pointer', p.id, 'pointer.body', p.id] : ['pointer.body', p.id, 'pointer', p.id]));
  }
  if (unused) {
    const model = nodes.find((n) => n.id === 'model')!;
    if (model.kind !== 'group') throw new Error('Expected model group');
    model.children.push('unused-declaration');
    nodes.push({ ...common, id: 'unused-declaration', parent_id: 'model', kind: 'input', label: 'unused', ports: [port('out', 'output')] });
    links.push(['unused-declaration', 'out', 'model', 'unused']);
  }
  return { graph_id: 'connector-readability', scope: 'language_model', coverage: 'complete', symbols: [], parameters: [], repetitions: [], diagnostics: [], nodes,
    edges: links.map(([s, p, t, q]) => ({ id: `${s}.${p}->${t}.${q}`, source: { node_id: s!, port_id: p! }, target: { node_id: t!, port_id: q! }, kind: 'data', provenance: [] })) };
}

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
