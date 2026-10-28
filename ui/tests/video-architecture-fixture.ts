import type { components } from '../src/api/generated/types';
import { frame } from './embedding-fixtures';

type S = components['schemas'];

/** Small hand-authored UI oracle, explicitly not a native checkpoint graph. */
export function videoFixture() {
  const graph: S['ArchitectureGraph'] = {
    graph_id: 'synthetic-video', scope: 'video_generation', coverage: 'partial',
    symbols: [], nodes: [], edges: [], repetitions: [], parameters: [],
    diagnostics: [{ code: 'ui_fixture', message: 'Synthetic UI branches, not checkpoint acceptance.' }],
  };
  const port = (id: string, direction: 'input' | 'output', value_type = 'latent tensor'): S['ArchitecturePort'] =>
    ({ id, label: id, direction, shape: null, value_type });
  const group = (id: string, label: string, parent?: S['ArchitectureGroupNode']) => {
    const node: S['ArchitectureGroupNode'] = { id, label, kind: 'group', children: [],
      ports: [port('in', 'input'), port('out', 'output')], parameter_ids: [], references: [],
      attributes: [], provenance: [], ...(parent ? { parent_id: parent.id } : {}) };
    parent?.children.push(id); graph.nodes.push(node); return node;
  };
  const edge = (source: string, sp: string, target: string, tp: string) => graph.edges.push({
    id: `${source}.${sp}-${target}.${tp}`, source: { node_id: source, port_id: sp },
    target: { node_id: target, port_id: tp }, kind: 'data', provenance: [],
  });
  const root = group('pipeline', 'Synthetic image-to-video pipeline');
  root.ports = [port('prompt', 'input', 'text'), port('image', 'input', 'required initial RGB image'),
    port('configuration', 'input', 'schedule and mode'), port('video', 'output', 'RGB video frames')];
  const conditioning = group('conditioning', 'Text and image conditioning', root);
  conditioning.ports = [port('prompt', 'input', 'text'), port('image', 'input', 'RGB image'), port('out', 'output')];
  edge(root.id, 'prompt', conditioning.id, 'prompt'); edge(root.id, 'image', conditioning.id, 'image');
  const loop = group('iteration', 'One denoising iteration', root);
  edge(conditioning.id, 'out', loop.id, 'in');
  const inventory: S['TensorInventory'] = { tensors: [], coverage: 'complete', diagnostics: [] };
  for (const role of ['high_noise', 'low_noise']) {
    const expert = group(role, role === 'high_noise' ? 'High noise expert' : 'Low noise expert', loop);
    expert.attributes = [{ name: 'condition', value: role === 'high_noise' ? 't >= 900' : 't < 900', provenance: [] }];
    edge(loop.id, 'in', expert.id, 'in');
    const repetition: S['ArchitectureRepetition'] = { id: `${role}-layers`, parent_id: expert.id, label: `${role} blocks`, instances: [] };
    for (const index of [0, 1]) {
      const block = group(`${role}-block-${index}`, `${role} block ${index}`, expert);
      repetition.instances.push({ node_id: block.id, index, variant: 'wan-ui-fixture' });
      edge(index ? `${role}-block-0` : expert.id, index ? 'out' : 'in', block.id, 'in');
      const id = `${block.id}-q`, name = `video.${role}.blocks.${index}.self_attn.q.weight`;
      graph.parameters.push({ id: `${id}-parameter`, name, binding: 'quantized',
        logical_shape: [{ kind: 'constant', value: 2 }, { kind: 'constant', value: 2 }],
        storage: [{ name, dtype: 'F8_E4M3', shape: [2, 2] },
          { name: name.replace('.weight', '.scale_weight'), dtype: 'F32', shape: [] }],
        inspection: { status: 'available', tensor_id: id }, provenance: [] });
      inventory.tensors.push({ id, name, path: name.split('.'), shape: [2, 2], rank: 2, numel: 4,
        storage_dtype: 'F8_E4M3', storage_format: 'fp8-e4m3-scaled', logical_dtype: 'float32' });
      graph.nodes.push({ id, parent_id: block.id, kind: 'operation', label: `${role} Q ${index}`, operation: 'linear',
        ports: [port('in', 'input'), port('out', 'output')], parameter_ids: [`${id}-parameter`],
        references: [{ kind: 'parameter', parameter_id: `${id}-parameter` }], attributes: [], provenance: [] });
      block.children.push(id); edge(block.id, 'in', id, 'in'); edge(id, 'out', block.id, 'out');
    }
    edge(`${role}-block-1`, 'out', expert.id, 'out'); graph.repetitions.push(repetition);
  }
  loop.children.push('select');
  graph.nodes.push({ id: 'select', parent_id: loop.id, kind: 'operation', label: 'Select timestep expert',
    operation: 'timestep_expert_selection', formula: 'out = high if t >= 0.900*1000 else low',
    ports: [port('high', 'input'), port('low', 'input'), port('out', 'output')],
    parameter_ids: [], references: [], attributes: [], provenance: [] });
  edge('high_noise', 'out', 'select', 'high'); edge('low_noise', 'out', 'select', 'low');
  edge('select', 'out', loop.id, 'out');
  const vae = group('vae', 'VAE decoding', root); edge(loop.id, 'out', vae.id, 'in');
  edge(vae.id, 'out', root.id, 'video');
  return { response: { status: 'available' as const, model_id: 'fixture/video', graph, diagnostics: [] }, inventory };
}

export const videoValues = (id: string) => id.startsWith('high_noise') ? [0.25, -0.5, 0.125, 0.5] : [1, -2, 0.5, 2];

export function videoStream(id: string, name: string, kind: string) {
  const values = videoValues(id), sorted = [...values].sort((a, b) => a - b);
  const percentile = (q: number) => sorted[Math.floor(3 * q)]! * (1 - (3 * q) % 1) + sorted[Math.ceil(3 * q)]! * ((3 * q) % 1);
  const mean = values.reduce((a, b) => a + b, 0) / 4;
  let metadata: object, data = new Uint8Array();
  if (kind === 'data') {
    metadata = { kind: 'tensor', tensor_id: id, name, shape: [2, 2], dtype: 'float32', byte_order: 'little', layout: 'c', byte_length: 16 };
    data = new Uint8Array(new Float32Array(values).buffer);
  } else if (kind === 'statistics') {
    metadata = { kind: 'tensor_statistics', tensor_id: id, count: 4, finite_count: 4, non_finite_count: 0,
      minimum: sorted[0], maximum: sorted[3], mean, stddev: Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / 4),
      percentiles: { p01: percentile(.01), p05: percentile(.05), p50: percentile(.5), p95: percentile(.95), p99: percentile(.99) }, byte_length: 0 };
  } else {
    const counts = new Uint32Array(400);
    values.forEach((v, i) => { const bin = Math.min(99, Math.floor((v - sorted[0]!) / (sorted[3]! - sorted[0]!) * 100));
      counts[Math.floor(i / 2) * 100 + bin]!++; counts[200 + bin * 2 + i % 2]!++; });
    metadata = { kind: 'tensor_distributions', tensor_id: id, rows: 2, columns: 2, bin_count: 100,
      binning: 'linear-full-range', domain_minimum: sorted[0], domain_maximum: sorted[3], dtype: 'uint32', byte_order: 'little', byte_length: 1600,
      sections: [{ name: 'row_counts', shape: [2, 100], offset: 0, byte_length: 800 }, { name: 'column_counts', shape: [100, 2], offset: 800, byte_length: 800 }] };
    data = new Uint8Array(counts.buffer);
  }
  return Buffer.from([...frame(1, new TextEncoder().encode(JSON.stringify(metadata))), ...(data.length ? frame(2, data) : []), ...frame(4)]);
}
