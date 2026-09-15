import type { Graph, GraphNode } from './graph';
import type { ProjectedNode } from './projection';

/** Labels are presentation aliases; inspection always receives the original record. */
export function displayLabel(node: ProjectedNode, graph: Graph): string {
  if (node.presentation === 'mlp') return 'MLP';
  if (node.instances) {
    if (node.presentation === 'range') return node.label;
    const name = graph.repetitions.find((r) => r.id === node.repetitionId)!.label.match(/encoder|predictor|decoder/i)?.[0] ?? 'Stack';
    return `${name[0]!.toUpperCase()}${name.slice(1)} ×${node.instances.length}`;
  }
  const record = node.record;
  if (!record) return node.label;
  const instance = graph.repetitions.flatMap((r) => r.instances).find((i) => i.node_id === record.id);
  if (instance) return `Layer ${instance.index}`;
  if (record.references.some((r) => r.kind === 'tokenizer')) return 'Tokenizer capability';
  if (record.kind === 'group') {
    const path = record.references.find((r) => r.kind === 'module')?.name ?? record.label;
    const end = path.split('.').at(-1)!;
    return ({ linear_attn: 'Linear attention', self_attn: 'Full attention', language_model: 'Language model' })[end as 'linear_attn'] ?? end.replaceAll('_', ' ');
  }
  const parameter = graph.parameters.find((p) => record.parameter_ids.includes(p.id));
  const module = parameter?.name.split('.').at(-2);
  if (module && ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'lm_head'].includes(module)) {
    return ({ gate_proj: 'Gate projection', up_proj: 'Up projection', down_proj: 'Down projection', q_proj: 'Q projection', k_proj: 'K projection', v_proj: 'V projection', o_proj: 'Output projection', lm_head: 'LM head' })[module as 'gate_proj'];
  }
  const aliases: Record<string, string> = { symbolic_token_ids: 'Token IDs', symbolic_thw_positions: 'THW positions',
    symbolic_padding_mask: 'Padding mask', current_sequence_padding_mask: 'Current mask', embedding_lookup: 'Embedding',
    rms_norm_zero_centered: 'RMSNorm', rms_norm: 'RMSNorm', vocabulary_logits: 'Logits', silu: 'SiLU', multiply: 'Multiply' };
  return aliases[record.operation ?? ''] ?? node.label;
}

export function instanceOf(graph: Graph, nodeId: string | null) {
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  let node: GraphNode | undefined = nodeId ? nodes.get(nodeId) : undefined;
  while (node) {
    for (const repetition of graph.repetitions) {
      const instance = repetition.instances.find((i) => i.node_id === node!.id);
      if (instance) return { repetition, instance };
    }
    node = node.parent_id ? nodes.get(node.parent_id) : undefined;
  }
  return undefined;
}

/** Exact order, including non-periodic variants; only compress a verified period. */
export function patternSummary(instances: Graph['repetitions'][number]['instances']): string {
  const variants = [...new Set(instances.map((i) => i.variant))];
  const symbols = variants.map((v, i) => v === 'linear_attention' ? 'L' : v === 'full_attention' ? 'F' : String(i + 1));
  const sequence = instances.map((i) => symbols[variants.indexOf(i.variant)]!);
  let period = sequence.length;
  for (let candidate = 1; candidate < sequence.length; candidate++) {
    if (sequence.length % candidate === 0 && sequence.every((v, i) => v === sequence[i % candidate])) { period = candidate; break; }
  }
  const pattern = sequence.slice(0, period).join(' · ');
  const legend = variants.map((v, i) => `${symbols[i]} = ${v.replaceAll('_', ' ')}`).join('; ');
  return `${pattern}${period < sequence.length ? ` × ${sequence.length / period}` : ''} (${legend})`;
}
