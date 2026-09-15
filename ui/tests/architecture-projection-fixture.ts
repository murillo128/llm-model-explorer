import type { components } from '../src/api/generated/types';
import type { Graph, GraphNode } from '../src/architecture-explorer/graph';

type S = components['schemas'];
type Group = S['ArchitectureGroupNode'];
type Port = S['ArchitecturePort'];
export type ProjectionVariant = 'linear_attention' | 'full_attention';
export interface ProjectionFixtureOptions {
  count?: number;
  /** Exact instance order; it is never sorted or interpreted as a repeating template. */
  variants?: ProjectionVariant[];
  /** A second, independently selectable predictor repetition; also removes tokenizer context. */
  secondStack?: boolean | number;
  tokenizer?: boolean;
  hiddenSize?: number;
  partial?: boolean;
  repetitions?: boolean;
}

/** Deliberately authored topology, independent of checkpoints, caches and layout coordinates.
 * Parameter identities are real within this fixture; their numerical data is not supplied.
 * All declared interfaces survive, including auxiliaries unused by a particular variant.
 */
export function makeProjectionFixture(options: ProjectionFixtureOptions = {}): Graph {
  const count = options.count ?? options.variants?.length ?? 24;
  const hidden = options.hiddenSize ?? 16;
  const secondCount = typeof options.secondStack === 'number' ? options.secondStack : options.secondStack ? 3 : 0;
  if (!Number.isSafeInteger(count) || count < 1 || !Number.isSafeInteger(secondCount) || secondCount < 0 ||
    !Number.isSafeInteger(hidden) || hidden < 1 || options.variants && options.variants.length !== count) {
    throw new Error('Projection fixture requires positive size/count and an exact variant sequence.');
  }
  const variants = options.variants ?? Array.from({ length: count }, (_, i) => i % 4 === 3 ? 'full_attention' : 'linear_attention');
  const visual = secondCount > 0;
  const graph: Graph = {
    graph_id: `authored-projection-${variants.map((v) => v === 'full_attention' ? 'F' : 'L').join('')}-${secondCount}-${hidden}-${!visual && options.tokenizer !== false}-${!!options.partial}-${options.repetitions !== false}`,
    scope: visual ? 'visual_encoder_predictor' : 'language_model', coverage: options.partial ? 'partial' : 'complete',
    symbols: [{ name: 'B', meaning: 'Batch' }, { name: 'S', meaning: 'Symbolic sequence length' }],
    nodes: [], edges: [], repetitions: [], parameters: [], diagnostics: [],
  };
  const sequence: S['ArchitectureShape'] = [{ kind: 'symbol', name: 'B' }, { kind: 'symbol', name: 'S' }];
  const activation: S['ArchitectureShape'] = [...sequence, { kind: 'constant', value: hidden }];
  const port = (id: string, direction: Port['direction'], shape: S['ArchitectureShape'] = activation): Port => ({ id, direction, label: id, shape });
  const input = (id: string, shape: S['ArchitectureShape'] = activation) => port(id, 'input', shape);
  const output = (id = 'out', shape: S['ArchitectureShape'] = activation) => port(id, 'output', shape);
  const auxiliary = () => ['positions', 'mask', 'current_mask'].map((id) => input(id, sequence));
  const provenance: S['ArchitectureProvenance'][] = [{ kind: 'description', source: 'authored-projection-fixture', rule: 'small deterministic graph; no checkpoint support claim' }];

  function node(id: string, label: string, ports: Port[], parent?: Group, kind: S['ArchitectureLeafNode']['kind'] = 'operation', operation?: string): S['ArchitectureLeafNode'] {
    const result: S['ArchitectureLeafNode'] = { id, label, kind, ports, parameter_ids: [], references: [], attributes: [], provenance,
      ...(parent ? { parent_id: parent.id } : {}), ...(operation ? { operation } : {}) };
    graph.nodes.push(result); parent?.children.push(id); return result;
  }
  function group(id: string, label: string, ports: Port[], parent?: Group): Group {
    const result: Group = { id, label, kind: 'group', ports, children: [], parameter_ids: [], references: [], attributes: [], provenance,
      ...(parent ? { parent_id: parent.id } : {}) };
    graph.nodes.push(result); parent?.children.push(id); return result;
  }
  function edge(source: GraphNode, from: string, target: GraphNode, to: string, kind: S['ArchitectureEdge']['kind'] = 'data') {
    graph.edges.push({ id: `edge-${graph.edges.length}`, source: { node_id: source.id, port_id: from },
      target: { node_id: target.id, port_id: to }, kind, provenance });
  }
  function parameter(owner: GraphNode, module: string) {
    const id = `parameter.${module}.weight`;
    owner.parameter_ids.push(id);
    owner.references.push({ kind: 'module', name: module }, { kind: 'parameter', parameter_id: id });
    graph.parameters.push({ id, name: `${module}.weight`, logical_shape: [{ kind: 'constant', value: hidden }, { kind: 'constant', value: hidden }],
      binding: 'unresolved', storage: [], inspection: { status: 'unavailable', reason: 'unresolved_binding', message: 'Authored structural fixture has no numerical weight payload.' }, provenance });
  }
  function operation(id: string, label: string, parent: Group, operationName = label, inputs = ['x'], outputs = ['out']) {
    return node(id, label, [...inputs.map((id) => input(id)), ...outputs.map((id) => output(id))], parent, 'operation', operationName);
  }
  function layer(parent: Group, id: string, index: number, variant: ProjectionVariant, module: string) {
    const result = group(id, `Layer ${index} · ${variant === 'full_attention' ? 'Full attention' : 'Linear attention'}`,
      [input('x'), ...auxiliary(), output()], parent);
    result.references.push({ kind: 'module', name: module });
    const norm = operation(`${id}.input-norm`, 'Input norm', result, 'rms_norm'); parameter(norm, `${module}.input_layernorm`);
    const stateNames = variant === 'full_attention' ? ['K', 'V'] : ['conv', 'delta'];
    const attention = group(`${id}.attention`, variant === 'full_attention' ? 'Full attention' : 'Linear attention',
      [input('x'), ...auxiliary(), ...stateNames.map((s) => input(`prior_${s}`)), output(), ...stateNames.map((s) => output(`next_${s}`))], result);
    attention.references.push({ kind: 'module', name: `${module}.self_attn` });
    edge(result, 'x', norm, 'x'); edge(norm, 'out', attention, 'x');
    // All layer auxiliaries are declared, but only consumers justified by source edges are forwarded.
    for (const aux of variant === 'full_attention' ? ['positions', 'mask'] : ['current_mask']) edge(result, aux, attention, aux);
    for (const state of stateNames) {
      const prior = node(`${id}.prior-${state}`, `Prior ${state}`, [output()], result, 'state');
      const next = node(`${id}.next-${state}`, `Next ${state}`, [input('x')], result, 'state');
      edge(prior, 'out', attention, `prior_${state}`, 'state'); edge(attention, `next_${state}`, next, 'x', 'state');
    }
    if (variant === 'full_attention') {
      const projections = ['Q', 'K', 'V'].map((name) => {
        const projection = operation(`${id}.attention.${name}`, `${name} projection`, attention, 'linear');
        parameter(projection, `${module}.self_attn.${name.toLowerCase()}_proj`); edge(attention, 'x', projection, 'x'); return projection;
      });
      const rotary = ['Q', 'K'].map((name, i) => {
        const rotation = node(`${id}.attention.rope-${name}`, `RoPE ${name}`, [input('x'), input('positions', sequence), output()], attention, 'operation', 'rotary_embedding');
        edge(projections[i]!, 'out', rotation, 'x'); edge(attention, 'positions', rotation, 'positions'); return rotation;
      });
      const core = node(`${id}.attention.core`, 'Attention', [input('Q'), input('K'), input('V'), input('mask', sequence), input('prior_K'), input('prior_V'),
        output(), output('next_K'), output('next_V')], attention, 'operation', 'scaled_dot_product_attention');
      edge(rotary[0]!, 'out', core, 'Q'); edge(rotary[1]!, 'out', core, 'K'); edge(projections[2]!, 'out', core, 'V');
      edge(attention, 'mask', core, 'mask');
      for (const state of stateNames) { edge(attention, `prior_${state}`, core, `prior_${state}`, 'state'); edge(core, `next_${state}`, attention, `next_${state}`, 'state'); }
      const out = operation(`${id}.attention.output`, 'Output projection', attention, 'linear'); parameter(out, `${module}.self_attn.o_proj`);
      edge(core, 'out', out, 'x'); edge(out, 'out', attention, 'out');
    } else {
      const projection = operation(`${id}.attention.input`, 'Input projection', attention, 'linear'); parameter(projection, `${module}.self_attn.in_proj`);
      const conv = node(`${id}.attention.conv`, 'Convolution', [input('x'), input('current_mask', sequence), input('prior_conv'), output(), output('next_conv')], attention, 'operation', 'convolution');
      const delta = operation(`${id}.attention.delta`, 'Delta update', attention, 'delta_rule', ['x', 'prior_delta'], ['out', 'next_delta']);
      const out = operation(`${id}.attention.output`, 'Output projection', attention, 'linear'); parameter(out, `${module}.self_attn.out_proj`);
      edge(attention, 'x', projection, 'x'); edge(projection, 'out', conv, 'x'); edge(attention, 'current_mask', conv, 'current_mask');
      edge(conv, 'out', delta, 'x'); edge(delta, 'out', out, 'x'); edge(out, 'out', attention, 'out');
      for (const [state, consumer] of [['conv', conv], ['delta', delta]] as const) {
        edge(attention, `prior_${state}`, consumer, `prior_${state}`, 'state'); edge(consumer, `next_${state}`, attention, `next_${state}`, 'state');
      }
    }
    const first = operation(`${id}.residual-1`, 'Residual addition 1', result, 'add', ['residual', 'update']);
    edge(result, 'x', first, 'residual'); edge(attention, 'out', first, 'update');
    const postNorm = operation(`${id}.post-norm`, 'Post-attention norm', result, 'rms_norm'); parameter(postNorm, `${module}.post_attention_layernorm`);
    edge(first, 'out', postNorm, 'x');
    const gate = operation(`${id}.gate`, 'Gate projection', result, 'linear'); parameter(gate, `${module}.mlp.gate_proj`);
    const up = operation(`${id}.up`, 'Up projection', result, 'linear'); parameter(up, `${module}.mlp.up_proj`);
    const silu = operation(`${id}.silu`, 'SiLU', result, 'silu');
    const multiply = operation(`${id}.multiply`, 'Elementwise multiply', result, 'multiply', ['gate', 'up']);
    const down = operation(`${id}.down`, 'Down projection', result, 'linear'); parameter(down, `${module}.mlp.down_proj`);
    for (const member of [silu, multiply]) member.references.push({ kind: 'module', name: `${module}.mlp` });
    edge(postNorm, 'out', gate, 'x'); edge(postNorm, 'out', up, 'x'); edge(gate, 'out', silu, 'x');
    edge(silu, 'out', multiply, 'gate'); edge(up, 'out', multiply, 'up'); edge(multiply, 'out', down, 'x');
    const second = operation(`${id}.residual-2`, 'Residual addition 2', result, 'add', ['residual', 'update']);
    edge(first, 'out', second, 'residual'); edge(down, 'out', second, 'update'); edge(second, 'out', result, 'out');
    return result;
  }

  if (!visual && options.tokenizer !== false) node('tokenizer', 'Tokenizer capability', [], undefined, 'context').references.push({ kind: 'tokenizer' });
  const source = node(visual ? 'visual-input' : 'token-ids', visual ? 'Symbolic visual input' : 'Token IDs', [output('out', sequence)], undefined, 'input');
  const auxiliaries = ['positions', 'mask', 'current_mask'].map((id) => node(id.replace('_', '-'), id, [output('out', sequence)], undefined, 'input'));
  const root = group('model', visual ? 'Encoder / predictor model' : 'Language model', [input('tokens', sequence), ...auxiliary(), output()]);
  edge(source, 'out', root, 'tokens'); auxiliaries.forEach((n, i) => edge(n, 'out', root, ['positions', 'mask', 'current_mask'][i]!));
  const embedding = node('embedding', visual ? 'Patch preparation' : 'Embedding', [input('tokens', sequence), output()], root, 'operation', visual ? 'patch_projection' : 'embedding');
  parameter(embedding, visual ? 'model.patch_embed' : 'model.embed_tokens'); edge(root, 'tokens', embedding, 'tokens');
  function stack(parent: Group, prefix: string, orderedVariants: ProjectionVariant[], module: string) {
    const repetition: S['ArchitectureRepetition'] = { id: `${prefix || 'decoder'}-layers`, parent_id: parent.id,
      label: prefix ? `${prefix[0]!.toUpperCase()}${prefix.slice(1)} layers` : 'Decoder layers', instances: [] };
    if (options.repetitions !== false) graph.repetitions.push(repetition);
    let previous: GraphNode | undefined;
    for (const [index, variant] of orderedVariants.entries()) {
      const current = layer(parent, `${prefix ? `${prefix}.` : ''}layer-${index}`, index, variant, `${module}.layers.${index}`);
      repetition.instances.push({ node_id: current.id, index, variant });
      if (previous) edge(previous, 'out', current, 'x');
      else edge(parent === root ? embedding : parent, parent === root ? 'out' : 'x', current, 'x');
      for (const aux of ['positions', 'mask', 'current_mask']) edge(parent, aux, current, aux);
      previous = current;
    }
    return previous!;
  }
  let tail: GraphNode;
  if (visual) {
    const encoder = group('encoder', 'Encoder', [input('x'), ...auxiliary(), output()], root);
    const predictor = group('predictor', 'Predictor', [input('x'), ...auxiliary(), output()], root);
    edge(embedding, 'out', encoder, 'x'); edge(encoder, 'out', predictor, 'x');
    for (const current of [encoder, predictor]) for (const aux of ['positions', 'mask', 'current_mask']) edge(root, aux, current, aux);
    edge(stack(encoder, 'encoder', variants, 'model.encoder'), 'out', encoder, 'out');
    edge(stack(predictor, 'predictor', Array.from({ length: secondCount }, () => 'full_attention'), 'model.predictor'), 'out', predictor, 'out');
    tail = predictor;
  } else tail = stack(root, '', variants, 'model');
  const finalNorm = operation('final-norm', 'Final norm', root, 'rms_norm'); parameter(finalNorm, 'model.norm');
  edge(tail, 'out', finalNorm, 'x'); edge(finalNorm, 'out', root, 'out');
  const head = node(visual ? 'representations' : 'head', visual ? 'Representations' : 'LM head', [input('x'), output()], undefined,
    visual ? 'output' : 'operation', visual ? undefined : 'linear');
  if (!visual) parameter(head, 'lm_head'); edge(root, 'out', head, 'x');
  if (options.partial) {
    const unknown = node('unknown-component', 'Unknown component', [input('x'), output()], root, 'operation', 'unknown');
    unknown.description = 'An authored unknown region, left explicit.';
    graph.diagnostics.push({ code: 'unknown_structure', message: 'Fixture intentionally includes an unresolved component.', node_id: unknown.id });
  }
  return graph;
}
