import type { components } from './generated/types';
import { requireProtocol as require } from './errors';
import { product, validateSchema } from './validation';

type S = components['schemas'];
export interface ArchitectureContext {
  modelId: string;
  inventory: S['TensorInventory'];
  tokenizerAvailable: boolean;
}
function unique<T>(values: T[], key: (value: T) => string) {
  const map = new Map(values.map((value) => [key(value), value]));
  require(map.size === values.length, 'Duplicate architecture identity');
  return map;
}
/** Contextual checks supplement the generated closed schemas; no model-family inference. */
export function validateArchitecture(value: unknown, context: ArchitectureContext): S['ArchitectureResponse'] {
  const response = validateSchema('ArchitectureResponse', value);
  require(response.model_id === context.modelId, 'Architecture model mismatch');
  if (response.status === 'unavailable') {
    require(response.diagnostics.every((d) => !d.node_id && !d.parameter_id), 'Unavailable diagnostic target');
    require(!['restart_required', 'cache_unavailable'].includes(response.reason) || response.requires_restart, 'Restart required');
    return response;
  }
  const graph = response.graph;
  const nodes = unique(graph.nodes, (n) => n.id);
  const parameters = unique(graph.parameters, (p) => p.id);
  unique([...graph.nodes, ...graph.parameters, ...graph.edges, ...graph.repetitions], (r) => r.id);
  const symbols = unique(graph.symbols, (s) => s.name);
  const childSets = new Map(graph.nodes.filter((n) => n.kind === 'group').map((n) => [n.id, new Set(n.children)]));
  const ports = new Map(graph.nodes.map((n) => [n.id, unique(n.ports, (p) => p.id)]));
  const diagnostics = [...graph.diagnostics, ...response.diagnostics];
  for (const d of diagnostics) {
    require(!d.node_id || nodes.has(d.node_id), 'Diagnostic node closure');
    require(!d.parameter_id || parameters.has(d.parameter_id), 'Diagnostic parameter closure');
  }
  function shape(dims: S['ArchitectureShape']) {
    if (!dims) return;
    for (const d of dims) {
      if (d.kind === 'symbol') require(symbols.has(d.name), 'Undeclared shape symbol');
      if (d.kind === 'expression') require(d.symbols.every((s) => symbols.has(s)), 'Undeclared expression symbol');
    }
    if (dims.every((d) => d.kind === 'constant')) product(dims.map((d) => d.value));
  }
  for (const n of nodes.values()) {
    if (n.parent_id) {
      const parent = nodes.get(n.parent_id);
      require(parent?.kind === 'group' && childSets.get(parent.id)!.has(n.id), 'Parent/child disagreement');
    }
    if (n.kind === 'group') {
      require(new Set(n.children).size === n.children.length, 'Duplicate child');
      require(n.children.every((id) => nodes.get(id)?.parent_id === n.id), 'Child/parent disagreement');
    }
    require(n.parameter_ids.every((id) => parameters.has(id)), 'Parameter closure');
    for (const ref of n.references) {
      if (ref.kind === 'parameter') require(parameters.has(ref.parameter_id), 'Reference closure');
      if (ref.kind === 'tokenizer') require(context.tokenizerAvailable, 'Missing tokenizer capability');
    }
    n.ports.forEach((p) => shape(p.shape));
  }
  function terminate(links: Map<string, string | undefined>) {
    const done = new Set<string>();
    for (const start of links.keys()) {
      const path = new Set<string>();
      let key: string | undefined = start;
      while (key !== undefined && !done.has(key)) {
        require(links.has(key) && !path.has(key), 'Unresolved or cyclic architecture link');
        path.add(key); key = links.get(key);
      }
      for (const id of path) done.add(id);
    }
  }
  terminate(new Map(graph.nodes.map((n) => [n.id, n.parent_id])));
  terminate(new Map(graph.parameters.map((p) => [p.id, p.binding === 'alias' ? p.alias_of : undefined])));
  for (const repetition of graph.repetitions) {
    const parent = nodes.get(repetition.parent_id);
    require(parent?.kind === 'group', 'Repetition parent');
    unique(repetition.instances, (i) => i.node_id);
    const positions = new Map(parent.children.map((id, i) => [id, i]));
    let previousIndex = -1, previousPosition = -1;
    for (const instance of repetition.instances) {
      const node = nodes.get(instance.node_id);
      const position = positions.get(instance.node_id);
      require(node?.kind === 'group' && node.parent_id === parent.id && position !== undefined, 'Repetition membership');
      require(instance.index > previousIndex && position > previousPosition, 'Repetition order');
      previousIndex = instance.index; previousPosition = position;
    }
  }
  for (const edge of graph.edges) {
    const source = ports.get(edge.source.node_id)?.get(edge.source.port_id);
    const target = ports.get(edge.target.node_id)?.get(edge.target.port_id);
    require(source && target, 'Edge endpoint closure');
    const sn = nodes.get(edge.source.node_id)!, tn = nodes.get(edge.target.node_id)!;
    require(source.direction === 'output' && target.direction === 'input' ||
      sn.kind === 'group' && tn.parent_id === sn.id && source.direction === 'input' && target.direction === 'input' ||
      tn.kind === 'group' && sn.parent_id === tn.id && source.direction === 'output' && target.direction === 'output', 'Edge directions');
    require(sn.parent_id === tn.parent_id || tn.parent_id === sn.id || sn.parent_id === tn.id, 'Edge bypasses boundary');
    const a = source.shape, b = target.shape;
    const mismatch = a && b && (a.length !== b.length || a.some((d, i) => d.kind === 'constant' && b[i]?.kind === 'constant' && d.value !== b[i].value));
    require(!mismatch || diagnostics.some((d) => d.node_id === sn.id || d.node_id === tn.id), 'Undiagnosed dimension mismatch');
  }
  const inventory = unique(context.inventory.tensors, (t) => t.id);
  const nativeBindings = new Map<string, S['ArchitectureParameter']>();
  for (const p of parameters.values()) {
    shape(p.logical_shape);
    p.storage.forEach((s) => product(s.shape));
    if (p.binding === 'fused_region') require(p.storage.some((s) => s.name === p.region.storage_name), 'Region storage closure');
    if (p.inspection.status !== 'available') continue;
    require(p.binding === 'native' || p.binding === 'alias', 'Non-native inspection');
    const dims = p.logical_shape;
    require(dims && [1, 2].includes(dims.length) && dims.every((d) => d.kind === 'constant'), 'Inspection geometry');
    const geometry = dims.map((d) => { require(d.kind === 'constant', 'Inspection dimension'); return d.value; });
    let native: S['ArchitectureParameter'] = p;
    const path: string[] = [];
    while (native.binding === 'alias') {
      path.push(native.id);
      native = nativeBindings.get(native.id) ?? parameters.get(native.alias_of)!;
    }
    path.forEach((id) => nativeBindings.set(id, native));
    require(native.binding === 'native' && JSON.stringify(native.logical_shape) === JSON.stringify(dims) &&
      native.inspection.status === 'available' && native.inspection.tensor_id === p.inspection.tensor_id, 'Alias identity/geometry');
    require(native.storage.some((s) => s.name === native.name && JSON.stringify(s.shape) === JSON.stringify(geometry) &&
      ['F32', 'F16', 'BF16', 'float32', 'float16', 'bfloat16'].includes(s.dtype) && !['scales', 'packed', 'packed_data'].includes(s.role ?? '')), 'Native storage geometry');
    const tensor = inventory.get(p.inspection.tensor_id);
    require(tensor && JSON.stringify(tensor.shape) === JSON.stringify(geometry) && tensor.name === native.name &&
      tensor.rank === geometry.length && tensor.numel === product(geometry), 'Inventory membership/geometry');
    require(native.storage.some((s) => s.name === tensor.name && s.dtype === tensor.storage_dtype), 'Inventory storage identity');
  }
  return response;
}
