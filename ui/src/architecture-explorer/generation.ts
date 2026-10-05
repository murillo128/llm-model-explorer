import type { Graph, GraphNode } from './graph';
import type { Endpoint, ProjectedEdge, Projection } from './projection';

export const isGeneration = (node: GraphNode | undefined) => node?.kind === 'group' &&
  node.attributes.some((a) => a.name === 'semantic_role' && a.value === 'autoregressive_generation');

/** This consumes the backend-validated finite pattern; labels are never selectors. */
export function generationPatterns(graph: Graph) {
  const records = new Map(graph.nodes.map((n) => [n.id, n]));
  return graph.nodes.filter(isGeneration).map((owner) => {
    if (owner.kind !== 'group') throw new Error('Invalid generation owner.');
    const children = owner.children.map((id) => records.get(id)!);
    const operation = (name: string) => children.find((n) => n.operation === name)!;
    return { owner, state: operation('generation_sequence_state'), prepare: operation('generation_prepare_inputs'),
      select: operation('generation_greedy_next_token'), append: operation('generation_append_token'),
      model: children.find((n) => n.kind === 'group')! };
  });
}

/** State-phase correspondence is not a continuous same-step source path. Each
 * real segment stays separate, with its state record and phase in inspection. */
export function projectGeneration(graph: Graph, base: Projection): Projection {
  for (const pattern of generationPatterns(graph)) {
    const { owner, state, prepare, append } = pattern;
    if (!base.nodes.some((n) => n.id === owner.id && n.expanded) ||
      !base.nodes.some((n) => n.id === state.id)) continue;
    const incident = base.edges.filter((e) => e.source.node_id === state.id || e.target.node_id === state.id);
    const incoming = (port: string) => incident.find((e) => e.target.node_id === state.id && e.target.port_id === port)!;
    const outgoing = (port: string) => incident.find((e) => e.source.node_id === state.id && e.source.port_id === port)!;
    const initial = incoming('initial'), current = outgoing('current'), next = incoming('next'), final = outgoing('final');
    if (!initial || !current || !next || !final) throw new Error('Generation phase interfaces are unavailable.');
    const ep = (node_id: string, port_id: string): Endpoint => ({ node_id, port_id });
    const relationship = (kind: 'entry' | 'return' | 'exit', source: Endpoint, target: Endpoint,
      segments: ProjectedEdge[], phase: 'initial' | 'next' | 'final'): ProjectedEdge => ({
      id: `generation:${owner.id}:${phase}`, source, target, kind: 'state',
      paths: segments.flatMap((e) => e.paths), originalEdgeIds: [...new Set(segments.flatMap((e) => e.originalEdgeIds))],
      relationship: { owner: 'generation', kind, groupId: owner.id, stateId: state.id, phase },
    });
    base.nodes = base.nodes.filter((n) => n.id !== state.id);
    base.edges = base.edges.filter((e) => !incident.includes(e));
    base.edges.push(
      relationship('entry', initial.source, ep(prepare.id, 'sequence'), [initial, current], 'initial'),
      relationship('return', ep(append.id, 'updated'), ep(prepare.id, 'sequence'), [next, current], 'next'),
      relationship('exit', ep(append.id, 'updated'), final.target, [next, final], 'final'),
    );
  }
  const represented = new Set(base.edges.flatMap((e) => e.originalEdgeIds));
  return { ...base, hiddenEdgeIds: base.hiddenEdgeIds.filter((id) => !represented.has(id)) };
}
