/** Fixture-specific independent oracle for the accepted split Attention boundary.
 * Keep ordered paths and multiplicity; an unordered edge-ID union is insufficient. */
import assert from 'node:assert/strict';
import type { Graph } from '../src/architecture-explorer/graph';
import type { Endpoint } from '../src/architecture-explorer/projection';

export interface BoundarySegment { source: Endpoint; target: Endpoint; paths: string[][] }
const ep = (node_id: string, port_id: string): Endpoint => ({ node_id, port_id });
export const attentionInput = ep('layer-3.attention', 'x');
export const normalizationAlias = ep('external:output:["layer-3.input-norm","out"]', 'output:["layer-3.input-norm","out"]');
const same = (a: Endpoint, b: Endpoint) => a.node_id === b.node_id && a.port_id === b.port_id;

export function assertAttentionInputProvenance(graph: Graph, segments: BoundarySegment[]) {
  const incoming = segments.filter((s) => same(s.target, attentionInput));
  const outgoing = segments.filter((s) => same(s.source, attentionInput));
  assert.equal(incoming.length, 1, 'Exactly one external producer must reach Attention x');
  assert.deepEqual(incoming[0]!.source, normalizationAlias, 'Do not substitute an unrelated external signal');
  assert.equal(outgoing.length, 3, 'Preserve all three fan-out branches without duplicates');
  const expectedEdge = (source: Endpoint, target: Endpoint) => {
    const matches = graph.edges.filter((e) => same(e.source, source) && same(e.target, target));
    assert.equal(matches.length, 1, 'Authored source segment must be unambiguous');
    return matches[0]!.id;
  };
  const prefix = expectedEdge(ep('layer-3.input-norm', 'out'), attentionInput);
  assert.deepEqual(incoming[0]!.paths, [[prefix]]);
  const complete: string[][] = [];
  const expected: string[][] = [];
  for (const name of ['Q', 'K', 'V']) {
    const target = ep(`layer-3.attention.${name}`, 'x');
    const branches = outgoing.filter((s) => same(s.target, target));
    assert.equal(branches.length, 1, `Missing or duplicate ${name} endpoint`);
    const suffix = expectedEdge(attentionInput, target);
    assert.deepEqual(branches[0]!.paths, [[suffix]], 'Internal evidence must name the exact source edge');
    for (const before of incoming[0]!.paths) for (const after of branches[0]!.paths) complete.push([...before, ...after]);
    expected.push([prefix, suffix]);
  }
  assert.deepEqual(complete, expected, 'Complete ordered paths and their multiplicity must survive the split');
}
