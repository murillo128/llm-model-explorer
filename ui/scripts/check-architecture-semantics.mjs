/** Compare producer exports with the accepted semantic fingerprints: pre-grouping
 * topology/bindings and issue #269's reviewed canonical operation-card formulas.
 * Run with Node 24 after backend/tests/architecture_grouping_cases.py.
 * Uses issue #119's independent multiset oracle, never production projection.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { semanticSnapshot } from '../tests/architecture-invariants.ts';

export function fingerprints(directory, caseNames) {
  return Object.fromEntries(caseNames.map((name) => {
    const { graph, names } = JSON.parse(readFileSync(`${directory}/${name}.json`, 'utf8'));
    // Grouping adds this architectural annotation. Every computational attribute,
    // accepted formula, binding and endpoint stays in the oracle.
    graph.nodes.forEach((node) => { node.attributes = node.attributes.filter((a) => a.name !== 'semantic_role'); });
    const snapshot = semanticSnapshot(graph, names);
    return [name, Object.fromEntries(Object.entries(snapshot).map(([section, value]) => [section,
      createHash('sha256').update(JSON.stringify(value)).digest('hex')]))];
  }));
}

if (process.argv[1]?.endsWith('/check-architecture-semantics.mjs')) {
  const expected = JSON.parse(readFileSync(new URL('../../backend/tests/fixtures/architecture-semantics-baseline.json', import.meta.url), 'utf8'));
  const actual = fingerprints(process.argv[2], Object.keys(expected.cases));
  assert.deepEqual(actual, expected.cases, 'Operation-level semantics changed from the accepted baseline');
  console.log(`PASS: ${Object.keys(actual).length} reviewed cases preserve ${expected.baseline} semantics with ${expected.formula_contract} formulas`);
}
