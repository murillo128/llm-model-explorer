import { describe, expect, it } from 'vitest';
import fixture from '../../../api/fixtures/architecture.json';
import type { components } from './generated/types';
import { validateResponse, validateSchema } from './validation';

type Edit = { path: string[]; value?: unknown; delete?: boolean };
function applyEdits(value: unknown, edits: Edit[]): unknown {
  const result: unknown = structuredClone(value);
  for (const edit of edits) {
    let target = result as Record<string, unknown>;
    for (const key of edit.path.slice(0, -1)) target = target[key] as Record<string, unknown>;
    const key = edit.path.at(-1)!;
    if (edit.delete) delete target[key];
    else target[key] = structuredClone(edit.value);
  }
  return result;
}

describe('generated architecture structure (contextual semantics belong to conformance oracles)', () => {
  for (const test of fixture.cases) it(test.name, () => {
    const base = 'base' in test ? fixture[test.base as 'template_response' | 'compact_response'] : fixture.response;
    const value = applyEdits(base, test.edits);
    const validate = () => {
      validateSchema('ArchitectureResponse', value);
      validateResponse('getArchitecture', value);
    };
    if (test.schema_valid) expect(validate).not.toThrow();
    else expect(validate).toThrow();
  });
});

it('keeps generated discriminants usable without a handwritten graph contract', () => {
  const response: components['schemas']['ArchitectureResponse'] = validateSchema('ArchitectureResponse', fixture.response);
  if (response.status === 'available') {
    const graph = response.graph;
    for (const node of graph.nodes) {
      if (node.kind === 'group') expect(node.children).toBeDefined();
    }
    for (const parameter of graph.parameters) {
      if (parameter.binding === 'alias') expect(parameter.alias_of).toBeTypeOf('string');
      if (parameter.inspection.status === 'available') expect(parameter.inspection.tensor_id).toBeTypeOf('string');
    }
  }
});

describe('generated inventory capability envelope', () => {
  for (const test of fixture.inventory_cases) it(test.name, () => {
    const validate = () => validateSchema('TensorInventory', test.value);
    if (test.valid) expect(validate).not.toThrow();
    else expect(validate).toThrow();
  });
});

it('rejects non-finite architecture attribute numbers', () => {
  for (const value of [NaN, Infinity, -Infinity]) {
    const response = applyEdits(fixture.response, [{ path: ['graph', 'nodes', '5', 'attributes', '0', 'value'], value }]);
    expect(() => validateSchema('ArchitectureResponse', response)).toThrow();
  }
});
