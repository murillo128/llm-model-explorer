import { describe, expect, it } from 'vitest';
import fixtures from '../../../api/fixtures/architecture.json';
import { validateArchitecture } from './architecture-validation';
import { validateSchema } from './validation';

function edit(value: unknown, edits: { path: string[]; value?: unknown; delete?: boolean }[]) {
  const result = structuredClone(value);
  for (const change of edits) {
    let parent = result as Record<string, unknown>;
    for (const key of change.path.slice(0, -1)) parent = parent[key] as Record<string, unknown>;
    if (change.delete) delete parent[change.path.at(-1)!];
    else parent[change.path.at(-1)!] = structuredClone(change.value);
  }
  return result;
}
describe('UI contextual validation against the independently published oracle cases', () => {
  for (const test of fixtures.cases) it(test.name, () => {
    const context = edit(fixtures.context, 'context_edits' in test ? test.context_edits : []) as typeof fixtures.context;
    const base = 'base' in test ? fixtures[test.base as 'template_response' | 'compact_response'] : fixtures.response;
    const check = () => validateArchitecture(edit(base, test.edits), {
      modelId: context.session.model_id, inventory: validateSchema('TensorInventory', context.inventory), tokenizerAvailable: context.tokenizer_available,
    });
    if (test.valid) expect(check).not.toThrow(); else expect(check).toThrow();
  });

  it('rejects malformed SmolLM2 NF4 packed storage metadata', () => {
    const test = fixtures.cases.find((entry) => entry.name === 'bnb-nf4-dq-complete-logical-inspection');
    expect(test).toBeDefined();
    const context = edit(fixtures.context, test!.context_edits) as typeof fixtures.context;
    const response = edit(fixtures.response, test!.edits);
    const malformed = edit(response, [{ path: ['graph', 'parameters', '2', 'storage', '0', 'shape'], value: [1, 1] }]);
    expect(() => validateArchitecture(malformed, {
      modelId: context.session.model_id, inventory: validateSchema('TensorInventory', context.inventory),
      tokenizerAvailable: context.tokenizer_available,
    })).toThrow();
  });
});
