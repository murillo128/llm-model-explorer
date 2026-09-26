import { describe, expect, it } from 'vitest';
import fixtures from '../../../api/fixtures/architecture.json';
import { validateArchitecture } from './architecture-validation';

const context = {
  modelId: fixtures.context.session.model_id,
};

describe('architecture transport boundary', () => {
  it('accepts the published schema-valid response', () => {
    expect(validateArchitecture(fixtures.response, context)).toEqual(fixtures.response);
  });
  it('rejects a response for another model', () => {
    expect(() => validateArchitecture({ ...fixtures.response, model_id: 'other' }, context))
      .toThrow('Architecture model mismatch');
  });
  it('rejects malformed response records through the generated schema', () => {
    expect(() => validateArchitecture({ ...fixtures.response, graph: null }, context))
      .toThrow('Invalid ArchitectureResponse schema');
  });
  it('requires restart for a missing prepared artifact', () => {
    const unavailable = { status: 'unavailable', model_id: context.modelId, reason: 'cache_unavailable', requires_restart: false, diagnostics: [] };
    expect(() => validateArchitecture(unavailable, context)).toThrow('Restart required');
  });

  it('reconstructs compact expert identities before navigation', () => {
    const response = validateArchitecture(fixtures.compact_response, context);
    expect(response.status).toBe('available');
    if (response.status !== 'available') throw new Error('Expected an available graph');
    expect(response.graph.compact_components).toBeUndefined();
    expect(response.graph.nodes.map((node) => node.id)).toEqual(['root', 'expert0', 'expert1']);
    expect(fixtures.compact_response.graph.nodes.map((node) => node.id)).toEqual(['root']);
  });
});
