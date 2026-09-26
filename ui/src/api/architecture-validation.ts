import type { components } from './generated/types';
import { requireProtocol } from './errors';
import { validateSchema } from './validation';
import { expandCompactGraph } from './compact-architecture';

type S = components['schemas'];
export interface ArchitectureContext {
  modelId: string;
}

/** Check the transport boundary; graph semantics are validated by the backend. */
export function validateArchitecture(value: unknown, context: ArchitectureContext): S['ArchitectureResponse'] {
  const response = validateSchema('ArchitectureResponse', value);
  requireProtocol(response.model_id === context.modelId, 'Architecture model mismatch');
  if (response.status === 'unavailable') {
    requireProtocol(response.diagnostics.every((d) => !d.node_id && !d.parameter_id), 'Unavailable diagnostic target');
    requireProtocol(!['restart_required', 'cache_unavailable'].includes(response.reason) || response.requires_restart, 'Restart required');
    return response;
  }
  return { ...response, graph: expandCompactGraph(response.graph) };
}
