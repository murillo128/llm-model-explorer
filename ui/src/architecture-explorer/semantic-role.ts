import type { GraphNode } from './graph';

/** Optional architectural annotation; never infer an edge or an instance identity. */
export function semanticRole(node: GraphNode | undefined): string | undefined {
  const value = node?.attributes.find((attribute) => attribute.name === 'semantic_role')?.value;
  return typeof value === 'string' ? value : undefined;
}
