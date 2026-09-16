import type { ProjectedNode } from './projection';

/** Presentation selection is local view state, never a fabricated source ID. */
export const cardSelection = (node: ProjectedNode) => node.record?.id ?? node.id;

export const cardDoubleClick = (node: ProjectedNode) =>
  node.kind === 'group' && !node.expanded ? 'expand' : 'inspect';

export function cardNavigation(node: ProjectedNode, isolated: boolean, structureOnly: boolean) {
  const action = isolated ? 'View in model' : 'Explore component';
  const target = node.presentation === 'mlp' ? node.id
    : node.record && ['group', 'operation'].includes(node.record.kind) ? node.record.id : undefined;
  const reason = structureOnly ? 'Choose a concrete instance to view in model.'
    : !target ? 'This presentation has no component navigation target.' : undefined;
  return { action, target: reason ? undefined : target, reason };
}
