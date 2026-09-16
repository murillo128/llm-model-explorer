import type { ProjectedNode } from './projection';

/** Presentation selection is local view state, never a fabricated source ID. */
export const cardSelection = (node: ProjectedNode) => node.record?.id ?? node.id;

export const cardExpandable = (node: ProjectedNode) => node.kind === 'group' && Boolean(
  node.record?.kind === 'group' && node.record.children.length ||
  node.presentation === 'mlp' && node.sourceIds.length || node.repetitionId && node.instances?.length);

export const cardDoubleClick = (node: ProjectedNode) =>
  cardExpandable(node) ? node.expanded ? 'collapse' : 'expand' : undefined;

/** activeRoot uses concrete source identity, including a rebound shared instance. */
export function cardNavigation(node: ProjectedNode, activeRoot: string | undefined, structureOnly: boolean) {
  const target = node.presentation === 'mlp' ? node.id
    : node.record && ['group', 'operation'].includes(node.record.kind) ? node.record.id : undefined;
  const action = target && target === activeRoot ? 'View in model' : 'Explore component';
  const reason = structureOnly ? 'Choose a concrete instance to view in model.'
    : !target ? 'This presentation has no component navigation target.' : undefined;
  return { action, target: reason ? undefined : target, reason };
}
