import type { Graph, GraphSnapshot, GraphView } from './graph';
import type { ProjectionOptions } from './projection';
import { componentScope } from './scope';
import { instanceOf } from './presentation';

export function projectionOptions(view: GraphSnapshot): ProjectionOptions {
  return { expanded: view.expanded, repetitions: view.repetitions, dimensions: view.dimensions,
    exhaustive: view.exhaustive, showUnused: view.showUnused, showContext: view.showContext, deriveMlp: view.deriveMlp,
    ...(view.stateScope ? { stateScope: view.stateScope } : {}), ...(view.scope ? { scope: view.scope } : {}) };
}

/** Copy only small view state. Neither source records nor geometry enter history. */
export function snapshotView(view: GraphSnapshot, viewport = view.viewport): GraphSnapshot {
  return { selected: view.selected, dimensions: view.dimensions, edge: view.edge, focus: view.focus,
    activeStack: view.activeStack, expanded: [...view.expanded],
    repetitions: Object.fromEntries(Object.entries(view.repetitions).map(([id, window]) => [id, { ...window }])),
    exhaustive: view.exhaustive, showUnused: view.showUnused, showContext: view.showContext,
    deriveMlp: view.deriveMlp, stateScope: view.stateScope, scope: view.scope, shared: view.shared ? { ...view.shared } : undefined,
    viewport: viewport ? { ...viewport } : undefined };
}

export function enterComponent(view: GraphView, graph: Graph, id: string, viewport = view.viewport) {
  const scope = componentScope(graph, id);
  if (view.scope === id) return;
  const previous = snapshotView(view, viewport);
  if (!view.scope) view.globalView = previous;
  view.history = [...view.history, previous].slice(-16);
  const expanded = view.expanded.filter((id) => scope.members.has(id) || id.startsWith('mlp:') && scope.members.has(id.slice(4)));
  const instance = instanceOf(graph, scope.derived?.parentId ?? id);
  // A concrete child leaves shared geometry; Back retains the verified shared snapshot.
  view.update({ scope: id, shared: undefined, selected: id, edge: null, focus: id, activeStack: instance?.repetition.id ?? null,
    expanded: [...new Set([...expanded, id])], exhaustive: false, stateScope: undefined,
    deriveMlp: scope.derived ? true : view.deriveMlp, viewport: undefined });
}

export function backFromComponent(view: GraphView) {
  const previous = view.history.pop() ?? view.globalView;
  if (!previous) return;
  view.update(snapshotView(previous));
  if (!view.scope) { view.history = []; view.globalView = undefined; }
}

export function returnToModel(view: GraphView) {
  if (view.globalView) view.update(snapshotView(view.globalView));
  view.scope = undefined; view.shared = undefined; view.history = []; view.globalView = undefined;
}

export function expandComponent(view: GraphView, graph: Graph): ProjectionOptions {
  if (!view.scope) return projectionOptions(view);
  const scope = componentScope(graph, view.scope);
  // All concrete instances inside this component; never expand outside it.
  return { ...projectionOptions(view), expanded: [...scope.members, scope.id], repetitions: {},
    deriveMlp: Boolean(scope.derived), exhaustive: false, stateScope: undefined, showUnused: true, showContext: true };
}
