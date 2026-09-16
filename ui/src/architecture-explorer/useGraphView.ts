import { useSyncExternalStore } from 'react';
import type { GraphView } from './graph';

/** Subscribe both representations to the same current scope, with stable layout inputs. */
export function useGraphView(view: GraphView) {
  useSyncExternalStore(view.subscribe, view.getRevision);
  return view.getProjectionOptions();
}
