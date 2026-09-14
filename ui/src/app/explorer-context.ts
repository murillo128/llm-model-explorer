import type { ArchitectureSelection } from '../architecture-explorer/ArchitectureCanvas';
import { createContext, useContext } from 'react';
import type { ComponentType } from 'react';
import type { ApiClient } from '../api/client';
import type { Lifetime } from './lifetime';
import type { Session, TensorDescriptor, ViewStatus } from './session-controller';

export interface ExplorerContextValue {
  client: ApiClient;
  session: Session;
  sessionId: string;
  selectedTensor: TensorDescriptor | null;
  /** Disposed synchronously on selection, tool, session, or backend changes. */
  selection: Lifetime;
  reportStatus: (status: ViewStatus) => void;
}
export interface ExplorerSlots {
  tensor?: ComponentType<ExplorerContextValue>;
  tokenizer?: ComponentType<ExplorerContextValue>;
  architecture?: ComponentType<ExplorerContextValue>;
  inspectArchitecture?: (selection: ArchitectureSelection) => void;
}
export const ExplorerContext = createContext<ExplorerContextValue | null>(null);
export function useExplorerContext() {
  const context = useContext(ExplorerContext);
  if (!context) throw new Error('Explorer must be composed inside an active session');
  return context;
}
