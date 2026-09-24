import { ApiClient } from '../api/client';
import { ApiFailure } from '../api/errors';
import type { components } from '../api/generated/types';
import { Lifetime } from './lifetime';

type Schemas = components['schemas'];
export type TensorDescriptor = Schemas['TensorDescriptor'];
export type ModelSummary = Schemas['ModelSummary'];
export type Session = Schemas['Session'];
export type Explorer = 'Tensor Explorer' | 'Tokenizer Explorer' | 'Architecture Explorer';
export type ViewStatus = 'idle' | 'loading' | 'streaming' | 'complete' | 'cancelled' | 'failed' | 'expired-session';
export type SessionStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;
export const sessionStorageKey = (backend: string) => `llm-model-explorer:session:${backend}`;

export interface ShellState {
  models: ModelSummary[];
  catalogueDiagnostics: Schemas['CatalogueDiagnostic'][];
  catalogue: 'loading' | 'complete' | 'failed';
  session: Session | null;
  sessionStatus: 'idle' | 'loading' | 'ready' | 'closing' | 'failed' | 'expired-session';
  message: string;
  tensors: TensorDescriptor[];
  inventoryCoverage: 'complete' | 'partial';
  inventoryDiagnostics: Schemas['TensorInventory']['diagnostics'];
  inventory: 'idle' | 'loading' | 'complete' | 'failed';
  selected: TensorDescriptor | null;
  explorer: Explorer;
  view: Lifetime;
  viewRevision: number;
  viewStatus: ViewStatus;
  storageAvailable: boolean;
}

export function isExpired(error: unknown) {
  return error instanceof ApiFailure && error.detail?.code === 'session_not_found';
}

// Do not render arbitrary backend messages/details (which may contain private paths).
function failureMessage(error: unknown, action: string) {
  if (error instanceof ApiFailure && error.kind === 'protocol') return `${action}: the backend returned an invalid response. Retry.`;
  if (error instanceof ApiFailure && error.detail?.code === 'model_content_changed') return 'Model content changed. Close this session and start a fresh session.';
  return `${action}: the backend request failed. Check the connection and retry.`;
}

/** One instance per mounted backend in one tab. No global current model/session. */
export class SessionController {
  private state: ShellState = {
    models: [], catalogueDiagnostics: [], catalogue: 'loading', session: null, sessionStatus: 'idle', message: '',
    tensors: [], inventoryCoverage: 'complete', inventoryDiagnostics: [], inventory: 'idle', selected: null, explorer: 'Tensor Explorer',
    view: new Lifetime(), viewRevision: 0, viewStatus: 'idle', storageAvailable: true,
  };
  private readonly listeners = new Set<() => void>();
  private catalogueRequest = new Lifetime();
  private sessionRequest = new Lifetime();
  private inventoryRequest = new Lifetime();
  private active = false;
  retrySession: () => void = () => {};
  constructor(readonly client: ApiClient, readonly backend: string, private readonly storage: SessionStorage | null) {}
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  getSnapshot = () => this.state;
  private update(patch: Partial<ShellState>) {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener();
  }
  private remember(id: string | null) {
    try {
      if (!this.storage) throw new Error('Storage unavailable');
      if (id) this.storage.setItem(sessionStorageKey(this.backend), id);
      else this.storage.removeItem(sessionStorageKey(this.backend));
    } catch { this.update({ storageAvailable: false }); }
  }
  private replaceView(patch: Partial<ShellState> = {}) {
    this.state.view.dispose();
    this.update({ view: new Lifetime(), viewRevision: this.state.viewRevision + 1, viewStatus: 'idle', ...patch });
  }
  start = () => {
    this.active = true;
    this.loadModels();
    let id: string | null = null;
    try {
      if (!this.storage) throw new Error('Storage unavailable');
      id = this.storage.getItem(sessionStorageKey(this.backend));
    } catch { this.update({ storageAvailable: false }); }
    if (id) this.recover(id);
  };
  dispose = () => {
    this.active = false;
    this.catalogueRequest.dispose();
    this.sessionRequest.dispose();
    this.inventoryRequest.dispose();
    this.state.view.dispose();
  };
  loadModels = () => {
    this.catalogueRequest.dispose();
    const request = this.catalogueRequest = new Lifetime();
    this.update({ catalogue: 'loading' });
    void this.client.listModels(request.signal).then(request.guard(({ models, diagnostics }) => {
      this.update({ catalogue: 'complete', models, catalogueDiagnostics: diagnostics });
    }), request.guard(() => this.update({ catalogue: 'failed', models: [], catalogueDiagnostics: [] })));
  };
  private beginSession() {
    this.sessionRequest.dispose();
    this.inventoryRequest.dispose();
    const request = this.sessionRequest = new Lifetime();
    this.replaceView({ session: null, sessionStatus: 'loading', selected: null, tensors: [], inventoryCoverage: 'complete', inventoryDiagnostics: [], inventory: 'idle', message: '' });
    return request;
  }
  private acceptSession(session: Session) {
    this.remember(session.id);
    this.replaceView({ session, sessionStatus: 'ready', message: '' });
    this.loadInventory();
  }
  private sessionFailure(error: unknown) {
    if (isExpired(error)) this.expire();
    else this.update({ sessionStatus: 'failed', message: failureMessage(error, 'Could not open session') });
  }
  recover = (id?: string) => {
    let stored = id;
    if (!stored) {
      try { stored = this.storage?.getItem(sessionStorageKey(this.backend)) ?? undefined; } catch { /* Create remains available. */ }
    }
    if (!stored) return;
    const sessionId = stored;
    this.retrySession = () => this.recover(sessionId);
    const request = this.beginSession();
    void this.client.getSession(stored, request.signal).then(request.guard((session) => this.acceptSession(session)), request.guard((error: unknown) => this.sessionFailure(error)));
  };
  chooseModel = (modelId: string) => {
    if (!this.state.models.some((model) => model.id === modelId)) return;
    this.retrySession = () => this.chooseModel(modelId);
    const request = this.beginSession();
    this.remember(null);
    // Let POST finish so a superseded creation's returned ID can be released.
    void this.client.createSession({ model_id: modelId }).then((session) => {
      if (!this.active || !request.isCurrent()) {
        void this.client.deleteSession(session.id).catch(() => {});
        return;
      }
      this.acceptSession(session);
    }, request.guard((error: unknown) => this.sessionFailure(error)));
  };
  loadInventory = () => {
    const session = this.state.session;
    if (!session || this.state.sessionStatus !== 'ready') return;
    this.inventoryRequest.dispose();
    const request = this.inventoryRequest = new Lifetime();
    this.replaceView({ selected: null, tensors: [], inventoryCoverage: 'complete', inventoryDiagnostics: [], inventory: 'loading', message: '' });
    void this.client.listTensors(session.id, request.signal).then(request.guard(({ tensors, coverage, diagnostics }) => {
      this.update({ tensors, inventoryCoverage: coverage, inventoryDiagnostics: diagnostics, inventory: 'complete' });
    }), request.guard((error: unknown) => {
      if (isExpired(error)) this.expire();
      else this.update({ inventory: 'failed', message: failureMessage(error, 'Could not load tensors') });
    }));
  };
  selectTensor = (tensor: TensorDescriptor) => {
    if (this.state.sessionStatus !== 'ready' || !this.state.tensors.includes(tensor)) return;
    this.replaceView({ selected: tensor });
  };
  switchExplorer = (explorer: Explorer) => {
    if (explorer !== this.state.explorer) this.replaceView({ explorer });
  };
  reportStatus = (view: Lifetime, status: ViewStatus) => {
    if (view !== this.state.view || !view.isCurrent()) return;
    if (status === 'expired-session') this.expire();
    else this.update({ viewStatus: status });
  };
  private expire() {
    this.sessionRequest.dispose();
    this.inventoryRequest.dispose();
    this.remember(null);
    this.replaceView({ session: null, sessionStatus: 'expired-session', selected: null, tensors: [], inventoryCoverage: 'complete', inventoryDiagnostics: [], inventory: 'idle',
      message: 'Session expired. Select a model to start a fresh session; runtime state was not restored.' });
  }
  closeSession = () => {
    const session = this.state.session;
    if (!session || this.state.sessionStatus === 'closing') return;
    this.sessionRequest.dispose();
    this.inventoryRequest.dispose();
    const request = this.sessionRequest = new Lifetime();
    this.replaceView({ sessionStatus: 'closing', selected: null, message: '' });
    const finish = () => {
      this.remember(null);
      this.update({ session: null, sessionStatus: 'idle', inventory: 'idle', tensors: [], message: 'Session closed.' });
    };
    void this.client.deleteSession(session.id).then(request.guard(finish), request.guard((error: unknown) => {
      if (isExpired(error)) finish();
      else {
        this.update({ sessionStatus: 'ready', message: failureMessage(error, 'Could not close session') });
        if (this.state.inventory === 'loading') this.loadInventory();
      }
    }));
  };
}
