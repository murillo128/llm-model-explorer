import { ApiClient } from '../api/client';
import { ApiFailure } from '../api/errors';
import type { components } from '../api/generated/types';
import { ModelDiagnostics, finding } from './model-diagnostics';
import { Feedback } from './feedback';
import type { FeedbackState } from './feedback';
import { Lifetime } from './lifetime';
import type { ModelState, ObservationStatus } from '../api/model-events';
import { RefreshPresentation } from './refresh-presentation';

type Schemas = components['schemas'];
export type TensorDescriptor = Schemas['TensorDescriptor'];
export type ModelSummary = Schemas['ModelSummary'];
export type Session = Schemas['Session'];
export type Explorer = 'Tensor Explorer' | 'Tokenizer Explorer' | 'Architecture Explorer';
export type ViewStatus = 'idle' | 'loading' | 'streaming' | 'complete' | 'cancelled' | 'failed' | 'expired-session';
export type SessionStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;
export const sessionStorageKey = (backend: string) => `llm-model-explorer:session:${backend}`;

export interface ShellState extends FeedbackState {
  selectedModelId: string | null;
  modelGeneration: number;
  refreshAttempt: number;
  refreshStatus: 'idle' | 'updating' | 'waiting';
  observation: ObservationStatus | 'idle';
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
    selectedModelId: null, modelGeneration: 0, refreshAttempt: 0, refreshStatus: 'idle', observation: 'idle',
    connection: 'connecting', toasts: [],
    models: [], catalogueDiagnostics: [], catalogue: 'loading', session: null, sessionStatus: 'idle', message: '',
    tensors: [], inventoryCoverage: 'complete', inventoryDiagnostics: [], inventory: 'idle', selected: null, explorer: 'Tensor Explorer',
    view: new Lifetime(), viewRevision: 0, viewStatus: 'idle', storageAvailable: true,
  };
  readonly diagnostics = new ModelDiagnostics();
  readonly feedback = new Feedback((state) => this.update(state));
  readonly presentation = new RefreshPresentation();
  explorerClient = (view: Lifetime) => {
    const feedback = this.feedback.observe(view, true);
    return this.client.observe(activity => {
      feedback(activity);
      if (view === this.state.view && view.isCurrent() && activity.failure) this.contentFailure(activity.failure);
    });
  };
  private subscription: (() => void) | undefined;
  private subscriptionId = 0;
  private backendEpoch: string | undefined;
  private latest: ModelState | undefined;
  private candidate: Session | undefined;
  private refreshBusy = false;
  private refreshQueued = false;
  private retryTimer: ReturnType<typeof setTimeout> | undefined;
  private retryCount = 0;
  private invalidRevision: string | undefined;
  private tensorBookmark: { id: string; name: string } | undefined;
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
    this.stopRefresh();
    this.feedback.reset();
    this.catalogueRequest.dispose();
    this.sessionRequest.dispose();
    this.inventoryRequest.dispose();
    this.state.view.dispose();
  };
  private stopSubscription() { this.subscriptionId++; this.subscription?.(); this.subscription = undefined; }
  private cancelRetry() { clearTimeout(this.retryTimer); this.retryTimer = undefined; }
  private stopRefresh() {
    this.stopSubscription(); this.cancelRetry(); this.refreshQueued = false;
    this.latest = undefined; this.backendEpoch = undefined; this.retryCount = 0;
    this.invalidRevision = undefined;
    this.tensorBookmark = undefined; this.presentation.clear();
    if (this.candidate) { this.retire(this.candidate); this.candidate = undefined; }
  }
  private retire(session: Session) {
    // One bounded retry, no retained queue of retired IDs. Never delete a shared artifact.
    void this.client.deleteSession(session.id).catch(async error => {
      if (isExpired(error)) return;
      try { await this.client.deleteSession(session.id); }
      catch { if (this.active) this.feedback.notify('Could not release an old session. Its local resources may remain until the backend restarts.', 'error'); }
    });
  }
  retryObservation = () => { this.observeModel(); };
  private observeModel() {
    this.stopSubscription();
    const modelId = this.state.selectedModelId;
    if (!modelId || !this.active) return;
    const identity = this.subscriptionId;
    this.update({ observation: 'connecting' });
    this.subscription = this.client.watchModel(modelId, {
      status: observation => { if (identity === this.subscriptionId && this.active) this.update({ observation }); },
      state: event => { if (identity === this.subscriptionId && this.active) this.observeState(event); },
    });
  }
  private observeState(event: ModelState) {
    const epochChanged = this.backendEpoch !== undefined && event.epoch !== this.backendEpoch;
    const changed = epochChanged || event.status !== this.latest?.status || event.model_revision !== this.latest?.model_revision;
    this.backendEpoch = event.epoch; this.latest = event;
    if (changed) { this.cancelRetry(); this.retryCount = 0; if (epochChanged || event.model_revision !== this.invalidRevision) this.invalidRevision = undefined; }
    if (event.status === 'unavailable') {
      this.sessionRequest.dispose();
      if (this.candidate) { this.retire(this.candidate); this.candidate = undefined; }
      this.refreshQueued = false;
      this.invalidate('waiting');
      return;
    }
    if (this.candidate) {
      const candidate = this.candidate; this.candidate = undefined;
      if (candidate.model_revision === event.model_revision) {
        this.acceptSession(candidate, true); return;
      }
      this.retire(candidate);
      this.refreshModel(); return;
    }
    if (!epochChanged && this.state.session?.model_revision === event.model_revision) return;
    if (this.refreshBusy) { if (changed) this.refreshQueued = true; return; }
    // Stable failed admission waits for a new state or explicit Retry.
    if (this.state.refreshStatus === 'waiting' && !changed) return;
    this.refreshModel();
  }
  private contentFailure(error: unknown): boolean {
    if (!this.state.selectedModelId || !(error instanceof ApiFailure) ||
      !['model_content_changed', 'session_not_found'].includes(error.detail?.code ?? '')) return false;
    const revision = this.state.session?.model_revision;
    if (revision && revision === this.invalidRevision) { this.invalidate('waiting'); return true; }
    this.invalidRevision = revision;
    this.refreshModel(); return true;
  }
  private invalidate(status: 'updating' | 'waiting') {
    this.retrySession = this.retryRefresh;
    if (this.state.session) {
      const session = this.state.session;
      this.tensorBookmark = this.state.selected ? { id: this.state.selected.id, name: this.state.selected.name } : this.tensorBookmark;
      this.presentation.save({ backend: this.backend, modelId: session.model_id, attempt: this.state.refreshAttempt + 1 });
      this.inventoryRequest.dispose(); this.diagnostics.activate(null); this.remember(null);
      this.replaceView({ session: null, selected: null, tensors: [], inventory: 'idle', inventoryDiagnostics: [], inventoryCoverage: 'complete' });
      this.retire(session);
    }
    this.update({ refreshStatus: status, sessionStatus: status === 'updating' ? 'loading' : 'failed',
      message: status === 'waiting' ? 'Waiting for valid model. Repair the model or retry.' : 'Updating model…' });
  }
  retryRefresh = () => { this.retryCount = 0; this.invalidRevision = undefined; this.cancelRetry(); this.refreshModel(); };
  private refreshModel() {
    const modelId = this.state.selectedModelId;
    if (!this.active || !modelId) return;
    if (this.refreshBusy) { this.refreshQueued = true; return; }
    this.cancelRetry();
    if (this.candidate) { this.retire(this.candidate); this.candidate = undefined; }
    this.invalidate('updating');
    this.update({ refreshAttempt: this.state.refreshAttempt + 1 });
    this.presentation.save({ backend: this.backend, modelId, attempt: this.state.refreshAttempt });
    this.retrySession = this.retryRefresh;
    this.sessionRequest.dispose();
    const request = this.sessionRequest = new Lifetime();
    this.refreshBusy = true; this.refreshQueued = false;
    void this.feedback.track(request, () => this.client.createSession({ model_id: modelId })).then(session => {
      if (!this.active || !request.isCurrent() || this.state.selectedModelId !== modelId) { this.retire(session); return; }
      if (session.model_id !== modelId) { this.retire(session); this.invalidate('waiting'); return; }
      // POST pins a snapshot. A fresh subscription confirms current state; old
      // revision tokens are equality tokens and cannot order this snapshot.
      this.candidate = session;
      this.refreshQueued = false;
      this.observeModel();
    }, request.guard((error: unknown) => {
      this.invalidate('waiting');
      const transient = error instanceof ApiFailure && (error.kind === 'transport' || (error.kind === 'http' && (error.status ?? 0) >= 500));
      if (transient && this.retryCount < 3) {
        const delay = 1000 * 2 ** this.retryCount++;
        this.retryTimer = setTimeout(() => { this.retryTimer = undefined; this.refreshModel(); }, delay);
      }
    })).finally(() => {
      this.refreshBusy = false;
      if (this.refreshQueued && this.active) { this.refreshQueued = false; this.refreshModel(); }
    });
  }
  loadModels = () => {
    const session = this.state.session;
    this.catalogueRequest.dispose();
    const request = this.catalogueRequest = new Lifetime();
    this.update({ catalogue: 'loading' });
    void this.feedback.track(request, () => this.client.listModels(request.signal)).then(request.guard(({ models, diagnostics }) => {
      this.update({ catalogue: 'complete', models, catalogueDiagnostics: diagnostics });
    }), request.guard((error: unknown) => {
      this.update({ catalogue: 'failed' });
      if (session && this.state.session === session && error instanceof ApiFailure && error.kind !== 'transport' && error.kind !== 'cancelled') {
        this.feedback.notify(failureMessage(error, 'Could not refresh models'), 'error');
      }
    }));
  };
  private beginSession() {
    this.diagnostics.activate(null);
    this.feedback.reset();
    this.sessionRequest.dispose();
    this.inventoryRequest.dispose();
    const request = this.sessionRequest = new Lifetime();
    this.replaceView({ session: null, sessionStatus: 'loading', selected: null, tensors: [], inventoryCoverage: 'complete', inventoryDiagnostics: [], inventory: 'idle', message: '' });
    return request;
  }
  private acceptSession(session: Session, refreshed = false) {
    this.diagnostics.activate(session);
    this.remember(session.id);
    this.replaceView({ session, selectedModelId: session.model_id, sessionStatus: 'ready', refreshStatus: 'idle', message: '' });
    if (!refreshed) this.observeModel();
    this.loadInventory();
    if (refreshed) { this.loadModels(); this.feedback.notify('Model updated.', 'info'); }
  }
  private sessionFailure(error: unknown) {
    if (isExpired(error)) this.expire();
    else { this.update({ sessionStatus: 'failed', message: failureMessage(error, 'Could not open session') }); if (this.state.selectedModelId) this.observeModel(); }
  }
  recover = (id?: string) => {
    if (this.state.sessionStatus === 'loading' && this.sessionRequest.isCurrent()) return;
    let stored = id;
    if (!stored) {
      try { stored = this.storage?.getItem(sessionStorageKey(this.backend)) ?? undefined; } catch { /* Create remains available. */ }
    }
    if (!stored) return;
    const sessionId = stored;
    this.retrySession = () => this.recover(sessionId);
    const request = this.beginSession();
    void this.feedback.track(request, () => this.client.getSession(sessionId, request.signal)).then(request.guard((session) => this.acceptSession(session)), request.guard((error: unknown) => this.sessionFailure(error)));
  };
  chooseModel = (modelId: string) => {
    if (!this.state.models.some((model) => model.id === modelId)) return;
    this.stopRefresh();
    this.update({ selectedModelId: modelId, modelGeneration: this.state.modelGeneration + 1, refreshStatus: 'idle' });
    this.retrySession = () => { if (this.state.sessionStatus !== 'loading') this.chooseModel(modelId); };
    const request = this.beginSession();
    this.remember(null);
    // Let POST finish so a superseded creation's returned ID can be released.
    void this.feedback.track(request, () => this.client.createSession({ model_id: modelId })).then((session) => {
      if (!this.active || !request.isCurrent()) {
        this.retire(session);
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
    void this.feedback.track(request, () => this.client.listTensors(session.id, request.signal)).then(request.guard(({ tensors, coverage, diagnostics }) => {
      this.diagnostics.observe(session, 'Tensor inventory', diagnostics.map((d) => finding(session.model_id, session.id, 'Tensor inventory', d, 'warning')));
      this.update({ tensors, inventoryCoverage: coverage, inventoryDiagnostics: diagnostics, inventory: 'complete' });
      if (this.tensorBookmark) {
        const bookmark = this.tensorBookmark; this.tensorBookmark = undefined;
        const named = tensors.filter(t => t.name === bookmark.name);
        const selected = tensors.find(t => t.id === bookmark.id && t.name === bookmark.name) ?? (named.length === 1 ? named[0] : undefined);
        if (selected) { if (this.state.explorer === 'Tensor Explorer') this.replaceView({ selected }); else this.update({ selected }); }
        else this.feedback.notify('The selected tensor is no longer available in this model.', 'info');
      }
    }), request.guard((error: unknown) => {
      if (this.contentFailure(error)) return;
      if (isExpired(error)) this.expire();
      else this.update({ inventory: 'failed', message: failureMessage(error, 'Could not load tensors') });
    }));
  };
  selectTensor = (tensor: TensorDescriptor) => {
    if (this.state.sessionStatus !== 'ready' || !this.state.tensors.includes(tensor)) return;
    this.replaceView({ selected: tensor });
  };
  switchExplorer = (explorer: Explorer) => {
    if (explorer !== this.state.explorer) { this.cancelRetry(); this.presentation.clear(); this.replaceView({ explorer }); }
  };
  reportStatus = (view: Lifetime, status: ViewStatus) => {
    if (view !== this.state.view || !view.isCurrent()) return;
    if (status === 'expired-session' && this.state.selectedModelId) this.refreshModel();
    else if (status === 'expired-session') this.expire();
    else this.update({ viewStatus: status });
  };
  private expire() {
    this.diagnostics.activate(null);
    this.feedback.reset();
    this.sessionRequest.dispose();
    this.inventoryRequest.dispose();
    this.remember(null);
    this.replaceView({ session: null, sessionStatus: 'expired-session', selected: null, tensors: [], inventoryCoverage: 'complete', inventoryDiagnostics: [], inventory: 'idle',
      toasts: [], message: 'Session expired. Select a model to start a fresh session; runtime state was not restored.' });
  }
  closeSession = () => {
    const session = this.state.session;
    if (this.state.sessionStatus === 'closing') return;
    this.stopRefresh();
    this.update({ selectedModelId: null, modelGeneration: this.state.modelGeneration + 1, refreshStatus: 'idle', observation: 'idle' });
    this.sessionRequest.dispose();
    this.inventoryRequest.dispose();
    if (!session) {
      this.remember(null); this.replaceView({ session: null, sessionStatus: 'idle', message: '' }); return;
    }
    const request = this.sessionRequest = new Lifetime();
    this.feedback.reset();
    this.replaceView({ sessionStatus: 'closing', selected: null, message: '' });
    const finish = () => {
      this.diagnostics.activate(null);
      this.remember(null);
      this.update({ session: null, sessionStatus: 'idle', inventory: 'idle', tensors: [], message: '' });
      this.feedback.notify('Session closed.', 'info');
    };
    void this.feedback.track(request, () => this.client.deleteSession(session.id)).then(request.guard(finish), request.guard((error: unknown) => {
      if (isExpired(error)) finish();
      else {
        this.update({ selectedModelId: session.model_id, sessionStatus: 'ready', message: '' });
        this.observeModel();
        this.feedback.notify(failureMessage(error, 'Could not close session'), 'error', { label: 'Retry close', run: this.closeSession });
        if (this.state.inventory === 'loading') this.loadInventory();
      }
    }));
  };
}
