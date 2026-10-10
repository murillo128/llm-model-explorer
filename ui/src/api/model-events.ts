import type { components } from './generated/types';
import { requireProtocol } from './errors';
import { validateSchema } from './validation';

export type ModelState = components['schemas']['ModelState'];
export type ObservationStatus = 'connecting' | 'live' | 'reconnecting' | 'protocol-error' | 'observation-error';
export interface ModelObserver {
  state: (state: ModelState) => void;
  status: (status: ObservationStatus) => void;
}

/** One native connection; callbacks remain fenced even if queued before close. */
export function watchModel(base: string, modelId: string, observer: ModelObserver): () => void {
  if (typeof EventSource === 'undefined') return () => {};
  const source = new EventSource(`${base.replace(/\/+$/, '')}/models/events?model_id=${encodeURIComponent(modelId)}`);
  let active = true, epoch: string | undefined, sequence = -1;
  const close = () => { active = false; source.close(); };
  const receive = (kind: 'ModelState' | 'ModelObservationError') => (raw: Event) => {
    if (!active) return;
    try {
      const event = raw as MessageEvent<string>;
      requireProtocol(typeof event.data === 'string' && event.data.length <= 65_536 && new TextEncoder().encode(event.data).length <= 65_536, 'Oversized model event');
      const value = validateSchema(kind, JSON.parse(event.data));
      if (value.model_id !== modelId) return;
      requireProtocol(Number.isSafeInteger(value.sequence) && event.lastEventId === `${value.epoch}:${value.sequence}`, 'Invalid model event identity');
      if (value.epoch === epoch && value.sequence <= sequence) return;
      epoch = value.epoch; sequence = value.sequence;
      observer.status(kind === 'ModelState' ? 'live' : 'observation-error');
      if ('status' in value) observer.state(value);
    } catch { close(); observer.status('protocol-error'); }
  };
  source.addEventListener('model-state', receive('ModelState'));
  source.addEventListener('observation-error', receive('ModelObservationError'));
  source.addEventListener('error', () => { if (active) observer.status('reconnecting'); });
  return close;
}
