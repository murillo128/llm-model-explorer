import { useRef, useSyncExternalStore } from 'react';
import { ModelDiagnostics, modelSuppliedExplanation } from './model-diagnostics';
import type { Diagnostic } from './model-diagnostics';
const labels = { info: 'Info', warning: 'Warning', error: 'Error' };
export function DiagnosticDetails({ records }: { records: Diagnostic[] }) {
  return <ul className="diagnostic-list">{records.map((d) => <li key={d.id} data-severity={d.severity}>
    <strong>{labels[d.severity]} · {d.capability}</strong>
    {d.context && <p>{d.context}</p>}<p>{d.message}</p><code>{d.code}</code>
  </li>)}</ul>;
}
export function ModelSuppliedBadge() {
  return <details className="model-supplied"><summary>ⓘ Model-supplied</summary><p role="note"><strong>Info</strong> · {modelSuppliedExplanation}</p></details>;
}
export function DiagnosticBand({ store }: { store: ModelDiagnostics }) {
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot);
  const section = useRef<HTMLElement>(null);
  const records = state.records.filter((d) => d.capability === 'Architecture' && !state.dismissed.includes(d.id));
  const top = records[0];
  return <section ref={section} tabIndex={-1} className="diagnostic-band" aria-label="Architecture notices">
    {state.modelSupplied && <ModelSuppliedBadge />}
    {top && <div className="diagnostic-notice" data-severity={top.severity}>
      <div role="status" aria-live="polite" aria-atomic="true"><strong><span aria-hidden="true">{top.severity === 'info' ? 'ⓘ' : '⚠'}</span> {labels[top.severity]}</strong>
        {records.length > 1 && <span> · {records.filter((d) => d.severity === 'error').length} errors · {records.filter((d) => d.severity === 'warning').length} warnings · {records.filter((d) => d.severity === 'info').length} info</span>}
        <p>{top.context && `${top.context}: `}{top.message}</p></div>
      <div className="diagnostic-actions"><details><summary>View details</summary><DiagnosticDetails records={records} /></details>
        <button type="button" aria-label="Dismiss architecture notices" onClick={() => { store.dismiss(records.map((d) => d.id)); (section.current?.closest<HTMLElement>('.architecture-explorer') ?? section.current)?.focus(); }}>Dismiss</button></div>
    </div>}
  </section>;
}
export function ModelDiagnosticInformation({ store }: { store: ModelDiagnostics }) {
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot);
  const warnings = state.records.filter((d) => d.severity === 'warning').length;
  const errors = state.records.filter((d) => d.severity === 'error').length;
  return <section aria-label="Diagnostics"><h3>Diagnostics <small>{warnings} warnings · {errors} errors</small></h3>
    {state.graphInformation && <section aria-label="Architecture information">
      <h4>Architecture</h4><p>Graph: <code>{state.graphInformation.graph_id}</code></p>
      <p>{state.graphInformation.coverage === 'partial' ? 'Partial architecture coverage' : 'Complete within declared scope'} · {state.graphInformation.scope.replaceAll('_', ' ')}</p>
    </section>}
    {!state.architectureObserved && <p>Architecture diagnostics have not been retrieved.</p>}
    {!state.records.length && <p>No diagnostics in the capabilities observed so far.</p>}
    <DiagnosticDetails records={state.records} />
    {state.modelSupplied && <p>{modelSuppliedExplanation}</p>}
  </section>;
}
