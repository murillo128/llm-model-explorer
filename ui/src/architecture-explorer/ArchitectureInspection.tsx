import { useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { components } from '../api/generated/types';
import type { ExplorerContextValue } from '../app/explorer-context';
import { Lifetime } from '../app/lifetime';
import { TensorExplorer } from '../explorers/TensorExplorer';
import type { ArchitectureSelection } from './ArchitectureCanvas';
import { formatShape } from './graph';
import type { Graph } from './graph';

type S = components['schemas'];
function Provenance({ records }: { records: S['ArchitectureProvenance'][] }) {
  return <ul>{records.map((p, i) => <li key={i}>{p.kind}: {p.source}{p.revision && ` · revision ${p.revision}`}{p.rule && ` · rule ${p.rule}`}</li>)}</ul>;
}

/** One modal generation; numeric children never own the graph's selection lifetime. */
export function ArchitectureInspection({ context, graph, inventory, selected, onClose, diagnostics: responseDiagnostics = [] }: {
  context: ExplorerContextValue; graph: Graph; inventory: S['TensorInventory'];
  selected: ArchitectureSelection; onClose: () => void;
  diagnostics?: S['ArchitectureDiagnostic'][];
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const title = useId();
  const [lifetime] = useState(() => new Lifetime());
  const [choice, setChoice] = useState('');
  const { node } = selected;
  const ids = new Set([...node.parameter_ids, ...node.references.flatMap((r) => r.kind === 'parameter' ? [r.parameter_id] : [])]);
  const parameters = graph.parameters.filter((p) => ids.has(p.id));
  const parameter = parameters.find((p) => p.id === choice);
  const inspection = parameter?.inspection;
  const tensor = inspection?.status === 'available'
    ? inventory.tensors.find((t) => t.id === inspection.tensor_id) : undefined;
  const diagnostics = [...graph.diagnostics, ...responseDiagnostics].filter((d) => d.node_id === node.id || (parameter && d.parameter_id === parameter.id));
  const close = () => { lifetime.dispose(); onClose(); };
  useLayoutEffect(() => {
    const element = dialog.current!;
    element.showModal();
    const detach = context.selection.onDispose(lifetime.dispose);
    return () => {
      lifetime.dispose(); detach(); element.close();
      if (selected.trigger.isConnected) selected.trigger.focus({ preventScroll: true });
    };
  }, [context.selection, lifetime, selected.trigger]);
  return createPortal(<dialog ref={dialog} className="architecture-inspection" aria-labelledby={title}
    onCancel={(event) => { event.preventDefault(); close(); }}
    onKeyDownCapture={(event) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); }
      if (event.key === 'Tab') {
        const controls = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('button, select, input, textarea, a[href], [tabindex]'))
          .filter((element) => element.tabIndex >= 0 && !element.matches(':disabled') && element.getClientRects().length > 0);
        const first = controls[0], last = controls.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    }}>
    <header className="architecture-inspection-heading"><h2 id={title}>{node.label}</h2><button autoFocus onClick={close} aria-label="Close inspection">×</button></header>
    <div className="architecture-inspection-details">
      <p>{selected.modelId} · {node.id}{node.parent_id && ` · parent ${node.parent_id}`} · {node.operation ?? node.kind}</p>
      {node.description && <p>{node.description}</p>}
      {node.formula && <pre>{node.formula}</pre>}
      {node.ports.map((p) => <p key={p.id}>{p.direction} {p.label}: {formatShape(p.shape)}</p>)}
      {node.references.map((r, i) => r.kind === 'module' ? <p key={i}>Module: {r.name}</p> : r.kind === 'tokenizer' ? <p key={i}>Tokenizer context for this model</p> : null)}
      {node.attributes.map((a, i) => <div key={i}><p>{a.name}: {JSON.stringify(a.value)}</p><Provenance records={a.provenance} /></div>)}
      <Provenance records={node.provenance} />
      {parameters.length > 0 && <label>Parameter <select aria-label="Inspect parameter" value={choice} onChange={(event) => setChoice(event.target.value)}>
        <option value="">Choose a parameter…</option>
        {parameters.map((p) => <option key={p.id} value={p.id}>{p.name} · {p.binding}</option>)}
      </select></label>}
      {parameter && <>
        <p>{parameter.name} · {parameter.binding} · logical shape {formatShape(parameter.logical_shape)}</p>
        {parameter.binding === 'alias' && <p>Alias of {parameter.alias_of}</p>}
        {parameter.binding === 'fused_region' && <p>{parameter.region.storage_name}: {parameter.region.description}</p>}
        {parameter.storage.map((s, i) => <p key={i}>Storage: {s.name} · {s.dtype} · [{s.shape.join(' × ')}]{s.role && ` · ${s.role}`}</p>)}
        <Provenance records={parameter.provenance} />
        {parameter.inspection.status === 'unavailable' && <p role="status">{parameter.inspection.reason.replaceAll('_', ' ')}: {parameter.inspection.message}</p>}
      </>}
      {diagnostics.map((d, i) => <p key={i} role="status">{d.message}</p>)}
    </div>
    {tensor && (tensor.rank === 1 || tensor.rank === 2) && <InspectionWeight key={choice} context={context} tensor={tensor} parent={lifetime} />}
  </dialog>, document.body);
}

function InspectionWeight({ context, tensor, parent }: { context: ExplorerContextValue; tensor: S['TensorDescriptor']; parent: Lifetime }) {
  const [selection] = useState(() => new Lifetime());
  useLayoutEffect(() => {
    const detach = parent.onDispose(selection.dispose);
    return () => { selection.dispose(); detach(); };
  }, [parent, selection]);
  return <TensorExplorer {...context} selectedTensor={tensor} selection={selection} showInformation={false} />;
}
