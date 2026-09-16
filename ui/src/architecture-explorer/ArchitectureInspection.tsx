import { useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { components } from '../api/generated/types';
import type { ExplorerContextValue } from '../app/explorer-context';
import { Lifetime } from '../app/lifetime';
import { TensorExplorer } from '../explorers/TensorExplorer';
import type { ArchitectureSelection } from './ArchitectureCanvas';
import { formatShape } from './graph';
import type { Graph } from './graph';
import { ownParameters } from './card-summary';

type S = components['schemas'];
function Provenance({ records }: { records: S['ArchitectureProvenance'][] }) {
  return <ul>{records.map((p, i) => <li key={i}>{p.kind}: {p.source}{p.revision && ` · revision ${p.revision}`}{p.rule && ` · rule ${p.rule}`}</li>)}</ul>;
}

/** Publish only effect-owned generations, including a fresh one after StrictMode replay. */
function useChildLifetime(parent: Lifetime) {
  const [lifetime, setLifetime] = useState<Lifetime | null>(null);
  useLayoutEffect(() => {
    const current = new Lifetime();
    const detach = parent.onDispose(current.dispose);
    // Children may subscribe only after their effect-owned lifetime is installed.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLifetime(current);
    return () => { current.dispose(); detach(); };
  }, [parent]);
  return lifetime;
}

/** One modal generation; numeric children never own the graph's selection lifetime. */
export function ArchitectureInspection({ context, graph, inventory, selected, onClose, diagnostics: responseDiagnostics = [] }: {
  context: ExplorerContextValue; graph: Graph; inventory: S['TensorInventory'];
  selected: ArchitectureSelection; onClose: () => void;
  diagnostics?: S['ArchitectureDiagnostic'][];
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const title = useId();
  const lifetime = useChildLifetime(context.selection);
  const [choice, setChoice] = useState(selected.parameterId ?? '');
  const { node } = selected;
  const parameters = selected.structureOnly ? [] : ownParameters(node, new Map(graph.parameters.map((p) => [p.id, p])));
  const templateInstance = graph.templates?.flatMap((t) => t.instances).find((i) => i.node_id === selected.templateInstanceId);
  const members = new Set(templateInstance?.nodes.map((m) => m.node_id));
  const external = templateInstance ? graph.edges.filter((e) => members.has(e.source.node_id) !== members.has(e.target.node_id)) : [];
  const parameter = parameters.find((p) => p.id === choice);
  const inspection = parameter?.inspection;
  const tensor = inspection?.status === 'available'
    ? inventory.tensors.find((t) => t.id === inspection.tensor_id) : undefined;
  const diagnostics = selected.structureOnly ? [] : [...graph.diagnostics, ...responseDiagnostics].filter((d) => d.node_id === node.id || (parameter && d.parameter_id === parameter.id));
  const close = () => { lifetime?.dispose(); onClose(); };
  useLayoutEffect(() => {
    const element = dialog.current!;
    element.showModal();
    return () => {
      element.close();
      if (selected.trigger.isConnected) selected.trigger.focus({ preventScroll: true });
    };
  }, [selected.trigger]);
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
      {selected.structureOnly ? <p>Shared structure: {selected.structureOnly.label} · {selected.structureOnly.role}. No instance selected; choose an instance for weights.</p>
        : <p>{selected.modelId} · {node.id}{node.parent_id && ` · parent ${node.parent_id}`} · {node.operation ?? node.kind}</p>}
      {!!external.length && <details><summary>Concrete instance interface connections</summary>{external.map((edge) => <p key={edge.id}>
        {graph.nodes.find((n) => n.id === edge.source.node_id)?.label} · <code>{edge.source.node_id}:{edge.source.port_id}</code> → {graph.nodes.find((n) => n.id === edge.target.node_id)?.label} · <code>{edge.target.node_id}:{edge.target.port_id}</code><br />Original edge: <code>{edge.id}</code>
      </p>)}</details>}
      {node.description && <p>{node.description}</p>}
      {node.formula && <pre>{node.formula}</pre>}
      {node.ports.map((p) => <p key={p.id}>{p.direction} {p.label}: {formatShape(p.shape)}</p>)}
      {!!selected.filteredInputs?.length && <p>Declared inputs without internal consumers: {selected.filteredInputs.join(', ')}. Their branches are filtered in the compact graph; enable Unused interfaces to show them.</p>}
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
    {lifetime?.isCurrent() && tensor && (tensor.rank === 1 || tensor.rank === 2) && <InspectionWeight key={choice} context={context} tensor={tensor} parent={lifetime} />}
  </dialog>, document.body);
}

function InspectionWeight({ context, tensor, parent }: { context: ExplorerContextValue; tensor: S['TensorDescriptor']; parent: Lifetime }) {
  const selection = useChildLifetime(parent);
  return selection?.isCurrent() && <TensorExplorer {...context} selectedTensor={tensor} selection={selection} showInformation={false} />;
}
