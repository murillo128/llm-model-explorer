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
import { interfaceIndex, resolveInterfaceEndpoints } from './interfaces';

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
  const parameters = selected.structureOnly || !node || selected.boundary ? [] : ownParameters(node, new Map(graph.parameters.map((p) => [p.id, p])));
  const templateInstance = graph.templates?.flatMap((t) => t.instances).find((i) => i.node_id === selected.templateInstanceId);
  const members = new Set(templateInstance?.nodes.map((m) => m.node_id));
  const external = templateInstance ? graph.edges.filter((e) => members.has(e.source.node_id) !== members.has(e.target.node_id)) : [];
  const parameter = parameters.find((p) => p.id === choice);
  const inspection = parameter?.inspection;
  const tensor = inspection?.status === 'available'
    ? inventory.tensors.find((t) => t.id === inspection.tensor_id) : undefined;
  const diagnostics = selected.structureOnly ? [] : [...new Map([...graph.diagnostics, ...responseDiagnostics].map((d) => [JSON.stringify([d.code, d.node_id, d.parameter_id, d.message]), d])).values()].filter((d) => d.node_id === node?.id || (parameter && d.parameter_id === parameter.id));
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
    <header className="architecture-inspection-heading"><h2 id={title}>{selected.boundary ? 'Interface inspection' : node?.label ?? 'Model'}</h2><button autoFocus onClick={close} aria-label="Close inspection">×</button></header>
    <div className="architecture-inspection-details">
      {node && <>
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
      </>}
      <InterfaceDetails graph={graph} selected={selected} />
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

function InterfaceDetails({ graph, selected }: { graph: Graph; selected: ArchitectureSelection }) {
  if (selected.structureOnly) return null;
  const index = interfaceIndex(graph), boundary = selected.boundary;
  const ids = new Set(resolveInterfaceEndpoints(graph, boundary?.endpoints ?? []).map((p) => p.node_id));
  const owner = boundary?.owner.id ?? selected.node?.id;
  const interfaces = index.interfaces.filter((item) => ids.size ? ids.has(item.node.id) : item.owner.id === owner);
  const nodes = new Map(interfaces.map((item) => [item.node.id, item.node]));
  if (boundary) for (const id of ids) { const node = graph.nodes.find((n) => n.id === id); if (node) nodes.set(id, node); }
  return <>
    {boundary?.owner.kind === 'model' && <p>Presentation boundary · {graph.scope} · {graph.coverage}. No source module or parameter binding.</p>}
    {boundary?.owner.kind === 'model' && !ids.size && <p>Source components: {index.roots.map((n) => n.label).join(', ')}</p>}
    {[...nodes.values()].map((node) => <details key={node.id} open={Boolean(boundary)}>
      <summary>{node.label} · interface metadata</summary>
      <p>{node.id} · {node.kind}{node.operation && ` · ${node.operation}`}{node.parent_id && ` · parent ${node.parent_id}`}</p>
      {node.description && <p>{node.description}</p>}
      {node.ports.map((p) => <p key={p.id}>{p.direction} {p.label} · {p.id}: {formatShape(p.shape)}</p>)}
      {node.attributes.map((a, i) => <div key={i}>{a.name}: {JSON.stringify(a.value)}<Provenance records={a.provenance} /></div>)}
      <Provenance records={node.provenance} />
      {graph.diagnostics.filter((d) => d.node_id === node.id).map((d, i) => <p key={i}>{d.message}</p>)}
      {graph.edges.filter((e) => e.source.node_id === node.id || e.target.node_id === node.id).map((e) => <div key={e.id}>
        <p>{e.source.node_id}:{e.source.port_id} → {e.target.node_id}:{e.target.port_id} · {e.kind} · {e.id}</p><Provenance records={e.provenance} />
      </div>)}
      {!graph.edges.some((e) => e.source.node_id === node.id || e.target.node_id === node.id) && <p>No source connection.</p>}
    </details>)}
    {selected.node && index.notices.has(selected.node.id) && <p role="status">{index.notices.get(selected.node.id)}</p>}
  </>;
}

function InspectionWeight({ context, tensor, parent }: { context: ExplorerContextValue; tensor: S['TensorDescriptor']; parent: Lifetime }) {
  const selection = useChildLifetime(parent);
  return selection?.isCurrent() && <TensorExplorer {...context} selectedTensor={tensor} selection={selection} showInformation={false} />;
}
