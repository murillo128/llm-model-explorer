import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { createPortal } from 'react-dom';
import type { components } from '../api/generated/types';
import type { ExplorerContextValue } from '../app/explorer-context';
import { TensorExplorer } from '../explorers/TensorExplorer';
import { InspectionDialog } from './InspectionDialog';
import { useChildLifetime } from './useChildLifetime';
import { neuronWeight, projectionWeights, validNeuronIndex } from './neuron-selection';
import type { NeuronModule, NeuronSelectionContext, NeuronViewState, ProjectionWeight } from './neuron-selection';
import type { useNeuronCapabilities } from './useNeuronCapabilities';
import './neurons.css';

type S = components['schemas'];
export function NeuronInspector({ context, capabilities: result, inventory, state, onChange, onBack, tokenizerAvailable, experiment, experimentTarget }: {
  context: ExplorerContextValue; capabilities: ReturnType<typeof useNeuronCapabilities>; inventory?: S['TensorInventory'] | undefined;
  state: NeuronViewState; onChange: (patch: Partial<NeuronViewState>) => void; onBack: () => void; tokenizerAvailable: boolean;
  experiment?: ((context: NeuronSelectionContext) => ReactNode) | undefined;
  experimentTarget: HTMLDivElement | null;
}) {
  const scroll = useRef<HTMLDivElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const capabilities = result.capabilities;
  const module = capabilities?.modules.find(item => item.module_id === state.moduleId);
  const pendingScroll = useRef<number | null>(state.scroll);
  useLayoutEffect(() => { heading.current?.focus(); }, []);
  useLayoutEffect(() => {
    // Loading content has no scroll range. Restore once its real content is
    // laid out, then let later user scrolling own the offset.
    if (!capabilities || !scroll.current || pendingScroll.current === null) return;
    scroll.current.scrollTop = pendingScroll.current;
    pendingScroll.current = null;
  }, [capabilities]);
  return <section className="neuron-inspector explorer-card" aria-label="Neuron inspector">
    <header><h2 ref={heading} tabIndex={-1}>Neuron inspector</h2><button onClick={onBack}>Back to graph</button></header>
    <div className="neuron-inspector-scroll" ref={scroll} onScroll={event => {
      if (capabilities && pendingScroll.current === null) onChange({ scroll: event.currentTarget.scrollTop });
    }}>
      {!capabilities && (result.error ? <p role="alert">{result.error} <button onClick={result.retry}>Retry neuron capabilities</button></p> : <p role="status">Retrieving neuron capabilities…</p>)}
      {capabilities && <>
        {(['observation', 'intervention', 'generation'] as const).map(name => {
          const feature = capabilities[name];
          return feature.status === 'unavailable' && <p key={name} role="status"><strong>{name[0]!.toUpperCase() + name.slice(1)} unavailable.</strong> {feature.reason.message} {feature.reason.recovery}</p>;
        })}
        {capabilities.modules.length > 0 && <label className="neuron-module">Concrete MLP
          <select value={module?.module_id ?? ''} onChange={event => { if (event.target.value !== state.moduleId) onChange({ moduleId: event.target.value || null, index: 0, scroll: 0 }); }}>
            <option value="">Choose a concrete module…</option>
            {capabilities.modules.map(item => <option key={item.module_id} value={item.module_id}>Layer {item.layer_index} · {item.module.name}</option>)}
          </select>
        </label>}
        {state.moduleId && !module && <p role="status">The previous module is unavailable in this session. Choose a current module.</p>}
        {module && <SelectedNeuron context={context} capabilities={capabilities} module={module} index={state.index}
          inventory={inventory} onIndex={index => onChange({ index })} tokenizerAvailable={tokenizerAvailable} experiment={experiment} experimentTarget={experimentTarget} />}
      </>}
    </div>
  </section>;
}

function SelectedNeuron({ context, capabilities, module, index, inventory, onIndex, tokenizerAvailable, experiment, experimentTarget }: {
  context: ExplorerContextValue; capabilities: S['NeuronCapabilities']; module: NeuronModule; index: number;
  inventory?: S['TensorInventory'] | undefined; onIndex: (index: number) => void; tokenizerAvailable: boolean;
  experiment?: ((context: NeuronSelectionContext) => ReactNode) | undefined;
  experimentTarget: HTMLDivElement | null;
}) {
  const [draft, setDraft] = useState({ moduleId: module.module_id, index, value: String(index) });
  // Invalid text is local editing state. A changed authoritative coordinate always
  // supersedes it, so the displayed index and actionable binding cannot diverge.
  const input = draft.moduleId === module.module_id && draft.index === index ? draft.value : String(index);
  const [weight, setWeight] = useState<{ projection: ProjectionWeight; trigger: HTMLElement } | null>(null);
  const valid = input.trim() !== '' && validNeuronIndex(Number(input), module);
  const lifetime = useChildLifetime(context.selection);
  const selected = useMemo<NeuronSelectionContext | null>(() => lifetime ? {
    client: context.client, session: context.session, selection: lifetime, capabilities, module, neuronIndices: valid ? [index] : [],
  } : null, [context.client, context.session, lifetime, capabilities, module, index, valid]);
  const bound = inventory && weight && neuronWeight(module, weight.projection, inventory);
  return <>
    <p className="neuron-context">Layer {module.layer_index} · <code>{module.module.name}</code></p>
    <label className="neuron-index">Intermediate index
      <input aria-describedby="neuron-index-help neuron-index-error" aria-invalid={!valid} type="number" min={0} max={module.intermediate_size - 1} step={1} value={input}
        onChange={event => {
          const value = event.target.value;
          const accepted = value.trim() !== '' && validNeuronIndex(Number(value), module);
          setDraft({ moduleId: module.module_id, index: accepted ? Number(value) : index, value });
          if (accepted) onIndex(Number(value));
        }} />
    </label>
    <p id="neuron-index-help">Zero-based coordinate · 0–{module.intermediate_size - 1}</p>
    {!valid && <p id="neuron-index-error" role="alert">Enter a whole index from 0 to {module.intermediate_size - 1}.</p>}
    <div className="neuron-formula"><code>a[j] = SiLU(g[j]) * u[j]</code><p>After gate × up, before the down projection.</p></div>
    <h3>Linked weights</h3>
    {!inventory && <p role="status">Weight inventory is unavailable. Retry architecture retrieval to validate the bindings.</p>}
    <ul className="neuron-weights">{projectionWeights.map(projection => {
      const binding = module.weights[projection];
      const target = inventory && neuronWeight(module, projection, inventory);
      return <li key={projection}>
        <div><strong>{projection} · {binding.axis} {valid ? index : '—'}</strong><code>{target && 'tensor' in target ? target.tensor.name : binding.tensor_id}</code>
          <span>[{binding.shape.join(' × ')}]</span>
          {target && 'reason' in target && <p role="status">{target.reason}</p>}
        </div>
        <button disabled={!valid || !target || !('tensor' in target)} aria-label={`Open ${projection} weight`}
          onClick={event => setWeight({ projection, trigger: event.currentTarget })}>Open</button>
      </li>;
    })}</ul>
    <p className="neuron-explanation">This coordinate selects an activation, not a weight cell. The complete weight matrices remain read-only.</p>
    {!tokenizerAvailable && <p role="status">Tokenizer unavailable. Weights and graph remain available; execution needs a local tokenizer.</p>}
    {selected && experiment && experimentTarget && createPortal(experiment(selected), experimentTarget)}
    {weight && bound && 'tensor' in bound && valid && lifetime?.isCurrent() && <NeuronWeightInspection key={bound.tensor.id} context={context} parent={lifetime}
      module={module} index={index} projection={weight.projection} tensor={bound.tensor} axis={bound.axis} trigger={weight.trigger} onClose={() => setWeight(null)} />}
  </>;
}

function NeuronWeightInspection({ context, parent, module, index, projection, tensor, axis, trigger, onClose }: {
  context: ExplorerContextValue; parent: ExplorerContextValue['selection']; module: NeuronModule; index: number;
  projection: ProjectionWeight; tensor: S['TensorDescriptor']; axis: 'row' | 'column'; trigger: HTMLElement; onClose: () => void;
}) {
  const selection = useChildLifetime(parent);
  const coordinate = useMemo(() => axis === 'row' ? { selectedRows: [index], revealRow: { row: index } }
    : { selectedColumns: [index], revealColumn: { column: index } }, [axis, index]);
  return <InspectionDialog heading={`Layer ${module.layer_index} · ${projection} weight`} trigger={trigger} onClose={() => { selection?.dispose(); onClose(); }}>
    <div className="architecture-inspection-details"><p><code>{module.module.name}</code> · neuron {index} · <strong>{axis} {index}</strong></p>
      <p>Complete native matrix · [{tensor.shape.join(' × ')}] · highlighted {axis} {index}</p></div>
    {selection?.isCurrent() && <TensorExplorer {...context} selectedTensor={tensor} selection={selection} showInformation={false} {...coordinate} />}
  </InspectionDialog>;
}
