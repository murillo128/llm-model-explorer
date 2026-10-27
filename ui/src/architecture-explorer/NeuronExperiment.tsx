import { useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { PromptTokenizer } from '../tokenizer/PromptTokenizer';
import type { CurrentTokenization } from '../tokenizer/PromptTokenizer';
import type { NeuronSelectionContext } from './neuron-selection';
import { NeuronExperimentController } from './neuron-experiment-controller';
import type { ExperimentResult } from './neuron-experiment-controller';
import { conditionLabel, experimentKey, experimentSnapshot, exportConfiguration, initialExperiment, percentagePoints, signed } from './neuron-experiment';
import type { ConditionDraft, ExperimentConfig, ExperimentSnapshot } from './neuron-experiment';
import './neuron-experiment.css';

export function NeuronExperiment({ context, tokenizerAvailable }: { context: NeuronSelectionContext; tokenizerAvailable: boolean }) {
  return <section className="neuron-experiment-card explorer-card" aria-label="Experiment">
    <header className="neuron-experiment-heading"><h2>Experiment</h2><span>Text continuation · SmolLM2 Base · explicit runs</span></header>
    <div className="neuron-experiment-body">
      <PromptTokenizer client={context.client} sessionId={context.session.id} signal={context.selection.signal}
        tokenizerAvailable={tokenizerAvailable} downstream={current => <ExperimentControls context={context} current={current} />} />
    </div>
  </section>;
}

function ExperimentControls({ context, current }: { context: NeuronSelectionContext; current: CurrentTokenization | undefined }) {
  const [config, setConfig] = useState<ExperimentConfig>(() => initialExperiment(context.capabilities));
  const [includeText, setIncludeText] = useState(false);
  const [exportError, setExportError] = useState('');
  const nextCondition = useRef(1);
  const controller = useMemo(() => new NeuronExperimentController(context.client), [context.client]);
  const state = useSyncExternalStore(controller.subscribe, controller.getSnapshot);
  const id = useId();
  const controlsKey = JSON.stringify([context.session, context.module.module_id, context.neuronIndices, config]);
  useEffect(() => () => controller.dispose(), [controller]);
  // Edits cancel only this consumer. A→B→A still has a new tokenizer signal.
  useEffect(() => () => { void controller.cancel(); }, [controller, controlsKey, current?.signal]);
  let snapshot: ExperimentSnapshot | undefined, invalid = '';
  try { snapshot = experimentSnapshot(context, current, config); }
  catch (error) { invalid = error instanceof Error ? error.message : 'Invalid configuration.'; }
  const result = state.result;
  const stale = !!result && (!snapshot || result.tokenSignal !== current?.signal || result.tokenSignal.aborted
    || experimentKey(result.snapshot) !== experimentKey(snapshot));
  const busy = controller.busy;
  const limits = context.capabilities.limits;
  const patchCondition = (index: number, patch: Partial<ConditionDraft>) => setConfig(previous => ({ ...previous,
    conditions: previous.conditions.map((value, i) => i === index ? { ...value, ...patch } : value) }));
  const status = state.status === 'preparing' ? 'Preparing / queued…'
    : state.status === 'running' ? 'Running… Waiting for a complete result; no token streaming is promised.'
    : state.status === 'cancelling' ? 'Cancelling…'
    : state.status === 'cancelled' ? 'Cancelled. No partial result was accepted.'
    : state.status === 'failed' || state.status === 'unsupported' ? state.message
    : state.status === 'complete' && !stale ? 'Complete.' : 'Idle. Run explicitly when the configuration is ready.';
  function download() {
    if (!result || stale || busy) return;
    try {
      const json = exportConfiguration(result.snapshot, result.metadata, includeText);
      const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'llmex-neuron-experiment.json';
      anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 0); setExportError('');
    } catch (error) { setExportError(error instanceof Error ? error.message : 'Configuration could not be exported.'); }
  }
  return <>
    <div className="experiment-toolbar">
      <div role="group" aria-label="Experiment mode" className="experiment-modes">{(['activations', 'compare'] as const).map(mode =>
        <button key={mode} aria-pressed={config.mode === mode} onClick={() => setConfig(previous => ({ ...previous, mode }))}>{mode === 'activations' ? 'Activations' : 'Compare'}</button>)}</div>
      <span className="experiment-target">Layer {context.module.layer_index} · neuron {context.neuronIndices.join(', ') || '—'}</span>
      <button className="experiment-run" disabled={!snapshot || busy} onClick={() => { if (snapshot && current) controller.run(snapshot, current.signal, context.selection.signal); }}>
        {result && !stale ? 'Run again' : 'Run'}</button>
      {busy && <button disabled={state.status === 'cancelling'} onClick={() => { void controller.cancel(); }}>Cancel</button>}
    </div>
    <p className="experiment-limits">{limits ? `Limits: ${limits.max_input_tokens} prompt tokens · ${limits.max_neurons} neurons in one module · ${limits.max_new_tokens} new tokens · ${limits.context_length} total context.` : 'Execution unavailable.'}</p>
    {config.mode === 'compare' && <>
      <fieldset className="experiment-conditions"><legend>Conditions · same selected coordinate</legend>
        <div className="experiment-baseline"><strong>Baseline</strong><span>Unmodified · always first</span></div>
        {config.conditions.map((condition, index) => <div className="experiment-condition" key={condition.id}>
          <strong>{condition.id}</strong>
          <label>Operation<select aria-label={`${condition.id} operation`} value={condition.mode} onChange={event => patchCondition(index, { mode: event.target.value as ConditionDraft['mode'] })}>
            <option value="zero">zero · set to 0</option><option value="scale">scale · multiply</option><option value="add">add · offset</option>
          </select></label>
          {condition.mode !== 'zero' ? <label>{condition.mode === 'scale' ? 'Factor' : 'Delta'}<input aria-label={`${condition.id} ${condition.mode === 'scale' ? 'factor' : 'delta'}`} type="text" inputMode="decimal" value={condition.value} onChange={event => patchCondition(index, { value: event.target.value })} /></label>
            : <span className="experiment-zero">a[j] = 0</span>}
          <label>Scope<select aria-label={`${condition.id} scope`} value={condition.scope} onChange={event => patchCondition(index, { scope: event.target.value as ConditionDraft['scope'] })}>
            <option value="all">all positions</option><option value="prompt">prompt only</option><option value="continuation">continuation only</option>
          </select></label>
          <button aria-label={`Remove ${condition.id}`} onClick={() => setConfig(previous => ({ ...previous, conditions: previous.conditions.filter((_, i) => i !== index) }))}>Remove</button>
        </div>)}
        <button disabled={context.capabilities.intervention.status !== 'available' || config.conditions.length >= (limits?.max_conditions ?? 1) - 1}
          onClick={() => { const condition: ConditionDraft = { id: `condition-${nextCondition.current++}`, mode: 'add', value: '1', scope: 'all' }; setConfig(previous => ({ ...previous, conditions: [...previous.conditions, condition] })); }}>Add condition</button>
      </fieldset>
      <div className="experiment-options"><label>Top-k<input type="number" min={1} max={limits?.max_top_k} value={config.topK} onChange={event => setConfig(previous => ({ ...previous, topK: event.target.value }))} /></label>
        <label>New tokens<input type="number" min={0} max={limits?.max_new_tokens} value={config.maxNewTokens} onChange={event => setConfig(previous => ({ ...previous, maxNewTokens: event.target.value }))} /></label><span>Greedy · 0 new tokens = table only</span></div>
      <p className="experiment-note">Continuation-only scope cannot change the first next-token distribution: it comes from the shared prompt.</p>
    </>}
    <div className="experiment-status" role={state.status === 'failed' ? 'alert' : 'status'} data-experiment-status={state.status}>{status}</div>
    {invalid && <p className="experiment-validation" role="status">{invalid}</p>}
    {result && <section className="experiment-result" aria-label="Experiment result" data-stale={stale || undefined}>
      {stale && <p className="experiment-stale" role="status">Previous result is stale. It belongs to the snapshot below, not the current controls. Run to replace it.</p>}
      {state.status !== 'complete' && <p>Previous completed result retained. The latest attempt is {busy ? 'pending' : state.status}.</p>}
      <div className="experiment-result-heading"><h3>{result.metadata.kind === 'neuron_activations' ? 'Prompt activations' : 'Next-token probability · same context'}</h3>
        <span>Layer {result.metadata.module.layer_index} · neuron {result.metadata.neuron_indices.join(', ')} · {result.metadata.token_ids.length} prompt positions</span></div>
      <ExperimentReadout result={result} />
      <details className="experiment-snapshot"><summary>Result configuration and execution identity</summary>
        <dl><dt>Model / revision</dt><dd>{result.metadata.model_id} · {result.metadata.model_revision}</dd><dt>Session</dt><dd>{result.metadata.session_id}</dd>
          <dt>Module / point</dt><dd>{result.metadata.module.module_id} · {result.metadata.module.module.name} · {result.metadata.module.point}</dd>
          <dt>Runtime</dt><dd>{result.metadata.execution.adapter} {result.metadata.execution.adapter_version} · runtime {result.metadata.execution.runtime_version} · PyTorch {result.metadata.execution.torch_version} · {result.metadata.execution.device} · {result.metadata.execution.dtype}</dd></dl>
        {result.metadata.kind === 'neuron_comparison' && <p>{result.metadata.conditions.map(conditionLabel).join('; ')} · top-k {result.metadata.top_k} · {result.metadata.max_new_tokens} new tokens · greedy</p>}
      </details>
      <div className="experiment-export"><button onClick={download} disabled={stale || busy} aria-describedby={`${id}-export-warning`}>Export configuration</button>
        <label><input type="checkbox" checked={includeText} onChange={event => setIncludeText(event.target.checked)} />Include user text (sensitive)</label></div>
      <p id={`${id}-export-warning`} className="experiment-note">Token IDs can disclose your prompt even without text. This explicit download is a recipe, not the checkpoint or a guarantee of bitwise replay. Results stay in this view’s memory.</p>
      {exportError && <p role="alert">{exportError}</p>}
    </section>}
    <p className="experiment-note">An activation or intervention effect does not identify a semantic concept or a causal circuit owned by this neuron.</p>
  </>;
}

export function ExperimentReadout({ result }: { result: ExperimentResult }) {
  const { metadata, values, snapshot } = result;
  if (metadata.kind === 'neuron_activations') return <div className="experiment-table-scroll" tabIndex={0} role="region" aria-label="Signed activation values by prompt position">
    <table><caption>Signed float32 activation, after gate × up and before down projection. Focus or hover a value for its exact decimal.</caption>
      <thead><tr><th scope="col">Position</th><th scope="col">Token ID</th><th scope="col">Token</th>{metadata.neuron_indices.map(index => <th scope="col" key={index}>a[{index}]</th>)}</tr></thead>
      <tbody>{metadata.token_ids.map((token, position) => <tr key={position}><th scope="row">{position}</th><td>{token}</td><td className="experiment-token">{snapshot.input.tokens[position]?.decoded}</td>
        {metadata.neuron_indices.map((index, column) => { const exact = signed(values[position * metadata.neuron_indices.length + column]!); return <td key={index}><span tabIndex={0} title={exact} aria-label={`Position ${position}, token ${token}, neuron ${index}: ${exact}`}>{exact}</span></td>; })}</tr>)}</tbody>
    </table></div>;
  const columns = metadata.candidate_token_ids.length;
  return <>
    <p className="experiment-note">Full-vocabulary probabilities, shown as percentages. Candidate union in server order; the displayed subset is not renormalized. Δ is percentage points from baseline.</p>
    <div className="experiment-table-scroll" tabIndex={0} role="region" aria-label="Next-token probabilities and percentage-point deltas"><table>
      <thead><tr><th scope="col">Candidate ID</th>{metadata.conditions.map(condition => <th scope="col" key={condition.id}>{conditionLabel(condition)}</th>)}</tr></thead>
      <tbody>{metadata.candidate_token_ids.map((token, column) => <tr key={token}><th scope="row">{token}</th>{metadata.conditions.map((condition, row) => {
        const value = values[row * columns + column]!, baseline = values[column]!;
        return <td key={condition.id}><span tabIndex={0} title={`Probability ${value}; delta ${value - baseline}`} aria-label={`${condition.id}, token ${token}: probability ${value}, delta ${percentagePoints(value, baseline)}`}>
          {(value * 100).toFixed(4)}%{row > 0 && <small>Δ {percentagePoints(value, baseline)}</small>}</span></td>;
      })}</tr>)}</tbody>
    </table></div>
    <details><summary>Server top-k order per condition</summary>{metadata.conditions.map((condition, index) => <p key={condition.id}>{conditionLabel(condition)}: {metadata.top_token_ids[index]!.join(', ')}</p>)}</details>
    <h3>Free continuations · separate contexts</h3><p className="experiment-note">Each branch continues its own text. Positions in divergent branches are not aligned causal comparisons.</p>
    <div className="experiment-continuations">{metadata.continuations.map((continuation, index) => <article key={continuation.condition_id} aria-label={`${continuation.condition_id} continuation`}>
      <h4>{conditionLabel(metadata.conditions[index]!)}</h4><p>{continuation.finish_reason === 'not_requested' ? 'No generation requested.' : `${continuation.token_ids.length} tokens · ${continuation.finish_reason === 'eos' ? 'EOS reached' : 'length limit'}`}</p>
      {continuation.finish_reason !== 'not_requested' && <pre>{continuation.text}</pre>}
    </article>)}</div>
  </>;
}
