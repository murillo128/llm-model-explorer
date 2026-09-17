import { useEffect, useId, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { Graph } from './graph';
import type { ProjectionOptions } from './projection';
import { instanceOf, patternSummary } from './presentation';
import { componentLabel } from './browser-model';
import type { Template } from './shared-structure';
import { interfaceIndex } from './interfaces';

type Repetition = Graph['repetitions'][number];
export interface NavigationItem { id: string; label: string; kind: 'node' | 'stack' }
export interface ControlSelection {
  label: string; detail: string; edge: boolean; nodeId?: string;
  inspect?: ((trigger: HTMLElement) => void) | undefined;
  explore?: (() => void) | undefined;
  shared?: (() => void) | undefined;
  viewInModel?: (() => void) | undefined;
  center: () => void; clear: () => void;
}
interface Props {
  shared: { template: Template; instanceId: string | null; choose: (id: string | null) => void } | undefined;
  family: { label: string; explore: () => void; clear: () => void } | undefined;
  returnContext: (() => void) | undefined;
  graph: Graph; focus: string | null; stack: Repetition | undefined;
  instanceId: string | undefined; visibleInstances: string[]; options: ProjectionOptions;
  breadcrumbs: NavigationItem[]; selection: ControlSelection | undefined;
  isolation: { back: () => void; viewInModel: (() => void) | undefined; expand: () => void; nodeIds: Set<string>; excludedEdges: string[] } | undefined;
  picker: RefObject<HTMLButtonElement | null>;
  reveal: (id: string) => void; navigate: (item: NavigationItem) => void;
  overview: () => void; fit: () => void; expandAll: () => void; collapseAll: () => void;
  chooseInstance: (id: string) => void; exploreStack: (id: string, start?: number) => void;
  windowSize: number; focusLayer: (() => void) | undefined; focusMlp: (() => void) | undefined;
  stateFocus: (() => void) | undefined; toggleSelected: (() => void) | undefined;
  preferences: (patch: Partial<ProjectionOptions>) => void;
  zoomIn: () => void; zoomOut: () => void; filtered: boolean;
}

export function ArchitectureControls(props: Props) {
  const { graph, stack, selection, options, picker } = props;
  const [popover, setPopover] = useState<'options' | null>(null);
  const [outsideOpen, setOutsideOpen] = useState(false);
  const controls = useRef<HTMLDivElement>(null);
  const optionsTrigger = useRef<HTMLButtonElement>(null), popup = useRef<HTMLDivElement>(null);
  const optionsId = useId();
  const close = (restore = true) => {
    setPopover(null);
    if (restore) optionsTrigger.current?.focus();
  };
  useEffect(() => {
    if (popover === 'options') popup.current?.querySelector<HTMLButtonElement>('button')?.focus();
    if (!popover) return;
    const outside = (event: PointerEvent) => {
      if (!controls.current?.contains(event.target as Node)) setPopover(null);
    };
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [popover]);
  const run = (action: () => void) => { close(); action(); };
  const choose = (id: string) => { close(); props.reveal(id); };
  const current = stack?.instances.findIndex((i) => i.node_id === props.instanceId) ?? -1;
  const window = stack && options.repetitions?.[stack.id];
  const visible = stack?.instances.filter((i) => props.visibleInstances.includes(i.node_id)) ?? [];
  const outside = popover === 'options' && props.isolation && !props.shared ? graph.nodes.filter((node) =>
    !props.isolation!.nodeIds.has(node.id) && interfaceIndex(graph).eligible(node.id)) : [];
  return <div className="architecture-controls" ref={controls} onKeyDown={(event) => {
    if (popover && event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); }
  }} onBlur={(event) => {
    if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget)) setPopover(null);
  }}>
    <div className="architecture-toolbar" aria-label="Graph navigation">
      {props.isolation && <span className="architecture-isolated-state">{props.shared ? 'Shared structure' : 'Isolated component'}</span>}
      <nav className="architecture-breadcrumbs" aria-label="Architecture focus">
        <button ref={picker} onClick={props.overview} aria-label="Model overview" aria-current={!props.focus && !stack && !props.isolation ? 'location' : undefined}>Model</button>
        {props.breadcrumbs.map((item, i) => <span key={`${item.kind}:${item.id}`}>
          <span aria-hidden="true">/</span><button title={item.label} aria-current={i === props.breadcrumbs.length - 1 ? 'location' : undefined}
            onClick={() => props.navigate(item)}>{item.label}</button>
        </span>)}
      </nav>
      {graph.coverage === 'partial' && <span className="architecture-partial">Partial coverage</span>}
      {props.family ? <div className="architecture-selection" aria-label="Graph selection">
        <span title={props.family.label}>{props.family.label}</span><button onClick={props.family.explore}>Explore structure</button>
        <button aria-label="Clear family selection" onClick={() => { picker.current?.focus(); props.family!.clear(); }}>×</button>
      </div> : selection && <div className="architecture-selection" aria-label="Graph selection" data-node-id={selection.nodeId}>
        <span title={selection.detail}>{selection.label}</span>
        {selection.inspect && <button aria-label={selection.edge ? 'Inspect connection' : 'Inspect selected'} onClick={(event) => selection.inspect?.(event.currentTarget)}>Inspect</button>}
        {selection.explore && <button onClick={selection.explore}>Explore component</button>}
        {selection.viewInModel && <button onClick={selection.viewInModel}>View in model</button>}
        {selection.shared && <button onClick={selection.shared}>Explore structure</button>}
        <button aria-label={selection.edge ? 'Center connection' : 'Center selected'} onClick={selection.center}>Center</button>
        <button aria-label={selection.edge ? 'Clear connection selection' : 'Clear node selection'} onClick={() => { picker.current?.focus(); selection.clear(); }}>×</button>
      </div>}
      <button onClick={props.fit}>Fit view</button>
      <button ref={optionsTrigger} aria-expanded={popover === 'options'} aria-controls={optionsId} aria-haspopup="dialog"
        onClick={() => setPopover(popover === 'options' ? null : 'options')}>View options</button>
    </div>
    {(props.isolation || stack || props.returnContext) && <div className="architecture-context-row">
      <div className="architecture-context-navigation" aria-label={props.isolation ? 'Component navigation' : stack ? 'Stack navigation' : 'Model components'}>
        {props.returnContext && !props.isolation && <button onClick={props.returnContext}>Back</button>}
        {props.isolation ? <>
          <button onClick={props.isolation.back}>Back</button>
          {!selection?.viewInModel && <button disabled={!props.isolation.viewInModel} onClick={props.isolation.viewInModel}>View in model</button>}
          <button onClick={props.isolation.expand}>Expand component</button>
          {props.shared && <>
            <button aria-label="Previous shared instance" disabled={props.shared.template.instances.findIndex((i) => i.node_id === props.shared!.instanceId) <= 0}
              onClick={() => { const at = props.shared!.template.instances.findIndex((i) => i.node_id === props.shared!.instanceId); props.shared!.choose(props.shared!.template.instances[at - 1]!.node_id); }}>←</button>
            <select aria-label="Shared structure instance" value={props.shared.instanceId ?? ''} onChange={(event) => props.shared!.choose(event.target.value || null)}>
              <option value="">Structure only · choose an instance for weights</option>
              {props.shared.template.instances.map((item, position) => {
                const location = instanceOf(graph, item.node_id);
                return <option key={item.node_id} value={item.node_id}>{location ? `${location.repetition.label} / Layer ${location.instance.index}` : graph.nodes.find((n) => n.id === item.node_id)?.label} · family instance {position + 1} of {props.shared!.template.instances.length}</option>;
              })}
            </select>
            <button aria-label="Next shared instance" disabled={!props.shared.instanceId || props.shared.template.instances.at(-1)?.node_id === props.shared.instanceId}
              onClick={() => { const at = props.shared!.template.instances.findIndex((i) => i.node_id === props.shared!.instanceId); props.shared!.choose(props.shared!.template.instances[at + 1]!.node_id); }}>→</button>
            <span role="status">{props.shared.instanceId ? 'Exact instance selected' : 'No instance selected; weights require a choice.'}</span>
          </>}
        </> : stack ? <>
          <span className="architecture-stack-name" title={patternSummary(stack.instances)}>{stack.label} <span>({stack.instances.length})</span></span>
          <button aria-label={`Previous instance of ${stack.label}`} disabled={current <= 0} onClick={() => props.chooseInstance(stack.instances[current - 1]!.node_id)}>←</button>
          <label className="architecture-instance-picker"><span className="visually-hidden">Instance</span>
            <select aria-label={`Expand instance of ${stack.label}`} value={props.instanceId ?? ''} onChange={(event) => props.chooseInstance(event.target.value)}>
              <option value="" disabled>Choose instance…</option>
              {stack.instances.map((i) => <option key={i.node_id} value={i.node_id}>Instance {i.index} · {i.variant.replaceAll('_', ' ')}</option>)}
            </select>
          </label>
          <button aria-label={`Next instance of ${stack.label}`} disabled={current < 0 || current === stack.instances.length - 1} onClick={() => props.chooseInstance(stack.instances[current + 1]!.node_id)}>→</button>
          {props.instanceId && <button onClick={() => props.chooseInstance(props.instanceId!)}>Open instance</button>}
          {options.stateScope && <span className="architecture-state-focus">State dependencies</span>}
          <span className="architecture-visible-range">{visible.length ? `Visible ${visible.map((i) => i.index).join(', ')}` : 'Stack collapsed'}</span>
          {window && <>
            <button aria-label={`Previous window ${stack.label}`} disabled={window.start === 0} onClick={() => props.exploreStack(stack.id, Math.max(0, window.start - props.windowSize))}>‹</button>
            <button aria-label={`Next window ${stack.label}`} disabled={window.start + window.count >= stack.instances.length} onClick={() => props.exploreStack(stack.id, window.start + props.windowSize)}>›</button>
          </>}
        </> : null}
      </div>
    </div>}
    {popover === 'options' && <div ref={popup} className="architecture-control-popover architecture-options" role="dialog" aria-label="View options" id={optionsId}>
      <div className="architecture-option-actions">
        <button onClick={() => run(props.expandAll)}>{props.isolation ? 'Show all operations in model' : 'Show all operations'}</button>
        <button onClick={() => run(props.collapseAll)}>{props.isolation ? 'Collapse model' : 'Collapse all'}</button>
        <button disabled={!selection} onClick={() => selection && run(selection.center)}>Center selection</button>
        <button aria-label="Zoom graph in" onClick={props.zoomIn}>Zoom in</button>
        <button aria-label="Zoom graph out" onClick={props.zoomOut}>Zoom out</button>
        {props.toggleSelected && <button onClick={() => run(props.toggleSelected!)}>Toggle selected group</button>}
        {stack && !props.isolation && <button onClick={() => run(() => props.exploreStack(stack.id))}>Explore stack</button>}
        {props.focusLayer && <button onClick={() => run(props.focusLayer!)}>{options.stateScope ? 'Back to layer' : 'Focus layer'}</button>}
        {props.focusMlp && <button onClick={() => run(props.focusMlp!)}>Focus MLP</button>}
        {props.stateFocus && <button onClick={() => run(props.stateFocus!)}>State dependencies</button>}
      </div>
      <fieldset><legend>Presentation</legend>
        <label><input type="checkbox" checked={options.dimensions === true} onChange={(event) => props.preferences({ dimensions: event.target.checked })} /> Show dimensions</label>
        <label><input type="checkbox" checked={options.showUnused === true} onChange={(event) => props.preferences({ showUnused: event.target.checked })} /> Unused interfaces</label>
        <label><input type="checkbox" checked={options.showContext !== false} onChange={(event) => props.preferences({ showContext: event.target.checked })} /> Context</label>
        <label><input type="checkbox" checked={options.deriveMlp !== false} onChange={(event) => props.preferences({ deriveMlp: event.target.checked })} /> Group MLP</label>
      </fieldset>
      <details><summary>Graph details{graph.diagnostics.length ? ` · ${graph.diagnostics.length} diagnostics` : ''}</summary>
        <p>{graph.coverage === 'partial' ? 'Partial architecture coverage' : 'Complete within declared scope'} · {graph.scope.replaceAll('_', ' ')}</p>
        {props.filtered && <p>Unconsumed interface branches filtered</p>}
        {options.stateScope && <p>State dependencies only; other flows are filtered</p>}
        {selection && <p>{selection.detail}</p>}
        {props.isolation && !props.shared && <details onToggle={(event) => setOutsideOpen(event.currentTarget.open)}>
          <summary>Outside component: {outside.length} components · {props.isolation.excludedEdges.length} connections</summary>
          <p>Choose a component to reveal it in model context. Declared interfaces remain available through their owner ports and interface search.</p>
          {outsideOpen && outside.map((node) =>
            <p key={node.id}><button onClick={() => choose(node.id)}>{componentLabel(node, graph)}</button> <code>{node.id}</code></p>)}
        </details>}
        <p>Graph: {graph.graph_id}</p>
        {graph.diagnostics.map((d, i) => <p key={i}>{d.message}</p>)}
      </details>
    </div>}
  </div>;
}
