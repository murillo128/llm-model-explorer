import { useEffect, useId, useMemo, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { Graph, GraphNode } from './graph';
import type { ProjectionOptions } from './projection';
import { displayLabel, patternSummary } from './presentation';

type Repetition = Graph['repetitions'][number];
export interface NavigationItem { id: string; label: string; kind: 'node' | 'stack' }
export interface ControlSelection {
  label: string; detail: string; edge: boolean; nodeId?: string;
  inspect?: ((trigger: HTMLElement) => void) | undefined;
  explore?: (() => void) | undefined;
  center: () => void; clear: () => void;
}
interface Props {
  graph: Graph; focus: string | null; stack: Repetition | undefined;
  instanceId: string | undefined; visibleInstances: string[]; options: ProjectionOptions;
  breadcrumbs: NavigationItem[]; selection: ControlSelection | undefined;
  isolation: { back: () => void; viewInModel: () => void; expand: () => void; nodeIds: Set<string>; excludedEdges: string[] } | undefined;
  picker: RefObject<HTMLButtonElement | null>;
  reveal: (id: string) => void; navigate: (item: NavigationItem) => void;
  overview: () => void; fit: () => void; expandAll: () => void; collapseAll: () => void;
  chooseInstance: (id: string) => void; exploreStack: (id: string, start?: number) => void;
  windowSize: number; focusLayer: (() => void) | undefined; focusMlp: (() => void) | undefined;
  stateFocus: (() => void) | undefined; toggleSelected: (() => void) | undefined;
  preferences: (patch: Partial<ProjectionOptions>) => void;
  zoomIn: () => void; zoomOut: () => void; filtered: boolean;
}

/** Presentation-only aliases. Source records and identity stay with the canvas. */
function nodeLabel(node: GraphNode, graph: Graph) {
  return displayLabel({ id: node.id, kind: node.kind, label: node.label, record: node, sourceIds: [node.id], ports: [], expanded: false }, graph);
}

export function ArchitectureControls(props: Props) {
  const { graph, stack, selection, options, picker } = props;
  const [popover, setPopover] = useState<'search' | 'options' | null>(null);
  const [query, setQuery] = useState(''), [active, setActive] = useState(0);
  const [outsideOpen, setOutsideOpen] = useState(false);
  const controls = useRef<HTMLDivElement>(null), search = useRef<HTMLInputElement>(null);
  const optionsTrigger = useRef<HTMLButtonElement>(null), popup = useRef<HTMLDivElement>(null);
  const listId = useId(), optionsId = useId();
  const index = useMemo(() => {
    const labels = new Map(graph.nodes.map((node) => [node.id, nodeLabel(node, graph)]));
    const records = new Map(graph.nodes.map((node) => [node.id, node]));
    const instances = new Map(graph.repetitions.flatMap((r) => r.instances.map((i) => [i.node_id, { r, i }] as const)));
    return graph.nodes.map((node) => {
      const parents: string[] = [];
      let parent = node.parent_id ? records.get(node.parent_id) : undefined;
      while (parent) {
        parents.unshift(labels.get(parent.id)!);
        const repetition = instances.get(parent.id);
        if (repetition) parents.unshift(repetition.r.label);
        parent = parent.parent_id ? records.get(parent.parent_id) : undefined;
      }
      const instance = instances.get(node.id);
      if (instance) parents.push(instance.r.label, instance.i.variant.replaceAll('_', ' '));
      const context = parents.join(' / ');
      // Only identity provenance is a component-path alias. General description
      // names (for example a fixture or producer name) are not node identities.
      const sources = node.provenance.filter((p) => p.kind === 'description' &&
        p.rule === 'Semantic source key in the reviewed packaged description').map((p) => p.source).join(' ');
      return { id: node.id, label: labels.get(node.id)!, context, fullLabel: node.label,
        search: `${labels.get(node.id)} ${context} ${node.label} ${node.id} ${sources} ${node.references.filter((r) => r.kind === 'module').map((r) => r.name).join(' ')}`.toLocaleLowerCase() };
    });
  }, [graph]);
  const matches = useMemo(() => {
    const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    return index.filter((item) => terms.every((term) => item.search.includes(term)));
  }, [index, query]);
  const close = (restore = true) => {
    setPopover(null);
    if (restore) (popover === 'search' ? picker.current : optionsTrigger.current)?.focus();
  };
  useEffect(() => {
    if (popover === 'search') search.current?.focus();
    if (popover === 'options') popup.current?.querySelector<HTMLButtonElement>('button')?.focus();
    if (!popover) return;
    const outside = (event: PointerEvent) => {
      if (!controls.current?.contains(event.target as Node)) setPopover(null);
    };
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [popover]);
  useEffect(() => {
    if (popover === 'search') document.getElementById(`${listId}-${active}`)?.scrollIntoView({ block: 'nearest' });
  }, [active, listId, popover]);
  const run = (action: () => void) => { close(); action(); };
  const choose = (id: string) => { close(); props.reveal(id); };
  const current = stack?.instances.findIndex((i) => i.node_id === props.instanceId) ?? -1;
  const window = stack && options.repetitions?.[stack.id];
  const visible = stack?.instances.filter((i) => props.visibleInstances.includes(i.node_id)) ?? [];
  const entryNodes = graph.nodes.filter((node) => (!node.parent_id || graph.nodes.some((p) => p.id === node.parent_id && !p.parent_id)) &&
    !graph.repetitions.some((r) => r.parent_id === node.id || r.instances.some((i) => i.node_id === node.id)));
  return <div className="architecture-controls" ref={controls} onKeyDown={(event) => {
    if (popover && event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); }
  }} onBlur={(event) => {
    if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget)) setPopover(null);
  }}>
    <div className="architecture-toolbar" aria-label="Graph navigation">
      {props.isolation && <span className="architecture-isolated-state">Isolated component</span>}
      <nav className="architecture-breadcrumbs" aria-label="Architecture focus">
        <button onClick={props.overview} aria-label="Model overview" aria-current={!props.focus && !stack && !props.isolation ? 'location' : undefined}>Model</button>
        {props.breadcrumbs.map((item, i) => <span key={`${item.kind}:${item.id}`}>
          <span aria-hidden="true">/</span><button title={item.label} aria-current={i === props.breadcrumbs.length - 1 ? 'location' : undefined}
            onClick={() => props.navigate(item)}>{item.label}</button>
        </span>)}
      </nav>
      {graph.coverage === 'partial' && <span className="architecture-partial">Partial coverage</span>}
      <button ref={picker} aria-expanded={popover === 'search'} aria-controls={listId} aria-haspopup="dialog"
        onClick={() => { setQuery(''); setActive(0); setPopover(popover === 'search' ? null : 'search'); }}>Find component</button>
      <button onClick={props.fit}>Fit view</button>
      <button ref={optionsTrigger} aria-expanded={popover === 'options'} aria-controls={optionsId} aria-haspopup="dialog"
        onClick={() => setPopover(popover === 'options' ? null : 'options')}>View options</button>
    </div>
    {(props.isolation || stack || entryNodes.length > 0 || graph.repetitions.length > 0 || selection) && <div className="architecture-context-row">
      <div className="architecture-context-navigation" aria-label={props.isolation ? 'Component navigation' : stack ? 'Stack navigation' : 'Model components'}>
        {props.isolation ? <>
          <button onClick={props.isolation.back}>Back</button>
          <button onClick={props.isolation.viewInModel}>View in model</button>
          <button onClick={props.isolation.expand}>Expand component</button>
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
        </> : <>
          {graph.repetitions.map((r) => <button key={r.id} aria-label={`Explore stack ${r.label}`} onClick={() => props.exploreStack(r.id)}>{r.label} <span>×{r.instances.length}</span></button>)}
          {entryNodes.map((node) => <button key={node.id} onClick={() => props.reveal(node.id)}>{nodeLabel(node, graph)}</button>)}
        </>}
      </div>
      {selection && <div className="architecture-selection" aria-label="Graph selection" data-node-id={selection.nodeId}>
        <span title={selection.detail}>{selection.label}</span>
        {selection.inspect && <button aria-label={selection.edge ? 'Inspect connection' : 'Inspect selected'} onClick={(event) => selection.inspect?.(event.currentTarget)}>Inspect</button>}
        {selection.explore && <button onClick={selection.explore}>Explore component</button>}
        <button aria-label={selection.edge ? 'Center connection' : 'Center selected'} onClick={selection.center}>Center</button>
        <button aria-label={selection.edge ? 'Clear connection selection' : 'Clear node selection'} onClick={() => { picker.current?.focus(); selection.clear(); }}>×</button>
      </div>}
    </div>}
    {popover === 'search' && <div className="architecture-control-popover architecture-search" role="dialog" aria-label="Find component" id={listId}>
      <input ref={search} role="combobox" aria-label="Search components" aria-expanded="true" aria-controls={`${listId}-results`} aria-autocomplete="list"
        aria-activedescendant={matches[active] ? `${listId}-${active}` : undefined} placeholder="Name, layer or component path…" value={query}
        onChange={(event) => { setQuery(event.target.value); setActive(0); }} onKeyDown={(event) => {
          if (['ArrowDown', 'ArrowUp', 'Home', 'End', 'Enter'].includes(event.key)) {
            // Home/End remain text-caret controls; only Alt+Home/End jump through results.
            if ((event.key === 'Home' || event.key === 'End') && !event.altKey) return;
            event.preventDefault(); event.stopPropagation();
            if (event.key === 'Enter') { if (matches[active]) choose(matches[active].id); }
            else setActive(event.key === 'Home' ? 0 : event.key === 'End' ? matches.length - 1 : Math.max(0, Math.min(matches.length - 1, active + (event.key === 'ArrowDown' ? 1 : -1))));
          }
        }} />
      <div className="architecture-search-results" role="listbox" aria-label="Components" id={`${listId}-results`}>
        {matches.map((item, i) => <div role="option" key={item.id} id={`${listId}-${i}`} aria-selected={active === i} data-node-id={item.id}
          title={item.fullLabel} onPointerDown={(event) => event.preventDefault()} onClick={() => choose(item.id)}>
          <strong>{item.label}</strong><span>{item.context || 'Model component'}{props.isolation && !props.isolation.nodeIds.has(item.id) ? ' · Outside component; reveal in model' : ''}</span>
        </div>)}
      </div>
      <span className="architecture-search-count" role="status">{matches.length ? `${matches.length} components · ↑ ↓ to choose · Enter to reveal` : 'No matching components'}</span>
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
        {props.isolation && <details onToggle={(event) => setOutsideOpen(event.currentTarget.open)}>
          <summary>Outside component: {graph.nodes.length - props.isolation.nodeIds.size} source records · {props.isolation.excludedEdges.length} connections</summary>
          <p>All source interfaces and connections remain in the model. Choose a record to reveal it in model context, then Inspect for its complete interface.</p>
          {outsideOpen && graph.nodes.filter((node) => !props.isolation!.nodeIds.has(node.id)).map((node) =>
            <p key={node.id}><button onClick={() => choose(node.id)}>{nodeLabel(node, graph)}</button> <code>{node.id}</code></p>)}
        </details>}
        <p>Graph: {graph.graph_id}</p>
        {graph.diagnostics.map((d, i) => <p key={i}>{d.message}</p>)}
      </details>
    </div>}
  </div>;
}
