import { useLayoutEffect, useMemo, useRef } from 'react';
import type { KeyboardEvent, RefObject } from 'react';
import type { Graph, GraphView } from './graph';
import { browserIndex } from './browser-model';

function BrowserIcon({ kind }: { kind: 'component' | 'block' | 'family' }) {
  return <svg data-icon={kind} aria-hidden="true" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2">
    {kind === 'component' ? <path d="M2 4h5l1.5 2H14v7H2zM2 4V2h4l2 2h5v2" /> : kind === 'family'
      ? <><path d="M5 2h9v9M3 4h9v9" /><rect x="1" y="6" width="9" height="8" rx="1" /></>
      : <><rect x="3" y="3" width="10" height="10" rx="1" /><path d="M6 3v10M3 6h10" /></>}
  </svg>;
}
interface Props {
  graph: Graph; view: GraphView; searchRef: RefObject<HTMLInputElement | null>;
  select: (id: string) => void; selectFamily: (id: string) => void; toggle: (id: string) => void; expanded: (id: string) => boolean;
  inside: (id: string) => boolean; exploreStack: (id: string) => void;
}
export function ArchitectureBrowser({ graph, view, searchRef, select, selectFamily, toggle, expanded, inside, exploreStack }: Props) {
  const entries = useMemo(() => browserIndex(graph), [graph]);
  const byId = useMemo(() => new Map(entries.map((entry) => [entry.node.id, entry])), [entries]);
  const scroll = useRef<HTMLDivElement>(null);
  const { query, families, selectedFamily } = view.browser;
  const searching = Boolean(query.trim());
  const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches = entries.filter((entry) => terms.every((term) => entry.search.includes(term)));
  const visible = searching ? matches : entries.filter(({ node }) => {
    let parent = node.parent_id;
    while (parent) {
      if (!expanded(parent)) return false;
      parent = byId.get(parent)?.node.parent_id;
    }
    return true;
  });
  useLayoutEffect(() => {
    if (scroll.current) scroll.current.scrollTop = searching ? view.browser.searchScroll : view.browser.treeScroll;
  }, [searching, view]);
  const update = (patch: Partial<GraphView['browser']>) => view.update({ browser: { ...view.browser, ...patch } });
  function keys(event: KeyboardEvent<HTMLButtonElement>, id: string, expandable: boolean) {
    const buttons = [...(scroll.current?.querySelectorAll<HTMLButtonElement>('[data-browser-name]') ?? [])];
    const at = buttons.indexOf(event.currentTarget);
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      buttons[event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : Math.max(0, Math.min(buttons.length - 1, at + (event.key === 'ArrowDown' ? 1 : -1)))]?.focus();
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      if (expandable && !expanded(id)) toggle(id); else buttons[at + 1]?.focus();
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      if (expandable && expanded(id)) toggle(id);
      else buttons.find((button) => button.dataset.browserName === byId.get(id)?.node.parent_id)?.focus();
    } else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); searchRef.current?.focus(); }
  }
  function row(id: string, shared = false) {
    const entry = byId.get(id);
    if (!entry) return null;
    const { node, label, path, depth } = entry;
    const expandable = node.kind === 'group' && node.children.length > 0;
    const isExpanded = expanded(id), selected = view.selectionMode === 'source' && view.selected === id && !view.edge;
    const detail = `${node.label} · ${path} · ${id}${inside(id) ? '' : ' · Model context; outside current canvas'}`;
    return <div key={id} className="architecture-browser-row" data-node-id={id} data-selected={selected} data-outside={!inside(id)}
      style={{ paddingLeft: `${searching || shared ? 0 : Math.min(depth, 4) * 12}px` }}>
      {expandable ? <button className="architecture-browser-disclosure" aria-label={`${isExpanded ? 'Collapse' : 'Expand'} component ${label}`}
        aria-expanded={isExpanded} onDoubleClick={(event) => event.stopPropagation()}
        onClick={(event) => { if (event.detail < 2) toggle(id); }}>{isExpanded ? '−' : '+'}</button> : <span className="architecture-browser-spacer" />}
      <button data-browser-name={id} className="architecture-browser-name" role={searching || shared ? undefined : 'treeitem'}
        aria-level={searching || shared ? undefined : depth + 1} aria-expanded={expandable ? isExpanded : undefined}
        aria-selected={searching || shared ? undefined : selected} aria-pressed={searching || shared ? selected : undefined}
        aria-label={`Select component ${label}`} aria-description={detail} title={detail}
        onClick={() => select(id)} onDoubleClick={() => { if (expandable) toggle(id); }} onKeyDown={(event) => keys(event, id, expandable)}>
        <BrowserIcon kind={expandable ? 'component' : 'block'} /><span><span>{label}</span>{(searching || shared) && <small>{path}</small>}</span>
        <span role="tooltip" className="architecture-browser-tooltip">{detail}</span>
      </button>
    </div>;
  }
  return <>
    <div className="architecture-browser-search"><input ref={searchRef} type="search" aria-label="Search components" placeholder="Search components…" value={query}
      onChange={(event) => update({ query: event.target.value })} onKeyDown={(event) => {
        if (event.key === 'Enter' && searching && matches[0]) { event.preventDefault(); select(matches[0].node.id); }
        if (event.key === 'ArrowDown') { event.preventDefault(); scroll.current?.querySelector<HTMLButtonElement>('[data-browser-name]')?.focus(); }
        if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); update({ query: '' }); }
      }} />
      {query && <button aria-label="Clear component search" onClick={() => { update({ query: '' }); searchRef.current?.focus(); }}>×</button>}
    </div>
    <div className="architecture-browser-scroll" ref={scroll} onScroll={(event) => {
      update({ [searching ? 'searchScroll' : 'treeScroll']: event.currentTarget.scrollTop });
    }}>
      <h3>Model</h3>
      <div role={searching ? 'group' : 'tree'} aria-label={searching ? 'Model search results' : 'Model components'}>
        {visible.map((entry) => row(entry.node.id))}
        {searching && !visible.length && <p className="architecture-browser-empty">No matching components</p>}
      </div>
      {!searching && graph.repetitions.length > 0 && <details open className="architecture-browser-stacks"><summary>Repetition windows</summary>
        {graph.repetitions.map((stack) => <button key={stack.id} onClick={() => exploreStack(stack.id)} aria-label={`Explore stack ${stack.label}`}>{stack.label} · {stack.instances.length} instances</button>)}
      </details>}
      <h3>Shared</h3>
      <div role="group" aria-label="Shared structures">
        {graph.templates?.map((family) => {
          const familyMatch = terms.every((term) => `${family.label} ${family.id}`.toLocaleLowerCase().includes(term));
          const instances = family.instances.filter((i) => familyMatch || terms.every((term) => byId.get(i.node_id)?.search.includes(term)));
          if (searching && !instances.length && !familyMatch) return null;
          const open = families.includes(family.id);
          return <div key={family.id} className="architecture-browser-family" data-family-id={family.id}>
            <div className="architecture-browser-row" data-selected={selectedFamily === family.id}>
              <button className="architecture-browser-disclosure" aria-label={`${open ? 'Hide' : 'Show'} instances of ${family.label}`} aria-expanded={open}
                onClick={() => update({ families: open ? families.filter((id) => id !== family.id) : [...families, family.id] })}>{open ? '−' : '+'}</button>
              <button data-browser-name={family.id} className="architecture-browser-name" aria-label={`Select shared family ${family.label}`} title={`${family.label} · ${family.instances.length} instances · ${family.id}`}
                aria-pressed={selectedFamily === family.id} onClick={() => selectFamily(family.id)} onKeyDown={(event) => {
                  if (event.key === 'ArrowRight' && !open || event.key === 'ArrowLeft' && open) {
                    event.preventDefault();
                    update({ families: open ? families.filter((id) => id !== family.id) : [...families, family.id] });
                  } else keys(event, family.id, false);
                }}>
                <BrowserIcon kind="family" /><span>{family.label}<small>{family.instances.length} instances</small></span>
              </button>
            </div>
            {(open || searching) && <div role="group" aria-label={`${family.label} instances`}>{instances.map((i) => row(i.node_id, true))}</div>}
          </div>;
        })}
        {!graph.templates?.length && <p className="architecture-browser-empty">No verified shared structures</p>}
      </div>
    </div>
  </>;
}
