import { useId, useLayoutEffect, useMemo, useRef } from 'react';
import type { KeyboardEvent, ReactNode, RefObject } from 'react';
import type { Graph, GraphView } from './graph';
import { browserIndex, browserPredicates, interfaceSearch } from './browser-model';
import { interfaceIndex } from './interfaces';
import type { BoundarySelection } from './interfaces';

type SectionName = 'model' | 'repetitions' | 'shared';

function BrowserIcon({ kind }: { kind: 'component' | 'block' | 'family' | 'stack' }) {
  return <svg data-icon={kind} aria-hidden="true" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2">
    {kind === 'component' ? <path d="M2 4h5l1.5 2H14v7H2zM2 4V2h4l2 2h5v2" /> : kind === 'family'
      ? <><path d="M5 2h9v9M3 4h9v9" /><rect x="1" y="6" width="9" height="8" rx="1" /></>
      : kind === 'stack' ? <><path d="m2 5 6-3 6 3-6 3z" /><path d="m2 8 6 3 6-3M2 11l6 3 6-3" /></>
      : <><rect x="3" y="3" width="10" height="10" rx="1" /><path d="M6 3v10M3 6h10" /></>}
  </svg>;
}

function Section({ name, label, open, controlRef, bodyRef, bodyId, toggle, heading, children }: {
  name: SectionName; label: string; open: boolean; controlRef: RefObject<HTMLButtonElement | null>;
  bodyRef: RefObject<HTMLDivElement | null>; bodyId: string; toggle: () => void; heading?: ReactNode; children: ReactNode;
}) {
  return <section className="architecture-browser-section" data-browser-section={name}>
    <div className="architecture-browser-section-header">
      <button ref={controlRef} className="architecture-browser-section-disclosure" aria-label={`${open ? 'Collapse' : 'Expand'} ${label} section`}
        aria-expanded={open} aria-controls={bodyId} onClick={toggle}>
        <svg aria-hidden="true" viewBox="0 0 16 16"><path d="m5 3 5 5-5 5" /></svg>
      </button>
      <h3>{heading ?? label}</h3>
    </div>
    {open && <div ref={bodyRef} id={bodyId} className="architecture-browser-section-body">{children}</div>}
  </section>;
}
interface Props {
  graph: Graph; view: GraphView; searchRef: RefObject<HTMLInputElement | null>;
  select: (id: string) => void; selectFamily: (id: string) => void; toggle: (id: string) => void;
  exploreStack: (id: string) => void;
  selectBoundary?: (selection: BoundarySelection) => void;
}
export function ArchitectureBrowser({ graph, view, searchRef, select, selectFamily, toggle, exploreStack, selectBoundary }: Props) {
  const interfaces = useMemo(() => interfaceIndex(graph), [graph]);
  const { inside, expanded } = browserPredicates(graph, view);
  const entries = useMemo(() => browserIndex(graph), [graph]);
  const byId = useMemo(() => new Map(entries.map((entry) => [entry.node.id, entry])), [entries]);
  const scroll = useRef<HTMLDivElement>(null);
  const sectionPrefix = useId();
  const modelControl = useRef<HTMLButtonElement>(null), repetitionControl = useRef<HTMLButtonElement>(null), sharedControl = useRef<HTMLButtonElement>(null);
  const modelBody = useRef<HTMLDivElement>(null), repetitionBody = useRef<HTMLDivElement>(null), sharedBody = useRef<HTMLDivElement>(null);
  const sectionControls = { model: modelControl, repetitions: repetitionControl, shared: sharedControl };
  const sectionBodies = { model: modelBody, repetitions: repetitionBody, shared: sharedBody };
  const { query, families, selectedFamily } = view.browser;
  const searching = Boolean(query.trim());
  const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches = entries.filter((entry) => terms.every((term) => entry.search.includes(term)));
  const interfaceMatches = (id: string) => searching ? interfaces.interfaces.filter((item) => item.owner.id === id && terms.every((term) => interfaceSearch(item.node).includes(term))) : [];
  const selectEntry = (id: string) => {
    const ports = interfaceMatches(id);
    if (ports.length && selectBoundary) selectBoundary({ kind: 'boundary', owner: ports[0]!.owner, endpoints: ports.map((p) => p.endpoint) });
    else select(id);
  };
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
  const setQuery = (next: string) => {
    const nextSearching = Boolean(next.trim());
    if (!searching && nextSearching) update({ query: next, sectionsBeforeSearch: { ...view.browser.sections },
      sections: { model: true, repetitions: false, shared: true } });
    else if (searching && !nextSearching) update({ query: next,
      sections: view.browser.sectionsBeforeSearch ?? view.browser.sections, sectionsBeforeSearch: null });
    else update({ query: next });
  };
  const toggleSection = (name: SectionName) => {
    const body = sectionBodies[name].current;
    const returnFocus = body?.contains(document.activeElement);
    update({ sections: { ...view.browser.sections, [name]: !view.browser.sections[name] } });
    if (returnFocus) requestAnimationFrame(() => sectionControls[name].current?.focus());
  };
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
    const expandable = node.kind === 'group' && node.children.some(interfaces.eligible);
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
        onClick={() => selectEntry(id)} onDoubleClick={() => { if (expandable) toggle(id); }} onKeyDown={(event) => keys(event, id, expandable)}>
        <span className="architecture-browser-icon"><BrowserIcon kind={expandable ? 'component' : 'block'} /></span>
        <span className="architecture-browser-copy"><span className="architecture-browser-primary">{label}</span>
          {(searching || shared) && <small className="architecture-browser-meta">{path}</small>}
          {interfaceMatches(id).length > 0 && <small className="architecture-browser-meta">Ports: {interfaceMatches(id).map((p) => p.label).join(', ')}</small>}
        </span>
        <span role="tooltip" className="architecture-browser-tooltip">{detail}</span>
      </button>
    </div>;
  }
  const modelSelected = view.boundary?.owner.kind === 'model';
  const modelInterfaceMatches = interfaces.outer.kind === 'model' ? interfaceMatches(interfaces.outer.id) : [];
  const visibleFamilies = (graph.templates ?? []).flatMap((family) => {
    const familyMatch = terms.every((term) => `${family.label} ${family.id}`.toLocaleLowerCase().includes(term));
    const instances = family.instances.filter((instance) => familyMatch || terms.every((term) => byId.get(instance.node_id)?.search.includes(term)));
    return searching && !instances.length && !familyMatch ? [] : [{ family, instances }];
  });
  return <>
    <div className="architecture-browser-search"><input ref={searchRef} type="search" aria-label="Search components" placeholder="Search components…" value={query}
      onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => {
        if (event.key === 'Enter' && searching) {
          event.preventDefault();
          if (interfaces.outer.kind === 'model' && interfaceMatches(interfaces.outer.id).length) selectEntry(interfaces.outer.id);
          else if (matches[0]) selectEntry(matches[0].node.id);
        }
        if (event.key === 'ArrowDown') { event.preventDefault(); scroll.current?.querySelector<HTMLButtonElement>('[data-browser-name]')?.focus(); }
        if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setQuery(''); }
      }} />
      {query && <button aria-label="Clear component search" onClick={() => { setQuery(''); searchRef.current?.focus(); }}>×</button>}
    </div>
    <div className="architecture-browser-scroll" ref={scroll} onScroll={(event) => {
      update({ [searching ? 'searchScroll' : 'treeScroll']: event.currentTarget.scrollTop });
    }}>
      <Section name="model" label="Model" open={view.browser.sections.model} controlRef={modelControl} bodyRef={modelBody}
        bodyId={`${sectionPrefix}-model`} toggle={() => toggleSection('model')}
        heading={interfaces.outer.kind === 'model' && selectBoundary ? <button className="architecture-browser-section-name" data-selected={modelSelected}
          aria-label="Select Model boundary" aria-pressed={modelSelected}
          onClick={() => selectBoundary({ kind: 'boundary', owner: interfaces.outer, endpoints: [] })}>Model</button> : undefined}>
        <div role={searching ? 'group' : 'tree'} aria-label={searching ? 'Model search results' : 'Model components'}>
          {visible.map((entry) => row(entry.node.id))}
          {modelInterfaceMatches.length > 0 && <div className="architecture-browser-row" data-node-id={interfaces.outer.id} data-selected={modelSelected}>
            <span className="architecture-browser-spacer" />
            <button data-browser-name={interfaces.outer.id} className="architecture-browser-name" aria-label="Select Model interface"
              aria-description={`Model · ${interfaces.outer.id}`} onClick={() => selectEntry(interfaces.outer.id)} onKeyDown={(event) => keys(event, interfaces.outer.id, false)}>
              <span className="architecture-browser-icon"><BrowserIcon kind="component" /></span>
              <span className="architecture-browser-copy"><span className="architecture-browser-primary">Model</span>
                <small className="architecture-browser-meta">Ports: {modelInterfaceMatches.map((p) => p.label).join(', ')}</small>
              </span>
            </button>
          </div>}
          {searching && !visible.length && !modelInterfaceMatches.length && <p className="architecture-browser-empty">No matching components</p>}
        </div>
      </Section>
      <Section name="repetitions" label="Repetition windows" open={view.browser.sections.repetitions} controlRef={repetitionControl}
        bodyRef={repetitionBody} bodyId={`${sectionPrefix}-repetitions`} toggle={() => toggleSection('repetitions')}>
        <div role="group" aria-label="Repetition windows">
          {searching ? <p className="architecture-browser-empty">Repetition windows are hidden while filtering components</p>
            : graph.repetitions.length > 0 ? graph.repetitions.map((stack) => <div key={stack.id} className="architecture-browser-row" data-stack-id={stack.id}>
              <span className="architecture-browser-spacer" />
              <button data-browser-name={stack.id} className="architecture-browser-name" onClick={() => exploreStack(stack.id)}
                onKeyDown={(event) => keys(event, stack.id, false)} aria-label={`Explore stack ${stack.label}`}
                aria-description={`${stack.label} · ${stack.instances.length} instances · ${stack.id}`} title={`${stack.label} · ${stack.instances.length} instances · ${stack.id}`}>
                <span className="architecture-browser-icon"><BrowserIcon kind="stack" /></span>
                <span className="architecture-browser-copy"><span className="architecture-browser-primary">{stack.label}</span>
                  <small className="architecture-browser-meta">{stack.instances.length} instances</small>
                </span>
              </button>
            </div>) : <p className="architecture-browser-empty">No repetition windows</p>}
        </div>
      </Section>
      <Section name="shared" label="Shared" open={view.browser.sections.shared} controlRef={sharedControl} bodyRef={sharedBody}
        bodyId={`${sectionPrefix}-shared`} toggle={() => toggleSection('shared')}>
        <div role="group" aria-label="Shared structures">
          {visibleFamilies.map(({ family, instances }) => {
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
                  <span className="architecture-browser-icon"><BrowserIcon kind="family" /></span>
                  <span className="architecture-browser-copy"><span className="architecture-browser-primary">{family.label}</span>
                    <small className="architecture-browser-meta">{family.instances.length} instances</small>
                  </span>
                </button>
              </div>
              {(open || searching) && <div role="group" aria-label={`${family.label} instances`}>{instances.map((instance) => row(instance.node_id, true))}</div>}
            </div>;
          })}
          {!visibleFamilies.length && <p className="architecture-browser-empty">{searching && graph.templates?.length
            ? 'No matching shared structures' : 'No verified shared structures'}</p>}
        </div>
      </Section>
    </div>
  </>;
}
