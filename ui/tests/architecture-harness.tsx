import { interfaceFixture } from './architecture-interface-fixture';
import { ArchitectureInspection } from '../src/architecture-explorer/ArchitectureInspection';
import type { ArchitectureSelection } from '../src/architecture-explorer/ArchitectureCanvas';
import { ApiClient } from '../src/api/client';
import { Lifetime } from '../src/app/lifetime';
import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArchitectureCanvas } from '../src/architecture-explorer/ArchitectureCanvas';
import { summaryFixture } from './architecture-summary-fixture';
import { GraphViews } from '../src/architecture-explorer/graph';
import { contractResponse, referenceFixture } from './architecture-fixtures';
import { makeProjectionFixture } from './architecture-projection-fixture';
import { makeTemplateFixture } from './architecture-template-fixture';
import { makeExplicitFixture } from './architecture-explicit-fixture';
import '../src/app/styles.css';

function fixture(name: string) {
  if (name.startsWith('interface-')) return { ...contractResponse, model_id: name, graph: interfaceFixture(name.includes('hybrid') ? 'hybrid' : name.includes('visual') ? 'visual' : 'dense', name.includes('many')) };
  if (name === 'summaries') return { ...contractResponse, model_id: name, graph: summaryFixture(new URLSearchParams(location.search).has('long')) };
  if (name === 'empty-group') {
    const graph = structuredClone(contractResponse.graph);
    const root = graph.nodes.find((n) => n.kind === 'group' && !n.parent_id)!;
    graph.graph_id = 'empty-group';
    graph.nodes = [{ ...root, kind: 'group', id: 'empty', label: 'Empty group', children: [], ports: [] }];
    graph.edges = []; graph.repetitions = []; graph.parameters = [];
    return { ...contractResponse, model_id: name, graph };
  }
  if (name === 'templates' || name === 'templates-absent') {
    const graph = makeTemplateFixture();
    if (name === 'templates-absent') delete graph.templates;
    return { model_id: name, status: 'available' as const, diagnostics: [], graph };
  }
  if (name === 'components-large') return { model_id: name, status: 'available' as const, diagnostics: [], graph: makeExplicitFixture({ count: 48 }) };
  if (name === 'components') return { model_id: name, status: 'available' as const, diagnostics: [], graph: makeExplicitFixture() };
  if (name === 'mixed-stacks') return { model_id: 'mixed-stacks', status: 'available' as const, diagnostics: [],
    graph: makeProjectionFixture({ variants: ['full_attention', 'linear_attention', 'full_attention', 'full_attention', 'linear_attention'], secondStack: 2 }) };

  if (name === 'partial') return { model_id: 'partial', status: 'available' as const, diagnostics: [],
    graph: makeProjectionFixture({ count: 1, partial: true, repetitions: false }) };
  if (name === 'contract') return contractResponse;
  if (name === 'hybrid' || name === 'connections' || name === 'visual-stacks') return {
    model_id: `authored-${name}`, status: 'available' as const, diagnostics: [],
    graph: makeProjectionFixture(name === 'connections' ? { count: 4 } : name === 'visual-stacks' ? { count: 3, secondStack: 2 } : {}),
  };
  return referenceFixture(name);
}

function Harness() {
  const [views] = useState(() => new GraphViews());
  const [name, setName] = useState(() => new URLSearchParams(location.search).get('fixture') ?? 'contract');
  const [shown, setShown] = useState(true);
  const [inspection, setInspection] = useState('');
  const [native, setNative] = useState<ArchitectureSelection | null>(null);
  const [context] = useState(() => ({ client: new ApiClient({ backendBaseUrl: 'http://127.0.0.1:1' }), session: { id: 'fixture-session', model_id: name }, sessionId: 'fixture-session', selection: new Lifetime(), selectedTensor: null, reportStatus: () => {} }));
  const [response, setResponse] = useState(() => {
    const response = structuredClone(fixture(name));
    if (location.search.includes('inert')) {
      response.graph.nodes[1]!.label = '<img src=x onerror=alert(1)>';
      response.graph.nodes.find((node) => node.id === 'linear0')!.ports.find((port) => port.id === 'out')!.shape =
        [{ kind: 'expression', text: 'window.alert(1)', symbols: [] }];
    }
    return response;
  });
  return <div style={{ height: '100dvh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
    <div><select aria-label="Fixture" value={name} onChange={(e) => { setName(e.target.value); setResponse(fixture(e.target.value)); }}>
      {['interface-dense', 'interface-hybrid', 'interface-visual', 'interface-many', 'contract', 'summaries', 'empty-group', 'templates', 'templates-absent', 'partial', 'mixed-stacks', 'hybrid', 'connections', 'components', 'components-large', 'visual-stacks', 'qwen3', 'qwen35', 'vjepa2', 'smollm2'].map((n) => <option key={n}>{n}</option>)}
    </select><button onClick={() => setShown(!shown)}>Toggle explorer</button><output style={{ display: 'block', height: 20, overflow: 'hidden' }}>{inspection}</output></div>
    <div contentEditable suppressContentEditableWarning aria-label="Untransformed prompt">Prompt remains outside graph camera</div>
    {shown && <ArchitectureCanvas key={name} graph={response.graph} modelId={response.model_id} sessionId="fixture-session"
      view={views.get(response.model_id, response.graph)} onDismissInspection={() => setInspection('')} onInspect={(selection) => { setInspection(selection.structureOnly ? `Structure only: ${selection.structureOnly.role}` : `${selection.graphId}: ${selection.node?.id ?? JSON.stringify(selection.boundary)}`); if (name.startsWith('interface-')) setNative(selection); }} />}
    {native && <ArchitectureInspection context={context} graph={response.graph} inventory={{ tensors: [], coverage: 'complete', diagnostics: [] }} selected={native} onClose={() => setNative(null)} />}
  </div>;
}
createRoot(document.getElementById('root')!).render(<Harness />);
