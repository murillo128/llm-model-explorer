import { interfaceFixture } from './architecture-interface-fixture';
import { overviewFixture } from './architecture-overview-fixture';
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
import { makeTemplateFixture, makeVjepaBrowserFixture } from './architecture-template-fixture';
import { makeExplicitFixture } from './architecture-explicit-fixture';
import { rotaryContextFixture } from './architecture-routing-fixture';
import '../src/app/styles.css';

function fixture(name: string) {
  if (name === 'routing-context') return { ...contractResponse, model_id: name, graph: rotaryContextFixture() };
  if (name.startsWith('overview-')) return { ...contractResponse, model_id: name, graph: overviewFixture(name.slice(9) as Parameters<typeof overviewFixture>[0]) };
  if (name === 'interface-long-ports') {
    const graph = interfaceFixture('hybrid');
    const declaredInput = graph.nodes.find((node) => node.id === 'Token IDs')!;
    declaredInput.label = 'An unusually long attention mask input name';
    declaredInput.parent_id = 'language';
    const boundary = graph.nodes.find((node) => node.id === 'language')!;
    if (boundary.kind !== 'group') throw new Error('Fixture language boundary must be a group');
    boundary.children.unshift(declaredInput.id);
    boundary.ports.find((port) => port.id === 'Token IDs')!.label = 'opaque boundary input';
    const operation = graph.nodes.find((node) => node.id === 'Embedding')!;
    operation.ports.find((port) => port.id === 'Token IDs')!.label = 'An unusually long attention mask input name';
    operation.ports.find((port) => port.id === 'out')!.label = 'An unusually long hidden state output name';
    return { ...contractResponse, model_id: name, graph };
  }
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
  if (name === 'browser-vjepa') return { model_id: name, status: 'available' as const, diagnostics: [], graph: makeVjepaBrowserFixture() };
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

function configuredFixture(name: string) {
  const response = structuredClone(fixture(name));
  const params = new URLSearchParams(location.search);
  if (params.has('inert')) {
    response.graph.nodes[1]!.label = '<img src=x onerror=alert(1)>';
    response.graph.nodes.find((node) => node.id === 'linear0')!.ports.find((port) => port.id === 'out')!.shape =
      [{ kind: 'expression', text: 'window.alert(1)', symbols: [] }];
  }
  const longBrowserNode = params.has('long-browser')
    ? response.graph.nodes.find((node) => node.id.endsWith('layer-31.attention.Q')) : undefined;
  if (longBrowserNode) longBrowserNode.label = 'Q projection with an intentionally long public component name that must remain fully available';
  return response;
}

function Harness() {
  const [views] = useState(() => new GraphViews());
  const [name, setName] = useState(() => new URLSearchParams(location.search).get('fixture') ?? 'contract');
  const [shown, setShown] = useState(true);
  const [inspection, setInspection] = useState('');
  const [native, setNative] = useState<ArchitectureSelection | null>(null);
  const [context] = useState(() => ({ client: new ApiClient({ backendBaseUrl: 'http://127.0.0.1:1' }), session: { id: 'fixture-session', model_id: name }, sessionId: 'fixture-session', selection: new Lifetime(), selectedTensor: null, reportStatus: () => {} }));
  const [response, setResponse] = useState(() => configuredFixture(name));
  return <div style={{ height: '100dvh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
    <div><select aria-label="Fixture" value={name} onChange={(e) => { setName(e.target.value); setResponse(configuredFixture(e.target.value)); }}>
      {['overview-compact', 'overview-synthetic', 'overview-wide', 'overview-fanout', 'overview-training', 'interface-dense', 'interface-hybrid', 'interface-hybrid-many', 'interface-visual', 'interface-many', 'routing-context', 'contract', 'summaries', 'empty-group', 'templates', 'templates-absent', 'browser-vjepa', 'partial', 'mixed-stacks', 'hybrid', 'connections', 'components', 'components-large', 'visual-stacks', 'qwen3', 'qwen35', 'vjepa2', 'smollm2'].map((n) => <option key={n}>{n}</option>)}
    </select><button onClick={() => setShown(!shown)}>Toggle explorer</button><output style={{ display: 'block', height: 20, overflow: 'hidden' }}>{inspection}</output></div>
    <div contentEditable suppressContentEditableWarning aria-label="Untransformed prompt">Prompt remains outside graph camera</div>
    {shown && <ArchitectureCanvas key={name} graph={response.graph} modelId={response.model_id} sessionId="fixture-session"
      view={views.get(response.model_id, response.graph)} onDismissInspection={() => { setInspection(''); setNative(null); }} onInspect={(selection) => { setInspection(selection.structureOnly ? `Structure only: ${selection.structureOnly.role}` : `${selection.graphId}: ${selection.node?.id ?? JSON.stringify(selection.boundary)}`); if (name.startsWith('interface-') || new URLSearchParams(location.search).has('native-inspection')) setNative(selection); }} />}
    {native && <ArchitectureInspection context={context} graph={response.graph} inventory={{ tensors: [], coverage: 'complete', diagnostics: [] }} selected={native} onClose={() => setNative(null)} />}
  </div>;
}
createRoot(document.getElementById('root')!).render(<Harness />);
