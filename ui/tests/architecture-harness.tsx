import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArchitectureCanvas } from '../src/architecture-explorer/ArchitectureCanvas';
import { GraphViews } from '../src/architecture-explorer/graph';
import { contractResponse, referenceFixture } from './architecture-fixtures';
import { makeProjectionFixture } from './architecture-projection-fixture';
import '../src/app/styles.css';

function fixture(name: string) {
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
  const [name, setName] = useState('contract');
  const [shown, setShown] = useState(true);
  const [inspection, setInspection] = useState('');
  const [response, setResponse] = useState(() => {
    const response = structuredClone(contractResponse);
    if (location.search.includes('inert')) {
      response.graph.nodes[1]!.label = '<img src=x onerror=alert(1)>';
      response.graph.nodes.find((node) => node.id === 'linear0')!.ports.find((port) => port.id === 'out')!.shape =
        [{ kind: 'expression', text: 'window.alert(1)', symbols: [] }];
    }
    return response;
  });
  return <div style={{ height: '100dvh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
    <div><select aria-label="Fixture" value={name} onChange={(e) => { setName(e.target.value); setResponse(fixture(e.target.value)); }}>
      {['contract', 'partial', 'mixed-stacks', 'hybrid', 'connections', 'visual-stacks', 'qwen3', 'qwen35', 'vjepa2', 'smollm2'].map((n) => <option key={n}>{n}</option>)}
    </select><button onClick={() => setShown(!shown)}>Toggle explorer</button><output>{inspection}</output></div>
    <div contentEditable suppressContentEditableWarning aria-label="Untransformed prompt">Prompt remains outside graph camera</div>
    {shown && <ArchitectureCanvas key={name} graph={response.graph} modelId={response.model_id} sessionId="fixture-session"
      view={views.get(response.model_id, response.graph)} onInspect={(selection) => setInspection(`${selection.graphId}: ${selection.node.id}`)} />}
  </div>;
}
createRoot(document.getElementById('root')!).render(<Harness />);
