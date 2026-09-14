import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArchitectureCanvas } from '../src/architecture-explorer/ArchitectureCanvas';
import { GraphViews } from '../src/architecture-explorer/graph';
import { contractResponse, referenceFixture } from './architecture-fixtures';
import '../src/app/styles.css';

function Harness() {
  const [views] = useState(() => new GraphViews());
  const [name, setName] = useState('contract');
  const [shown, setShown] = useState(true);
  const [inspection, setInspection] = useState('');
  const [response, setResponse] = useState(() => {
    const response = structuredClone(contractResponse);
    if (location.search.includes('inert')) {
      response.graph.nodes[1]!.label = '<img src=x onerror=alert(1)>';
      response.graph.nodes[1]!.ports[1]!.shape = [{ kind: 'expression', text: 'window.alert(1)', symbols: [] }];
    }
    return response;
  });
  return <div style={{ height: '100dvh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
    <div><select aria-label="Fixture" value={name} onChange={(e) => { setName(e.target.value); setResponse(e.target.value === 'contract' ? contractResponse : referenceFixture(e.target.value)); }}>
      {['contract', 'qwen3', 'qwen35', 'vjepa2', 'smollm2'].map((n) => <option key={n}>{n}</option>)}
    </select><button onClick={() => setShown(!shown)}>Toggle explorer</button><output>{inspection}</output></div>
    <div contentEditable suppressContentEditableWarning aria-label="Untransformed prompt">Prompt remains outside graph camera</div>
    {shown && <ArchitectureCanvas key={name} graph={response.graph} modelId={response.model_id} sessionId="fixture-session"
      view={views.get(response.model_id, response.graph)} onInspect={(selection) => setInspection(`${selection.graphId}: ${selection.node.id}`)} />}
  </div>;
}
createRoot(document.getElementById('root')!).render(<Harness />);
