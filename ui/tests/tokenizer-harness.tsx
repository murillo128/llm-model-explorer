import { EditorView } from '@codemirror/view';
import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ApiClient } from '../src/api/client';
import { PromptTokenizer } from '../src/tokenizer/PromptTokenizer';
import type { Tokenization } from '../src/tokenizer/annotations';
import '../src/app/styles.css';

interface Pending { session: string; text: string; add_special_tokens: boolean; aborted: boolean }
declare global {
  interface Window {
    tokenizerHarness: {
      requests: Pending[];
      source: () => string;
      selection: () => number[];
      complete: (index: number, data: Tokenization, status?: number) => void;
      fail: (index: number) => void;
    };
  }
}
const resolvers: ((value: Response) => void)[] = [];
const rejecters: ((reason: unknown) => void)[] = [];
window.tokenizerHarness = {
  requests: [],
  source: () => EditorView.findFromDOM(document.querySelector('.cm-content')!)!.state.doc.toString(),
  selection: () => { const selection = EditorView.findFromDOM(document.querySelector('.cm-content')!)!.state.selection.main; return [selection.from, selection.to]; },
  complete: (index, data, status = 200) => resolvers[index]!(new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } })),
  fail: (index) => rejecters[index]!(new Error('simulated late transport failure')),
};
// Use the production typed client, but deliberately ignore abort when resolving
// transport promises so a real browser exercises the generation fence as well.
const client = new ApiClient({ backendBaseUrl: 'https://fixture.example' }, async (input, init) => {
  const request: Pending = { session: new URL(String(input)).pathname.split('/')[2]!, ...JSON.parse(init!.body as string), aborted: false };
  window.tokenizerHarness.requests.push(request);
  init?.signal?.addEventListener('abort', () => { request.aborted = true; });
  return new Promise<Response>((resolve, reject) => { resolvers.push(resolve); rejecters.push(reject); });
});
function Harness() {
  const [session, setSession] = useState('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa');
  const [special, setSpecial] = useState(true);
  const [available, setAvailable] = useState(true);
  return <main className="app-shell">
    <h1>Tokenizer Explorer</h1>
    <div><button onClick={() => setSession((value) => value.startsWith('a') ? 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb' : 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')}>Change session</button>
      <label><input type="checkbox" checked={special} onChange={(event) => setSpecial(event.target.checked)} />Add special tokens</label>
      <label><input type="checkbox" checked={available} onChange={(event) => setAvailable(event.target.checked)} />Tokenizer available</label></div>
    <PromptTokenizer client={client} sessionId={session} addSpecialTokens={special} tokenizerAvailable={available} />
  </main>;
}
createRoot(document.getElementById('root')!).render(<StrictMode><Harness /></StrictMode>);
