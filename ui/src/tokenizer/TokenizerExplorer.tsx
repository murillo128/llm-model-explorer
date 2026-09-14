import { useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import type { ApiClient } from '../api/client';
import type { ExplorerContextValue } from '../app/explorer-context';
import { MatrixExplorer } from '../matrix-explorer';
import { PromptTokenizer } from './PromptTokenizer';
import type { CurrentTokenization } from './PromptTokenizer';
import { EmbeddingController } from './embedding-controller';
import type { EmbeddingState } from './embedding-controller';
import './embeddings.css';

interface Props {
  client: ApiClient;
  sessionId: string;
  signal?: AbortSignal;
  addSpecialTokens?: boolean;
  tokenizerAvailable?: boolean;
}

/** The prompt subtree keeps its original layout, source, and native history. */
export function TokenizerWorkspace({ client, sessionId, ...props }: Props) {
  const [tokenRow, setTokenRow] = useState<number | null>(null);
  const [matrixRow, setMatrixRow] = useState<number | null>(null);
  function selectTokenRow(row: number | null) {
    setTokenRow(row);
    if (row !== null) setMatrixRow(null);
  }
  function selectMatrixRow(row: number | null) {
    setMatrixRow(row);
    if (row !== null) setTokenRow(null);
  }
  return <div className="tokenizer-workspace">
    <PromptTokenizer {...props} client={client} sessionId={sessionId} activeRow={matrixRow ?? tokenRow} onRowSelect={selectTokenRow}
      downstream={current => <section className="input-embeddings" aria-label="Input embeddings">
        {current ? <EmbeddingRegion client={client} sessionId={sessionId} current={current} highlightedRow={tokenRow} onRowSelect={selectMatrixRow} />
          : <><h2>Input embeddings</h2><p role="status">Waiting for current tokenization.</p></>}
      </section>} />
  </div>;
}

function EmbeddingRegion({ client, sessionId, current, highlightedRow, onRowSelect }: {
  client: ApiClient; sessionId: string; current: CurrentTokenization;
  highlightedRow: number | null; onRowSelect: (row: number | null) => void;
}) {
  const { data, signal } = current;
  const [result, setResult] = useState<{ data: typeof data; signal: AbortSignal; state: EmbeddingState }>();
  useEffect(() => {
    if (!data.tokens.length) return;
    const controller = new EmbeddingController(client, sessionId, data.tokens.map(token => token.id), signal, (state, allocate) => {
      const update = () => setResult({ data, signal, state });
      if (allocate) flushSync(update); else update();
    });
    queueMicrotask(controller.start);
    return controller.dispose;
  }, [client, sessionId, data, signal]);
  const state = result?.data === data && result.signal === signal && !signal.aborted ? result.state : undefined;
  const status = !data.tokens.length ? 'No tokens to embed.'
    : state?.status === 'complete' ? `${data.tokens.length} token rows · ${state.source!.descriptor.shape[1]} hidden dimensions`
    : state?.status === 'unsupported' ? 'Input embeddings are unavailable for this model. Tokenization remains usable.'
    : state?.status === 'failed' ? 'Could not load input embeddings. Edit the prompt or retry tokenization.'
    : state?.status === 'cancelled' ? 'Input embedding lookup cancelled.'
    : state?.status === 'streaming' ? 'Streaming input embeddings…' : 'Loading input embeddings…';
  return <>
    <header><h2>Input embeddings</h2><p role="status">{status}</p></header>
    {state?.source && <MatrixExplorer source={state.source} highlightedRow={highlightedRow} onRowSelect={onRowSelect}
      label="Input embeddings; rows are token sequence positions, columns are hidden dimensions" />}
  </>;
}

export function TokenizerExplorer({ client, sessionId, selection, tokenizerAvailable = true }: ExplorerContextValue & { tokenizerAvailable?: boolean }) {
  return <TokenizerWorkspace client={client} sessionId={sessionId} signal={selection.signal} tokenizerAvailable={tokenizerAvailable} />;
}
