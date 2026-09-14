import { useEffect, useState } from 'react';
import { flushSync } from 'react-dom';
import type { ReactNode } from 'react';
import type { ApiClient } from '../api/client';
import type { ExplorerContextValue } from '../app/explorer-context';
import { MatrixExplorer, PanelHeader } from '../matrix-explorer';
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
  const [link, setLink] = useState<{
    signal: AbortSignal; tokenRow: number | null; matrixRow: number | null;
    revealRow: { row: number } | null;
  }>();
  function selectTokenRow(row: number | null, current: CurrentTokenization | undefined, activate = false) {
    if (!current || current.signal.aborted) { setLink(undefined); return; }
    setLink(previous => ({ signal: current.signal, tokenRow: row,
      matrixRow: row !== null || previous?.signal !== current.signal ? null : previous.matrixRow,
      revealRow: activate && row !== null ? { row } : previous?.signal === current.signal ? previous.revealRow : null,
    }));
  }
  function selectMatrixRow(row: number | null, current: CurrentTokenization) {
    if (current.signal.aborted) return;
    setLink(previous => ({ signal: current.signal, matrixRow: row,
      tokenRow: row !== null || previous?.signal !== current.signal ? null : previous.tokenRow,
      revealRow: previous?.signal === current.signal ? previous.revealRow : null,
    }));
  }
  return <div className="tokenizer-workspace">
    <PromptTokenizer {...props} client={client} sessionId={sessionId}
      header={<PanelHeader identity={<h2>Prompt / Tokens</h2>} />}
      activeRow={link ? { signal: link.signal, row: link.matrixRow ?? link.tokenRow } : undefined}
      onRowSelect={selectTokenRow} onRowActivate={(row, current) => selectTokenRow(row, current, true)}
      downstream={current => <section className="input-embeddings" aria-label="Input embeddings">
        {current ? <EmbeddingRegion client={client} sessionId={sessionId} current={current}
          highlightedRow={link?.signal === current.signal ? link.tokenRow : null}
          revealRow={link?.signal === current.signal ? link.revealRow : null}
          onRowSelect={row => selectMatrixRow(row, current)} />
          : <EmbeddingHeader status="Waiting for current tokenization." />}
      </section>} />
  </div>;
}

function EmbeddingHeader({ summary, status, actions }: { summary?: ReactNode; status: string; actions?: ReactNode }) {
  return <PanelHeader identity={<h2>Input Embeddings</h2>} summary={summary}
    status={status && <span role="status">{status}</span>} actions={actions} />;
}

function EmbeddingRegion({ client, sessionId, current, highlightedRow, revealRow, onRowSelect }: {
  client: ApiClient; sessionId: string; current: CurrentTokenization;
  revealRow: { row: number } | null;
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
    : state?.status === 'complete' ? ''
    : state?.status === 'unsupported' ? 'Input embeddings are unavailable for this model. Tokenization remains usable.'
    : state?.status === 'failed' ? 'Could not load input embeddings. Edit the prompt or retry tokenization.'
    : state?.status === 'cancelled' ? 'Input embedding lookup cancelled.'
    : state?.status === 'streaming' ? 'Streaming input embeddings…' : 'Loading input embeddings…';
  const source = state?.source;
  const summary = source ? <span className="embedding-shape">[{source.descriptor.shape.join(' × ')}] · {source.descriptor.logical_dtype}</span> : undefined;
  return source ? <MatrixExplorer source={source} highlightedRow={highlightedRow} revealRow={revealRow} onRowSelect={onRowSelect}
    header={controls => <EmbeddingHeader summary={summary} status={status} actions={controls} />}
    label="Input embeddings; rows are token sequence positions, columns are hidden dimensions" />
    : <EmbeddingHeader status={status} />;
}

export function TokenizerExplorer({ client, sessionId, selection, tokenizerAvailable = true }: ExplorerContextValue & { tokenizerAvailable?: boolean }) {
  return <TokenizerWorkspace client={client} sessionId={sessionId} signal={selection.signal} tokenizerAvailable={tokenizerAvailable} />;
}
