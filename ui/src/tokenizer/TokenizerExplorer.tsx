import { useEffect, useLayoutEffect, useRef, useState } from 'react';
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
  const [linkedSignal, setLinkedSignal] = useState<AbortSignal>();
  const [link, setLink] = useState<{
    signal: AbortSignal; tokenRow: number | null; matrixRow: number | null;
    revealRow: { row: number } | null;
  }>();
  function selectTokenRow(row: number | null, current: CurrentTokenization | undefined, activate = false) {
    if (!current || current.signal.aborted || linkedSignal !== current.signal) { setLink(undefined); return; }
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
      activeRow={link && linkedSignal === link.signal ? { signal: link.signal, row: link.matrixRow ?? link.tokenRow } : undefined}
      onRowSelect={selectTokenRow} onRowActivate={(row, current) => selectTokenRow(row, current, true)}
      downstream={current => <section className="input-embeddings" aria-label="Input embeddings">
        <EmbeddingRegion client={client} sessionId={sessionId} current={current}
          highlightedRow={link && link.signal === current?.signal ? link.tokenRow : null}
          revealRow={link && link.signal === current?.signal ? link.revealRow : null}
          onLinkedGeneration={setLinkedSignal}
          onRowSelect={row => { if (current) selectMatrixRow(row, current); }} />
      </section>} />
  </div>;
}

function EmbeddingHeader({ summary, status, actions }: { summary?: ReactNode; status: string; actions?: ReactNode }) {
  return <PanelHeader identity={<h2>Input Embeddings</h2>} summary={summary}
    status={status && <span role="status">{status}</span>} actions={actions} />;
}

interface EmbeddingSnapshot {
  generation: number;
  signal: AbortSignal;
  state: EmbeddingState;
  renderingFailed: () => void;
}

function EmbeddingRegion({ client, sessionId, current, highlightedRow, revealRow, onRowSelect, onLinkedGeneration }: {
  client: ApiClient; sessionId: string; current: CurrentTokenization | undefined;
  revealRow: { row: number } | null;
  highlightedRow: number | null; onRowSelect: (row: number | null) => void;
  onLinkedGeneration: (signal: AbortSignal | undefined) => void;
}) {
  const data = current?.data, signal = current?.signal;
  const generation = useRef(0);
  const [result, setResult] = useState<{ latest?: EmbeddingSnapshot; complete?: EmbeddingSnapshot }>({});
  useEffect(() => {
    if (!data?.tokens.length || !signal) return;
    const identity = ++generation.current;
    const controller = new EmbeddingController(client, sessionId, data.tokens.map(token => token.id), signal, (state, allocate) => {
      const snapshot = { generation: identity, signal, state, renderingFailed: controller.renderingFailed };
      const update = () => setResult(previous => ({ latest: snapshot,
        ...(state.status === 'complete' && state.source ? { complete: snapshot }
          : previous.complete ? { complete: previous.complete } : {}),
      }));
      if (allocate) flushSync(update); else update();
    });
    queueMicrotask(controller.start);
    return controller.dispose;
  }, [client, sessionId, data, signal]);
  const latest = result.latest?.signal === signal && !signal?.aborted ? result.latest : undefined;
  const state = latest?.state;
  // Keep the completed renderer mounted. A replacement receives progressive
  // DATA in a hidden sibling, then becomes visible in the same commit that
  // disposes the old renderer. At most two matrix allocations are live.
  const visible = result.complete ?? (state?.source ? latest : undefined);
  const staging = state?.source && latest !== visible ? latest : undefined;
  const linked = visible?.signal === signal && !!signal && !signal.aborted
    && (state?.status === 'streaming' || state?.status === 'complete');
  useLayoutEffect(() => { onLinkedGeneration(linked ? signal : undefined); }, [linked, signal, onLinkedGeneration]);
  const status = !data ? 'Waiting for current tokenization.'
    : !data.tokens.length ? 'No tokens to embed.'
    : state?.status === 'complete' ? ''
    : state?.status === 'unsupported' ? 'Input embeddings are unavailable for this model. Tokenization remains usable.'
    : state?.status === 'failed' ? 'Could not load input embeddings. Edit the prompt or retry tokenization.'
    : state?.status === 'cancelled' ? 'Input embedding lookup cancelled.'
    : state?.status === 'streaming' ? 'Streaming input embeddings…' : 'Loading input embeddings…';
  const stale = !!visible && !linked;
  const message = status + (stale ? ' Previous matrix is stale; token linkage is disabled.' : '');
  return <div className="embedding-layers" data-embeddings={stale ? 'stale' : 'current'}>
    {!visible && <EmbeddingHeader status={message} />}
    {[visible, staging].map(snapshot => {
      const source = snapshot?.state.source;
      if (!snapshot || !source) return null;
      const hidden = snapshot === staging;
      const summary = <span className="embedding-shape">[{source.descriptor.shape.join(' × ')}] · {source.descriptor.logical_dtype}</span>;
      return <div key={snapshot.generation} className="embedding-layer" data-staging={hidden || undefined}
        aria-hidden={hidden || undefined} inert={hidden || undefined}>
        <MatrixExplorer source={source} highlightedRow={!hidden && linked ? highlightedRow : null}
          revealRow={!hidden && linked ? revealRow : null}
          onRowSelect={row => { if (!hidden && linked) onRowSelect(row); }}
          onRenderingStateChange={state => {
            if (hidden && ['failed', 'lost', 'needs-reconstruction'].includes(state)) snapshot.renderingFailed();
          }}
          header={controls => <EmbeddingHeader summary={summary} status={message} actions={controls} />}
          label={stale ? 'Previous input embeddings (stale); token linkage disabled'
            : 'Input embeddings; rows are token sequence positions, columns are hidden dimensions'} />
      </div>;
    })}
  </div>;
}

export function TokenizerExplorer({ client, sessionId, selection, tokenizerAvailable = true }: ExplorerContextValue & { tokenizerAvailable?: boolean }) {
  return <TokenizerWorkspace client={client} sessionId={sessionId} signal={selection.signal} tokenizerAvailable={tokenizerAvailable} />;
}
