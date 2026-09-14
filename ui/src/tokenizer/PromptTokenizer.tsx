import { useEffect, useId, useMemo, useRef, useState } from 'react';
import type { ApiClient } from '../api/client';
import { ApiFailure } from '../api/errors';
import type { ReactNode } from 'react';
import { InlineEditor } from './InlineEditor';
import type { Tokenization } from './annotations';
import './tokenizer.css';

interface Props {
  client: Pick<ApiClient, 'tokenize'>;
  sessionId: string;
  addSpecialTokens?: boolean;
  tokenizerAvailable?: boolean;
  signal?: AbortSignal;
  header?: ReactNode;
  activeRow?: { signal: AbortSignal; row: number | null } | undefined;
  onRowSelect?: (row: number | null, current: CurrentTokenization | undefined) => void;
  onRowActivate?: (row: number, current: CurrentTokenization | undefined) => void;
  downstream?: (result: CurrentTokenization | undefined) => ReactNode;
}
interface Editor { text: string; generation: number; composing: boolean; promptly: boolean }
export interface CurrentTokenization { data: Tokenization; signal: AbortSignal }
interface Result {
  editor: Editor; context: object;
  data?: Tokenization; error?: string; signal?: AbortSignal;
}

/** The editor document owns source and history. Responses change decorations
 * only; every request/result is fenced by editor and context generations. */
export function PromptTokenizer({ client, sessionId, addSpecialTokens = true, tokenizerAvailable = true, signal, header, activeRow, onRowSelect, onRowActivate, downstream }: Props) {
  const id = useId();
  const [editor, setEditor] = useState<Editor>({ text: '', generation: 0, composing: false, promptly: false });
  const [result, setResult] = useState<Result>();
  const generation = useRef(0);
  const requestGeneration = useRef(0);
  const pending = useRef<AbortController | null>(null);
  // Reference identity also fences settled results across configuration A→B→A.
  const context = useMemo(() => ({ client, sessionId, addSpecialTokens, tokenizerAvailable, signal }),
    [client, sessionId, addSpecialTokens, tokenizerAvailable, signal]);
  const current = result?.editor === editor && result.context === context && tokenizerAvailable && !signal?.aborted ? result : undefined;

  function edit(text: string, isComposing: boolean, promptly = false) {
    pending.current?.abort();
    onRowSelect?.(null, undefined);
    setEditor({ text, generation: ++generation.current, composing: isComposing, promptly });
  }

  useEffect(() => {
    const request = ++requestGeneration.current;
    if (editor.composing || !tokenizerAvailable || signal?.aborted) return;
    const controller = new AbortController();
    pending.current = controller;
    let active = true;
    const abort = () => controller.abort();
    signal?.addEventListener('abort', abort, { once: true });
    const stamp = { editor, context };
    const isCurrent = () => active && !controller.signal.aborted && requestGeneration.current === request && generation.current === editor.generation;
    const timer = window.setTimeout(() => {
      void client.tokenize(sessionId, { text: editor.text, add_special_tokens: addSpecialTokens }, controller.signal).then((data) => {
        if (!isCurrent()) return;
        if (data.text !== editor.text || data.add_special_tokens !== addSpecialTokens) {
          setResult({ ...stamp, error: 'The tokenizer returned a result for different input. Edit the prompt or retry.' });
        } else setResult({ ...stamp, data, signal: controller.signal });
      }, (error: unknown) => {
        if (!isCurrent()) return;
        const message = error instanceof ApiFailure && error.detail?.code === 'unsupported_representation'
          ? 'Tokenizer unavailable for this model.'
          : error instanceof ApiFailure && error.kind === 'cancelled' ? 'Tokenization cancelled. Edit the prompt or retry.'
          : 'Could not tokenize the current prompt. Check the backend connection and retry.';
        setResult({ ...stamp, error: message });
      });
    }, editor.promptly ? 0 : 150);
    return () => { active = false; clearTimeout(timer); controller.abort(); signal?.removeEventListener('abort', abort); };
  }, [editor, context, client, sessionId, addSpecialTokens, tokenizerAvailable, signal]);

  const message = !tokenizerAvailable ? 'Tokenizer unavailable for this model. You can still edit the prompt.'
    : signal?.aborted ? 'Session view closed.'
    : editor.composing ? 'Composing text… Tokenization will resume when composition finishes.'
    : current?.error ?? (current?.data ? `${current.data.tokens.length} tokens · current prompt` : 'Tokenizing… Previous boundaries are hidden.');
  const tokenization = current?.data && current.signal ? { data: current.data, signal: current.signal } : undefined;
  const prompt = <section className="prompt-tokenizer" aria-label="Live prompt tokenization">
    <label className="section-label" htmlFor={id}>Prompt</label>
    <p id={`${id}-help`} className="tokenizer-help">Edit the prompt directly. Gray brackets and IDs annotate source spans. Gray token text is an annotation without a source span.</p>
    <InlineEditor id={id} result={current?.data} onEdit={edit}
      activeRow={activeRow?.signal === current?.signal ? activeRow?.row : null}
      onRowSelect={row => onRowSelect?.(row, tokenization)}
      onRowActivate={row => onRowActivate?.(row, tokenization)} />
    <p id={`${id}-status`} role={current?.error ? 'alert' : 'status'} className="tokenizer-status">{message}</p>
    {current?.error && <button type="button" className="button" onClick={() => edit(editor.text, false, true)}>Retry tokenization</button>}
  </section>;
  return <>{header ? <section className="prompt-panel" aria-label="Prompt / Tokens">{header}{prompt}</section> : prompt}
    {downstream?.(tokenization)}</>;
}
