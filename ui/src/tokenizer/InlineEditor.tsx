import { useLayoutEffect, useRef } from 'react';
import { EditorState, StateEffect, StateField } from '@codemirror/state';
import { Decoration, EditorView, WidgetType, keymap } from '@codemirror/view';
import type { DecorationSet } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { annotate } from './annotations';
import type { SourceSelection, Tokenization } from './annotations';

const annotationsChanged = StateEffect.define<DecorationSet>();
const annotationsStale = StateEffect.define<boolean>();
const annotationField = StateField.define<{ decorations: DecorationSet; stale: boolean }>({
  create: () => ({ decorations: Decoration.none, stale: true }),
  update(value, transaction) {
    let next = transaction.docChanged
      ? { decorations: value.decorations.map(transaction.changes), stale: true } : value;
    for (const effect of transaction.effects) {
      if (effect.is(annotationsChanged)) next = { decorations: effect.value, stale: false };
      if (effect.is(annotationsStale)) next = { ...next, stale: effect.value };
    }
    return next;
  },
  provide: (field) => [
    EditorView.decorations.from(field, value => value.decorations),
    EditorView.editorAttributes.from(field, value => ({ 'data-annotations': value.stale ? 'stale' : 'current' })),
  ],
});

class AnnotationWidget extends WidgetType {
  constructor(readonly text: string, readonly ids: string, readonly className: string, readonly description: string, readonly indices: readonly number[] = []) { super(); }
  eq(other: AnnotationWidget) { return this.text === other.text && this.ids === other.ids && this.className === other.className && this.description === other.description && this.indices.join() === other.indices.join(); }
  toDOM() {
    const node = document.createElement('span');
    node.className = this.className;
    node.setAttribute('aria-label', this.description);
    node.textContent = this.text;
    if (this.indices.length) node.dataset.tokenIndices = this.indices.join(' ');
    if (this.ids) {
      const label = document.createElement('span'); label.className = 'token-ids';
      this.ids.split(', ').forEach((id, i) => {
        if (i) label.append(', ');
        const token = document.createElement('span');
        token.textContent = id;
        token.dataset.tokenIndex = String(this.indices[i]);
        token.tabIndex = 0;
        token.setAttribute('role', 'button');
        token.setAttribute('aria-label', `Token ${id}, sequence ${this.indices[i]}`);
        label.append(token);
      });
      node.append(label);
    }
    return node;
  }
}

function decorations(result: Tokenization): DecorationSet {
  const { groups, unmapped } = annotate(result);
  const ranges = [];
  for (const group of groups) {
    const ids = group.tokens.map((token) => token.id).join(', ');
    const description = `Token IDs ${ids}${group.tokens.some((token) => token.special) ? '; includes source-typed special token' : ''}`;
    // A source token may cross lines. Annotate each exact line segment, including
    // empty newline spans, without inserting/removing document characters.
    let from = group.start;
    while (from < group.end) {
      const newline = result.text.indexOf('\n', from);
      const to = newline >= 0 ? Math.min(newline, group.end) : group.end;
      const opening = new AnnotationWidget('[', ids, 'token-opening', description, group.tokens.map(token => token.index));
      ranges.push(Decoration.widget({ widget: opening, side: 1 }).range(from));
      ranges.push(Decoration.widget({ widget: new AnnotationWidget(']', '', 'token-closing', 'End source span'), side: to === from ? 2 : -2 }).range(to));
      if (from < to) ranges.push(Decoration.mark({ class: 'source-annotation', attributes: {
        'data-token-ids': ids, 'data-token-indices': group.tokens.map(token => token.index).join(' '), 'aria-label': description,
        style: `min-width: ${Math.max(0, ids.length * 6.7 - 8)}px`,
      } }).range(from, to));
      // Newline-only annotations still need room for their associated ID.
      else ranges.push(Decoration.widget({ widget: new AnnotationWidget('', '', 'empty-token-space', ''), side: 3 }).range(from));
      if (to === group.end) break;
      from = to + 1;
    }
  }
  for (const { token, anchor } of unmapped) {
    const native = token.token || token.decoded || '(empty)';
    ranges.push(Decoration.widget({ widget: new AnnotationWidget(
      native + (token.special ? '' : ' (no source span)'), String(token.id), 'unmapped-annotation',
      `Sequence ${token.index}, token ${token.id}, no source span: ${native}`, [token.index],
    ), side: -1 }).range(anchor));
  }
  return Decoration.set(ranges, true);
}

interface Props {
  id: string;
  result: Tokenization | undefined;
  activeRow?: number | null | undefined;
  onRowSelect?: ((row: number | null) => void) | undefined;
  onRowActivate?: ((row: number) => void) | undefined;
  onEdit: (text: string, composing: boolean, promptly?: boolean) => void;
  onContentHeight?: ((height: number) => void) | undefined;
  onSourceSelection?: ((ranges: readonly SourceSelection[]) => void) | undefined;
}

/** CodeMirror's immutable document/history remain separate from decoration-only
 * transactions. Widgets and brackets never enter source, selections or copy. */
export function InlineEditor({ id, result, onEdit, activeRow = null, onRowSelect, onRowActivate, onContentHeight, onSourceSelection }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const edit = useRef(onEdit);
  const measure = useRef(onContentHeight);
  const selection = useRef(onSourceSelection);
  useLayoutEffect(() => { selection.current = onSourceSelection; }, [onSourceSelection]);
  useLayoutEffect(() => { measure.current = onContentHeight; }, [onContentHeight]);
  useLayoutEffect(() => { edit.current = onEdit; }, [onEdit]);
  useLayoutEffect(() => {
    let disposed = false;
    const reportHeight = (editor: EditorView) => editor.requestMeasure({
      key: measure,
      read: view => view.contentHeight,
      write: height => { if (!disposed) measure.current?.(height); },
    });
    const instance = new EditorView({ parent: host.current!, state: EditorState.create({
      extensions: [history(), keymap.of([...defaultKeymap, ...historyKeymap]), annotationField,
        EditorState.tabSize.of(4), ...(measure.current ? [EditorView.lineWrapping] : []),
        EditorView.contentAttributes.of({ id, role: 'textbox', 'aria-label': 'Prompt', 'aria-multiline': 'true', 'aria-describedby': `${id}-help ${id}-status`, spellcheck: 'false' }),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) edit.current(update.state.doc.toString(), update.view.composing);
          // Blur/DOM selection loss is not an intentional caret placement. Keep
          // the native source range when inspecting another surface. Document
          // edits always report their mapped selection for the next generation.
          if (update.docChanged || (update.selectionSet && update.view.hasFocus)) {
            selection.current?.(update.state.selection.ranges.map(({ from, to }) => ({ from, to })));
          }
          if (update.geometryChanged || update.docChanged) reportHeight(update.view);
        }),
        EditorView.domEventHandlers({
          compositionstart: (_event, editor) => { edit.current(editor.state.doc.toString(), true); },
          compositionend: (_event, editor) => {
            queueMicrotask(() => { if (!disposed) edit.current(editor.state.doc.toString(), false, true); });
          },
        }),
      ],
    }) });
    view.current = instance;
    reportHeight(instance);
    return () => { disposed = true; view.current = null; instance.destroy(); };
  }, [id]);
  useLayoutEffect(() => {
    const instance = view.current;
    if (!instance) return;
    instance.dispatch({ effects: result && result.text === instance.state.doc.toString()
      ? annotationsChanged.of(decorations(result)) : annotationsStale.of(true) });
  }, [result]);
  useLayoutEffect(() => {
    const root = host.current!;
    const rowFor = (target: EventTarget | null) => {
      const node = target instanceof Element ? target.closest<HTMLElement>('[data-token-index], [data-token-indices]') : null;
      return node && root.contains(node) ? Number(node.dataset.tokenIndex ?? node.dataset.tokenIndices?.split(' ')[0]) : null;
    };
    const enter = (event: Event) => onRowSelect?.(rowFor(event.target));
    const leave = (event: Event) => onRowSelect?.(rowFor((event as MouseEvent).relatedTarget));
    const press = (event: Event) => {
      if (event.target instanceof Element && event.target.closest('[data-token-index]')) {
        // Annotation interaction must not move the editor's native selection.
        event.preventDefault(); enter(event);
      }
    };
    const activate = (event: Event) => {
      if (!(event.target instanceof Element) || !event.target.closest('[data-token-index]')) return;
      event.preventDefault();
      const row = rowFor(event.target);
      if (row !== null) onRowActivate?.(row);
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') { press(event); activate(event); }
    };
    root.addEventListener('pointerover', enter); root.addEventListener('pointerout', leave);
    root.addEventListener('focusin', enter); root.addEventListener('focusout', leave);
    root.addEventListener('click', activate);
    root.addEventListener('mousedown', press); root.addEventListener('keydown', key);
    return () => {
      root.removeEventListener('pointerover', enter); root.removeEventListener('pointerout', leave);
      root.removeEventListener('focusin', enter); root.removeEventListener('focusout', leave);
      root.removeEventListener('click', activate);
      root.removeEventListener('mousedown', press); root.removeEventListener('keydown', key);
    };
  }, [onRowSelect, onRowActivate]);
  useLayoutEffect(() => {
    for (const node of host.current!.querySelectorAll<HTMLElement>('[data-token-index], [data-token-indices]')) {
      const indices = node.dataset.tokenIndex ?? node.dataset.tokenIndices ?? '';
      node.toggleAttribute('data-active-token', activeRow !== null && indices.split(' ').includes(String(activeRow)));
    }
  }, [activeRow, result]);
  return <div ref={host} className="tokenizer-editor" />;
}
