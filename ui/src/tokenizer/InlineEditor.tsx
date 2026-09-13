import { useLayoutEffect, useRef } from 'react';
import { EditorState, StateEffect, StateField } from '@codemirror/state';
import { Decoration, EditorView, WidgetType, keymap } from '@codemirror/view';
import type { DecorationSet } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { annotate } from './annotations';
import type { Tokenization } from './annotations';

const annotationsChanged = StateEffect.define<DecorationSet>();
const annotationField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(value, transaction) {
    if (transaction.docChanged) return Decoration.none;
    for (const effect of transaction.effects) if (effect.is(annotationsChanged)) return effect.value;
    return value;
  },
  provide: (field) => EditorView.decorations.from(field),
});

class AnnotationWidget extends WidgetType {
  constructor(readonly text: string, readonly ids: string, readonly className: string, readonly description: string) { super(); }
  eq(other: AnnotationWidget) { return this.text === other.text && this.ids === other.ids && this.className === other.className && this.description === other.description; }
  toDOM() {
    const node = document.createElement('span');
    node.className = this.className;
    node.setAttribute('aria-label', this.description);
    node.textContent = this.text;
    if (this.ids) {
      const label = document.createElement('span'); label.className = 'token-ids'; label.textContent = this.ids;
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
      const opening = new AnnotationWidget('[', ids, 'token-opening', description);
      ranges.push(Decoration.widget({ widget: opening, side: 1 }).range(from));
      ranges.push(Decoration.widget({ widget: new AnnotationWidget(']', '', 'token-closing', 'End source span'), side: to === from ? 2 : -2 }).range(to));
      if (from < to) ranges.push(Decoration.mark({ class: 'source-annotation', attributes: {
        'data-token-ids': ids, 'aria-label': description,
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
      `Sequence ${token.index}, token ${token.id}, no source span: ${native}`,
    ), side: -1 }).range(anchor));
  }
  return Decoration.set(ranges, true);
}

interface Props {
  id: string;
  result: Tokenization | undefined;
  onEdit: (text: string, composing: boolean, promptly?: boolean) => void;
}

/** CodeMirror's immutable document/history remain separate from decoration-only
 * transactions. Widgets and brackets never enter source, selections or copy. */
export function InlineEditor({ id, result, onEdit }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const edit = useRef(onEdit);
  useLayoutEffect(() => { edit.current = onEdit; }, [onEdit]);
  useLayoutEffect(() => {
    let disposed = false;
    const instance = new EditorView({ parent: host.current!, state: EditorState.create({
      extensions: [history(), keymap.of([...defaultKeymap, ...historyKeymap]), annotationField,
        EditorState.tabSize.of(4),
        EditorView.contentAttributes.of({ id, role: 'textbox', 'aria-label': 'Prompt', 'aria-multiline': 'true', 'aria-describedby': `${id}-help ${id}-status`, spellcheck: 'false' }),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) edit.current(update.state.doc.toString(), update.view.composing);
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
    return () => { disposed = true; view.current = null; instance.destroy(); };
  }, [id]);
  useLayoutEffect(() => {
    const instance = view.current;
    if (!instance) return;
    instance.dispatch({ effects: annotationsChanged.of(result && result.text === instance.state.doc.toString() ? decorations(result) : Decoration.none) });
  }, [result]);
  return <div ref={host} className="tokenizer-editor" />;
}
