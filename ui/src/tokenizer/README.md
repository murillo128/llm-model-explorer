# Reusable prompt tokenization

`PromptTokenizer` accepts the typed client's `tokenize` capability, `sessionId`,
optional `addSpecialTokens` (default `true`), optional known tokenizer availability,
and an optional owning view's abort signal. It owns its editable source. The shell's
`TokenizerExplorer` adapter supplies its session/client/lifetime context; callers
can embed the same component in later inference views.

The CodeMirror document is the only source/editor. Inline decoration widgets
supply gray brackets and IDs, and marks associate exact source spans. Unmapped
native representations are non-editable widgets at their sequence anchors.
Annotation strings use text nodes, never HTML. Decoration-only transactions do
not change the document, selection or undo history. Clipboard copy/paste uses the
editor's document, so annotations never enter source text or tokenizer requests.
CodeMirror supplies established composition, selection and history handling; its
modules load lazily when the standalone explorer opens.

Edits increment a local generation and abort the pending request immediately.
Requests coalesce for 150 ms; composition suppresses requests and its completion
requests the latest text promptly. Results must match generation, editor snapshot,
client, session, echoed source and options. Cleanup also fences promises from
transports that ignore abort. Superseded errors and cancellations are ignored.
Pending/error/composition states show no previous token boundaries.

`annotations.ts` converts code-point offsets to UTF-16 and unions overlapping
source ranges without duplicating source characters. Typed special IDs with spans
remain editable source. Missing and zero-width spans have native-text annotations
at sequence anchors, never fabricated character ranges. Source annotations reserve
enough width for their IDs. Multiline spans retain
their exact source characters and annotate each line segment. Native scrolling
keeps long prompts and annotations reachable.

Only the backend tokenizes. Browser fixtures are synthetic contract examples,
including deliberately non-concatenating decoded text; their IDs are not claims
about any particular model. See `ui/evidence/tokenizer.md` for validation.
