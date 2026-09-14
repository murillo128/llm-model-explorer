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
Pending/error/composition states retain transaction-mapped decorations with reduced
emphasis and accessible stale status. Only a matching current response atomically
replaces them. Deleted source ranges follow CodeMirror’s normal change mapping.

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


## Downstream input embeddings

`TokenizerWorkspace` composes the unchanged prompt subtree and a matrix region
below it. `PromptTokenizer.downstream` receives only its current successful result
and the request-generation abort signal. A pending/error/IME/session/options
change removes that authoritative result immediately while its mapped annotations
remain stale. The embedding region itself stays mounted. `TokenizerExplorer` in
`TokenizerExplorer.tsx` adapts the application's session lifetime to this workspace.

`EmbeddingController` calls the shared typed `streamInputEmbeddings` method.
Validated META synchronously mounts the Matrix Explorer before DATA can arrive
in the same network chunk. `StreamWords` consumes split float32 words directly;
only the renderer retains an inspection copy. No histogram/statistics are inferred.
Resubscribing after any received bytes restarts the lookup at offset zero;
StrictMode's synchronous pre-DATA detach/reattach can reuse the untouched stream.
Every callback is fenced by the tokenizer abort signal and transport epoch.
The last complete renderer remains mounted during recomputation. A replacement
streams into a hidden, inert sibling renderer and is promoted only on COMPLETE.
Failed/superseded staging renderers are disposed without replacing the prior
matrix. At most two renderer allocations are retained, with no extra scalar cache.

`TokenizerWorkspace` frames the unchanged PromptTokenizer surface with a compact
`PanelHeader` and gives Input Embeddings the same header through
`MatrixExplorer.header`. Matrix shape and logical dtype appear once beside its
identity; loading/error status uses the reserved header slot. Camera controls,
inspection and magnifier remain entirely in the shared matrix implementation.

Existing ID text gets per-sequence focus/interaction wiring; overlapping IDs
remain separately addressable without repeating source text. Linked rows use
`MatrixExplorer.highlightedRow`, and matrix inspection uses `onRowSelect` to mark
the existing annotation. Hover/focus is transient; click/tap or Enter/Space emits
`onRowActivate`, creating a fresh `revealRow` intent without changing scale.
Link state is stamped with the current tokenization's abort signal. Prompt and
matrix consume it only when the visible matrix and current tokenization have that
same generation, so session/options/source replacement
cannot revive stale selection. Source replacement releases the old matrix and
starts a fresh camera; remount does not restore old linkage.

These interactions never dispatch editor document or selection transactions.
They do not automatically scroll/focus the Prompt panel. `embeddings.css` styles
the panel framing and active annotation cues; `tokenizer.css` also marks stale
annotations without dimming source text.
