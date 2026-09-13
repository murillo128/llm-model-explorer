# Tokenizer Explorer

## Status

**Dedicated UI specification accepted.**

This document owns the Tokenizer Explorer's UI behavior, layout, and presentation. Cross-cutting product scope, session/model ownership, tokenizer result semantics, and API schemas remain owned by their existing specifications and are not repeated here.

## Purpose and boundary

The Tokenizer Explorer is an editable text-inspection surface for the tokenizer associated with the current session model. Its purpose is to make the relationship between user-entered text and the tokenizer's actual output immediately visible.

The UI consumes the tokenizer result defined by the API contract. It does not reimplement tokenizer logic in the browser, and it does not attempt to reconstruct or teach BPE, SentencePiece, or another tokenizer algorithm step by step.

The prompt/tokenization surface defined here is a reusable UI component. The standalone Tokenizer Explorer uses it directly, and later inference-oriented views may embed the same component above downstream stages without creating a second tokenizer presentation.

## Visual reference

The current wireframe is the `Inference Explorer` frame in the Miro dashboard:

https://miro.com/app/board/uXjVHnoDEYY=/?moveToWidget=3458764683523016635

The Miro board is a visual reference for the combined prompt/tokenization area. This document is normative if the board and specification diverge. Only the narrow input-embedding lookup described below is an accepted downstream extension; other inference content in the wider wireframe remains future scope.

## Live editing and tokenization

The explorer has one primary editable text surface. Tokenization updates in real time as the user types; there is no separate prompt view followed by a second token view that repeats the same text.

Each edit retokenizes the current text and updates the visible token boundaries and token metadata in place. Token boundaries are allowed to change anywhere in the text after an edit; the UI must not assume that previously rendered token boundaries remain stable.

The rendered tokenizer state must correspond to the latest text currently in the editor. A result for an older input must not replace the presentation for a newer input if responses complete out of order.

## Inline token presentation

Ordinary tokens are presented inline as annotations of the prompt itself. The prompt text is not duplicated on another line merely to show tokenization.

For each ordinary token:

- the opening and closing token brackets are gray;
- the text between those brackets is black and represents the token's corresponding user-entered text;
- whitespace belonging to the token is preserved, so a token whose source span begins with a space visually includes that space inside its brackets;
- the token ID is shown in gray as secondary metadata visually associated with that same token, normally aligned beneath it.

Conceptually, a token is therefore rendered as a gray `[` + black source text + gray `]`, with its gray token ID associated with that span. The text itself must not be repeated in a separate token-only row.

When source offsets are available from the tokenizer result, they are used to associate tokens with the exact input substring. API offsets count Unicode code points and must be converted explicitly to JavaScript/DOM UTF-16 positions without normalization.

Overlapping source spans are grouped around their union, with every associated token ID retained in sequence order. The source character appears once, including when multiple byte-level tokens cover the same emoji. Do not split overlapping spans into invented disjoint boundaries.

When a reliable nonempty source span is unavailable, show the tokenizer-native/decoded representation as an explicitly unmapped annotation at its sequence position. This placement does not claim a character mapping. Per-token decoded strings need not concatenate to the input.

Annotations are separate from the editable value and never enter tokenizer requests or source clipboard text. Asynchronous decoration changes preserve native selection, caret, undo/redo, paste and IME composition. Pending results hide previous boundaries while keeping the prompt editable.

## Special tokens

The API `special` flag identifies a special token ID, not whether its text was inserted. A source-typed special with a supplied source span remains black editable source, with gray brackets and metadata like any other source span.

Backend-inserted special tokens have no source span. Their textual representation and token ID are gray non-editable annotations, separate from the prompt value.

Special tokens are placed at the sequence position reported by the tokenizer, for example before or after ordinary prompt tokens when the tokenizer inserts beginning/end markers. They must remain visually distinguishable from black user-entered text without introducing a separate color scheme per token.

## Visual semantics

The Tokenizer Explorer intentionally uses a minimal monochrome distinction:

- black: user-entered text represented by ordinary token spans;
- gray: token brackets, token IDs, special tokens, and supporting tokenizer annotations.

Token identity is not encoded by assigning a different color to every token. The important visual relationship is the exact segmentation of the editable text and its associated metadata.

## Accepted input-embedding extension

The post-PoC Tokenizer Explorer may consume the input-embedding lookup defined
in `../api/contract.md` for the latest successful tokenizer result. Submit its
real token IDs in sequence order, including special tokens and duplicates,
under the same session. Associate each matrix row with that sequence position
using the echoed IDs; never infer IDs from displayed text or download the full
vocabulary embedding table to gather rows in the browser.

The matrix must correspond to the current editor result and session. An older
lookup must not replace a newer one; while tokenization/lookup is pending or
fails, an older matrix must not be presented as current. Empty input and
unsupported lookup are explicit states. Consume the float32 matrix progressively
using the common rules in `rendering.md`, preserving one weight per rendered
pixel. This contract establishes data association and scope; detailed matrix
composition and interaction are owned by the separate UI implementation issue.

## Reuse in later inference views

Later inference exploration may place the tokenizer component directly above embeddings or other model stages. That composition must reuse this same live prompt/tokenization component rather than introducing a parallel tokenizer UI with different interaction or visual semantics.

Beyond the accepted input-embedding row lookup above, this document does not
define downstream computation stages. Tensor rendering rules remain in
`rendering.md`; positional encoding, transformer execution, logits, and
generation require future design.
