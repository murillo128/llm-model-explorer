# Neuron view inside Architecture Explorer

## Integration and eligibility

This accepted view lives within the existing Architecture workspace, using the
[shared visual language](visual-language.md). It adds no fourth global explorer,
shell changes, separate graph or replacement matrix renderer. Preserve graph
camera, expansion, concrete instance and selection while opening/closing it.

Read the session's [neuron capabilities](../api/neuron-explorer.md). Only that
descriptor enables observation/intervention/generation actions. A static graph,
model-owned description, tensor name or inspectable weight never implies
executability. Show each unavailable capability's reason and recovery action;
keep independent graph, tensor and tokenizer inspection usable. Do not show fake
activations, fabricated semantic labels or results for unsupported models.

Select a concrete MLP from the capability list, or follow its verified binding
from the current graph. A representative/shared structure needs an explicit
concrete instance. Retain exact module identity, intervention point and neuron
index visibly; never substitute layer zero. Missing graph linkage still permits
the explicit module selector. Choose 1–8 unique indices within its actual width.

## Weights, prompt and observation

Explain `a[j] = SiLU(gate(x)[j]) * up(x)[j]` before `down_proj` and the distinction
between an activation coordinate and a weight cell. Existing weight navigation
opens the complete bound gate/up matrices at row `j` and down matrix at column
`j`, through current tensor IDs. Preserve native matrix geometry and exact scalar
identity; selection is an overlay, not sliced/recolored weight storage.

Compose the existing inline tokenizer surface. Execute only the latest successful
exact sequence from this session; editing/tokenizing supersedes pending execution
and makes older results visibly stale and non-actionable until rerun. Never add
BOS, tokenize a second hidden prompt or apply a chat template. Show effective
limits before execution and validate locally for feedback; the backend remains
authoritative. Keep accessible labels, keyboard operation and visible focus.

Observation displays signed float32 values by prompt position and selected neuron
in request order. Duplicate token IDs remain separate positions. Keep sign and
exact numeric inspection available; luminosity alone must not erase negative
values. Do not mistake weights, absolute values or generated-token values for
prompt activations. Any matrix representation uses existing exact-cell rules.

## Experiment and comparison

Keep one immutable baseline first and up to three labelled intervention
conditions. Each chooses exactly zero, scale or add, with the applicable finite
factor/delta and prompt/continuation/all scope. No seed, sampling or temperature
controls. One common top-k and a greedy continuation length (including zero)
apply to every condition. Explain that continuation-only scope cannot affect the
first next-token distribution, which comes from the prompt.

The common-context table displays the union of top-k candidates, every condition's
full-vocabulary probability and its delta from baseline. Token ID is the stable
identity; decoded text is presentation only. Label probability, percent and
percentage-point deltas accurately; never renormalize the displayed subset.
Maintain wire candidate/condition order by default. An explicit presentation sort
must retain the exact ID association for all columns.

Show free continuations in separate labelled condition blocks with EOS/length or
no-generation status. These contexts diverge; equal output indices do not form a
controlled comparison. Explain that an isolated effect is not evidence of a
semantic concept or a causal circuit belonging to one neuron.

## Cancellation, replacement and export

Use existing operation status and cancellation patterns. Headers provide the
operation ID while bounded computation may delay META; do not invent percentage
progress or promise token streaming. Closing, superseding inputs, switching
session/model, content revision changes or disconnect releases affected consumers
and rejects late replies. A cancellation is distinct from failure. Partial bytes
never become completed evidence; retain no stale result as current. A failed
comparison must leave weights and the next baseline execution clean.

Keep results only in memory for the current browser session. Navigation within
that session may preserve them; session close/replacement, content change or page
reload discards them. Do not use local storage, IndexedDB, automatic downloads,
analytics payloads or backend artifacts for experiment content.

An explicit **Export configuration** action downloads a bounded (at most 1 MiB)
JSON manifest with `format: llmex-neuron-experiment`, `format_revision: 1`,
operation kind, public model/session/revision identity, execution versions/device/
dtype, effective limits, selected module/bindings, exact token IDs and ordered
neuron indices, plus comparison conditions/options when applicable. Warn at the
action that token IDs can disclose the prompt. Include neither raw activation/
probability arrays, generated text/IDs nor private filesystem locations. This is
an explicit configuration export, not a persisted experiment library or import
service. Preserve opaque identity honestly: replay needs independently matching
local content and versions plus a fresh valid session; a stored revision token
cannot prove cross-restart equivalence or authorize stale actions.

## Acceptance boundary

Use small deterministic fixtures for selection, signed-value/ID mapping,
condition/scope editing, delta units, divergent-output presentation, accessible
controls, cancellation and late-response rejection. Real browser checks cover
the bounded view at 1440×960 and 760px width without shell/matrix regressions.
Those checks do not establish real runtime effects. Integrated acceptance must
separately exercise the full local SmolLM2 Base checkpoint and distinguish its
evidence from mockups and fixtures.
