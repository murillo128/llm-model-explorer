# Neuron execution API

## Capability and identity

This is an accepted contract published ahead of runtime/UI implementation.
`GET /sessions/{session_id}/neuron-capabilities` returns `NeuronCapabilities`:
session/model identity and the session's `model_revision`, independent typed
`observation`, `intervention` and `generation` availability, effective limits,
execution identity and a bounded list of concrete modules. An unavailable feature
has a typed reason, safe explanation and recovery action. Availability describes
admitted capability, not a reservation or guarantee of free memory. Unsupported
models return an honest descriptor, not fake modules or an HTTP model-not-found.

`execution` and `limits` are null and `modules` empty when observation is
unavailable. Otherwise they are required, with at least one module; intervention
or generation availability implies observation. Generation may be unavailable
while prompt observation/comparison remain supported. Native execution support is
established by the [backend admission contract](../backend/neuron-execution.md),
independently of architecture, tokenizer and tensor capability descriptors.

Execution identity names family `smollm2`, fixed adapter `smollm2_native`, adapter
and runtime versions, PyTorch version, effective `cpu` or `cuda:N` device and
`float32` computation dtype. `model_revision` is the existing opaque pinned-session
snapshot token, not a portable hash. Exports include it for association but must
not claim it proves content equality across restarts. Public model identity may
include upstream revision metadata; the private content fingerprint stays private.

Each `NeuronModule` has an opaque `module_id`, an existing semantic `module`
reference, concrete `layer_index`, positive `intermediate_size` and `hidden_size`,
point `post_activation_product_pre_down_proj`, and three weight bindings. Gate/up
bindings identify existing inventory `tensor_id`s with axis `row`; down uses
`column`. Coordinate `j` selects that row/column. Shapes must agree with the
module and inventory. IDs, layers and module references are unique; modules are
in increasing layer order, complete for the admitted adapter (at most 64).
An optional `graph_binding` names an exact `graph_id` and concrete `node_id`
verified by the backend; never substitute a representative layer or infer a join
from a tensor name. Missing graph bindings leave the module selector usable.

Capability JSON and every META are bounded to the existing 1 MiB control budget.
Do not drop modules or silently truncate strings to fit. All responses use
`Cache-Control: no-store`; no public filesystem paths or private fingerprints.

## Requests and preflight

- `POST /sessions/{session_id}/neuron-activations` accepts
  `NeuronActivationsRequest` and returns kind `neuron_activations`.
- `POST /sessions/{session_id}/neuron-comparisons` accepts
  `NeuronComparisonRequest` and returns kind `neuron_comparison`.

Both echo the requested `model_revision`, one `module_id`, ordered `neuron_indices`
and ordered `token_ids`. Clients pass the exact successful tokenizer sequence from
that session, including duplicates and any special tokens already present. The
backend validates IDs against the admitted vocabulary and forwards them verbatim;
it never retokenizes, inserts BOS, pads the sequence or applies a chat template.
No tokenization receipt/history is required or retained. Provenance of the
submitted sequence is a client association requirement, not a server text log.

The effective `NeuronLimits` publishes `max_modules_per_request: 1`,
`max_neurons: 8`, `max_input_tokens: min(512, context_length)`,
`max_new_tokens: min(64, context_length - 1)`, `max_conditions: 4`,
`max_top_k: min(20, vocabulary_size)`, actual positive `context_length` and
`vocabulary_size`, and validated sorted unique `eos_token_ids` from checkpoint
configuration (empty if absent). Do not apply unrelated generation-config defaults.

Reject empty prompts, more than the effective token bound, duplicate/empty/more
than eight neuron indices, indices outside `[0, intermediate_size)`, token IDs
outside `[0, vocabulary_size)`, unknown modules and mismatched revision. Comparison
adds required `max_new_tokens` (0 through its effective bound), `top_k` (1 through
its effective bound), `decoding: greedy` and 1–4 ordered conditions. Require
`token_ids.length + max_new_tokens <= context_length`, even if EOS could stop
earlier. Validate all conditions and sizes before loading or allocating results.
No sampling, temperature, seed, batch, arbitrary hooks or extra request fields.

## Conditions, positions and numerical comparison

The first and only baseline is exactly `{id: "baseline", mode: "baseline"}`.
Remaining IDs are unique and distinct from `baseline`. Each condition contains
exactly one mode: `zero`, `scale` with finite `factor`, or
`add` with finite `delta`; and one scope: `prompt`,
`continuation` or `all`. Factors/deltas are applied in the effective computation
dtype, rounding to float32 (including ordinary underflow). Their absolute value
must not exceed `3.4028234663852886e38`. Negative and zero factors are valid. Every condition targets the same
module and ordered selected neurons; no per-condition module overrides.

At the declared point, for each selected `j`, zero sets `a[j] = 0`, scale sets
`a[j] = a[j] * factor`, and add sets `a[j] = a[j] + delta`. All other coordinates
are untouched. The original forward value is used at every position/condition;
interventions never accumulate or alter weights. Baseline cannot have scope,
factor, delta or any intervention.

For prompt length `T`, prompt positions are `0..T-1` and continuation input
positions are `T, T+1, ...`. Scope `prompt` covers the former, `continuation` the
latter, and `all` both. The logits at prompt position `T-1` predict the **first**
new token. Consequently continuation-only interventions leave that distribution
unchanged; their first possible effect is the distribution of the second new
token. Prompt-only effects may persist through attention/KV state during later
generation even though the hook is no longer active.

For each condition compute the next-token distribution at the exact common
prompt before any condition generates a different token. Use a stable softmax
over the **whole vocabulary**, calculated at least in float32. Take `top_k`
candidates for each condition by descending probability, breaking equal values
by ascending token ID. Define the shared candidate columns as the sorted,
unique ascending union of those lists (at most 80 IDs). Return every condition's
probability for every union candidate, including candidates outside its own
top-k. Never renormalize the union or top-k; a row sum may be less than one.
Ordering is based on the float32 probabilities that the result exposes.

Probability delta is `p(condition, candidate) - p(baseline, candidate)`.
The wire carries probabilities in `[0,1]`, not percentages, logits or deltas;
clients derive deltas from these values. A displayed percentage-point delta is
`100 * delta`, explicitly labelled, not a relative percentage change.

Free continuations use greedy argmax (ties choose the lowest token ID), starting
with that condition's common-prompt distribution and then its own generated
context. Include the selected EOS token in output IDs and stop there. Prompt EOS
alone does not stop generation. Stop at the requested length otherwise. Decode
the entire output sequence with the local tokenizer, without dropping special
tokens or applying cleanup; per-token strings need not concatenate correctly.
`max_new_tokens: 0` returns empty IDs/text and `finish_reason: not_requested` for
each condition. `eos` means the last ID is EOS, with no earlier EOS; `length`
means exactly the requested count with no EOS. These are ordinary completion,
not cancellation. Never align divergent continuations by index as shared context.

## LMEX layouts

Reuse [LMEX framing](binary-streaming.md), headers and terminal rules unchanged.
These are cancellable operations with `X-Operation-Id`. Headers may precede
bounded computation of the complete result so META can declare exact lengths.
No PROGRESS before META; no token-at-a-time guarantee. Failure/cancellation before
META uses the existing first-and-only ERROR/CANCELLED terminal.

Both metadata variants contain session/model/revision identity, execution
identity, effective limits, the selected full module descriptor, exact token and
neuron order, `dtype: float32`, `byte_order: little`, `layout: c`, ordered
`sections` and exact `byte_length`. Every DATA float is finite. Sections are
contiguous, without padding, starting at offset zero; their lengths are four
times shape products and their sum equals total byte length. Check all arithmetic
before allocation; frames can split rows but never float32 words.

| Kind | Section (only entry) | Shape and order |
| --- | --- | --- |
| `neuron_activations` | `activations` | `[T, N]`, prompt position outer, selected neuron inner |
| `neuron_comparison` | `probabilities` | `[C, U]`, request condition outer, ascending union candidate inner |

Observation is the **signed**, unintervened intermediate activation, not its
absolute value, a summary or a weight product. It has no generated-token rows.
Comparison META additionally echoes `conditions`, `top_k`, `max_new_tokens`,
`decoding`, `candidate_token_ids`, `top_token_ids` (one ranked list per condition)
and `continuations` (one record per condition, with matching `condition_id`).
Token IDs are bounded metadata identities, not float32 DATA. Text and short ID
lists belong in META; activation/probability arrays exist only in DATA.

Receivers validate schema, dimensions/bytes, unique and ordered identities, limits,
condition and continuation correspondence, candidate union, ID ranges, finiteness
and probability bounds. Top lists must rank their row's exposed candidates by
probability/tie rules. Row sums cannot exceed one beyond float32 rounding
(`1e-6`); only the producer can prove ranking against omitted vocabulary entries.
The originator additionally matches session/model/revision, module, neurons,
token IDs, conditions and options against its still-current request/capabilities.
An otherwise well-formed response for another request must never be displayed.
Incomplete/failed/cancelled bytes are not a successful result.

## Errors, privacy and lifetime

Use the existing error codes and status families:

| Outcome | Before headers | After headers |
| --- | --- | --- |
| Unsupported execution/capability | 422 `unsupported_representation` | same code in ERROR |
| Invalid IDs/fields/conditions/finite parameters | 422 `validation_error` | same code in ERROR |
| Exceeded declared/context/control-size bound | 422 `unsupported_size` | same code in ERROR |
| Insufficient resources | 503 `resource_exhausted` | same code in ERROR |
| Non-finite computed result | 500 `internal_error`, `details.reason: numerical_failure` | same Error in ERROR |
| Changed content or mismatched pinned revision | 409 `model_content_changed` | same code in ERROR |
| Cancellation | existing operation cancellation | CANCELLED, not Error |

Malformed JSON, missing session and other unexpected failures retain the common
contract. Numerical failures carry a safe generic message, no tensor dump or
prompt. No new top-level error code or protocol is introduced.

No automatic persistence of prompt/IDs/output/activations/comparisons in disk,
logs, traces or artifact cache. Results are ephemeral in backend operation memory
and browser session memory. Revision change invalidates old actions/results and
requires a new session. Explicit export is a small configuration manifest owned
by the UI specification, not a persistent experiment service.

## Conformance

`api/fixtures/neurons.json` supplies synthetic, independent values, request and
metadata boundaries, association checks and small LMEX streams. Its deliberately
small vocabulary/MLP are codec oracles, not advertised real-model admission.
Reuse the existing parser terminal/chunking suite; runtime numerical and full
checkpoint acceptance remain separate work.
