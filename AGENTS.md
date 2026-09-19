# AGENTS.md

Repository-wide instructions for ChatGPT, Codex, and other development agents.

## Mission and scope

The project's durable mission and accepted architecture are defined by `docs/spec/`, with `docs/spec/README.md` as the specification index. `docs/spec/product.md` owns cross-cutting product scope; documents under `docs/spec/backend/`, `docs/spec/api/`, and `docs/spec/ui/` own decisions inside those system boundaries. `README.md` is the project entry point and summary, but it does not override the specification.

Do not broaden the project, invent adjacent goals, or promote exploratory discussion into settled scope without an explicit repository or issue-level decision. Do not duplicate a normative decision across specification boundaries when one document already owns it.

This file owns repository-wide agent invariants and routes work to reusable skills. Skills define reusable procedure, issues define bounded task contracts, and repository documents define durable project knowledge.

## Load context progressively

For non-trivial work start with `AGENTS.md` and the controlling issue. Then read `docs/spec/README.md` and load only the specification documents relevant to the component being changed, followed by source/tests/config/evidence and the one workflow skill needed by the current role/action. Do not preload every specification file, document, skill, issue/PR history, result directory, or derived wiki.

For backend work, load the relevant documents under `docs/spec/backend/` plus API documents when the change affects the contract. For API work, load `docs/spec/api/` plus the specific backend/UI documents whose boundary is affected. For UI work, load the relevant documents under `docs/spec/ui/` plus API documents consumed by the UI.

`docs/spec/ui/tensor-explorer.md` and `docs/spec/ui/tokenizer-explorer.md` contain accepted explorer behavior. Read the relevant dedicated specification before changing an explorer. Features explicitly reserved for future design remain outside the proof of concept; route unspecified behavior through design rather than guessing.

On resume, verify branch, `HEAD`, worktree, controlling issue's current single workflow-state label, canonical execution context when applicable, and new material issue/PR discussion. Reuse unchanged inspected context rather than replaying history.

## Source-of-truth hierarchy

Unless project-specific documentation defines a stricter hierarchy:

1. Tests, formal checks, evaluation outputs, and captured evidence establish observed behavior.
2. Accepted specifications under `docs/spec/**`, accepted decisions, and other explicitly normative architecture documents establish durable intended behavior.
3. The controlling issue establishes the bounded execution contract and may specialize the accepted specification only within its explicit authority; it must not silently contradict normative project invariants.
4. `README.md` summarizes product state and navigation but does not override `docs/spec/**`.
5. PRs, checks, reviews, commits, and Git history preserve implementation and reproducible evidence.
6. Roadmaps/epics/planning establish planning/dependency status only within their declared authority.
7. Exploratory notes/research/drafts are provisional unless explicitly adopted.
8. `wiki/**` is agent-generated derived non-normative knowledge and never overrides stronger sources.
9. Chat is provisional until intentionally recorded in an authoritative repository/GitHub source.

Do not promote `OPEN`, `SPECULATIVE`, placeholder, exploratory, or wiki-derived statements into requirements without explicit adoption. Surface material conflicts rather than silently choosing.

## Skill-driven workflow

Load skills lazily by role:

- optional local runner provisioning/repair: `skills/codex-local-runner/SKILL.md`;
- design authority: `skills/design-github-issue/SKILL.md`;
- ordinary issue executor: `skills/spec-driven-codex-loop/SKILL.md`;
- final PR audit controller: `skills/codex-pr-audit/SKILL.md`;
- Git/GitHub mutations/publication: `skills/codex-github-operations/SKILL.md`;
- independent technical review: `skills/codex-independent-review/SKILL.md`;
- event-driven epic initialization/scheduling: `skills/codex-epic-scheduler/SKILL.md`;
- explicit manual multi-issue orchestration: `skills/codex-issue-orchestrator/SKILL.md`;
- model-owned Architecture Explorer view authoring/review: `skills/architecture-json-authoring/SKILL.md`;
- derived wiki curation: `skills/repository-wiki-curation/SKILL.md`.

The generic launcher for an execution target is the same. After launch, read the controlling issue: when it declares `execution_mode: epic-dag`, route to `codex-epic-scheduler`; otherwise route to `spec-driven-codex-loop`. Keep this decision in agent instructions, not duplicated in the launcher workflow.

## Workflow state

Every non-trivial controlling issue uses exactly one current workflow-state label:

- `queued`
- `execution-ready`
- `in-progress`
- `review-ready`
- `design-required`
- `investigation-required`
- `blocked`
- `completed`

The label is authoritative. State-only transitions normally produce no comment.

`queued` is normal waiting for a fully designed epic child; it is not a blocker. Only `codex-epic-scheduler` may automatically promote `queued -> execution-ready`. An actor that explicitly resolves `blocked`, `design-required`, or `investigation-required` may return an epic child to `queued`.

`review-ready` is the executor's successful terminal handoff: implementation/validation and issue-declared intermediate checkpoints are complete, the PR is ready, and integration freshness is stable when applicable. It hands control to `codex-pr-audit`; the executor does not perform a duplicate final independent review.

`completed` is post-merge. Executors and independent reviewers never merge, enable auto-merge, close issues, or set completed. After `codex-pr-audit` obtains `PASS` or `PASS_WITH_NOTES` with `final-capable: yes` on its current exact PR head, the audit controller has standing authority to merge that exact head, make `completed` observable, and close the issue. The reviewer verdict alone has no mutation authority.

## Codex model and delegation policy

Use the native roles and operating instructions in [`docs/codex-operations.md`](docs/codex-operations.md). Model, effort and profile overrides are optional in the controlling issue or epic through its dedicated `codex` TOML block. Without a block, use native local Codex settings; normal resume retains the saved session selection. Do not impose a profile file, model family or reasoning floor merely because an issue has no override. Explicit overrides apply to the next permitted inactive-session turn, not to active work. Astra xhigh for ordinary work and Astra Ultra for critical semantics/isolation/concurrency/security/numerical integration remain opt-in recommendations. Independent review requires fresh context and sufficient capability; the optional Astra Max/Ultra reviewer role must not accidentally inherit an economical helper model.

Keep one product change owner and at most three concurrent auxiliary subagents per session. Bounded exploration uses the `bounded-explorer` role (Terra medium); prescribed commands and CI/log collection use `command-runner` (Luna low). Explicitly select the role with fresh context; do not fork the owner's full history into an economical helper. The owner designs invariants/oracles and judges computational preservation. Helpers cannot edit source or expectations, weaken checks, mutate Git/GitHub, or launch competing heavy suites. Use a stable isolated snapshot and separate ports/output directories for checks. Escalate or disclose an authorized fallback if a helper model is unavailable; fail closed when required critical capability is unavailable.

Internal delegation does not change an epic's `max_parallel_workers`, issue holds, ownership or the final audit boundary. An implementation helper is not the independent final reviewer. Existing active turns retain their selections; adopt a different profile only through an explicitly requested new or inactive session.

## Epic workflow

An epic parent starts with a compact seed contract (`execution_mode: epic-dag`, child issue set, parallelism limit, optional integration branch), **not** a manually generated DAG. Its initial state is `execution-ready`; fully designed children normally start `queued`.

The first `codex-epic-scheduler` turn inspects all child contracts, derives direct dependency edges and serialization mutexes, validates acyclicity/closure, creates the integration branch from the current default branch when necessary, and persists exactly one parent comment marked `<!-- codex-epic-dag:v1 -->`. Only after that durable graph exists does it make the parent `in-progress` and activate the first wave.

Later scheduler turns consume the canonical graph; they do not re-infer it from edited prose. Every child activation receives exactly one canonical `<!-- codex-execution-context:v1 -->` comment with parent, integration branch, and exact base SHA before `execution-ready` becomes observable.

The single `.github/workflows/codex-issue-state.yml` dispatcher is the only `issues:labeled` workflow. It routes `execution-ready` to execution, `review-ready` to audit, and `completed`/`queued` child events to the unique active parent discovered from canonical DAG state. Duplicate/delayed scheduler wake-ups are expected and must remain idempotent.

## Design and execution discipline

For non-trivial changes, design a self-contained controlling issue; implement the smallest coherent outcome; preserve behavior outside scope; validate proportionally with repository-native checks; retain enough evidence for claims; and use independent review only at issue-declared intermediate risk boundaries plus the final audit controller.

Implement against the accepted specification rather than re-deriving architecture from framework defaults or current source layout. When implementation convenience conflicts with `docs/spec/**`, surface the conflict and route it through design instead of silently changing the contract.

Do not invent project-wide roadmaps, schemas, frameworks, provider abstractions, compatibility layers, storage systems, or process machinery merely because they might be useful. Do not mechanically implement every reviewer suggestion; judge findings against contract/materiality/invariants.

## Durable knowledge and wiki

Keep deliberate normative docs and generated memory distinct. `docs/spec/**` is normative accepted product/architecture specification. Other explicitly normative `docs/**` sources use normal design/execution workflow. `wiki/**` is generated, derived, non-normative memory maintained by `repository-wiki-curation`. GitHub issues own actionable discrepancies, unresolved decisions, suspected defects/drift, and bounded work contracts.

The curator has standing write authority only for `wiki/**`, and only after its adversarial review and hard write-boundary gate. It never modifies non-wiki paths. New curator-created issues use `curator-detected`.

## Implementation and check entry points

- Backend application and CLI: `backend/src/llm_model_explorer/app.py` and `cli.py`; domain modules and tests live under `backend/src/llm_model_explorer/` and `backend/tests/`.
- Independent API checks/fixtures: `api/validate_contract.py`; generated UI contract bindings: `ui/scripts/api-generator/generate.mjs`.
- UI composition/navigation: `ui/src/app/`; Tensor Explorer: `ui/src/explorers/`; reusable WebGL2 renderer: `ui/src/rendering/`; live tokenizer: `ui/src/tokenizer/`.
- Static architecture analysis/startup: `backend/src/llm_model_explorer/architecture_analysis/` and `architecture_service.py`; global graph and native weight modal: `ui/src/architecture-explorer/`; integrated acceptance: `acceptance/test_architecture.py` and `ui/acceptance/architecture.spec.ts`. Full local reference acceptance remains separate from fixture success; see `acceptance/architecture.md`.
- UI unit and browser checks: `ui/package.json`, `ui/src/**/*.test.*`, and `ui/tests/`.
- Integrated TCP/production-browser acceptance: `acceptance/check.sh`, `acceptance/test_*.py`, and `ui/acceptance/`; `.github/workflows/application-acceptance.yml` aggregates application gates. Reproduction and optional capability checks are documented in `acceptance/README.md`.

These are implementation navigation links, not alternative normative specifications. Preserve the separate Skillforge workflow scripts and dispatchers.

## Evidence and artifacts

Keep evidence proportional. Commit source, tests, configuration, small deterministic fixtures, concise reports, and compact reproducibility evidence. Keep large generated outputs, binaries, model weights, datasets, caches, traces, or bulky logs outside Git unless explicitly required and distributable. Never publish secrets/private data/restricted artifacts.

The runtime artifact cache described by the product specification is local derived state, not repository evidence. Do not commit cache contents or local model files.

## Git and GitHub behavior

- Keep changes scoped and use explicit paths.
- Agent-created commits follow `codex-github-operations` conventions.
- Do not rewrite shared valid history without explicit authority. The standing exception is the executor-owned `codex/issue-N` branch, before `review-ready`, using the exact-old-head `--force-with-lease` integration-rebase protocol.
- Direct commits to default branch require explicit user instruction except the narrow wiki-curator authority.
- An implementation workflow ends with a ready PR and `review-ready` handoff to audit.
- Before handoff, an epic child executor performs the final integration-freshness gate: if integration advances, reconcile/revalidate/republish and require fresh exact-head CI until stable.
- Replacing `in-progress` with `review-ready` is the executor's final GitHub mutation. Afterward only local teardown/bookkeeping/response composition is allowed until another controller returns it to execution.
- Outside the positive `codex-pr-audit` path, merge requires a later explicit user-facing instruction after review finds no material blocker.

## Project-specific invariants

The concise rules below are operational reminders. `docs/spec/**` is authoritative when more detail is needed.

### Product and boundaries

- The proof of concept has two user-facing capabilities: Tensor Explorer and Tokenizer Explorer. Detailed behavior belongs in their accepted dedicated specifications; future-analysis backlog entries do not expand current scope.
- The accepted post-PoC Architecture Explorer is implemented on the integration branch: static descriptions, blocking startup/cache retrieval, global canvas and native weight modal. Its dedicated specifications own behavior and selected coverage. Implementation and fixture success do not establish full local reference acceptance.
- The system has three independent boundaries: backend, API contract, and browser UI. Keep ownership aligned with `docs/spec/backend/`, `docs/spec/api/`, and `docs/spec/ui/`.
- The initial reference model is `HuggingFaceTB/SmolLM2-135M` Base, not Instruct.
- The proof of concept must be evolvable into step-by-step transformer inference without replacing the established backend/API/session/stream/cache/renderer boundaries.

### Backend and models

- The backend is Python with FastAPI/Starlette for HTTP and PyTorch for tensor computation. Do not add a project-owned C, C++, or Rust compute layer without an accepted design change.
- Heavy tensor computation uses PyTorch/native kernels rather than Python element-by-element loops. CPU and CUDA must expose the same logical API result; CUDA is optional and selected by backend configuration.
- The proof of concept supports local Hugging Face model directories only. Do not add generic model-provider abstractions or remote model download support without accepted design.
- The backend receives a local model root, discovers models beneath it, and keeps filesystem paths private. Model files are read-only.
- Model access is lazy. Do not require loading an entire model into RAM or GPU merely to inspect metadata or one tensor; use memory mapping/equivalent lazy access when the storage format permits it.
- Public model identity is derived from Hugging Face metadata when suitable, with directory-name fallback. Cache validity uses a content fingerprint independent from the model's local path.
- Physical storage formats belong to the backend. The main visualization path exposes logical tensor values in canonical `float32`; the UI must not need NF4, INT8, or other quantization decoders.

### API and streaming

- The API is contract-first and independent from backend implementation classes and UI code. Conventional HTTP behavior is described by the API contract/OpenAPI; large numeric results follow the binary streaming specification.
- Use explicit typed capabilities, not a universal stringly typed `execute` endpoint.
- JSON is for commands, descriptors, metadata, and small responses. Large numeric tensor payloads are binary and must not be encoded as JSON or Base64.
- Long operations expose an `operation_id`; the request that starts the operation carries its progressive binary response. Operations are cancelable and cancellation is a distinct outcome.
- Errors before streaming begins use HTTP plus structured JSON; errors after a stream starts use an error frame in the binary stream.
- The proof of concept has no public backwards-compatible API versioning such as `/v1`; backend and UI evolve together against the current accepted contract.

### Sessions, concurrency, and cache

- Multiple sessions may exist concurrently. A session is bound to exactly one `model_id` for its lifetime.
- Proof-of-concept session state is in backend memory and may survive UI refresh/reconnect while the backend process remains alive. Backend restart may discard sessions.
- Disk reads, cache hits, tokenization, and HTTP streams may run concurrently. Expensive GPU work is serialized through one execution queue per GPU device.
- Equivalent artifact-producing operations are deduplicated across sessions. Cancelling one consumer must not cancel shared work while another consumer still needs it.
- The artifact cache is shared across sessions, filesystem-only, disposable, and contains complete immutable reconstructible artifacts. Chunks are transport units, not cache units.
- Publish a cache artifact only after successful complete generation; cancelled or failed work must not become a valid cache entry. Model-derived artifact keys include the model content fingerprint.
- Do not add automatic cache GC, LRU, size limits, databases, Redis, or cache-management UI in the proof of concept unless an accepted design changes scope.

### UI and rendering

- The UI is React + TypeScript + Vite and is deployable independently from the backend, including on a different computer. Backend base URL is configurable.
- WebGL2 rendering lives in a reusable TypeScript renderer independent from React. React owns application composition; the renderer owns GPU resources and drawing.
- WebGL2 is for visualization and visual transformations, not authoritative model computation. Mathematical model operations belong to PyTorch on the backend.
- The UI consumes long-operation results progressively and must be able to begin rendering before a complete tensor has arrived.
- The proof-of-concept direct tensor visualization supports complete 1D and 2D tensors. Tensor shape in the API remains generic for arbitrary rank.
- Preserve **one authoritative logical scalar per matrix cell**, exact indexed sampling/hit testing, and uniform square-cell display scale. Matrix Explorer supports fit-width and focal-point zoom with a native 1:1 minimum; never interpolate, aggregate, minify below one device pixel per cell, or duplicate/recolor scalar storage. Oversized content uses normal scroll. Internal texture/band partitioning must preserve the complete tensor and exact logical cell identity.
- Tensor value controls luminosity through a configurable nonlinear sigmoid-like transfer using robust tensor statistics such as percentiles. Statistics are independent derived artifacts and may arrive after tensor bytes begin rendering.
- Color is a separate semantic channel for selection, activation state, clusters, highlighting, and similar overlays. Changing color or luminosity must not rewrite the underlying tensor values.

### Deployment and security

- Backend and UI are independent processes and may run on separate machines. Configure backend host/port, model root, artifact-cache directory, compute device, and CORS primarily through command-line options for the proof of concept.
- The proof of concept has no authentication and assumes a trusted local network or otherwise trusted environment. Do not invent user/account/authentication systems without accepted design.
