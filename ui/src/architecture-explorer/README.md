# Architecture canvas implementation

The accepted behavior lives in `docs/spec/ui/architecture-explorer.md`. This module
consumes generated contract types and validated backend topology. It does not
identify model families or generate model semantics.

`ArchitectureExplorer` retrieves only when mounted in an active session, validates
parameter actions against that session's inventory, and cancels obsolete requests.
Retrieval has a 15-second deadline and a 32 MiB decoded-body limit. Capability and
protocol failures remain local. No tokenization, tensor stream, analysis job or
polling request is started by the graph.

`GraphViews` belongs to one backend shell and retains only camera, expansion IDs,
dimensions and selection, keyed by model and graph identity. The canvas's
`onInspect` seam receives exact model/session/graph/node identity and the activating
DOM element. `ArchitectureExplorer` connects it to `ArchitectureInspection`, one
native modal outside the graph transform. The modal resolves parameter IDs and
semantic references against the validated graph/inventory, displays provenance,
formulas, storage and localized limitations, and composes the existing
`TensorExplorer` consumer for available native vectors/matrices. Parameter choice
never infers a layer or tensor ID from its name.

Every modal and weight choice has a child `Lifetime`. Closing, replacing a weight,
or disposing the session cancels only its own numeric handles; Matrix Explorer
unsubscription releases the viewport's retained CPU/GPU resources. Existing
controller and source guards reject late callbacks, including a return to the
same tensor. Escape closes from any descendant, Tab stays inside, and close
restores the activating element without altering graph camera or expansion.
The shared cell readout portals into its enclosing dialog when present. Tensor
metadata is already in the modal, so this composition omits the header popover
instead of stacking dialogs. Unsupported representations, fused regions,
unresolved bindings and high ranks have no numeric action.

## Graph implementation and dependencies

- `@xyflow/react` **12.11.6**, MIT, pinned in `ui/package.json` and its lockfile.
  [Source/license](https://github.com/xyflow/xyflow/blob/main/LICENSE).
- Nested parent-before-child nodes and explicit source/target handles follow
  [React Flow subflows](https://reactflow.dev/learn/layouting/sub-flows) and
  [handles](https://reactflow.dev/learn/customization/handles). Group inputs may
  forward internally as sources and group outputs may receive internal targets;
  both handle roles retain the original port identity.
- ELK was evaluated as a candidate. Its compound layout support is documented in
  [hierarchy handling](https://eclipse.dev/elk/reference/options/org-eclipse-elk-hierarchyHandling.html),
  with [compound-port limitations](https://github.com/eclipse-elk/elk/issues/1192).
  This implementation uses deterministic ordered containment packing instead of
  adding ELK: it preserves the contract's exact child/repetition order, supports
  all boundary ports, and has linear packing work. React Flow draws the original
  edges; layout does not invent a serial path from node placement.

`graph.ts` and `layout.ts` isolate the replaceable layout boundary. The worker uses
iterative walks rather than recursive containment. Its ten-second deadline,
termination on replacement/unmount, and explicit retry/collapse failure preserve
recoverability. A stale layout cannot update a later graph generation. Expanding a
group anchors its upper-left location in screen coordinates; component selection
can reveal and center any concrete instance without fitting the whole diagram.

Collapsed interiors retain semantic records in the original graph. Only edges
whose endpoints are visible are sent to React Flow; the contract requires explicit
group-boundary ports, so external crossings remain present unchanged. Viewport
culling affects DOM elements, not the graph, component picker or expanded layout.
Every instance remains separately labelled with its index and variant. No
representative is substituted for a different instance's parameter bindings.

## Validation boundaries

`api/fixtures/architecture.json` supplies the independent structural/contextual
oracle. UI tests run its positive and negative cases against generated schema and
context validation, then exercise exact endpoint retention and concrete-instance
selection in Chromium. Production shell tests cover on-demand retrieval, partial
inventory, malformed graphs, no-tokenizer V-JEPA and explorer restoration.

`ui/tests/architecture-fixtures.ts` also generates full-size **synthetic stress
graphs** using layer counts and hybrid order observed from pinned reference
`config.json` files. Their 32/40-operation interiors deliberately stress nested
groups, branches, long crossings and recurrence; they are not analyzer output or
claims about checkpoint topology. They report partial coverage with an explicit
fixture diagnostic. Config sources are the model/revision pairs in that file:
`https://huggingface.co/<model>/blob/<revision>/config.json`.

The browser tests attach node/edge counts, worker layout time, total interaction
time and Chromium heap usage for desktop and narrow viewports. Heap usage is a
point-in-time measurement including the browser application, not a peak or a
guaranteed performance bound. Actual analyzer/checkpoint topology and built-UI
tests against the prepared endpoint remain the backend and integrated acceptance
children's responsibility; fixtures do not establish those claims.

Reproduce from `ui/`:

```sh
npm run check
UI_TEST_PORT=18484 npm run test:browser -- tests/architecture.spec.ts tests/architecture-shell.spec.ts --project=desktop --project=narrow
```

Modal/browser reproduction (synthetic asymmetric streams, not checkpoint inference):

```sh
UI_TEST_PORT=18484 npm run test:browser -- tests/architecture-inspection.spec.ts --project=desktop --project=narrow
```
