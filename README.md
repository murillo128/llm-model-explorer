# Skillforge

A reusable GitHub repository template for skill-driven agentic software development.

Skillforge separates responsibilities deliberately: `AGENTS.md` owns repository-wide invariants/routing; `skills/` owns reusable procedure; project documentation owns durable project knowledge; `wiki/` is optional derived non-normative memory; and GitHub issues own bounded execution contracts/actionable findings.

## Starting a project

A repository created from this template runs `skills/repository-bootstrap/SKILL.md` once before normal non-trivial work. Bootstrap verifies the required workflow labels (including `queued` for epic children), replaces this template README with the actual project's README, adapts project-specific AGENTS invariants, hard-checks its narrow publication boundary, commits initialization atomically to default branch, and removes itself. Local runner setup is opt-in and happens only as a separate post-bootstrap handoff.

The canonical `murillo128/skillforge` template retains bootstrap and must never bootstrap or provision a repository runner against itself.

## Issue workflow automation

Skillforge uses one public issue-state dispatcher: `.github/workflows/codex-issue-state.yml`. It is the only workflow that reacts to `issues:labeled` and routes state transitions to reusable internal workflows:

- `execution-ready` launches/resumes the controlling issue through `.github/workflows/codex-execute-ready.yml`;
- `review-ready` launches an isolated exact-head PR audit through `.github/workflows/codex-review-ready.yml`;
- `completed` or deliberately restored `queued` on an epic child wakes the unique active parent whose canonical DAG contains that child.

The executor and auditor use a repository self-hosted runner labeled `codex` and the Codex App Server already shared with Desktop Remote Control. The dispatcher itself may run on GitHub-hosted infrastructure for control-plane routing.

Local runner infrastructure is optional. Without a matching self-hosted runner, the template remains valid but execution/audit jobs cannot run locally. `skills/codex-local-runner/SKILL.md` installs or repairs the repository-scoped runner only when explicitly requested; it never creates API keys or exposes inbound services.

## Epic DAG execution

An epic is designed as a small parent seed plus self-contained child issues. The parent seed declares `execution_mode: epic-dag`, a child issue list, a parallelism limit, and optionally an integration branch. **The DAG is not pre-generated during design.**

On the parent's first scheduler execution, `skills/codex-epic-scheduler/SKILL.md` reads all child contracts, derives the minimal direct dependency graph and serialization mutexes, validates it, initializes the integration branch when needed, and persists one canonical `codex-epic-dag:v1` parent comment. The parent then becomes `in-progress` and the scheduler activates the first deterministic dependency-ready wave.

Later wake-ups reconstruct state from GitHub labels plus that canonical graph. Selected children receive canonical execution context (epic, integration branch, exact base SHA) before moving from `queued` to `execution-ready`. The scheduler itself never implements, reviews, or integrates child work.

## Review and completion

Ordinary executors stop at a ready PR plus `review-ready`. The final review is performed in a fresh isolated audit context through `codex-pr-audit`/`codex-independent-review`. A positive final-capable exact-head audit has standing authority to merge that exact head, expose `completed`, and close the controlling issue. Audit failure returns the issue to execution; integration drift returns it for reconciliation without fabricating a technical failure.

## Optional derived wiki

`skills/repository-wiki-curation/SKILL.md` maintains optional `wiki/**` derived from repository/GitHub evidence. The wiki is non-normative. The curator may publish only `wiki/**` directly to default branch after adversarial review and a fail-closed path boundary; actionable discrepancies go to GitHub issues labeled `curator-detected`.

The core template remains generic. Project truth belongs in each repository created from it.
