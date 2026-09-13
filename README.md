# LLM Model Explorer

LLM Model Explorer is a browser-first interactive tool for understanding transformer models by inspecting and visualizing tensors, weights, activations, and matrix/vector operations.

## Current scope

The initial reference model is `HuggingFaceTB/SmolLM2-135M`, using the Base variant rather than Instruct. The target runtime is the browser, with WebGL2 and GLSL shaders as the working direction for GPU-side tensor visualization and interaction.

The explorer should make model computation inspectable rather than hiding it behind aggregate charts. Matrix and vector operations are intended to be reusable first-class views; for example, a matrix multiplication view should be able to show `[A] × [B] = [C]` and, when hovering an element of `C`, highlight the row of `A` and column of `B` that produced it.

## Data and visualization constraints

Weight values must remain exact unless an explicit design decision says otherwise. Do not introduce rounding, grouping, quantization, or lossy visual storage merely to simplify rendering.

Avoid maintaining duplicate GPU-side copies of the same weights solely for visualization. Prefer one authoritative GPU representation when the chosen WebGL2 format and operation allow it, and keep the relationship between stored values and visualized elements direct.

Color is a visualization concern. Visual color mapping and interaction should be performed in shaders or equivalent rendering logic without rewriting the underlying weight values just to change their appearance.

## Repository workflow

This repository inherits the Skillforge issue-driven development workflow. Durable product decisions belong in repository documentation, bounded implementation work belongs in GitHub issues, and non-trivial implementation should follow the repository's agent and review workflow.

The application implementation has not been scaffolded yet; this repository bootstrap establishes project scope and development invariants only.
