# LLM Model Explorer

Interactive browser-based explorer for understanding transformer models by visualizing tensors, weights, activations, and matrix operations directly in the browser.

## Initial target

- Model: `HuggingFaceTB/SmolLM2-135M` (Base)
- Runtime: browser
- Rendering / tensor visualization: WebGL2 + GLSL shaders
- Language: TypeScript
- Dev server / build: Vite

## Design constraints

- Keep a single GPU-side representation of tensor data whenever possible.
- Preserve a 1:1 relationship between weight values and visualized texels where the chosen GPU format allows it.
- Treat color as a visualization concern in shaders rather than duplicating or rewriting the underlying weights.
- Build reusable matrix/vector visualization primitives, including an interactive matrix multiplication view where hovering an output element highlights the contributing row and column.

## Development

```bash
npm install
npm run dev
```

The repository starts with a minimal browser/WebGL2 foundation so tensor storage and visualization can be validated before adding application-framework complexity.
