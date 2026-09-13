# Renderer boundary

This directory is reserved for reusable TypeScript WebGL2 rendering code. It has no
implementation yet. React components belong in `app/` or `components/`; they own
layout and ordinary controls. Rendering code owns GPU resources and drawing and
must not import React, application components, or JSX. ESLint checks direct imports
and JSX in this boundary. Backend model computation does not belong here.

See [UI architecture](../../../docs/spec/ui/architecture.md). Future renderer APIs
and tensor behavior are defined by their own controlling issues; this shell does
not predefine them.
