import type { Graph, GraphNode } from './graph';
import { formatShape } from './graph';
import type { ProjectedNode } from './projection';

export type Parameter = Graph['parameters'][number];
export const summaryLimit = 6;
export const summaryWidth = 300;

/** Exact own references only: containment and repeated structure confer no ownership. */
export function ownParameters(node: GraphNode, parameters: ReadonlyMap<string, Parameter>): Parameter[] {
  const ids = new Set([...node.parameter_ids, ...node.references.flatMap((r) => r.kind === 'parameter' ? [r.parameter_id] : [])]);
  return [...ids].flatMap((id) => { const parameter = parameters.get(id); return parameter ? [parameter] : []; });
}

export function relativeParameterName(node: GraphNode, parameter: Parameter): string {
  const modules = new Set(node.references.flatMap((r) => r.kind === 'module' ? [r.name] : []));
  // Only an explicit, unique module reference establishes a removable prefix.
  const owner = modules.size === 1 ? [...modules][0] : undefined;
  return owner && parameter.name.startsWith(`${owner}.`) ? parameter.name.slice(owner.length + 1) : parameter.name;
}

// Computational scalars already published by the operation descriptions. Do not
// turn semantic_role, axes, provenance or arbitrary configuration into constants.
const scalarAttributes: Record<string, readonly string[]> = {
  layer_norm: ['epsilon', 'eps'], rms_norm: ['epsilon', 'eps', 'axis', 'weight_offset'],
  rms_norm_zero_centered: ['epsilon', 'eps'], scale: ['factor'],
};
export function cardSummary(node: GraphNode | undefined, parameters: ReadonlyMap<string, Parameter>) {
  return {
    formula: node?.formula,
    parameters: node ? ownParameters(node, parameters) : [],
    constants: node?.attributes.filter((a) => scalarAttributes[node.operation ?? '']?.includes(a.name) &&
      !Array.isArray(a.value)) ?? [],
  };
}
export type CardSummary = ReturnType<typeof cardSummary>;

/** One bounded geometry contract for the layout worker and rendered card. Long
 * text uses a full-text disclosure; hover/focus never changes these dimensions. */
export function cardMetrics(node: ProjectedNode, summary: CardSummary, dimensions: boolean, annotation = false) {
  const hasSubtitle = Boolean(summary.formula || node.summary || annotation);
  const portStart = 54 + (hasSubtitle ? 24 : 0);
  const portGap = dimensions ? 40 : 24;
  const portRows = Math.max(node.ports.filter((p) => p.direction === 'input').length,
    node.ports.filter((p) => p.direction === 'output').length);
  const metadataTop = portStart + portRows * portGap + 4;
  const rowHeight = dimensions ? 42 : 26;
  const portLabelWidth = (direction: 'input' | 'output') => Math.min(120, Math.max(0, ...node.ports.filter((p) => p.direction === direction)
    .map((p) => dimensions ? Math.max((p.interfaceLabel ?? p.label).length * 6, formatShape(p.shape).length * 6) : (p.interfaceLabel ?? p.label).length * 6)));
  const height = Math.max(84, metadataTop + Math.min(summaryLimit, summary.parameters.length) * rowHeight +
    (summary.parameters.length > summaryLimit ? 26 : 0) + summary.constants.length * 24 + 10);
  return { width: summary.formula || summary.parameters.length || summary.constants.length || dimensions ? summaryWidth :
    node.ports.some((p) => p.interfaces?.length) ? Math.max(180, portLabelWidth('input') + portLabelWidth('output') + 48) : 180,
    height, portStart, portGap, metadataTop, rowHeight, portLabelWidth: { input: portLabelWidth('input'), output: portLabelWidth('output') } };
}
