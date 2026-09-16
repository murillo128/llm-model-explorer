import type { components } from '../api/generated/types';

export type Token = components['schemas']['Token'];
export type Tokenization = components['schemas']['TokenizeResponse'];
export interface SourceGroup { start: number; end: number; tokens: Token[] }
export interface UnmappedToken { token: Token; anchor: number }
export interface SourceSelection { readonly from: number; readonly to: number }

/** Intersect native UTF-16 selections with individual trusted source spans.
 * Annotation unions/anchors and vocabulary IDs cannot establish row identity. */
export function selectedTokenRows(result: Tokenization, ranges: readonly SourceSelection[]): number[] {
  const boundaries = utf16Boundaries(result.text);
  const nonempty = ranges.map(({ from, to }) => [Math.min(from, to), Math.max(from, to)] as const)
    .filter(([from, to]) => Number.isFinite(from) && Number.isFinite(to) && from < to);
  return [...new Set(result.tokens.filter(token => {
    if (token.start === undefined || token.end === undefined || token.end <= token.start) return false;
    const start = boundaries[token.start], end = boundaries[token.end];
    return start !== undefined && end !== undefined && nonempty.some(([from, to]) => start < to && end > from);
  }).map(token => token.index))].sort((a, b) => a - b);
}

/** API offsets are code points; JavaScript and editor offsets are UTF-16 units. */
export function utf16Boundaries(text: string): number[] {
  const boundaries = [0];
  for (const point of text) boundaries.push(boundaries[boundaries.length - 1]! + point.length);
  return boundaries;
}

/** Only associate spans supplied by the backend. This does no tokenization. */
export function annotate(result: Tokenization): { groups: SourceGroup[]; unmapped: UnmappedToken[] } {
  const boundaries = utf16Boundaries(result.text);
  const spans = result.tokens.filter((token) => token.start !== undefined && token.end !== undefined && token.end > token.start);
  const groups: SourceGroup[] = [];
  for (const token of [...spans].sort((a, b) => a.start! - b.start! || a.end! - b.end!)) {
    const start = boundaries[token.start!]!;
    const end = boundaries[token.end!]!;
    const previous = groups.at(-1);
    if (previous && start < previous.end) {
      previous.end = Math.max(previous.end, end);
      previous.tokens.push(token);
    } else groups.push({ start, end, tokens: [token] });
  }
  for (const group of groups) group.tokens.sort((a, b) => a.index - b.index);
  const mapped = new Set(spans);
  const unmapped = result.tokens.filter((token) => !mapped.has(token)).map((token) => {
    // This is a sequence anchor for an annotation, never a claimed source span.
    const next = spans.find((span) => span.index > token.index);
    const previous = spans.findLast((span) => span.index < token.index);
    return { token, anchor: next ? boundaries[next.start!]! : previous ? boundaries[previous.end!]! : 0 };
  });
  return { groups, unmapped };
}
