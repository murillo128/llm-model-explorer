import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { CardParameters, SummaryText } from './CardSummary';
import { cardMetrics, cardSummary, ownParameters, relativeParameterName } from './card-summary';
import type { Parameter } from './card-summary';
import type { GraphNode } from './graph';

const parameter = (id: string, name = id, rank = 2): Parameter => ({ id, name, binding: 'native',
  logical_shape: Array.from({ length: rank }, (_, i) => ({ kind: 'constant', value: i + 3 })),
  storage: [], inspection: { status: 'available', tensor_id: `inventory-${id}` }, provenance: [] });
const node = (): GraphNode => ({ id: 'norm7', label: 'layer norm', kind: 'operation', operation: 'layer_norm',
  formula: 'LayerNorm(x; weight, bias, epsilon)', ports: [], parameter_ids: ['weight', 'bias'],
  references: [{ kind: 'module', name: 'layers.7.norm' }, { kind: 'parameter', parameter_id: 'weight' }],
  attributes: [{ name: 'epsilon', value: 0.00003, provenance: [] }, { name: 'semantic_role', value: 'input_normalization', provenance: [] }], provenance: [] });
const parameters = new Map([parameter('weight', 'layers.7.norm.weight'), parameter('bias', 'layers.7.norm.bias', 1)].map((p) => [p.id, p]));

describe('source-backed card content', () => {
  it('deduplicates exact own references including group-owned parameters, without descending', () => {
    const record = node();
    record.references.push({ kind: 'parameter', parameter_id: 'bias' });
    expect(ownParameters(record, parameters).map((p) => p.id)).toEqual(['weight', 'bias']);
    const group: GraphNode = { ...record, kind: 'group', children: ['child'], parameter_ids: [], references: [] };
    expect(cardSummary(group, parameters).parameters).toEqual([]);
    group.references.push({ kind: 'parameter', parameter_id: 'weight' });
    expect(cardSummary(group, parameters).parameters.map((p) => p.id)).toEqual(['weight']);
  });
  it('preserves every path segment and unqualified name, and refuses ambiguous module prefixes', () => {
    const record = node();
    for (const name of ['q_proj.weight', 'weight', 'layers.8.norm.weight', 'other.weight'])
      expect(relativeParameterName(record, parameter('p', name))).toBe(name);
    expect(relativeParameterName(record, parameter('p', 'layers.7.norm.q_proj.weight'))).toBe('q_proj.weight');
    record.references.push({ kind: 'module', name: 'layers.7' });
    expect(relativeParameterName(record, parameter('p', 'layers.7.norm.weight'))).toBe('layers.7.norm.weight');
  });
  it('keeps nondefault scalar names/values and unknowns, omits absent constants and unrelated configuration', () => {
    const record = node();
    expect(cardSummary(record, parameters).constants.map(({ name, value }) => [name, value])).toEqual([['epsilon', 0.00003]]);
    record.attributes = [];
    expect(cardSummary(record, parameters).constants).toEqual([]);
    record.attributes = [{ name: 'eps', value: null, provenance: [] }];
    expect(cardSummary(record, parameters).constants[0]?.value).toBeNull();
    record.operation = 'linear';
    expect(cardSummary(record, parameters).constants).toEqual([]);
  });
  it.each([true, false])('shows only the supplied formula and tensors for biased=%s linear', (biased) => {
    const record = node(); record.operation = 'linear';
    record.parameter_ids = biased ? ['weight', 'bias'] : ['weight'];
    record.formula = biased ? 'x W^T + bias' : 'x W^T';
    const summary = cardSummary(record, parameters);
    expect(summary.formula).toBe(record.formula);
    expect(summary.parameters.map((p) => p.id)).toEqual(record.parameter_ids);
    delete record.formula;
    expect(cardSummary(record, parameters).formula).toBeUndefined();
  });
});

it('keeps constants and names with dimensions off, exposes logical shapes and exact independent matrix actions', () => {
  const record = node(), matrix = vi.fn(), inspect = vi.fn(), outer = vi.fn();
  const props = { node: record, summary: cardSummary(record, parameters), matrix, inspect, dimensions: false, top: 100 };
  const view = render(<div onClick={outer} onDoubleClick={outer} onKeyDown={outer}><CardParameters {...props} /></div>);
  expect(screen.getByLabelText('layers.7.norm.weight')).toHaveAccessibleName('layers.7.norm.weight');
  expect(screen.getByLabelText('epsilon = 0.00003')).toBeInTheDocument();
  expect(screen.queryByText('[3 × 4]')).not.toBeInTheDocument();
  expect(matrix).not.toHaveBeenCalled();
  const bias = screen.getByRole('button', { name: 'Inspect matrix layers.7.norm.bias' });
  fireEvent.click(bias); fireEvent.doubleClick(bias); fireEvent.keyDown(bias, { key: 'Enter' });
  expect(matrix).toHaveBeenCalledExactlyOnceWith('bias', bias);
  expect(outer).not.toHaveBeenCalled(); expect(inspect).not.toHaveBeenCalled();
  view.rerender(<CardParameters {...props} dimensions />);
  expect(screen.getByLabelText('[3 × 4]')).toBeInTheDocument();
  expect(screen.getByLabelText('[3]')).toBeInTheDocument();
  expect(screen.getByLabelText('epsilon = 0.00003')).toBeInTheDocument();
});

it.each(['unsupported_rank', 'unsupported_representation', 'requires_view', 'unresolved_binding'] as const)(
  'explains non-actionable %s without a matrix request', (reason) => {
    const record = node(), p = parameter('weight');
    p.inspection = { status: 'unavailable', reason, message: 'Supplied explanation.' };
    const matrix = vi.fn();
    render(<CardParameters node={record} summary={cardSummary(record, new Map([[p.id, p]]))} dimensions top={100} matrix={matrix} inspect={vi.fn()} />);
    const button = screen.getByRole('button', { name: 'Inspect matrix weight' });
    expect(button).toHaveAttribute('aria-disabled', 'true');
    expect(button).toHaveAccessibleDescription(`${reason.replaceAll('_', ' ')}: Supplied explanation.`);
    fireEvent.click(button); expect(matrix).not.toHaveBeenCalled();
  });

it('keeps long lists truthful and formulas inert with bounded, content-dependent geometry', () => {
  const record = node(); record.parameter_ids = Array.from({ length: 9 }, (_, i) => `p${i}`); record.references = [];
  const all = new Map(record.parameter_ids.map((id) => [id, parameter(id, `module.${'long.'.repeat(50)}${id}`)]));
  const summary = cardSummary(record, all), inspect = vi.fn();
  const projected = { id: record.id, label: record.label, kind: record.kind, record, ports: [], sourceIds: [record.id], expanded: false };
  const long = cardMetrics(projected, summary, false);
  expect(long.width).toBeLessThanOrEqual(300);
  expect(long.height).toBeGreaterThan(cardMetrics(projected, cardSummary(node(), parameters), false).height);
  render(<><SummaryText text={'<script>untrusted()</script>'.repeat(30)} /><CardParameters node={record} summary={summary} dimensions={false} top={100} matrix={vi.fn()} inspect={inspect} /></>);
  expect(document.querySelector('script')).toBeNull();
  expect(screen.getAllByRole('button', { name: /^Inspect matrix/ })).toHaveLength(6);
  const more = screen.getByRole('button', { name: 'Inspect 3 more tensors of layer norm' });
  expect(more).toHaveTextContent('+3 more'); fireEvent.click(more);
  expect(inspect).toHaveBeenCalledExactlyOnceWith(more);
});
