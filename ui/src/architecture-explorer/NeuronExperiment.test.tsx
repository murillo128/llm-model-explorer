import { render, screen, within } from '@testing-library/react';
import { expect, it } from 'vitest';
import fixtures from '../../../api/fixtures/neurons.json';
import { validateSchema } from '../api/validation';
import { ExperimentReadout } from './NeuronExperiment';
import type { ExperimentResult } from './neuron-experiment-controller';

function result(kind: 'activations' | 'probabilities'): ExperimentResult {
  const metadata = kind === 'activations' ? validateSchema('NeuronActivationsMetadata', fixtures.bases.activations)
    : validateSchema('NeuronComparisonMetadata', fixtures.bases.probabilities);
  return { metadata, values: new Float32Array(fixtures.numeric[kind]), tokenSignal: new AbortController().signal,
    snapshot: { session: { id: metadata.session_id, model_id: metadata.model_id, model_revision: metadata.model_revision },
      capabilities: validateSchema('NeuronCapabilities', fixtures.bases.capabilities), mode: kind === 'activations' ? 'activations' : 'compare',
      request: validateSchema('NeuronActivationsRequest', fixtures.bases.observation), input: { text: 'x x', add_special_tokens: false,
        tokens: [0, 1].map(index => ({ index, id: 1, token: 'x', decoded: 'x', special: false })) } } };
}
it('exposes signed exact values by position even for repeated token IDs and reversed neuron order', () => {
  render(<ExperimentReadout result={result('activations')} />);
  expect(screen.getByLabelText('Position 0, token 1, neuron 2: -1')).toHaveAttribute('tabindex', '0');
  expect(screen.getByLabelText('Position 0, token 1, neuron 0: +2')).toHaveAttribute('title', '+2');
  expect(screen.getByLabelText('Position 1, token 1, neuron 0: -3')).toBeVisible();
  expect(screen.getAllByRole('row')).toHaveLength(3);
});
it('shows backend candidate order, unchanged subset probabilities and percentage-point deltas separately from continuations', () => {
  const value = result('probabilities');
  if (value.metadata.kind !== 'neuron_comparison') throw new Error('fixture');
  value.metadata.continuations = [{ condition_id: 'baseline', token_ids: [2], text: 'First branch.', finish_reason: 'length' },
    { condition_id: 'zero-prompt', token_ids: [3], text: 'Different branch.', finish_reason: 'length' }];
  render(<ExperimentReadout result={value} />);
  const table = screen.getByRole('table');
  expect(within(table).getAllByRole('rowheader').map(cell => cell.textContent)).toEqual(['0', '1', '2']);
  expect(within(table).getAllByText('12.5000%')).toHaveLength(2);
  expect(within(table).getByText('Δ +37.5000 pp')).toBeVisible(); expect(within(table).getByText('Δ -37.5000 pp')).toBeVisible();
  expect(screen.getByRole('article', { name: 'baseline continuation' })).toHaveTextContent('First branch.');
  expect(screen.getByRole('article', { name: 'zero-prompt continuation' })).toHaveTextContent('Different branch.');
  expect(table).not.toHaveTextContent('branch');
});
