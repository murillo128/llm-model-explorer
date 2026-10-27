import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import type { ReactNode } from 'react';
import { ApiClient } from '../api/client';
import { Lifetime } from '../app/lifetime';
import { neuronArchitecture, neuronGraph, neuronCapabilities, neuronInventory, neuronSession, unavailableNeurons } from '../test/neuron-fixture';
import { deferred } from '../test/shell-fixtures';
import { ArchitectureExplorer } from './ArchitectureExplorer';
import { GraphViews } from './graph';
import type { ArchitectureSelection } from './ArchitectureCanvas';

vi.mock('./ArchitectureCanvas', () => ({ ArchitectureCanvas: ({ notices, onInspect }: { notices: ReactNode; onInspect: (value: ArchitectureSelection) => void }) =>
  <div>{notices}<button onClick={event => onInspect({ modelId: neuronSession.model_id, sessionId: neuronSession.id,
    graphId: neuronGraph.graph_id,
    node: neuronGraph.nodes.find(node => node.id === 'linear1')!,
    trigger: event.currentTarget })}>Inspect concrete MLP</button></div> }));
vi.mock('../explorers/TensorExplorer', () => ({ TensorExplorer: (props: { selectedTensor: { id: string }; selectedRows?: number[]; selectedColumns?: number[] }) =>
  <div aria-label="Bound matrix">{JSON.stringify({ tensor: props.selectedTensor.id, rows: props.selectedRows, columns: props.selectedColumns })}</div> }));

beforeEach(() => {
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; };
});

function setup() {
  const client = new ApiClient({ backendBaseUrl: 'https://fixture.example' });
  vi.spyOn(client, 'getArchitecture').mockResolvedValue(neuronArchitecture);
  vi.spyOn(client, 'listTensors').mockResolvedValue(neuronInventory);
  const getCapabilities = vi.spyOn(client, 'getNeuronCapabilities').mockResolvedValue(neuronCapabilities);
  const props = { client, session: neuronSession, sessionId: neuronSession.id, selectedTensor: null, selection: new Lifetime(),
    reportStatus: vi.fn(), views: new GraphViews(), tokenizerAvailable: true };
  return { props, getCapabilities, user: userEvent.setup() };
}

it('opens a concrete layer-7 neuron, validates index edits and navigates all three bound axes without fetching again', async () => {
  const { props, getCapabilities, user } = setup();
  render(<ArchitectureExplorer {...props} neuronExperiment={value => <output aria-label="Experiment selection">{value.module.module_id}:{value.neuronIndices.join(',')}</output>} />);
  await user.click(await screen.findByRole('button', { name: 'Inspect concrete MLP' }));
  await user.click(await screen.findByRole('button', { name: 'Explore neurons' }));
  expect(screen.getByLabelText('Concrete MLP')).toHaveValue('mlp-7');
  const input = screen.getByRole('spinbutton', { name: 'Intermediate index' });
  await user.clear(input); await user.type(input, '3');
  expect(screen.getByRole('alert')).toHaveTextContent('0 to 2');
  // Keep the experiment editor mounted, but provide no actionable indices.
  expect(screen.getByLabelText('Experiment selection').textContent).toBe('mlp-7:');
  expect(screen.getByRole('button', { name: 'Open gate weight' })).toBeDisabled();
  await user.clear(input); await user.type(input, '2');
  await user.selectOptions(screen.getByLabelText('Concrete MLP'), 'mlp-7');
  expect(input).toHaveValue(2);
  expect(screen.getByLabelText('Experiment selection')).toHaveTextContent('mlp-7:2');
  for (const [projection, expected] of [['gate', { tensor: 'gate7', rows: [2] }], ['up', { tensor: 'up7', rows: [2] }], ['down', { tensor: 'down7', columns: [2] }]] as const) {
    await user.click(screen.getByRole('button', { name: `Open ${projection} weight` }));
    expect(JSON.parse((await screen.findByLabelText('Bound matrix')).textContent!)).toEqual(expected);
    await user.click(screen.getByRole('button', { name: 'Close inspection' }));
  }
  expect(getCapabilities).toHaveBeenCalledOnce();
  expect(props.client.listTensors).toHaveBeenCalledOnce();
  expect(props.client.getArchitecture).toHaveBeenCalledOnce();
  await user.click(screen.getByRole('button', { name: 'Back to graph' }));
  expect(screen.queryByLabelText('Neuron inspector')).not.toBeInTheDocument();
});

it('offers explicit module selection without graph linkage and local unavailable explanations without blocking graph inspection', async () => {
  const { props, getCapabilities, user } = setup();
  const cap = structuredClone(neuronCapabilities); delete cap.modules[0]!.graph_binding;
  getCapabilities.mockResolvedValue(cap);
  const view = render(<ArchitectureExplorer {...props} tokenizerAvailable={false} />);
  await screen.findByRole('button', { name: 'Inspect concrete MLP' });
  await user.click(screen.getByRole('button', { name: 'Neurons' }));
  await user.selectOptions(await screen.findByLabelText('Concrete MLP'), 'mlp-7');
  expect(screen.getByText(/Tokenizer unavailable\. Weights and graph/)).toBeInTheDocument();
  view.unmount();
  getCapabilities.mockResolvedValue(unavailableNeurons);
  render(<ArchitectureExplorer {...props} />);
  await screen.findByRole('button', { name: 'Inspect concrete MLP' });
  expect(await screen.findByText(/Observation unavailable/)).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Open gate weight' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Inspect concrete MLP' })).toBeInTheDocument();
});

it('aborts and rejects late capabilities on revision replacement, including the same session and module IDs', async () => {
  const { props, getCapabilities, user } = setup();
  const old = deferred<typeof neuronCapabilities>(); getCapabilities.mockReturnValueOnce(old.promise);
  const view = render(<ArchitectureExplorer {...props} />);
  await screen.findByRole('button', { name: 'Inspect concrete MLP' });
  await user.click(screen.getByRole('button', { name: 'Neurons' }));
  await waitFor(() => expect(getCapabilities).toHaveBeenCalledOnce());
  const signal = getCapabilities.mock.calls[0]![1]!;
  getCapabilities.mockResolvedValue({ ...unavailableNeurons, model_revision: 'replacement' });
  props.selection.dispose();
  view.rerender(<ArchitectureExplorer {...props} session={{ ...props.session, model_revision: 'replacement' }} selection={new Lifetime()} />);
  await screen.findByRole('button', { name: 'Inspect concrete MLP' });
  await user.click(screen.getByRole('button', { name: 'Neurons' }));
  await screen.findByText(/Observation unavailable/);
  await act(() => old.resolve(neuronCapabilities));
  expect(signal.aborted).toBe(true);
  expect(screen.queryByLabelText('Concrete MLP')).not.toBeInTheDocument();
});
