import type { ReactNode } from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import fixture from '../../../api/fixtures/architecture.json';
import diagnosticCases from '../../../api/fixtures/model-defined-diagnostics.json';
import { validateSchema } from '../api/validation';
import { ApiClient } from '../api/client';
import { ApiFailure } from '../api/errors';
import { Lifetime } from '../app/lifetime';
import { ArchitectureExplorer } from './ArchitectureExplorer';
import { ModelDiagnostics } from '../app/model-diagnostics';
import { GraphViews } from './graph';

vi.mock('./ArchitectureCanvas', () => ({ ArchitectureCanvas: ({ modelId, notices }: { modelId: string; notices: ReactNode }) => <div>{notices}Graph for {modelId}</div> }));
const response = validateSchema('ArchitectureResponse', fixture.response);
const session = { id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', model_id: response.model_id };
const inventory = validateSchema('TensorInventory', fixture.context.inventory);
function setup() {
  const client = new ApiClient({ backendBaseUrl: 'https://example.test' });
  vi.spyOn(client, 'listTensors').mockResolvedValue(inventory);
  const retrieve = vi.spyOn(client, 'getArchitecture');
  const selection = new Lifetime();
  const diagnostics = new ModelDiagnostics(); diagnostics.activate(session);
  const props = { client, session, sessionId: session.id, selectedTensor: null, selection, reportStatus: vi.fn(), views: new GraphViews(), diagnostics, tokenizerAvailable: true };
  return { props, retrieve, selection };
}
it('fences late model/session results and aborts the obsolete retrieval', async () => {
  const { props, retrieve, selection } = setup();
  let resolve!: (value: typeof response) => void;
  retrieve.mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  const view = render(<ArchitectureExplorer {...props} />);
  await waitFor(() => expect(retrieve).toHaveBeenCalledOnce());
  const signal = retrieve.mock.calls[0]![2]!;
  selection.dispose();
  const next = { ...props, session: { ...session, model_id: 'replacement' }, selection: new Lifetime() };
  retrieve.mockResolvedValueOnce({ status: 'unavailable', model_id: 'replacement', reason: 'cache_unavailable', requires_restart: true, diagnostics: [] });
  props.diagnostics.activate(next.session);
  view.rerender(<ArchitectureExplorer {...next} />);
  await screen.findByLabelText('Architecture capability');
  await waitFor(() => expect(screen.getAllByText('The prepared architecture cache is unavailable.').length).toBeGreaterThan(0));
  await act(() => resolve(response));
  expect(signal.aborted).toBe(true);
  expect(props.diagnostics.getSnapshot().records.every((d) => JSON.parse(d.id)[0] === 'replacement')).toBe(true);
  expect(screen.queryByText(`Graph for ${response.model_id}`)).not.toBeInTheDocument();
});
it.each(['unsupported_architecture', 'analysis_failed', 'restart_required', 'unsupported_size', 'cache_unavailable'] as const)('localizes unavailable %s without polling', async (reason) => {
  const { props, retrieve } = setup();
  retrieve.mockResolvedValue({ status: 'unavailable', model_id: session.model_id, reason, requires_restart: ['restart_required', 'cache_unavailable'].includes(reason), diagnostics: [] });
  render(<ArchitectureExplorer {...props} />);
  await waitFor(() => expect(screen.queryByText('Retrieving prepared architecture…')).not.toBeInTheDocument());
  expect(retrieve).toHaveBeenCalledOnce();
  expect(screen.queryByRole('button', { name: /generat/i })).not.toBeInTheDocument();
});
it('cancels unmounted requests even when a transport ignores abort', async () => {
  const { props, retrieve } = setup();
  retrieve.mockReturnValue(new Promise(() => {}));
  const view = render(<ArchitectureExplorer {...props} />);
  await waitFor(() => expect(retrieve).toHaveBeenCalledOnce());
  view.unmount(); expect(retrieve.mock.calls[0]![2]!.aborted).toBe(true);
});

it('marks model-owned graphs without claiming source verification', async () => {
  const { props, retrieve } = setup();
  const owned = validateSchema('ArchitectureResponse', {
    ...fixture.response, graph: { ...fixture.response.graph, scope: 'model_defined' },
  });
  retrieve.mockResolvedValue(owned);
  render(<ArchitectureExplorer {...props} />);
  expect(await screen.findByRole('note', { hidden: true })).toHaveTextContent('equivalence to model code is not verified');
  expect(screen.getByText(`Graph for ${session.model_id}`)).toBeInTheDocument();
});

it.each(diagnosticCases)('shows the exact $name load finding beside the capability status', async (item) => {
  const { props, retrieve } = setup();
  retrieve.mockResolvedValue({
    status: 'unavailable', model_id: session.model_id,
    reason: item.code === 'unsupported_size' ? 'unsupported_size' : 'analysis_failed',
    requires_restart: true, diagnostics: [{ code: item.code, message: item.message }],
  });
  render(<ArchitectureExplorer {...props} />);
  expect(await screen.findByText(item.code === 'unsupported_size'
    ? 'This architecture exceeds the supported response size.'
    : 'Architecture preparation failed for this model.')).toBeInTheDocument();
  expect((await screen.findAllByText(item.message)).length).toBeGreaterThan(0);
});

it('shows the bounded protocol validation failure', async () => {
  const { props, retrieve } = setup();
  retrieve.mockRejectedValue(new ApiFailure('protocol', 'Invalid ArchitectureResponse schema'));
  render(<ArchitectureExplorer {...props} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Invalid ArchitectureResponse schema');
});
