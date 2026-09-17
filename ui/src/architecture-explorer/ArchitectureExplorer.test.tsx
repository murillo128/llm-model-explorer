import { act, render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import fixture from '../../../api/fixtures/architecture.json';
import { validateSchema } from '../api/validation';
import { ApiClient } from '../api/client';
import { Lifetime } from '../app/lifetime';
import { ArchitectureExplorer } from './ArchitectureExplorer';
import { GraphViews } from './graph';

vi.mock('./ArchitectureCanvas', () => ({ ArchitectureCanvas: ({ modelId }: { modelId: string }) => <div>Graph for {modelId}</div> }));
const response = validateSchema('ArchitectureResponse', fixture.response);
const session = { id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', model_id: response.model_id };
const inventory = validateSchema('TensorInventory', fixture.context.inventory);
function setup() {
  const client = new ApiClient({ backendBaseUrl: 'https://example.test' });
  vi.spyOn(client, 'listTensors').mockResolvedValue(inventory);
  const retrieve = vi.spyOn(client, 'getArchitecture');
  const selection = new Lifetime();
  const props = { client, session, sessionId: session.id, selectedTensor: null, selection, reportStatus: vi.fn(), views: new GraphViews(), tokenizerAvailable: true };
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
  view.rerender(<ArchitectureExplorer {...next} />);
  await screen.findByText('The prepared architecture cache is unavailable.');
  await act(() => resolve(response));
  expect(signal.aborted).toBe(true);
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
  expect(await screen.findByRole('note')).toHaveTextContent('equivalence to model code is not verified');
  expect(screen.getByText(`Graph for ${session.model_id}`)).toBeInTheDocument();
});
