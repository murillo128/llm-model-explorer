import { StrictMode, useEffect } from 'react';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import { deferred, json, models, sessionA, sessionB, tensors } from '../test/shell-fixtures';
import { App } from './App';
import type { ExplorerContextValue } from './explorer-context';
import { sessionStorageKey } from './session-controller';

const config = { backendBaseUrl: 'https://backend.example' };
afterEach(() => sessionStorage.clear());
function mockBackend() {
  const fetcher = vi.fn(async (input: string | URL | Request, options?: RequestInit) => {
    const url = String(input);
    if (url.endsWith('/models')) return json({ models, diagnostics: [] });
    if (url.endsWith('/tokenize')) return json({ ...JSON.parse(options!.body as string), tokens: [] });
    if (url.endsWith('/tensors')) return json({ tensors, coverage: 'complete', diagnostics: [] });
    if (options?.method === 'DELETE') return new Response(null, { status: 204 });
    if (options?.method === 'POST') return json(JSON.parse(options.body as string).model_id === models[1]!.id ? sessionB : sessionA, 201);
    return json(sessionA);
  });
  vi.stubGlobal('fetch', fetcher);
  return fetcher;
}
async function chooseAlpha() {
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Model' })).toBeEnabled());
  await userEvent.selectOptions(screen.getByRole('combobox'), models[0]!.id);
  await screen.findByRole('button', { name: /left.weight/ });
}

it('renders public model metadata and logical hierarchy, describes unsupported ranks without mounting a viewer', async () => {
  mockBackend();
  const slot = vi.fn((props: ExplorerContextValue) => <p>Viewer: {props.selectedTensor?.id}</p>);
  render(<StrictMode><App config={config} slots={{ tensor: slot }} /></StrictMode>);
  await chooseAlpha();
  expect(within(screen.getByRole('contentinfo')).getByText('ExampleArchitecture')).toBeInTheDocument();
  expect(screen.getByText('left', { selector: 'summary > span' })).toBeInTheDocument();
  expect(screen.getByText('right', { selector: 'summary > span' })).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: /left.weight/ }));
  expect(screen.getByText('Viewer: first')).toBeInTheDocument();
  expect(screen.queryByText('safetensors')).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Tensor information' }));
  expect(screen.getByText('safetensors')).toBeVisible();
  await userEvent.keyboard('{Escape}');
  await userEvent.click(screen.getByRole('button', { name: /right.weight/ }));
  expect(screen.getByText('Viewer: second')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: /cube/ }));
  expect(screen.getByText(/This rank-3 tensor/)).toBeInTheDocument();
  expect(screen.queryByText(/Viewer:/)).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: /scalar/ }));
  expect(screen.getByText(/This rank-0 tensor/)).toBeInTheDocument();
});

it('mounts and releases child-owned consumers and rejects stale status after A → B → A or tool switching', async () => {
  mockBackend();
  const contexts: ExplorerContextValue[] = [];
  const cancellations: string[] = [];
  function Probe(props: ExplorerContextValue) {
    useEffect(() => {
      contexts.push(props);
      const cancel = () => { cancellations.push(props.selectedTensor!.id); };
      const unregister = props.selection.onDispose(cancel);
      return unregister;
    }, [props]);
    return <p>Consumer {props.selectedTensor?.id}</p>;
  }
  render(<App config={config} slots={{ tensor: Probe, tokenizer: () => <p>Tokenizer editor</p> }} />); await chooseAlpha();
  await userEvent.click(screen.getByRole('button', { name: /left.weight/ }));
  const old = contexts.at(-1)!;
  await userEvent.click(screen.getByRole('button', { name: /right.weight/ }));
  await userEvent.click(screen.getByRole('button', { name: /left.weight/ }));
  act(() => old.reportStatus('failed'));
  expect(screen.queryByText('Operation failed.')).not.toBeInTheDocument();
  const current = contexts.at(-1)!;
  act(() => current.reportStatus('streaming'));
  expect(screen.getByText('Operation streaming.')).toBeInTheDocument();
  expect(contexts.at(-1)!.reportStatus).toBe(current.reportStatus);
  expect(cancellations).toEqual(['first', 'second']);
  await userEvent.click(screen.getByRole('button', { name: 'Tokenizer Explorer' }));
  expect(cancellations).toEqual(['first', 'second', 'first']);
  expect(screen.getByText('Session active.')).toBeInTheDocument();
});

it('recovers then expires a session honestly and allows fresh creation', async () => {
  const fetcher = mockBackend();
  sessionStorage.setItem(sessionStorageKey(config.backendBaseUrl), sessionA.id);
  fetcher.mockImplementation(async (input, options) => {
    if (String(input).endsWith('/models')) return json({ models, diagnostics: [] });
    if (options?.method === 'POST') return json(sessionA, 201);
    if (String(input).endsWith('/tensors')) return json({ tensors, coverage: 'complete', diagnostics: [] });
    return json({ code: 'session_not_found', message: '/srv/private/models' }, 404);
  });
  render(<App config={config} />);
  expect(await screen.findByText(/Session expired/)).toHaveTextContent('runtime state was not restored');
  expect(document.body.textContent).not.toContain('/srv/private');
  await chooseAlpha();
  expect(screen.getByText('Session active.')).toBeInTheDocument();
});

it('shows empty, malformed, and unreachable catalogues with working refresh', async () => {
  const fetcher = mockBackend();
  fetcher.mockResolvedValueOnce(json({ models: [], diagnostics: [] }));
  render(<App config={config} />);
  expect(await screen.findByText('No models available on this backend.')).toBeInTheDocument();
  fetcher.mockResolvedValueOnce(json({ models: [{ ...models[0], local_path: '/private/weights' }], diagnostics: [] }));
  await userEvent.click(screen.getByRole('button', { name: 'Refresh models' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('invalid response');
  expect(document.body.textContent).not.toContain('/private/weights');
  fetcher.mockRejectedValueOnce(new Error('/private/network/detail'));
  await userEvent.click(screen.getByRole('button', { name: 'Refresh models' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('unreachable');
  await userEvent.click(screen.getByRole('button', { name: 'Refresh models' }));
  await chooseAlpha();
});

it('shows rejected adapter diagnostics without offering an invalid composition as a model', async () => {
  const fetcher = mockBackend();
  fetcher.mockResolvedValueOnce(json({
    models: [],
    diagnostics: [{
      code: 'peft_composition_rejected', candidate: 'bad-adapter', base_model_id: 'org/base',
      message: 'LoRA composition with the local base was rejected: PEFT LoRA A/B dimensions do not match the base weight.',
    }],
  }));
  render(<App config={config} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('bad-adapter · org/base');
  expect(screen.getByRole('alert')).toHaveTextContent('A/B dimensions do not match the base weight');
  expect(screen.getByText('No selectable models are available.')).toBeInTheDocument();
  expect(screen.queryByRole('option', { name: /bad-adapter/ })).not.toBeInTheDocument();
  expect(screen.getByRole('combobox', { name: 'Model' })).toBeDisabled();
  expect(fetcher.mock.calls.some(([, options]) => options?.method === 'POST')).toBe(false);
});

it('backend replacement disposes the old view, fences old inventory, and uses backend-specific recovery', async () => {
  const fetcher = mockBackend();
  const pending = deferred<Response>();
  const { rerender } = render(<App config={config} />);
  await waitFor(() => expect(screen.getByRole('combobox')).toBeEnabled());
  fetcher.mockImplementationOnce(async () => json(sessionA, 201)).mockReturnValueOnce(pending.promise);
  await userEvent.selectOptions(screen.getByRole('combobox'), models[0]!.id);
  await screen.findByText('Loading tensor inventory…');
  rerender(<App config={{ backendBaseUrl: 'https://second.example' }} />);
  await waitFor(() => expect(screen.getByRole('combobox')).toBeEnabled());
  await act(async () => { pending.resolve(json({ tensors, coverage: 'complete', diagnostics: [] })); });
  expect(screen.queryByRole('button', { name: /left.weight/ })).not.toBeInTheDocument();
  expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('');
  expect(sessionStorage.getItem(sessionStorageKey(config.backendBaseUrl))).toBe(sessionA.id);
  expect(fetcher.mock.calls.some(([url]) => String(url).startsWith('https://second.example/sessions'))).toBe(false);
});


it('keeps model/session API actions in compact accessible controls and discloses raw metadata', async () => {
  const fetcher = mockBackend();
  fetcher.mockResolvedValueOnce(json({ models: [{ ...models[0], size_bytes: 272400000 }], diagnostics: [] }));
  render(<App config={config} />);
  await chooseAlpha();
  expect(screen.queryByRole('heading', { level: 1 })).not.toBeInTheDocument();
  expect(screen.queryByText('Tensor Explorer workspace')).not.toBeInTheDocument();
  expect(screen.getByText('272.4 MB')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Close session' })).not.toBeInTheDocument();
  const options = screen.getByRole('button', { name: 'Session options' });
  await userEvent.click(options);
  expect(options).toHaveAttribute('aria-expanded', 'true');
  expect(screen.getByText('272,400,000')).toBeVisible();
  await userEvent.keyboard('{Escape}');
  expect(options).toHaveFocus();
  expect(options).toHaveAttribute('aria-expanded', 'false');
  const requests = fetcher.mock.calls.length;
  await userEvent.click(screen.getByRole('button', { name: 'Refresh models' }));
  await waitFor(() => expect(fetcher.mock.calls.slice(requests).map(([url]) => url)).toEqual(['https://backend.example/models']));
  await userEvent.click(options);
  await userEvent.click(screen.getByRole('button', { name: 'Close session' }));
  await screen.findByText('Session closed.');
  expect(fetcher).toHaveBeenCalledWith(`https://backend.example/sessions/${sessionA.id}`, expect.objectContaining({ method: 'DELETE' }));
  expect(options).toHaveFocus();
  expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('');
});
