import { StrictMode, useEffect } from 'react';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import { deferred, json, models, sessionA, sessionB, tensors } from '../test/shell-fixtures';
import { App } from './App';
import type { ExplorerContextValue } from './explorer-context';
import { sessionStorageKey } from './session-controller';
import type { ModelSummary } from './session-controller';

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

it('uses uniform model labels while preserving SmolLM2 variants and exact session identifiers', async () => {
  const catalogue: ModelSummary[] = [
    { id: 'SmolLM2-135M', display_name: 'SmolLM2-135M', architectures: [], tokenizer_available: true },
    { id: 'HuggingFaceTB/SmolLM2-135M@bnb-nf4-dq', display_name: 'HuggingFaceTB/SmolLM2-135M (bnb-nf4-dq)', architectures: [], tokenizer_available: true },
    { id: 'SmolLM2-135M+peft-lora:smoltalk', display_name: 'SmolLM2-135M + LoRA smollm2-135m-smoltalk-lora', architectures: [], tokenizer_available: true },
    { id: 'HuggingFaceTB/SmolLM2-135M@bnb-nf4-dq+peft-lora:smoltalk', display_name: 'HuggingFaceTB/SmolLM2-135M (bnb-nf4-dq) + LoRA smollm2-135m-smoltalk-lora', architectures: [], tokenizer_available: true },
    { id: 'Contrastive-LM/CLM-v0.1-8B@e939398d4556fcd9400c76fa8c5a513202f42b0a', display_name: 'Contrastive-LM/CLM-v0.1-8B', architectures: [], tokenizer_available: true },
    { id: 'jaredpalmer/kev-0.8b-inspection@9a45d25eb2ab761841196625383fa1dff0e56c1e', display_name: 'jaredpalmer/kev-0.8b-inspection', architectures: [], tokenizer_available: true },
    { id: 'qwen@gptq-int4', display_name: 'Qwen3-0.6B-GPTQ-Int4 (gptq-int4)', architectures: [], tokenizer_available: true },
    { id: 'qwen35@nvfp4', display_name: 'Qwen3.5-0.8B-NVFP4 (nvfp4)', architectures: [], tokenizer_available: true },
  ];
  const fetcher = mockBackend();
  let selected = catalogue[0]!.id;
  fetcher.mockImplementation(async (input, options) => {
    if (String(input).endsWith('/models')) return json({ models: catalogue, diagnostics: [] });
    if (String(input).endsWith('/tensors')) return json({ tensors, coverage: 'complete', diagnostics: [] });
    if (options?.method === 'DELETE') return new Response(null, { status: 204 });
    if (options?.method === 'POST') selected = JSON.parse(options.body as string).model_id;
    return json({ ...sessionA, model_id: selected }, options?.method === 'POST' ? 201 : 200);
  });
  render(<App config={config} />);
  const picker = screen.getByRole('combobox', { name: 'Model' });
  await waitFor(() => expect(picker).toBeEnabled());
  const options = within(picker).getAllByRole('option').slice(1);
  expect(options.map((option) => option.textContent)).toEqual([
    'SmolLM2-135M', 'SmolLM2-135M (NF4 4-bit, double quantization)',
    'SmolLM2-135M + LoRA smollm2-135m-smoltalk-lora',
    'SmolLM2-135M (NF4 4-bit, double quantization) + LoRA smollm2-135m-smoltalk-lora',
    'CLM-v0.1-8B', 'kev-0.8b-inspection',
    'Qwen3-0.6B-GPTQ-Int4 (GPTQ 4-bit)', 'Qwen3.5-0.8B-NVFP4 (NVFP4 4-bit)',
  ]);
  expect(options.map((option) => option.getAttribute('value'))).toEqual(catalogue.map((entry) => entry.id));
  await userEvent.selectOptions(picker, catalogue[3]!.id);
  await screen.findByText('Session active.');
  expect(fetcher).toHaveBeenCalledWith('https://backend.example/sessions', expect.objectContaining({
    body: JSON.stringify({ model_id: catalogue[3]!.id }),
  }));
  await userEvent.click(screen.getByRole('button', { name: 'Session options' }));
  expect(screen.getByText(catalogue[3]!.id, { selector: 'dd' })).toBeVisible();
  expect(screen.getByText(catalogue[3]!.display_name, { selector: 'dd' })).toBeVisible();
});

it('keeps model choices distinct when shortened names or revision display names collide', async () => {
  const catalogue: ModelSummary[] = [
    { id: 'alice/shared', display_name: 'alice/shared', architectures: [], tokenizer_available: true },
    { id: 'bob/shared', display_name: 'bob/shared', architectures: [], tokenizer_available: true },
    { id: 'lab/revised@first', display_name: 'lab/revised', architectures: [], tokenizer_available: true },
    { id: 'lab/revised@second', display_name: 'lab/revised', architectures: [], tokenizer_available: true },
    { id: 'lab/custom@unknown', display_name: 'Custom (constructor)', architectures: [], tokenizer_available: true },
  ];
  mockBackend().mockResolvedValueOnce(json({ models: catalogue, diagnostics: [] }));
  render(<App config={config} />);
  const picker = screen.getByRole('combobox', { name: 'Model' });
  await waitFor(() => expect(picker).toBeEnabled());
  const options = within(picker).getAllByRole('option').slice(1);
  expect(options.map((option) => option.textContent)).toEqual([
    'alice/shared', 'bob/shared', 'lab/revised (lab/revised@first)',
    'lab/revised (lab/revised@second)', 'Custom (constructor)',
  ]);
  expect(new Set(options.map((option) => option.textContent)).size).toBe(catalogue.length);
  expect(options.map((option) => option.getAttribute('value'))).toEqual(catalogue.map((entry) => entry.id));
});

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

it('recovers under StrictMode without leaving a disposed recovery attempt in progress', async () => {
  mockBackend();
  sessionStorage.setItem(sessionStorageKey(config.backendBaseUrl), sessionA.id);
  render(<StrictMode><App config={config} /></StrictMode>);
  await screen.findByText('Session active.');
  expect(within(screen.getByRole('contentinfo')).getByText('Connected')).toBeVisible();
});

it('fences late operation failures across model and backend replacement without recreating current consumers for feedback', async () => {
  const fetcher = mockBackend();
  const contexts: ExplorerContextValue[] = [];
  const mounts = vi.fn();
  function Probe(props: ExplorerContextValue) {
    useEffect(() => { contexts.push(props); }, [props]);
    useEffect(() => { mounts(); }, []);
    return <p>Consumer</p>;
  }
  const { rerender } = render(<App config={config} slots={{ tensor: Probe }} />);
  await chooseAlpha();
  await userEvent.click(screen.getByRole('button', { name: /left.weight/ }));
  const stale = contexts.at(-1)!;
  const pending = deferred<Response>();
  fetcher.mockReturnValueOnce(pending.promise);
  let done!: Promise<unknown>;
  act(() => { done = stale.client.streamTensor(sessionA.id, 'first').done.catch(() => {}); });
  await userEvent.selectOptions(screen.getByRole('combobox'), models[1]!.id);
  await screen.findByRole('button', { name: /left.weight/ });
  await act(async () => { pending.reject(new TypeError('late private failure')); await done; });
  expect(within(screen.getByRole('list', { name: 'Application notifications' })).queryByRole('listitem')).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: /left.weight/ }));
  const current = contexts.at(-1)!;
  const before = mounts.mock.calls.length;
  fetcher.mockRejectedValueOnce(new TypeError('lost transport'));
  await act(async () => { await current.client.streamTensor(sessionB.id, 'first').done.catch(() => {}); });
  expect(within(screen.getByRole('list', { name: 'Application notifications' })).getAllByRole('listitem')).toHaveLength(1);
  expect(mounts).toHaveBeenCalledTimes(before);
  await userEvent.click(screen.getByRole('button', { name: 'Dismiss notification' }));
  expect(mounts).toHaveBeenCalledTimes(before);
  expect(within(screen.getByRole('contentinfo')).getByText('Disconnected')).toBeVisible();
  const backendPending = deferred<Response>();
  fetcher.mockReturnValueOnce(backendPending.promise);
  act(() => { done = current.client.streamTensor(sessionB.id, 'first').done.catch(() => {}); });
  rerender(<App config={{ backendBaseUrl: 'https://second.example' }} slots={{ tensor: Probe }} />);
  await waitFor(() => expect(screen.getByRole('combobox')).toBeEnabled());
  await act(async () => { backendPending.reject(new TypeError('late backend failure')); await done; });
  expect(within(screen.getByRole('list', { name: 'Application notifications' })).queryByRole('listitem')).not.toBeInTheDocument();
  expect(within(screen.getByRole('contentinfo')).getByText('Connected')).toBeVisible();
});

it('keeps reachable catalogue errors in a safe toast without moving the active workspace notice strip', async () => {
  const fetcher = mockBackend();
  render(<App config={config} />);
  await chooseAlpha();
  fetcher.mockResolvedValueOnce(json({ code: 'resource_exhausted', message: '/private/path' }, 503));
  await userEvent.click(screen.getByRole('button', { name: 'Refresh models' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Could not refresh models');
  expect(within(screen.getByRole('contentinfo')).getByText('Connected')).toBeVisible();
  expect(document.querySelector('.workspace-notices')).toBeEmptyDOMElement();
  expect(document.body.textContent).not.toContain('/private/path');
});
