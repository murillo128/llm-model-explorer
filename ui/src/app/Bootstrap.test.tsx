import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it, vi } from 'vitest';
import { Bootstrap } from './Bootstrap';

it('gates backend requests on validated configuration', async () => {
  let respond!: (response: Response) => void;
  const fetchMock = vi.fn().mockReturnValueOnce(new Promise<Response>((resolve) => { respond = resolve; }))
    .mockReturnValue(new Promise<Response>(() => {}));
  vi.stubGlobal('fetch', fetchMock);
  render(<Bootstrap />);
  expect(screen.getByRole('status')).toHaveTextContent('Loading application configuration');
  expect(screen.queryByRole('navigation')).not.toBeInTheDocument();
  respond(new Response('{"backend_base_url":"https://models.example/api/"}'));
  expect(await screen.findByRole('heading', { level: 1, name: 'Tensor Explorer' })).toBeInTheDocument();
  expect(screen.getByTestId('backend-url')).toHaveTextContent('https://models.example/api');
  expect(screen.getByText('Loading models…')).toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(fetchMock.mock.calls[1]?.[0]).toBe('https://models.example/api/models');

  await userEvent.click(screen.getByRole('button', { name: 'Tokenizer Explorer' }));
  expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Tokenizer Explorer');
  expect(screen.getByRole('button', { name: 'Tokenizer Explorer' })).toHaveAttribute('aria-current', 'page');
  expect(screen.getByText('Open a model session to use this explorer.')).toBeInTheDocument();
});

it('offers a working retry after invalid configuration', async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response('{}'))
    .mockResolvedValueOnce(new Response('{"backend_base_url":"http://localhost:9000"}'))
    .mockResolvedValue(new Response('{"models":[]}', { headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', fetchMock);
  render(<Bootstrap />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Set backend_base_url');
  expect(screen.queryByRole('navigation')).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Retry configuration' }));
  await waitFor(() => expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Tensor Explorer'));
  expect(screen.getByTestId('backend-url')).toHaveTextContent('http://localhost:9000');
});

it('aborts startup when unmounted', () => {
  const fetchMock = vi.fn().mockReturnValue(new Promise(() => {}));
  vi.stubGlobal('fetch', fetchMock);
  const { unmount } = render(<Bootstrap />);
  const options = fetchMock.mock.calls[0]?.[1] as RequestInit;
  unmount();
  expect(options.signal?.aborted).toBe(true);
});
