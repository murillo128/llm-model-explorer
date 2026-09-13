import { describe, expect, it, vi } from 'vitest';
import { loadRuntimeConfig, parseRuntimeConfig } from './runtime-config';

describe('public runtime configuration', () => {
  it.each([
    ['http://localhost:8000/', 'http://localhost:8000'],
    ['https://models.example/api///', 'https://models.example/api'],
    [' https://MODELS.example/ ', 'https://models.example'],
    ['http://[::1]:8000/', 'http://[::1]:8000'],
  ])('normalizes %s', (input, expected) => {
    expect(parseRuntimeConfig({ backend_base_url: input }).backendBaseUrl).toBe(expected);
  });

  it.each([
    null, [], {}, { backend_base_url: 42 },
    ...['', '/api', '//models.example', 'file:///models', 'ftp://models.example',
      'javascript:alert(1)', 'http:models.example', 'http://', 'https://user:secret@models.example',
      'https://models.example/?secret=hidden', 'https://models.example/#fragment',
      'https://models.example/?', 'https://models.example/#', 'https://models .example',
      'https://models.example/\npath', 'https://models.example\\path',
    ].map((backend_base_url) => ({ backend_base_url })),
  ])('rejects malformed, non-HTTP or non-public config: %j', (value) => {
    expect(() => parseRuntimeConfig(value)).toThrow('Set backend_base_url in runtime-config.json');
  });

  it('loads uncached config using the supplied abort signal', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"backend_base_url":"https://backend.example/"}'));
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();
    expect(await loadRuntimeConfig(controller.signal)).toEqual({ backendBaseUrl: 'https://backend.example' });
    expect(fetchMock).toHaveBeenCalledWith('/runtime-config.json', { cache: 'no-store', signal: controller.signal });
  });

  it.each([404, 500])('explains an HTTP %s failure', async (status) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status })));
    await expect(loadRuntimeConfig(new AbortController().signal)).rejects.toThrow(`HTTP ${status}`);
  });

  it('explains malformed JSON, including a static-server HTML fallback', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<!doctype html>')));
    await expect(loadRuntimeConfig(new AbortController().signal)).rejects.toThrow('not valid JSON');
  });

  it('explains network failure without exposing raw server details', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('private details')));
    await expect(loadRuntimeConfig(new AbortController().signal)).rejects.toThrow('Check the UI server and network');
  });
});
