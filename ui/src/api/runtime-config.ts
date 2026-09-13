/** Public deployment configuration; never model paths, credentials or model settings. */
export interface RuntimeConfig {
  readonly backendBaseUrl: string;
}

const configurationHelp =
  'Set backend_base_url in runtime-config.json to an absolute HTTP(S) backend URL without credentials, a query or a fragment, then retry.';

export function parseRuntimeConfig(value: unknown): RuntimeConfig {
  if (typeof value !== 'object' || value === null ||
      !('backend_base_url' in value) || typeof value.backend_base_url !== 'string') {
    throw new Error(configurationHelp);
  }

  const input = value.backend_base_url.trim();
  let url: URL;
  try {
    // Require an explicit scheme and authority, rather than URL's permissive repair.
    if (!/^https?:\/\//i.test(input) || /[\s\\]/.test(input)) throw new Error();
    url = new URL(input);
  } catch {
    throw new Error(configurationHelp);
  }
  if (!url.hostname || url.username || url.password || input.includes('?') || input.includes('#')) {
    throw new Error(configurationHelp);
  }

  return Object.freeze({ backendBaseUrl: url.href.replace(/\/+$/, '') });
}

export async function loadRuntimeConfig(signal: AbortSignal): Promise<RuntimeConfig> {
  let response: Response;
  try {
    response = await fetch(`${import.meta.env.BASE_URL}runtime-config.json`, {
      cache: 'no-store', signal,
    });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error('Could not load runtime-config.json. Check the UI server and network, then retry.', { cause: error });
  }
  if (!response.ok) {
    throw new Error(`Could not load runtime-config.json (HTTP ${response.status}). Publish the file alongside the UI, then retry.`);
  }
  let value: unknown;
  try {
    value = await response.json();
  } catch {
    throw new Error('runtime-config.json is not valid JSON. Correct the file and retry.');
  }
  return parseRuntimeConfig(value);
}
