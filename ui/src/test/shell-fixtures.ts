import type { ModelSummary, Session, TensorDescriptor } from '../app/session-controller';

export const models: ModelSummary[] = [
  { id: 'lab/alpha', display_name: 'Alpha', architectures: ['ExampleArchitecture'], tokenizer_available: true, parameter_count: 12 },
  { id: 'lab/beta', display_name: 'Beta', architectures: [], tokenizer_available: false },
];
export const sessionA: Session = { id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', model_id: models[0]!.id };
export const sessionB: Session = { id: 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb', model_id: models[1]!.id };
export const tensors: TensorDescriptor[] = [
  { id: 'first', name: 'left.weight', path: ['left', 'weight'], shape: [2, 3], rank: 2, numel: 6, storage_dtype: 'bfloat16', storage_format: 'safetensors', logical_dtype: 'float32' },
  { id: 'second', name: 'right.weight', path: ['right', 'weight'], shape: [3], rank: 1, numel: 3, storage_dtype: 'float32', logical_dtype: 'float32' },
  { id: 'third', name: 'cube', path: ['other', 'cube'], shape: [1, 2, 3], rank: 3, numel: 6, storage_dtype: 'int8', logical_dtype: 'float32' },
  { id: 'scalar', name: 'scalar', path: ['scalar'], shape: [], rank: 0, numel: 1, storage_dtype: 'float32', logical_dtype: 'float32' },
];
export function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
export function memoryStorage() {
  const entries = new Map<string, string>();
  return {
    getItem: (key: string) => entries.get(key) ?? null,
    setItem: (key: string, value: string) => { entries.set(key, value); },
    removeItem: (key: string) => { entries.delete(key); },
  };
}
export const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
