import { act, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import type { MatrixViewportOptions } from '../rendering/matrix-viewport';
import type { Inspection } from '../rendering/matrix-inspection';
import { MatrixExplorer } from './MatrixExplorer';
import type { MatrixSource, MatrixUpdates } from './types';

const fake = vi.hoisted(() => ({ fail: false, transferFails: false, views: [] as {
  options: MatrixViewportOptions; upload: ReturnType<typeof vi.fn>; transfer: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
}[] }));
vi.mock('../rendering/matrix-viewport', () => ({ MatrixViewport: class {
  matrix;
  constructor(host: HTMLElement, _descriptor: unknown, readonly options: MatrixViewportOptions) {
    if (fake.fail) throw new Error('No GPU');
    const region = document.createElement('div');
    region.className = 'matrix-scroll'; host.append(region);
    const upload = vi.fn();
    const transfer = vi.fn(() => { if (fake.transferFails) throw new Error('Invalid transfer'); });
    this.matrix = { renderer: { upload, setTransfer: transfer } };
    this.dispose = vi.fn(() => { options.onInspection?.(null); host.replaceChildren(); });
    fake.views.push({ options, upload, transfer, dispose: this.dispose });
  }
  refresh = vi.fn();
  setLinkedRow = vi.fn();
  setDistributionDomain = vi.fn();
  dispose;
} }));
beforeEach(() => { fake.fail = false; fake.transferFails = false; fake.views.length = 0; });
function source(): MatrixSource & { updates: MatrixUpdates[]; detach: ReturnType<typeof vi.fn> } {
  const updates: MatrixUpdates[] = [];
  const detach = vi.fn();
  return { descriptor: { shape: [2, 3], rank: 2, numel: 6, logical_dtype: 'float32' }, updates, detach,
    subscribe(sink) { updates.push(sink); return detach; } };
}
it('fences retained subscriptions and disposes their allocation even when unsubscribe throws', () => {
  const a = source(), b = source();
  const view = render(<MatrixExplorer source={a} />);
  view.rerender(<MatrixExplorer source={b} />);
  view.rerender(<MatrixExplorer source={a} />);
  for (const old of [a.updates[0]!, b.updates[0]!]) {
    old.values(new Float32Array([99]), 0);
    old.transfer({ slope: 2 });
    old.distribution('rows', new Uint32Array([99]), 0);
    old.distributionDomain({ minimum: -99, maximum: 99 });
  }
  expect(fake.views.every((v) => v.upload.mock.calls.length === 0 && v.transfer.mock.calls.length === 0)).toBe(true);
  expect(fake.views.slice(0, 2).every((v) => v.dispose.mock.calls.length === 1)).toBe(true);
  const values = new Float32Array([1, 2, 3]);
  a.updates[1]!.values(values, 0);
  expect(fake.views[2]!.upload.mock.calls[0]![0]).toBe(values);
  a.detach.mockImplementation(() => { throw new Error('Unsubscribe failed'); });
  expect(() => view.unmount()).toThrow('Unsubscribe failed');
  expect(fake.views[2]!.dispose).toHaveBeenCalledOnce();
  expect(() => a.updates[1]!.values(values, 0)).not.toThrow();
});
it('updates context and callbacks without restarting a source, and emits exact indices once per selection change', () => {
  const data = source();
  const first = vi.fn(), next = vi.fn(), row = vi.fn(), column = vi.fn();
  const view = render(<MatrixExplorer source={data} header="First context" onCellSelect={first} />);
  view.rerender(<MatrixExplorer source={data} header="Next context" label="Result" onCellSelect={next} onRowSelect={row} onColumnSelect={column} />);
  expect(screen.getByText('Next context')).toBeVisible();
  expect(data.updates).toHaveLength(1);
  const inspection: Inspection = { row: 1, column: 2, value: '5', left: 0, top: 0, draw: vi.fn() };
  act(() => { fake.views[0]!.options.onInspection!(inspection); fake.views[0]!.options.onInspection!({ ...inspection, value: '6' }); });
  expect(first).not.toHaveBeenCalled(); expect(next).toHaveBeenCalledExactlyOnceWith({ row: 1, column: 2 });
  expect(row).toHaveBeenCalledExactlyOnceWith(1); expect(column).toHaveBeenCalledExactlyOnceWith(2);
  act(() => fake.views[0]!.options.onInspection!(null));
  expect(next).toHaveBeenLastCalledWith(null); expect(row).toHaveBeenLastCalledWith(null); expect(column).toHaveBeenLastCalledWith(null);
});
it('reports allocation failure without subscribing and recovers on source replacement', () => {
  fake.fail = true;
  const a = source(), b = source();
  const view = render(<MatrixExplorer source={a} />);
  expect(screen.getByRole('alert')).toHaveTextContent('Exact rendering is unavailable');
  expect(a.updates).toHaveLength(0);
  fake.fail = false;
  view.rerender(<MatrixExplorer source={b} />);
  act(() => fake.views[0]!.options.onStateChange!('ready'));
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  expect(b.updates).toHaveLength(1);
});
it('resets domain labels across source reuse and ignores detached metadata', () => {
  const a = { ...source(), distributions: true }, b = { ...source(), distributions: true };
  const view = render(<MatrixExplorer source={a} />);
  act(() => a.updates[0]!.distributionDomain({ minimum: -2, maximum: 6 }));
  expect(screen.getByTitle('True finite minimum: -2')).toBeVisible();
  view.rerender(<MatrixExplorer source={b} />);
  view.rerender(<MatrixExplorer source={a} />);
  act(() => a.updates[0]!.distributionDomain({ minimum: -99, maximum: 99 }));
  expect(screen.getByLabelText('Distribution range')).toHaveTextContent('Bin domain unavailable');
  act(() => a.updates[1]!.distributionDomain({ minimum: null, maximum: null }));
  expect(screen.getByLabelText('Distribution range')).toHaveTextContent('No finite values');
});
it('releases an allocated viewport if initial transfer setup fails', () => {
  fake.transferFails = true;
  const data = source();
  render(<MatrixExplorer source={{ ...data, transfer: { slope: -1 } }} />);
  expect(fake.views[0]!.dispose).toHaveBeenCalledOnce();
  expect(data.updates).toHaveLength(0);
  expect(screen.getByRole('alert')).toBeVisible();
});
it('releases the viewport and fences callbacks when a parent subscription throws', () => {
  const data = source();
  let sink: MatrixUpdates | undefined;
  expect(() => render(<MatrixExplorer source={{ ...data, subscribe(updates) {
    sink = updates; throw new Error('Parent source failed');
  } }} />)).toThrow('Parent source failed');
  expect(fake.views.every((v) => v.dispose.mock.calls.length === 1)).toBe(true);
  sink!.values(new Float32Array([1]), 0);
  expect(fake.views.every((v) => v.upload.mock.calls.length === 0)).toBe(true);
});
