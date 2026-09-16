import { act, fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import type { MatrixViewportOptions } from '../rendering/matrix-viewport';
import type { Inspection } from '../rendering/matrix-inspection';
import { MatrixExplorer } from './MatrixExplorer';
import { PanelHeader } from './PanelHeader';
import { ViewerPanel } from './ViewerPanel';
import { TensorHeader } from '../components/TensorHeader';
import { tensors } from '../test/shell-fixtures';
import type { MatrixSource, MatrixUpdates } from './types';

const fake = vi.hoisted(() => ({ fail: false, transferFails: false, views: [] as {
  options: MatrixViewportOptions; upload: ReturnType<typeof vi.fn>; transfer: ReturnType<typeof vi.fn>;
  fitWidth: ReturnType<typeof vi.fn>; revealRow: ReturnType<typeof vi.fn>; dispose: ReturnType<typeof vi.fn>; refresh: ReturnType<typeof vi.fn>;
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
    fake.views.push({ options, upload, transfer, fitWidth: this.fitWidth, revealRow: this.revealRow, dispose: this.dispose, refresh: this.refresh });
  }
  fitWidth = vi.fn();
  revealRow = vi.fn();
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
it('uploads immediately, coalesces presentation, flushes atomically and cancels stale frames', () => {
  const frames = new Map<number, FrameRequestCallback>();
  let sequence = 0;
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { frames.set(++sequence, callback); return sequence; });
  vi.stubGlobal('cancelAnimationFrame', (id: number) => { frames.delete(id); });
  const a = source(), b = source();
  const view = render(<MatrixExplorer source={a} />);
  const updates = a.updates[0]!, renderer = fake.views[0]!;
  const chunks = [1, 2, 3].map(value => new Float32Array([value]));
  updates.values(chunks[0]!, 0);
  expect(renderer.refresh).toHaveBeenCalledOnce();
  updates.values(chunks[1]!, 1); updates.values(chunks[2]!, 2);
  expect(renderer.upload).toHaveBeenCalledTimes(3);
  chunks.forEach((chunk, offset) => {
    expect(renderer.upload.mock.calls[offset]![0]).toBe(chunk);
    expect(renderer.upload.mock.calls[offset]![1]).toBe(offset);
  });
  expect(renderer.refresh).toHaveBeenCalledOnce();
  expect(frames.size).toBe(1);
  const frame = [...frames.values()][0]!; frames.clear(); frame(0);
  expect(renderer.refresh).toHaveBeenCalledTimes(2);
  updates.values(new Float32Array([4]), 3);
  const cancelled = [...frames.values()][0]!;
  updates.flush!(); updates.flush!(); cancelled(0);
  expect(frames.size).toBe(0);
  expect(renderer.refresh).toHaveBeenCalledTimes(3);
  updates.values(new Float32Array([5]), 4);
  const obsolete = [...frames.values()][0]!;
  view.rerender(<MatrixExplorer source={b} />);
  expect(frames.size).toBe(0);
  obsolete(0); updates.values(new Float32Array([6]), 5); updates.flush!();
  expect(renderer.refresh).toHaveBeenCalledTimes(3);
  expect(renderer.upload).toHaveBeenCalledTimes(5);
  expect(renderer.dispose).toHaveBeenCalledOnce();
  view.unmount();
});
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
  const inspection: Inspection = { row: 1, column: 2, value: '5', left: 0, top: 0, width: 170, magnifier: true, draw: vi.fn() };
  act(() => { fake.views[0]!.options.onInspection!(inspection); fake.views[0]!.options.onInspection!({ ...inspection, value: '6' }); });
  expect(first).not.toHaveBeenCalled(); expect(next).toHaveBeenCalledExactlyOnceWith({ row: 1, column: 2 });
  expect(row).toHaveBeenCalledExactlyOnceWith(1); expect(column).toHaveBeenCalledExactlyOnceWith(2);
  act(() => fake.views[0]!.options.onInspection!({ ...inspection, magnifier: false }));
  expect(screen.queryByLabelText('9 by 9 matrix neighborhood')).not.toBeInTheDocument();
  expect(screen.getByRole('status')).toHaveTextContent('row 1 · column 2');
  expect(screen.getByRole('status')).toHaveTextContent('5');
  expect(next).toHaveBeenCalledTimes(1);
  act(() => fake.views[0]!.options.onInspection!(inspection));
  expect(screen.getByLabelText('9 by 9 matrix neighborhood')).toBeInTheDocument();
  expect(next).toHaveBeenCalledTimes(1);
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
  const header: NonNullable<import('./types').MatrixExplorerProps['header']> = (controls, domain) =>
    <TensorHeader tensor={tensors[0]!} domain={domain} actions={controls} />;
  const view = render(<MatrixExplorer source={a} header={header} />);
  fireEvent.click(screen.getByRole('button', { name: 'Tensor information' }));
  act(() => a.updates[0]!.distributionDomain({ minimum: -2, maximum: 6 }));
  expect(screen.getByTitle('True finite minimum: -2')).toBeVisible();
  expect(a.updates).toHaveLength(1);
  expect(fake.views[0]!.upload).not.toHaveBeenCalled();
  expect(fake.views[0]!.fitWidth).not.toHaveBeenCalled();
  view.rerender(<MatrixExplorer source={b} header={header} />);
  view.rerender(<MatrixExplorer source={a} header={header} />);
  act(() => a.updates[0]!.distributionDomain({ minimum: -99, maximum: 99 }));
  expect(screen.getByRole('dialog')).toHaveTextContent('Bin domain unavailable');
  expect(screen.queryByTitle('True finite minimum: -2')).not.toBeInTheDocument();
  act(() => a.updates[1]!.distributionDomain({ minimum: null, maximum: null }));
  expect(screen.getByRole('dialog')).toHaveTextContent('No finite values');
  expect(screen.queryByRole('region', { name: 'Distribution range' })).not.toBeInTheDocument();
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

it('keeps header, actions and scientific body in one card without restarting delivery', () => {
  const data = source();
  const view = render(<MatrixExplorer source={data} header={(controls) => <PanelHeader identity="Result" actions={controls} />} />);
  const card = view.container.querySelector('.viewer-panel')!;
  const title = card.querySelector('.matrix-explorer-header')!;
  const body = card.querySelector('.viewer-panel-body')!;
  const matrix = body.querySelector('.matrix-scroll');
  expect([...card.children]).toEqual([title, body]);
  expect(title).toContainElement(screen.getByText('Result'));
  expect(title).toContainElement(screen.getByRole('button', { name: 'Fit width' }));
  expect(matrix).toBeInTheDocument();
  act(() => data.updates[0]!.values(new Float32Array([1, 2, 3]), 0));
  view.rerender(<MatrixExplorer source={data} header={(controls) => <PanelHeader identity="Updated result"
    status="Streaming" actions={<><button>Cancel</button>{controls}</>} />} />);
  expect(view.container.querySelector('.viewer-panel')).toBe(card);
  expect([...card.children]).toEqual([title, body]);
  expect(title).toContainElement(screen.getByText('Updated result'));
  expect(title).toContainElement(screen.getByText('Streaming'));
  expect(title).toContainElement(screen.getByRole('button', { name: 'Cancel' }));
  expect(body.querySelector('.matrix-scroll')).toBe(matrix);
  fireEvent.click(screen.getByRole('button', { name: 'Fit width' }));
  expect(fake.views[0]!.fitWidth).toHaveBeenCalledOnce();
  expect(fake.views).toHaveLength(1);
  expect(fake.views[0]!.upload).toHaveBeenCalledOnce();
  expect(fake.views[0]!.dispose).not.toHaveBeenCalled();
  expect(data.updates).toHaveLength(1);
  expect(data.detach).not.toHaveBeenCalled();
});

it.each(['Waiting', 'Empty', 'Unavailable'])('keeps %s content inside the same card as its header without a renderer', (state) => {
  const view = render(<ViewerPanel header={<PanelHeader identity="Result" />}><p>{state}</p></ViewerPanel>);
  const card = view.container.querySelector('.viewer-panel')!;
  const title = card.querySelector('.matrix-explorer-header')!;
  const body = card.querySelector('.viewer-panel-body')!;
  expect([...card.children]).toEqual([title, body]);
  expect(title).toContainElement(screen.getByText('Result'));
  expect(body).toContainElement(screen.getByText(state));
  expect(fake.views).toHaveLength(0);
});

it('reveals only on a new semantic intent without restarting delivery or resetting the camera', () => {
  const data = source();
  const intent = { row: 1 };
  const view = render(<MatrixExplorer source={data} revealRow={intent} />);
  view.rerender(<MatrixExplorer source={data} revealRow={intent} highlightedRow={0} />);
  expect(fake.views[0]!.revealRow).toHaveBeenCalledExactlyOnceWith(1);
  view.rerender(<MatrixExplorer source={data} revealRow={{ row: 1 }} />);
  expect(fake.views[0]!.revealRow).toHaveBeenCalledTimes(2);
  expect(fake.views[0]!.fitWidth).not.toHaveBeenCalled();
  expect(data.updates).toHaveLength(1);
  expect(fake.views[0]!.upload).not.toHaveBeenCalled();
  view.rerender(<MatrixExplorer source={source()} />);
  expect(fake.views[1]!.revealRow).not.toHaveBeenCalled();
});
