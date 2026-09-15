import { startTransition, StrictMode, Suspense, useLayoutEffect, useState } from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { useCanvasCallback } from './useCanvasCallback';

it('keeps one event identity while reading the latest committed instance and arguments', () => {
  const inspect = vi.fn(), publish = vi.fn();
  function View({ instance }: { instance: string }) {
    const callback = useCanvasCallback((port: string) => inspect(instance, port));
    useLayoutEffect(() => { publish(callback); }, [callback]);
    return null;
  }
  const root = render(<View instance="first" />);
  const callback = publish.mock.calls[0]![0] as (port: string) => void;
  callback('input');
  root.rerender(<View instance="later" />);
  callback('output');
  expect(publish).toHaveBeenCalledTimes(1);
  expect(inspect.mock.calls).toEqual([['first', 'input'], ['later', 'output']]);
});

it('does not publish the context of a suspended, uncommitted navigation', async () => {
  const inspect = vi.fn(), attempted = vi.fn(), publish = vi.fn();
  let ready = false, release!: () => void;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  function View({ instance }: { instance: string }) {
    const callback = useCanvasCallback(() => inspect(instance));
    useLayoutEffect(() => { publish(callback); }, [callback]);
    attempted(instance);
    if (instance === 'later' && !ready) throw pending;
    return <span>{instance}</span>;
  }
  function Navigation() {
    const [instance, setInstance] = useState('first');
    return <><button onClick={() => startTransition(() => setInstance('later'))}>Next</button>
      <Suspense fallback="Waiting"><View instance={instance} /></Suspense></>;
  }
  render(<Navigation />);
  const callback = publish.mock.calls[0]![0] as () => void;
  fireEvent.click(screen.getByRole('button', { name: 'Next' }));
  expect(attempted).toHaveBeenCalledWith('later');
  expect(screen.getByText('first')).toBeVisible();
  callback();
  expect(inspect).toHaveBeenLastCalledWith('first');
  await act(async () => { ready = true; release(); await pending; });
  expect(screen.getByText('later')).toBeVisible();
  callback();
  expect(inspect).toHaveBeenLastCalledWith('later');
  expect(publish).toHaveBeenCalledTimes(1);
});

it('clears disposed handlers and keeps replacement sessions in separate slots under StrictMode', () => {
  const inspect = vi.fn(), publish = vi.fn();
  function View({ session }: { session: string }) {
    const callback = useCanvasCallback(() => inspect(session));
    useLayoutEffect(() => { publish(callback); }, [callback]);
    return null;
  }
  const root = render(<StrictMode><View key="old" session="old" /></StrictMode>);
  const old = publish.mock.lastCall![0] as () => void;
  old();
  root.rerender(<StrictMode><View key="new" session="new" /></StrictMode>);
  const current = publish.mock.lastCall![0] as () => void;
  expect(current).not.toBe(old);
  old(); current();
  expect(inspect.mock.calls).toEqual([['old'], ['new']]);
  root.unmount();
  old(); current();
  expect(inspect.mock.calls).toEqual([['old'], ['new']]);
});
