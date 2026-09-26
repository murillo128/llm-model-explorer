import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ToastHost } from './ToastHost';

afterEach(() => vi.useRealTimers());
it('keeps actionable errors, pauses informational expiry for hover/focus, and cleans replacement timers', () => {
  vi.useFakeTimers();
  const dismiss = vi.fn();
  const toasts = [{ id: 1, message: 'Session closed.', severity: 'info' as const },
    { id: 2, message: 'Could not close session.', severity: 'error' as const }];
  const { rerender } = render(<ToastHost toasts={toasts} onDismiss={dismiss} />);
  const info = screen.getByText('Session closed.').closest('li')!;
  fireEvent.mouseEnter(info);
  act(() => vi.advanceTimersByTime(10000));
  expect(dismiss).not.toHaveBeenCalled();
  fireEvent.focus(info.querySelector('button')!);
  fireEvent.mouseLeave(info);
  act(() => vi.advanceTimersByTime(10000));
  expect(dismiss).not.toHaveBeenCalled();
  fireEvent.blur(info.querySelector('button')!);
  act(() => vi.advanceTimersByTime(5000));
  expect(dismiss).toHaveBeenCalledExactlyOnceWith(1);
  dismiss.mockClear();
  rerender(<ToastHost toasts={[{ ...toasts[0]!, id: 3 }]} onDismiss={dismiss} />);
  rerender(<ToastHost toasts={[]} onDismiss={dismiss} />);
  act(() => vi.advanceTimersByTime(10000));
  expect(dismiss).not.toHaveBeenCalled();
});
