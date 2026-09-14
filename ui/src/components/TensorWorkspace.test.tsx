import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it, vi } from 'vitest';
import { TensorWorkspace } from './TensorWorkspace';

function Workspace() {
  return <TensorWorkspace enabled inventory={<p>Tree</p>}>{(restore) => <section>{restore}<p>Science</p></section>}</TensorWorkspace>;
}

it('restores visibility after remount and retains a mounted scientific subtree', async () => {
  const view = render(<Workspace />);
  const science = screen.getByText('Science');
  await userEvent.click(screen.getByRole('button', { name: 'Hide inventory' }));
  expect(screen.getByRole('button', { name: 'Show inventory' })).toHaveFocus();
  expect(screen.getByText('Science')).toBe(science);
  view.unmount();
  render(<Workspace />);
  expect(screen.getByRole('button', { name: 'Show inventory' })).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Show inventory' }));
  expect(screen.getByRole('button', { name: 'Hide inventory' })).toHaveFocus();
});

it.each(['malformed', 'unavailable'])('keeps hide/restore usable with %s browser preferences', async (mode) => {
  if (mode === 'malformed') localStorage.setItem('lmex.tensor-inventory.pane', '{invalid');
  else {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('disabled'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('disabled'); });
  }
  render(<Workspace />);
  await userEvent.click(screen.getByRole('button', { name: 'Hide inventory' }));
  await userEvent.click(screen.getByRole('button', { name: 'Show inventory' }));
  expect(screen.getByText('Tree')).toBeVisible();
});
