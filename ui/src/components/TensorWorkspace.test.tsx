import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it, vi } from 'vitest';
import { TensorWorkspace } from './TensorWorkspace';
import { readInventoryPreference, writeInventoryPreference } from './inventory-preferences';

function Workspace() {
  return <TensorWorkspace enabled inventory={<p>Tree</p>}><section><p>Science</p></section></TensorWorkspace>;
}

it('restores visibility after remount and retains a mounted scientific subtree', async () => {
  const view = render(<Workspace />);
  const science = screen.getByText('Science');
  const tree = screen.getByText('Tree');
  expect(screen.getByRole('heading', { name: 'Inventory' })).toBeVisible();
  await userEvent.click(screen.getByRole('button', { name: 'Collapse inventory' }));
  expect(screen.getByRole('button', { name: 'Expand inventory' })).toHaveFocus();
  expect(screen.getByText('Science')).toBe(science);
  expect(screen.getByText('Tree')).toBe(tree);
  expect(tree).not.toBeVisible();
  const restore = screen.getByRole('button', { name: 'Expand inventory' });
  expect(restore).toHaveTextContent('');
  expect(restore.querySelector('svg')).toHaveAttribute('aria-hidden', 'true');
  expect(restore.closest('section')).toBeNull();
  expect(restore).toHaveAccessibleDescription('Expand inventory');
  view.unmount();
  render(<Workspace />);
  expect(screen.getByRole('button', { name: 'Expand inventory' })).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Expand inventory' }));
  expect(screen.getByRole('button', { name: 'Collapse inventory' })).toHaveFocus();
});

it.each(['malformed', 'unavailable'])('keeps hide/restore usable with %s browser preferences', async (mode) => {
  if (mode === 'malformed') localStorage.setItem('lmex.tensor-inventory.pane', '{invalid');
  else {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('disabled'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('disabled'); });
  }
  render(<Workspace />);
  expect(screen.getByRole('heading', { name: 'Inventory' })).toBeVisible();
  await userEvent.click(screen.getByRole('button', { name: 'Collapse inventory' }));
  await userEvent.click(screen.getByRole('button', { name: 'Expand inventory' }));
  expect(screen.getByText('Tree')).toBeVisible();
});
it('keeps a user change across remounts when browser storage denies access', async () => {
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('disabled'); });
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('disabled'); });
  const view = render(<Workspace />);
  await userEvent.click(screen.getByRole('button', { name: 'Collapse inventory' }));
  view.unmount();
  render(<Workspace />);
  expect(screen.getByRole('button', { name: 'Expand inventory' })).toBeVisible();
});
it.each(['missing', 'stale'])('keeps a failed inventory write ahead of %s readable storage', async (mode) => {
  const key = 'lmex.tensor-inventory.pane';
  writeInventoryPreference(key, { visible: true, width: 344 });
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('read-only'); });
  if (mode === 'missing') vi.spyOn(Storage.prototype, 'getItem').mockReturnValue(null);
  const view = render(<Workspace />);
  expect(screen.getByRole('heading', { name: 'Inventory' })).toBeVisible();
  await userEvent.click(screen.getByRole('button', { name: 'Collapse inventory' }));
  view.unmount();
  expect(readInventoryPreference(key)).toEqual({ visible: false, width: mode === 'stale' ? 344 : 280 });
  render(<Workspace />);
  expect(screen.getByRole('button', { name: 'Expand inventory' })).toBeVisible();
});
