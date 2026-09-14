import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it } from 'vitest';
import { tensors } from '../test/shell-fixtures';
import { TensorHeader } from './TensorHeader';

it('previews metadata on hover/focus, pins on activation, and dismisses without reopening on restored focus', async () => {
  const user = userEvent.setup();
  render(<><TensorHeader tensor={tensors[0]!} /><button>Outside</button></>);
  expect(screen.getAllByText('left › weight')).toHaveLength(1);
  expect(screen.getByText('[2 × 3] · bfloat16')).toBeVisible();
  const trigger = screen.getByRole('button', { name: 'Tensor information' });
  await user.hover(trigger);
  expect(screen.getByRole('dialog')).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Close tensor information' })).not.toBeInTheDocument();
  await user.unhover(trigger);
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  await user.tab();
  expect(trigger).toHaveFocus();
  expect(screen.getByRole('dialog')).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Close tensor information' })).not.toBeInTheDocument();
  await user.keyboard('{Enter}');
  const dialog = screen.getByRole('dialog', { name: 'Tensor information' });
  expect(within(dialog).getByRole('button')).toHaveFocus();
  for (const text of ['Logical path', 'Rank', 'Elements', 'Storage dtype', 'Storage format', 'Logical dtype', 'safetensors', 'float32']) {
    expect(within(dialog).getByText(text)).toBeVisible();
  }
  expect(dialog.querySelectorAll('dt')).toHaveLength(6);
  expect(screen.queryByText(/One value per device pixel|Focus the matrix/)).not.toBeInTheDocument();
  await user.keyboard('{Escape}');
  expect(trigger).toHaveFocus();
  expect(trigger).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  await user.keyboard(' ');
  await user.tab();
  expect(screen.getByRole('button', { name: 'Outside' })).toHaveFocus();
  expect(screen.getByRole('dialog')).toBeVisible();
  await user.keyboard('{Escape}');
  expect(trigger).toHaveFocus();
  await user.click(trigger);
  await user.unhover(trigger);
  expect(screen.getByRole('dialog')).toBeVisible();
  await user.click(screen.getByRole('button', { name: 'Close tensor information' }));
  expect(trigger).toHaveFocus();
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  await user.click(trigger);
  await user.click(screen.getByRole('button', { name: 'Outside' }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

it('closes an ephemeral focus preview when keyboard focus leaves', async () => {
  render(<><TensorHeader tensor={tensors[0]!} /><button>Outside</button></>);
  await userEvent.tab();
  expect(screen.getByRole('dialog')).toBeVisible();
  await userEvent.tab();
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});
