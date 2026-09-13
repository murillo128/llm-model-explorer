import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it } from 'vitest';
import { tensors } from '../test/shell-fixtures';
import { TensorHeader } from './TensorHeader';

it('keeps one contextual identity and shape/dtype visible, with keyboard-accessible secondary metadata and help', async () => {
  render(<><TensorHeader tensor={tensors[0]!} /><button>Outside</button></>);
  expect(screen.getAllByText('left › weight')).toHaveLength(1);
  expect(screen.queryByText('left.weight')).not.toBeInTheDocument();
  expect(screen.getByText('[2 × 3] · bfloat16')).toBeVisible();
  expect(screen.queryByText('Logical path')).not.toBeInTheDocument();
  expect(screen.queryByText(/Focus the matrix/)).not.toBeInTheDocument();
  const trigger = screen.getByRole('button', { name: 'Tensor information and help' });
  trigger.focus();
  await userEvent.keyboard('{Enter}');
  const dialog = screen.getByRole('dialog', { name: 'Tensor information and help' });
  expect(trigger).toHaveAttribute('aria-expanded', 'true');
  expect(within(dialog).getByRole('button')).toHaveFocus();
  for (const text of ['Logical path', 'Rank', 'Elements', 'Storage dtype', 'Storage format', 'Logical dtype', 'safetensors', 'float32']) {
    expect(within(dialog).getByText(text)).toBeVisible();
  }
  expect(within(dialog).getByText(/Focus the matrix/)).toBeVisible();
  await userEvent.keyboard('{Escape}');
  expect(trigger).toHaveFocus();
  expect(trigger).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  await userEvent.keyboard(' ');
  await userEvent.tab();
  expect(screen.getByRole('button', { name: 'Outside' })).toHaveFocus();
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  await userEvent.click(trigger);
  await userEvent.click(screen.getByRole('button', { name: 'Outside' }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});
