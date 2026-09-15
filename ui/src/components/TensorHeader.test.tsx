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
  expect(dialog.querySelectorAll('dt')).toHaveLength(9);
  expect(dialog).toHaveTextContent('Bin domain unavailable');
  expect(within(dialog).getAllByText('Unavailable')).toHaveLength(2);
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

it('keeps authoritative full-range metadata inside the popover, including constant and nonfinite domains', async () => {
  const user = userEvent.setup();
  const view = render(<TensorHeader tensor={tensors[0]!} domain={{ minimum: -12345, maximum: 67890 }} />);
  expect(screen.queryByText('Full-range bins')).not.toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Tensor information' }));
  const dialog = screen.getByRole('dialog');
  expect(within(dialog).getByText('Full-range bins')).toBeVisible();
  expect(within(dialog).getByTitle('True finite minimum: -12345')).toHaveTextContent('-1.2e4');
  expect(within(dialog).getByTitle('True finite maximum: 67890')).toHaveTextContent('6.8e4');
  view.rerender(<TensorHeader tensor={tensors[0]!} domain={{ minimum: 3, maximum: 3 }} />);
  expect(dialog).toHaveTextContent('Constant · samples in bin 50; no value span');
  view.rerender(<TensorHeader tensor={tensors[0]!} domain={{ minimum: null, maximum: null }} />);
  expect(dialog).toHaveTextContent('No finite values · bin domain and true min/max unavailable');
  expect(within(dialog).getAllByText('Unavailable')).toHaveLength(2);
  expect(dialog.querySelector('[title]')).toBeNull();
  view.rerender(<TensorHeader tensor={{ ...tensors[0]!, rank: 1, shape: [6] }} />);
  expect(dialog).not.toHaveTextContent('Distribution domain');
});
