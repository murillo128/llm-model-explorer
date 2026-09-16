import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it, vi } from 'vitest';
import { ArchitectureWorkspace } from './ArchitectureWorkspace';

function Workspace() { return <ArchitectureWorkspace browser={<input aria-label="Retained browser" defaultValue="query" />}><section>Canvas</section></ArchitectureWorkspace>; }
it('retains both subtrees and focus through collapse, and saves only pane preferences', async () => {
  const instance = render(<Workspace />), canvas = screen.getByText('Canvas'), input = screen.getByLabelText('Retained browser');
  await userEvent.click(screen.getByRole('button', { name: 'Collapse browser' }));
  expect(screen.getByText('Canvas')).toBe(canvas); expect(screen.getByLabelText('Retained browser')).toBe(input);
  expect(input).not.toBeVisible(); expect(screen.getByRole('button', { name: 'Expand browser' })).toHaveFocus();
  expect(JSON.parse(localStorage.getItem('lmex.architecture-browser.pane')!)).toEqual({ visible: false, width: 280 });
  instance.unmount(); render(<Workspace />);
  await userEvent.click(screen.getByRole('button', { name: 'Expand browser' }));
  expect(screen.getByRole('button', { name: 'Collapse browser' })).toHaveFocus(); expect(screen.getByLabelText('Retained browser')).toHaveValue('query');
});
it.each(['malformed', 'unavailable'])('works when optional browser storage is %s', async (mode) => {
  if (mode === 'malformed') localStorage.setItem('lmex.architecture-browser.pane', '{invalid');
  else {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('disabled'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('disabled'); });
  }
  render(<Workspace />); await userEvent.click(screen.getByRole('button', { name: 'Collapse browser' }));
  await userEvent.click(screen.getByRole('button', { name: 'Expand browser' })); expect(screen.getByLabelText('Retained browser')).toBeVisible();
});
