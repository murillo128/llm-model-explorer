import { createRef } from 'react';
import type { ComponentProps } from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { GraphAction } from './GraphAction';
import { ArchitectureControls } from './ArchitectureControls';
import { contractResponse } from '../../tests/architecture-fixtures';

function controls(patch: Partial<ComponentProps<typeof ArchitectureControls>> = {}) {
  const command = () => vi.fn();
  const props: ComponentProps<typeof ArchitectureControls> = {
    graph: contractResponse.graph, focus: null, stack: undefined, shared: undefined, family: undefined,
    returnContext: undefined, instanceId: undefined, visibleInstances: [], options: { expanded: [] }, breadcrumbs: [],
    selection: undefined, isolation: undefined, picker: createRef(), navigate: command(), overview: command(),
    expandAll: command(), collapseAll: command(), chooseInstance: command(), exploreStack: command(), windowSize: 2,
    focusLayer: undefined, focusMlp: undefined, stateFocus: undefined, toggleSelected: undefined,
    preferences: command(), derivedMlpAvailable: false, filtered: false, ...patch,
  };
  render(<ArchitectureControls {...props} />);
  return props;
}
it('invokes each relocated global and contextual command once without coupling collapse to another action', () => {
  const focusLayer = vi.fn(), focusMlp = vi.fn(), stateFocus = vi.fn(), toggleSelected = vi.fn();
  const props = controls({ focusLayer, focusMlp, stateFocus, toggleSelected });
  for (const [label, command] of [['Collapse all', props.collapseAll], ['Show all operations', props.expandAll],
    ['Focus layer', focusLayer], ['Focus MLP', focusMlp], ['State dependencies', stateFocus], ['Toggle selected group', toggleSelected]] as const) {
    fireEvent.click(screen.getByRole('button', { name: label }));
    expect(command).toHaveBeenCalledTimes(1);
  }
  expect(props.overview).not.toHaveBeenCalled();
  expect(props.preferences).not.toHaveBeenCalled();
});
it('keeps isolated global expansion, Back and component commands outside preferences', () => {
  const isolation = { back: vi.fn(), expand: vi.fn(), viewInModel: undefined };
  const props = controls({ isolation });
  fireEvent.click(screen.getByRole('button', { name: 'Show all operations in model' }));
  fireEvent.click(screen.getByRole('button', { name: 'Back' }));
  fireEvent.click(screen.getByRole('button', { name: 'Expand component' }));
  expect(props.expandAll).toHaveBeenCalledTimes(1);
  expect(isolation.back).toHaveBeenCalledTimes(1); expect(isolation.expand).toHaveBeenCalledTimes(1);
  expect(screen.getByRole('button', { name: 'View in model' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'View options' }));
  const popup = screen.getByRole('dialog', { name: 'View options' });
  expect(within(popup).queryAllByRole('button')).toHaveLength(0);
  expect(within(popup).getAllByRole('checkbox')).toHaveLength(3);
  expect(props.preferences).not.toHaveBeenCalled();
});

it.each([['plus', 'Zoom in'], ['minus', 'Zoom out'], ['fit', 'Fit view']] as const)('camera action %s invokes only its command, once', (icon, label) => {
  const command = vi.fn();
  render(<GraphAction icon={icon} label={label} onClick={command} iconOnly />);
  const button = screen.getByRole('button', { name: label });
  fireEvent.focus(button); fireEvent.mouseEnter(button);
  expect(screen.getByRole('tooltip')).toHaveTextContent(label);
  expect(command).not.toHaveBeenCalled();
  fireEvent.click(button);
  expect(command).toHaveBeenCalledTimes(1);
});
