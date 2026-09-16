import { describe, expect, it } from 'vitest';
import { makeProjectionFixture } from '../../tests/architecture-projection-fixture';
import { projectGraph } from './projection';
import { cardDoubleClick, cardNavigation, cardSelection } from './card-actions';

describe('card actions preserve presentation and concrete target identity', () => {
  const graph = makeProjectionFixture({ count: 4 });
  const overview = projectGraph(graph, { expanded: ['model'] });
  const detail = projectGraph(graph, { expanded: ['model', 'layer-3', 'layer-3.attention'] });
  const expanded = projectGraph(graph, { expanded: ['model', 'layer-3', 'mlp:layer-3.gate'] });

  it('selects a repetition presentation without choosing an arbitrary instance', () => {
    const range = overview.nodes.find((node) => node.repetitionId)!;
    expect(cardSelection(range)).toBe(range.id);
    expect(graph.nodes.some((node) => node.id === cardSelection(range))).toBe(false);
    expect(cardDoubleClick(range)).toBe('expand');
    expect(cardNavigation(range, false, false)).toMatchObject({ action: 'Explore component', target: undefined, reason: expect.any(String) });
  });

  it('expands only collapsed groups and inspects expanded groups and operations', () => {
    const mlp = detail.nodes.find((node) => node.id === 'mlp:layer-3.gate')!;
    expect(cardDoubleClick(mlp)).toBe('expand');
    expect(cardDoubleClick(expanded.nodes.find((node) => node.id === mlp.id)!)).toBe('inspect');
    expect(cardDoubleClick(detail.nodes.find((node) => node.id === 'layer-3.attention.Q')!)).toBe('inspect');
    expect(cardDoubleClick(detail.nodes.find((node) => node.id === 'layer-3')!)).toBe('inspect');
  });

  it('resolves source operations and supported derived groups independently of selection', () => {
    for (const id of ['layer-3', 'layer-3.attention.Q', 'mlp:layer-3.gate']) {
      const node = detail.nodes.find((node) => node.id === id)!;
      expect(cardSelection(node)).toBe(id);
      expect(cardNavigation(node, false, false)).toEqual({ action: 'Explore component', target: id, reason: undefined });
      expect(cardNavigation(node, true, false)).toEqual({ action: 'View in model', target: id, reason: undefined });
    }
  });

  it('uses a rebound shared record instead of its anchor, and disables structure-only navigation', () => {
    const node = detail.nodes.find((node) => node.id === 'layer-3.attention.Q')!;
    const rebound = { ...node, id: 'layer-0.attention.Q' };
    expect(cardSelection(rebound)).toBe('layer-3.attention.Q');
    expect(cardNavigation(rebound, true, false).target).toBe('layer-3.attention.Q');
    expect(cardNavigation(rebound, true, true)).toMatchObject({ target: undefined, reason: 'Choose a concrete instance to view in model.' });
  });

  it('does not navigate external aliases or source context', () => {
    const isolated = projectGraph(graph, { expanded: ['layer-3.attention'], scope: 'layer-3.attention' });
    const external = isolated.nodes.find((node) => node.presentation === 'external')!;
    expect(cardNavigation(external, true, false).target).toBeUndefined();
    const context = overview.nodes.find((node) => node.kind === 'context')!;
    expect(cardNavigation(context, false, false).target).toBeUndefined();
  });
});
