import { describe, expect, it } from 'vitest';
import { ModelDiagnostics, finding, uniqueFindings } from './model-diagnostics';
const source = { code: 'unknown_region', message: 'Exact safe finding.', node_id: 'a' };
const warning = finding('model', 'graph-1', 'Architecture', source, 'warning');
describe('current model diagnostic identity', () => {
  it('deduplicates response/graph findings but retains distinct locations and severity', () => {
    expect(uniqueFindings([warning, { ...warning }, finding('model', 'graph-1', 'Architecture', { ...source, node_id: 'b' }, 'warning'),
      finding('model', 'graph-1', 'Architecture', source, 'error')])).toHaveLength(3);
  });
  it('retains dismissals across observations, exposes new findings and invalidates graph replacements', () => {
    const store = new ModelDiagnostics(), session = {};
    store.activate(session); store.graph(session, 'model', 'graph-1');
    store.observe(session, 'Architecture', [warning]); store.dismiss([warning.id]);
    store.observe(session, 'Architecture', [warning]);
    expect(store.getSnapshot().dismissed).toContain(warning.id);
    const escalated = finding('model', 'graph-1', 'Architecture', source, 'error');
    store.observe(session, 'Architecture', [escalated]);
    expect(store.getSnapshot().dismissed).not.toContain(escalated.id);
    store.graph(session, 'model', 'graph-2');
    expect(store.getSnapshot().dismissed).toEqual([]);
  });
  it('clears current observations and rejects late callbacks, including a same-ID session replacement', () => {
    const store = new ModelDiagnostics(), old = { id: 'same' }, next = { id: 'same' };
    store.activate(old); store.observe(old, 'Architecture', [warning], true);
    store.activate(next); store.observe(old, 'Architecture', [warning], true);
    expect(store.getSnapshot().records).toEqual([]); expect(store.getSnapshot().modelSupplied).toBe(false);
    store.observe(next, 'Architecture', [warning]); store.activate(null);
    expect(store.getSnapshot().records).toEqual([]);
    expect(new ModelDiagnostics().getSnapshot().records).toEqual([]);
  });
  it('retains same-generation dismissals after returning to a model, without retaining its observations', () => {
    const store = new ModelDiagnostics(), first = {}, second = {};
    store.activate(first); store.graph(first, 'model', 'graph-1'); store.observe(first, 'Architecture', [warning]); store.dismiss([warning.id]);
    store.activate(second); expect(store.getSnapshot().records).toEqual([]);
    store.graph(second, 'model', 'graph-1'); store.observe(second, 'Architecture', [warning]);
    expect(store.getSnapshot().dismissed).toContain(warning.id);
  });
});
