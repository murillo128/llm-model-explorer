import { StrictMode, useLayoutEffect } from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import fixture from '../../../api/fixtures/architecture.json';
import { ApiClient } from '../api/client';
import type { StreamOptions } from '../api/client';
import { validateSchema } from '../api/validation';
import { Lifetime } from '../app/lifetime';
import type { MatrixSource, MatrixUpdates } from '../matrix-explorer';
import { ArchitectureInspection } from './ArchitectureInspection';
import { finding } from '../app/model-diagnostics';

const sinks = vi.hoisted(() => [] as MatrixUpdates[]);
vi.mock('../matrix-explorer', () => ({ MatrixExplorer: ({ source }: { source: MatrixSource }) => {
  useLayoutEffect(() => {
    const updates = { values: vi.fn(), transfer: vi.fn(), distribution: vi.fn(), distributionDomain: vi.fn() };
    sinks.push(updates);
    return source.subscribe(updates);
  }, [source]);
  return <div>Scientific surface</div>;
} }));
beforeEach(() => {
  sinks.length = 0;
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; };
});
function setup() {
  const response = validateSchema('ArchitectureResponse', fixture.response);
  if (response.status !== 'available') throw new Error('Fixture');
  const graph = structuredClone(response.graph);
  const node = graph.nodes.find((n) => n.id === 'linear1')!;
  node.parameter_ids = graph.parameters.map((p) => p.id);
  const inventory = validateSchema('TensorInventory', fixture.context.inventory);
  const client = new ApiClient({ backendBaseUrl: 'https://example.test' });
  const handles: { cancel: ReturnType<typeof vi.fn>; options: StreamOptions; resolve: (value: unknown) => void }[] = [];
  for (const method of ['streamTensor', 'streamTensorStatistics', 'streamTensorDistributions'] as const) {
    vi.spyOn(client, method).mockImplementation((_session, _tensor, options = {}) => {
      let resolve!: (value: unknown) => void;
      const done = new Promise((r) => { resolve = r; });
      const cancel = vi.fn().mockResolvedValue(undefined);
      handles.push({ cancel, options, resolve });
      return { done, cancel, operationId: undefined } as ReturnType<ApiClient[typeof method]>;
    });
  }
  const trigger = document.createElement('button'); document.body.append(trigger);
  const session = { id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', model_id: response.model_id };
  const context = { client, session, sessionId: session.id, selection: new Lifetime(), selectedTensor: null, reportStatus: vi.fn() };
  const props = { context, graph, inventory, selected: { modelId: response.model_id, sessionId: session.id, graphId: graph.graph_id, node, trigger }, onClose: vi.fn() };
  return { props, handles, trigger, client };
}
it('replaces only its weight operations, fences late bytes/results, and leaves another shared-work consumer alive', async () => {
  const { props, handles, trigger, client } = setup();
  const other = client.streamTensor(props.context.sessionId, 'tensor_native');
  const view = render(<ArchitectureInspection {...props} />);
  expect(handles).toHaveLength(1);
  fireEvent.change(screen.getByLabelText('Inspect parameter'), { target: { value: 'weight' } });
  expect(handles).toHaveLength(4);
  expect(client.streamTensor).toHaveBeenLastCalledWith(props.context.sessionId, 'tensor_native', expect.anything());
  fireEvent.change(screen.getByLabelText('Inspect parameter'), { target: { value: 'alias' } });
  expect(handles).toHaveLength(7);
  expect(handles.slice(1, 4).every((h) => h.cancel.mock.calls.length === 1)).toBe(true);
  await act(async () => {
    handles[1]!.options.onData!(new Uint8Array(new Float32Array([42]).buffer), 0);
    handles[2]!.resolve({ kind: 'complete', metadata: { kind: 'tensor_statistics' } });
  });
  expect(sinks.every((s) => vi.mocked(s.values).mock.calls.length === 0 && vi.mocked(s.transfer).mock.calls.length === 0)).toBe(true);
  fireEvent.click(screen.getByRole('button', { name: 'Close inspection' }));
  expect(handles.slice(4).every((h) => h.cancel.mock.calls.length === 1)).toBe(true);
  expect(props.context.selection.isCurrent()).toBe(true);
  expect(handles[0]!.cancel).not.toHaveBeenCalled();
  expect(props.onClose).toHaveBeenCalledOnce();
  view.unmount(); expect(trigger).toHaveFocus(); trigger.remove();
  await other.cancel();
});
it.each(['quantized', 'fused', 'unresolved', 'volume'])('keeps %s as metadata without numeric operations', (id) => {
  const { props, handles, trigger } = setup();
  const view = render(<ArchitectureInspection {...props} />);
  fireEvent.change(screen.getByLabelText('Inspect parameter'), { target: { value: id } });
  expect(screen.getByRole('status')).toHaveTextContent('Numeric inspection unavailable.');
  expect(screen.queryByText('Scientific surface')).not.toBeInTheDocument();
  expect(handles).toHaveLength(0);
  view.unmount(); trigger.remove();
});
it('resolves reference-only parameters, displays provenance and text safely, and Escape disposes operations', () => {
  const { props, handles, trigger } = setup();
  props.selected.node.parameter_ids = [];
  props.selected.node.description = '<script>not executable</script>';
  props.selected.node.formula = 'Y = XWᵀ';
  const view = render(<ArchitectureInspection {...props} />);
  expect(screen.getByText('<script>not executable</script>')).toBeInTheDocument();
  expect(screen.getByText('Y = XWᵀ')).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Inspect parameter'), { target: { value: 'weight' } });
  expect(screen.getByText('storage: linear.weight')).toBeInTheDocument();
  fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
  expect(handles.every((h) => h.cancel.mock.calls.length === 1)).toBe(true);
  expect(props.onClose).toHaveBeenCalledOnce();
  view.unmount(); trigger.remove();
});
it('synchronously cancels a disposed session and rejects late callbacks', () => {
  const { props, handles, trigger } = setup();
  const view = render(<ArchitectureInspection {...props} />);
  fireEvent.change(screen.getByLabelText('Inspect parameter'), { target: { value: 'weight' } });
  act(() => props.context.selection.dispose());
  expect(handles.every((h) => h.cancel.mock.calls.length === 1)).toBe(true);
  handles[0]!.options.onData!(new Uint8Array(4), 0);
  expect(sinks[0]!.values).not.toHaveBeenCalled();
  view.unmount(); trigger.remove();
});

it('direct activation preselects the exact own reference and fences a reopened same-ID generation', () => {
  const { props, handles, trigger, client } = setup();
  const selected = { ...props.selected, parameterId: 'weight' };
  const first = render(<ArchitectureInspection {...props} selected={selected} />);
  expect(screen.getByLabelText('Inspect parameter')).toHaveValue('weight');
  expect(client.streamTensor).toHaveBeenCalledExactlyOnceWith(props.context.sessionId, 'tensor_native', expect.anything());
  const previousSink = sinks.at(-1)!;
  first.unmount();
  const second = render(<ArchitectureInspection {...props} selected={selected} />);
  expect(handles).toHaveLength(6);
  expect(handles.slice(0, 3).every((h) => h.cancel.mock.calls.length === 1)).toBe(true);
  const currentSink = sinks.at(-1)!;
  act(() => handles[0]!.options.onData!(new Uint8Array(new Float32Array([99]).buffer), 0));
  expect(previousSink.values).not.toHaveBeenCalled(); expect(currentSink.values).not.toHaveBeenCalled();
  act(() => handles[3]!.options.onData!(new Uint8Array(new Float32Array([-7]).buffer), 0));
  expect(currentSink.values).toHaveBeenCalledWith(new Float32Array([-7]), 0);
  second.unmount(); trigger.remove();
});

it('cannot activate an unrelated parameter or structure-only weight through an initial selection', () => {
  const { props, handles, trigger } = setup();
  props.selected.node.parameter_ids = [];
  props.selected.node.references = [];
  const view = render(<ArchitectureInspection {...props} selected={{ ...props.selected, parameterId: 'weight' }} />);
  expect(handles).toHaveLength(0);
  view.rerender(<ArchitectureInspection {...props} selected={{ ...props.selected, parameterId: 'weight', structureOnly: { label: 'Common', role: 'linear' } }} />);
  expect(handles).toHaveLength(0);
  view.unmount(); trigger.remove();
});

it('shows all normalized findings in real and presentation Model inspection, and only exact component findings elsewhere', () => {
  const { props, trigger } = setup();
  props.graph.scope = 'model_defined'; props.graph.coverage = 'partial';
  const root = props.graph.nodes.find((n) => n.id === 'root')!;
  root.provenance = [{ kind: 'description', source: '<script>source evidence</script>' }];
  const findings = [
    finding(props.selected.modelId, props.graph.graph_id, 'Architecture', { code: 'source_warning', node_id: 'linear0', message: 'Source warning.' }, 'warning'),
    finding(props.selected.modelId, props.graph.graph_id, 'Architecture', { code: 'interface_mapping_ambiguous', node_id: 'linear1', message: 'UI mapping warning.' }, 'warning'),
    finding(props.selected.modelId, props.graph.graph_id, 'Architecture', { code: 'parameter_warning', parameter_id: 'weight', message: 'Own parameter warning.' }, 'warning'),
    finding(props.selected.modelId, props.graph.graph_id, 'Architecture', { code: 'coverage_note', message: 'Coverage note.' }, 'warning'),
  ];
  const model = render(<ArchitectureInspection {...props} selected={{ ...props.selected, node: root }} diagnostics={findings} />);
  const modelDetails = screen.getByLabelText('Model architecture information');
  expect(modelDetails).toHaveTextContent('Partial architecture coverage · model defined');
  expect(modelDetails).toHaveTextContent('Model-supplied architecture.');
  expect(screen.getByRole('dialog')).toHaveTextContent('<script>source evidence</script>');
  expect(screen.getByLabelText('Applicable architecture diagnostics').querySelectorAll('li')).toHaveLength(4);
  model.unmount();
  const component = render(<ArchitectureInspection {...props} diagnostics={findings} />);
  expect(screen.getByLabelText('Applicable architecture diagnostics')).toHaveTextContent('UI mapping warning.');
  expect(screen.getByLabelText('Applicable architecture diagnostics')).toHaveTextContent('Own parameter warning.');
  expect(screen.getByLabelText('Applicable architecture diagnostics')).not.toHaveTextContent('Source warning.');
  expect(screen.queryByLabelText('Model architecture information')).not.toBeInTheDocument();
  component.unmount();
  const syntheticGraph = structuredClone(props.graph);
  const secondRoot = { ...syntheticGraph.nodes.find((n) => n.id === 'linear0')! };
  delete secondRoot.parent_id;
  syntheticGraph.nodes.push({ ...secondRoot, id: 'second-root' });
  const synthetic = render(<ArchitectureInspection {...props} graph={syntheticGraph} selected={{
    modelId: props.selected.modelId, sessionId: props.selected.sessionId, graphId: props.selected.graphId, trigger,
    boundary: { kind: 'boundary', owner: { kind: 'model', id: 'presentation:model' }, endpoints: [] } }} diagnostics={findings} />);
  expect(screen.getByLabelText('Model architecture information')).toHaveTextContent('Model-supplied architecture.');
  expect(screen.getByLabelText('Applicable architecture diagnostics').querySelectorAll('li')).toHaveLength(4);
  synthetic.unmount(); trigger.remove();
});

it('streams values through StrictMode effect replay and fences replaced/closed generations', () => {
  const { props, handles, trigger } = setup();
  const view = render(<StrictMode><ArchitectureInspection {...props} /></StrictMode>);
  fireEvent.change(screen.getByLabelText('Inspect parameter'), { target: { value: 'weight' } });
  const live = () => handles.filter((h) => !h.cancel.mock.calls.length);
  expect(live()).toHaveLength(3);
  const previous = live();
  const previousSink = sinks.at(-1)!;
  const bytes = new Uint8Array(new Float32Array([-2, 0, 2]).buffer);
  act(() => previous[0]!.options.onData!(bytes, 0));
  expect(previousSink.values).toHaveBeenCalledWith(new Float32Array([-2, 0, 2]), 0);
  fireEvent.change(screen.getByLabelText('Inspect parameter'), { target: { value: 'alias' } });
  expect(live()).toHaveLength(3);
  expect(previous.every((h) => h.cancel.mock.calls.length === 1)).toBe(true);
  const replacement = live();
  const replacementSink = sinks.at(-1)!;
  act(() => {
    previous[0]!.options.onData!(bytes, 0);
    replacement[0]!.options.onData!(bytes, 0);
  });
  expect(previousSink.values).toHaveBeenCalledOnce();
  expect(replacementSink.values).toHaveBeenCalledOnce();
  fireEvent.click(screen.getByRole('button', { name: 'Close inspection' }));
  expect(live()).toHaveLength(0);
  act(() => replacement[0]!.options.onData!(bytes, 0));
  expect(replacementSink.values).toHaveBeenCalledOnce();
  expect(props.context.selection.isCurrent()).toBe(true);
  view.unmount(); expect(trigger).toHaveFocus(); trigger.remove();
});
