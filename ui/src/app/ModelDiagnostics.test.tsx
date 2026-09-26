import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { DiagnosticBand, ModelDiagnosticInformation } from './ModelDiagnostics';
import { ModelDiagnostics, finding } from './model-diagnostics';
it.each(['info', 'warning', 'error'] as const)('presents %s explicitly and retains dismissed complete details', (severity) => {
  const store = new ModelDiagnostics(), session = {};
  store.activate(session); store.observe(session, 'Architecture', [finding('m', 'g', 'Architecture', { code: 'safe_code', message: 'Complete safe detail.' }, severity)]);
  const view = render(<><DiagnosticBand store={store} /><ModelDiagnosticInformation store={store} /></>);
  expect(screen.getByRole('status')).toHaveTextContent(severity[0]!.toUpperCase() + severity.slice(1));
  fireEvent.click(screen.getByRole('button', { name: 'Dismiss architecture notices' }));
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
  expect(screen.queryByRole('region', { name: 'Architecture notices' })).not.toBeInTheDocument();
  expect(screen.getByLabelText('Diagnostics')).toHaveTextContent('Complete safe detail.');
  view.rerender(<><DiagnosticBand store={store} /><ModelDiagnosticInformation store={store} /></>);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});
it.each(['provenance', 'findings'] as const)('dismisses %s independently while preserving full information', (first) => {
  const store = new ModelDiagnostics(), session = {};
  const rawId = 'long/raw/id/with_a_technical_identifier';
  const record = finding('m', 'g', 'Architecture', { code: 'interface_mapping_ambiguous', node_id: rawId,
    message: `Mapping for ${rawId} is ambiguous.` }, 'warning', `Input IDs · ${rawId}`, 'Input IDs');
  store.activate(session); store.graph(session, 'm', 'g', { graph_id: 'g', scope: 'model_defined', coverage: 'partial' });
  store.observe(session, 'Architecture', [record], true);
  render(<><DiagnosticBand store={store} /><ModelDiagnosticInformation store={store} /></>);
  const band = () => screen.getByRole('region', { name: 'Architecture notices' });
  expect(band().getElementsByClassName('diagnostic-notice')).toHaveLength(1);
  expect(band().querySelector('.diagnostic-notice [role="status"]')).toHaveTextContent('Input IDs: Mapping for the component is ambiguous.');
  expect(band().querySelector('.diagnostic-notice [role="status"]')).not.toHaveTextContent(rawId);
  fireEvent.click(screen.getByText('View details'));
  expect(band()).toHaveTextContent(rawId);
  const names = ['Dismiss model-supplied information', 'Dismiss architecture notices'];
  fireEvent.click(screen.getByRole('button', { name: names[first === 'provenance' ? 0 : 1]! }));
  expect(band().querySelector(first === 'provenance' ? '.model-supplied' : '.diagnostic-notice')).toBeNull();
  expect(band().querySelector(first === 'provenance' ? '.diagnostic-notice' : '.model-supplied')).not.toBeNull();
  fireEvent.click(screen.getByRole('button', { name: names[first === 'provenance' ? 1 : 0]! }));
  expect(screen.queryByRole('region', { name: 'Architecture notices' })).not.toBeInTheDocument();
  expect(screen.getByLabelText('Diagnostics')).toHaveTextContent(rawId);
  expect(screen.getByLabelText('Diagnostics')).toHaveTextContent('Model-supplied architecture.');
});
it('does not claim unobserved capabilities are healthy', () => {
  render(<ModelDiagnosticInformation store={new ModelDiagnostics()} />);
  expect(screen.getByText('Architecture diagnostics have not been retrieved.')).toBeInTheDocument();
});
it('shows compact graph metadata in information and clears it on session replacement or unavailable graph', () => {
  const store = new ModelDiagnostics(), session = {};
  store.activate(session);
  store.graph(session, 'm', 'g', { graph_id: 'g', scope: 'model_defined', coverage: 'complete' });
  const view = render(<ModelDiagnosticInformation store={store} />);
  expect(screen.getByLabelText('Architecture information')).toHaveTextContent('Graph: g');
  expect(screen.getByLabelText('Architecture information')).toHaveTextContent('Complete within declared scope · model defined');
  // Mutation before mounting avoids conflating store lifetime with React scheduling.
  view.unmount(); store.activate({});
  store.graph(session, 'm', 'late', { graph_id: 'late', scope: 'model_defined', coverage: 'partial' });
  const next = render(<ModelDiagnosticInformation store={store} />);
  expect(screen.queryByLabelText('Architecture information')).not.toBeInTheDocument();
  next.unmount(); store.activate(session);
  store.graph(session, 'm', 'g', { graph_id: 'g', scope: 'model_defined', coverage: 'complete' });
  store.graph(session, 'm', 'unavailable');
  render(<ModelDiagnosticInformation store={store} />);
  expect(screen.queryByLabelText('Architecture information')).not.toBeInTheDocument();
});
