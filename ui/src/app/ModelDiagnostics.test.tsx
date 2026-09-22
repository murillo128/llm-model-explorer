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
  expect(screen.getByLabelText('Diagnostics')).toHaveTextContent('Complete safe detail.');
  expect(screen.getByLabelText('Architecture notices')).toHaveFocus();
  view.rerender(<><DiagnosticBand store={store} /><ModelDiagnosticInformation store={store} /></>);
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});
it('does not claim unobserved capabilities are healthy', () => {
  render(<ModelDiagnosticInformation store={new ModelDiagnostics()} />);
  expect(screen.getByText('Architecture diagnostics have not been retrieved.')).toBeInTheDocument();
});
