import type { components } from '../api/generated/types';

type S = components['schemas'];
export type Severity = 'info' | 'warning' | 'error';
export interface Diagnostic {
  id: string; capability: 'Architecture' | 'Tensor inventory'; severity: Severity;
  code: string; message: string; context?: string;
}
export const modelSuppliedExplanation = 'Model-supplied architecture. Structure and weight bindings are validated; equivalence to model code is not verified.';
export function finding(model: string, generation: string, capability: Diagnostic['capability'],
  source: S['ArchitectureDiagnostic'], severity: Severity, context?: string): Diagnostic {
  return { id: JSON.stringify([model, generation, capability, source.code, source.node_id ?? '', source.parameter_id ?? '', severity, source.message]),
    capability, severity, code: source.code, message: source.message,
    ...(context || source.node_id || source.parameter_id ? { context: context ?? source.node_id ?? source.parameter_id! } : {}) };
}
export function uniqueFindings(records: Diagnostic[]) {
  return [...new Map(records.map((d) => [d.id, d])).values()].sort((a, b) =>
    ({ error: 0, warning: 1, info: 2 }[a.severity] - { error: 0, warning: 1, info: 2 }[b.severity]));
}
type GraphInformation = Pick<S['ArchitectureGraph'], 'graph_id' | 'scope' | 'coverage'>;
const empty = { graphInformation: undefined as GraphInformation | undefined, records: [] as Diagnostic[], dismissed: [] as string[], architectureObserved: false, modelSupplied: false };
/** Current observations only. No graphs, numeric resources, connection state or history. */
export class ModelDiagnostics {
  private state = empty;
  private session: object | null = null;
  private readonly listeners = new Set<() => void>();
  private readonly generations = new Map<string, string>();
  private dismissed = new Set<string>();
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  getSnapshot = () => this.state;
  private publish(patch: Partial<typeof empty>) {
    this.state = { ...this.state, ...patch }; this.listeners.forEach((listener) => listener());
  }
  activate(session: object | null) { this.session = session; this.publish({ ...empty, dismissed: [...this.dismissed] }); }
  observe(session: object, capability: Diagnostic['capability'], records: Diagnostic[], modelSupplied = false) {
    if (session !== this.session) return;
    this.publish({ records: uniqueFindings([...this.state.records.filter((d) => d.capability !== capability), ...records]),
      ...(capability === 'Architecture' ? { architectureObserved: true, modelSupplied } : {}) });
  }
  graph(session: object, model: string, generation: string, graphInformation?: GraphInformation) {
    if (session !== this.session) return;
    this.publish({ graphInformation: graphInformation && { graph_id: graphInformation.graph_id, scope: graphInformation.scope, coverage: graphInformation.coverage } });
    if (this.generations.get(model) !== generation) {
      this.dismissed = new Set([...this.dismissed].filter((id) => (JSON.parse(id) as string[])[0] !== model));
      this.generations.delete(model); this.generations.set(model, generation);
      while (this.generations.size > 8) {
        const oldest = this.generations.keys().next().value!; this.generations.delete(oldest);
        this.dismissed = new Set([...this.dismissed].filter((id) => (JSON.parse(id) as string[])[0] !== oldest));
      }
      this.publish({ dismissed: [...this.dismissed] });
    }
  }
  dismiss = (ids: string[]) => { ids.forEach((id) => this.dismissed.add(id)); this.publish({ dismissed: [...this.dismissed] }); };
}
