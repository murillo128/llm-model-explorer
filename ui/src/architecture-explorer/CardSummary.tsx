import { formatShape } from './graph';
import type { GraphNode } from './graph';
import { relativeParameterName, summaryLimit } from './card-summary';
import type { CardSummary as Summary } from './card-summary';

export function SummaryText({ text, fullText = text, className = '' }: { text: string; fullText?: string; className?: string }) {
  return <span className={`architecture-summary-text ${className}`} tabIndex={0} aria-label={fullText} onKeyDown={(event) => event.stopPropagation()}>
    <span>{text}</span><span className="architecture-summary-tooltip" aria-hidden="true">{fullText}</span>
  </span>;
}

export function CardParameters({ node, summary, dimensions, top, inspect, matrix }: {
  node: GraphNode; summary: Summary; dimensions: boolean; top: number;
  inspect: (trigger: HTMLElement) => void; matrix: (id: string, trigger: HTMLElement) => void;
}) {
  return <div className="architecture-card-parameters nodrag nopan" style={{ top }}
    onPointerDown={(event) => event.stopPropagation()} onClick={(event) => event.stopPropagation()}
    onDoubleClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
    {summary.parameters.slice(0, summaryLimit).map((parameter) => {
      const unavailable = parameter.inspection.status === 'unavailable'
        ? `${parameter.inspection.reason.replaceAll('_', ' ')}: ${parameter.inspection.message}` : undefined;
      return <div className="architecture-tensor-row" data-parameter-id={parameter.id} key={parameter.id}>
        <button className="architecture-matrix-action" aria-label={`Inspect matrix ${parameter.name}`}
          aria-disabled={Boolean(unavailable)} aria-description={unavailable}
          title={unavailable ?? `Inspect matrix ${parameter.name}`}
          onClick={(event) => { if (!unavailable && event.detail < 2) matrix(parameter.id, event.currentTarget); }}>
          <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" aria-hidden="true">
            <rect x="1.5" y="1.5" width="13" height="13" rx="1" /><path d="M6 2v12M10 2v12M2 6h12M2 10h12" />
          </svg>
          {unavailable && <span className="architecture-summary-tooltip" aria-hidden="true">{unavailable}</span>}
        </button>
        <div className="architecture-tensor-name">
          <SummaryText text={relativeParameterName(node, parameter)} fullText={parameter.name} />
          {dimensions && <SummaryText className="architecture-card-shape" text={formatShape(parameter.logical_shape)} />}
        </div>
      </div>;
    })}
    {summary.parameters.length > summaryLimit && <button className="architecture-more-parameters"
      aria-label={`Inspect ${summary.parameters.length - summaryLimit} more tensors of ${node.label}`}
      onClick={(event) => inspect(event.currentTarget)}>+{summary.parameters.length - summaryLimit} more</button>}
    {summary.constants.map((a, index) => <div className="architecture-constant" key={`${a.name}:${index}`}>
      <SummaryText text={`${a.name} = ${a.value === null ? 'unknown' : String(a.value)}`} />
    </div>)}
  </div>;
}
