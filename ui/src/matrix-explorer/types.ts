import type { ReactNode } from 'react';
import type { TensorDescriptor } from '../rendering/geometry';
import type { TransferParameters } from '../rendering/transfer';
import type { RendererState } from '../rendering/tensor-renderer';
import type { DistributionDomain } from '../rendering/distribution-scale';

/** Native zero-based coordinates; no viewport, texture-band or domain indices. */
export interface MatrixCell { readonly row: number; readonly column: number }

/** Synchronous uploads, with element offsets into consecutive C-order prefixes.
 * Calls made after this subscription is detached are ignored, including A → B → A.
 * Arrays are consumed immediately, never retained as a second scalar copy.
 */
export interface MatrixUpdates {
  values(values: Float32Array, offset: number): void;
  transfer(parameters: TransferParameters): void;
  /** Authoritative uint32 counts: rows [rows, 100], columns [100, columns]. */
  distribution(axis: 'rows' | 'columns', counts: Uint32Array, offset: number): void;
  /** Shared bin endpoints from the distribution producer, never percentile anchors. */
  distributionDomain(domain: DistributionDomain): void;
}

/** Keep this object stable for one generation. Replacement detaches the old
 * subscription and allocates a fresh viewport even when descriptors are equal.
 * Re-subscribing (including StrictMode) must restart delivery at offset zero.
 */
export interface MatrixSource {
  /** Logical rank-2 float32 descriptor; rank-1 strips remain supported. */
  readonly descriptor: TensorDescriptor;
  /** Omit/false for matrix-only; true reserves both aligned 100-bin surfaces. */
  readonly distributions?: boolean;
  readonly transfer?: TransferParameters;
  subscribe(updates: MatrixUpdates): () => void;
}

export interface MatrixExplorerProps {
  readonly source: MatrixSource;
  readonly label?: string;
  readonly header?: ReactNode | ((cameraControls: ReactNode) => ReactNode);
  readonly onRenderingStateChange?: (state: RendererState) => void;
  /** Display-only row context from a linked view; does not select a cell or move focus. */
  readonly highlightedRow?: number | null;
  /** New object identity requests minimal vertical reveal, preserving scale/X/focus.
   * Keep stable across renders; omit on source/generation replacement. */
  readonly revealRow?: { readonly row: number } | null;
  /** Hover/focus selection, cleared with null on leave, blur or replacement. */
  readonly onCellSelect?: (cell: MatrixCell | null) => void;
  readonly onRowSelect?: (row: number | null) => void;
  readonly onColumnSelect?: (column: number | null) => void;
}
