/** Test-only exhaustive comparison. Expected values remain owned by each oracle. */
export interface Cell { readonly row: number; readonly column: number }
export interface PixelComparison {
  readonly scenario: string;
  readonly width: number;
  readonly height: number;
  readonly frame: readonly (readonly (readonly number[])[])[];
  readonly cells: readonly (readonly (Cell | null)[])[];
  readonly expected: (cell: Cell, x: number, y: number) => readonly number[];
  readonly tolerance?: number;
}

export function comparePixels({ scenario, width, height, frame, cells, expected, tolerance = 0 }: PixelComparison): number {
  const fail = (message: string): never => { throw new Error(`${scenario}: ${message}`); };
  if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0 ||
      !Number.isSafeInteger(width * height * 4)) fail(`invalid dimensions ${width}×${height}`);
  if (!Number.isFinite(tolerance) || tolerance < 0) fail(`invalid tolerance ${tolerance}`);
  if (frame.length !== height || cells.length !== height) fail(`expected ${height} rows; pixels ${frame.length}, cells ${cells.length}`);
  let comparisons = 0, mismatches = 0;
  const details: string[] = [];
  for (let y = 0; y < height; y++) {
    const row = frame[y], cellRow = cells[y];
    if (!row || !cellRow || row.length !== width || cellRow.length !== width) fail(`device row ${y}: expected width ${width}; pixels ${row?.length}, cells ${cellRow?.length}`);
    for (let x = 0; x < width; x++) {
      const cell = cellRow![x];
      if (!cell || !Number.isSafeInteger(cell.row) || cell.row < 0 || !Number.isSafeInteger(cell.column) || cell.column < 0) fail(`device pixel ${x}:${y}: missing/invalid logical cell`);
      const actual = row![x];
      const wanted = expected(cell!, x, y);
      if (!actual || actual.length !== 4 || wanted.length !== 4) fail(`device pixel ${x}:${y}, cell ${cell!.row}:${cell!.column}: expected four channels; actual ${actual?.length}, expected ${wanted.length}`);
      for (let channel = 0; channel < 4; channel++) {
        comparisons++;
        const a = actual![channel]!, e = wanted[channel]!;
        if (!Number.isInteger(a) || a < 0 || a > 255 || !Number.isFinite(e) || !(Math.abs(a - e) <= tolerance)) {
          mismatches++;
          if (details.length < 8) details.push(`device pixel ${x}:${y}, cell ${cell!.row}:${cell!.column}, channel ${channel}: expected ${e}, actual ${a}`);
        }
      }
    }
  }
  if (comparisons !== width * height * 4) fail(`comparison count ${comparisons}, expected ${width * height * 4}`);
  if (mismatches) fail(`${mismatches} mismatched channels in ${comparisons} comparisons (tolerance ${tolerance})\n${details.join('\n')}`);
  return comparisons;
}
