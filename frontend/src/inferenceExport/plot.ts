import type { Data } from '../lib/plotly';
import type { PlotExportTable } from '../lib/plotExport';

/** Limit rendering only. Preserve each source's endpoints and explicit missing-value gaps. */
export function inferenceExportPlotData(table: PlotExportTable, maxPoints = 4000): Data[] {
  return table.columns.slice(1).map((column) => {
    const present: Array<{ row: number; segment: number }> = [];
    let segment = 0;
    let missing = false;
    column.values.forEach((value, row) => {
      if (value === null) { missing = true; return; }
      if (missing) segment += 1;
      missing = false;
      present.push({ row, segment });
    });
    const count = Math.min(present.length, Math.max(2, maxPoints));
    const x: Array<string | null> = [];
    const y: Array<number | null> = [];
    let previousSegment: number | null = null;
    for (let index = 0; index < count; index += 1) {
      const point = present[Math.floor(index * (present.length - 1) / Math.max(1, count - 1))];
      if (previousSegment !== null && previousSegment !== point.segment) { x.push(null); y.push(null); }
      x.push(String(table.columns[0].values[point.row]).replaceAll('.', '-').replace(' ', 'T'));
      y.push(column.values[point.row] as number);
      previousSegment = point.segment;
    }
    return { type: 'scattergl', mode: 'lines+markers', marker: { size: 3 }, name: column.name, x, y, connectgaps: false };
  });
}
