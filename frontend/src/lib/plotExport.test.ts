import { describe, expect, it } from 'vitest';
import { parquetReadObjects } from 'hyparquet';
import type { Data } from './plotly';
import {
  buildPlotExportTable,
  formatPlotTimestamp,
  normalizedDownloadName,
  plotTableToCsv,
  plotTableToParquet,
} from './plotExport';

function trace(name: string, x: unknown[], y: unknown[]): Data {
  return { type: 'scatter', mode: 'lines', name, x, y } as unknown as Data;
}

describe('plot export', () => {
  it('formats timestamps without applying a timezone conversion', () => {
    expect(formatPlotTimestamp('2026-08-25T14:03:09.999+02:00')).toBe('2026.08.25 14:03:09');
    expect(formatPlotTimestamp('2026-08-25 04:05')).toBe('2026.08.25 04:05:00');
    expect(formatPlotTimestamp('not-a-date')).toBeNull();
  });

  it('outer-joins graph columns by timestamp and retains duplicate frames', () => {
    const table = buildPlotExportTable([
      trace('Score', ['2026-01-01T00:00:00', '2026-01-01T00:00:00', '2026-01-01T00:01:00'], [1, 2, 3]),
      trace('Threshold', ['2026-01-01T00:00:00', '2026-01-01T00:02:00'], [10, 20]),
    ]);

    expect(table.rowCount).toBe(4);
    expect(table.seriesCount).toBe(2);
    expect(table.columns.map((column) => column.name)).toEqual(['timestamp', 'Score', 'Threshold']);
    expect(table.columns[0].values).toEqual([
      '2026.01.01 00:00:00',
      '2026.01.01 00:00:00',
      '2026.01.01 00:01:00',
      '2026.01.01 00:02:00',
    ]);
    expect(table.columns[1].values).toEqual([1, 2, 3, null]);
    expect(table.columns[2].values).toEqual([10, null, null, 20]);
  });

  it('exports numeric axes and expands heatmap rows into graph columns', () => {
    const table = buildPlotExportTable([{
      type: 'heatmap',
      name: 'Error',
      x: [0, 1],
      y: ['ROI A', 'ROI B'],
      z: [[1, 2], [3, Number.NaN]],
    } as unknown as Data]);
    expect(table.columns.map((column) => column.name)).toEqual(['x', 'Error · ROI A', 'Error · ROI B']);
    expect(table.columns[0].values).toEqual([0, 1]);
    expect(table.columns[1].values).toEqual([1, 2]);
    expect(table.columns[2].values).toEqual([3, null]);
  });

  it('writes quoted UTF-8 CSV with the requested timestamp representation', () => {
    const table = buildPlotExportTable([
      trace('Score, raw', ['2026-01-01T00:00:00'], [1.5]),
    ]);
    expect(plotTableToCsv(table)).toBe('\uFEFFtimestamp,"Score, raw"\r\n2026.01.01 00:00:00,1.5');
  });

  it('writes readable Parquet with one column per graph', async () => {
    const table = buildPlotExportTable([
      trace('Score', ['2026-01-01T00:00:00', '2026-01-01T00:01:00'], [1.25, null]),
      trace('Baseline', ['2026-01-01T00:00:00', '2026-01-01T00:01:00'], [0.5, 0.75]),
    ]);
    const buffer = await plotTableToParquet(table);
    const rows = await parquetReadObjects({ file: buffer });
    expect(rows).toEqual([
      { timestamp: '2026.01.01 00:00:00', Score: 1.25, Baseline: 0.5 },
      { timestamp: '2026.01.01 00:01:00', Score: null, Baseline: 0.75 },
    ]);
  });

  it('normalizes unsafe names and replaces an existing export extension', () => {
    expect(normalizedDownloadName('daily/score.csv', 'parquet')).toBe('daily_score.parquet');
    expect(normalizedDownloadName('   ', 'csv')).toBe('plot-data.csv');
  });
});
