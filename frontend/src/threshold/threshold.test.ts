import { describe, expect, it } from 'vitest';

import type { CsvDocument } from '../csvMerge/csvMerge';
import {
  buildThresholdDataset, calculateQuantileThreshold, canonicalSelectionRange, decimateThresholdRows, linearQuantile,
  numericCandidateColumns, parseThresholdNumber, thresholdExportTable, timestampCandidateColumns,
  thresholdAxisPlan, unionThresholdIntervals,
} from './threshold';

const document: CsvDocument = {
  fileName: 'sensors.csv', delimiter: ',', headers: ['time', 'a', 'b'], rows: [
    ['2026-01-01 00:00:00', '1', '1,5'],
    ['2026-01-01 00:00:10', '2', 'broken'],
    ['2026-01-01 00:00:20', '3', '3,5'],
    ['2026-01-01 00:02:00', '4', null],
    ['2026-01-01 00:02:10', '5', '5,5'],
    ['invalid', '6', '6,5'],
  ],
};

describe('threshold CSV data', () => {
  it('detects temporal/numeric columns and parses decimal commas', () => {
    expect(timestampCandidateColumns(document)).toEqual(['time']);
    expect(numericCandidateColumns(document, 'time')).toEqual(['a', 'b']);
    expect(parseThresholdNumber('1,25')).toEqual({ value: 1.25, invalid: false });
    expect(parseThresholdNumber('NaN')).toEqual({ value: null, invalid: true });
  });

  it('sorts timestamps, retains duplicates, reports invalid cells, and splits gaps', () => {
    const dataset = buildThresholdDataset(document, 'time', ['a', 'b']);
    expect(dataset.rows).toHaveLength(5);
    expect(dataset.invalidTimestampRows).toBe(1);
    expect(dataset.invalidNumericByColumn.b).toBe(1);
    expect(dataset.rows.map((row) => row.continuitySegment)).toEqual([0, 0, 0, 1, 1]);
  });
});

describe('quantile thresholds', () => {
  const rows = buildThresholdDataset(document, 'time', ['a', 'b']).rows;

  it('matches linear quantiles at endpoints and interpolated positions', () => {
    expect(linearQuantile([1, 2, 3, 4], 0)).toBe(1);
    expect(linearQuantile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(linearQuantile([1, 2, 3, 4], 1)).toBe(4);
    expect(() => linearQuantile([], 0.9)).toThrow(/no finite/i);
  });

  it('uses inclusive reference ranges and strict greater-than decisions', () => {
    const result = calculateQuantileThreshold(rows, 'a', 0.5, 'reference', '2026-01-01T00:00:00', '2026-01-01T00:00:20');
    expect(result.value).toBe(2);
    expect(result.inputCount).toBe(3);
    expect(result.exceedancePointCount).toBe(3);
    expect(result.intervals).toHaveLength(2);
    expect(result.intervals[0].start).toBe('2026-01-01T00:00:20');
  });

  it('canonicalizes plot drags in either direction', () => {
    expect(canonicalSelectionRange('2026-01-01 00:02', '2026-01-01 00:01')).toEqual({
      start: '2026-01-01T00:01:00', end: '2026-01-01T00:02:00',
    });
  });

  it('does not merge red intervals across continuity gaps', () => {
    const threshold = calculateQuantileThreshold(rows, 'a', 0, 'entire');
    const merged = unionThresholdIntervals([...threshold.intervals, { start: rows[0].timestamp, end: rows[1].timestamp, continuitySegment: 0 }]);
    expect(new Set(merged.map((item) => item.continuitySegment))).toEqual(new Set([0, 1]));
  });

  it('decimates only previews and exports every original row and threshold', () => {
    const many = Array.from({ length: 20_000 }, (_, index) => ({
      ...rows[index % rows.length], timestamp: `2026-01-${String(Math.floor(index / 1440) + 1).padStart(2, '0')}T${String(Math.floor(index / 60) % 24).padStart(2, '0')}:${String(index % 60).padStart(2, '0')}:00`, position: index,
      values: { a: index, b: index % 17 }, continuitySegment: 0,
    }));
    expect(decimateThresholdRows(many, ['a', 'b'], 8000).length).toBeLessThanOrEqual(8000);
    expect(linearQuantile(many.map((row) => Number(row.values.a)), 0.9)).toBeCloseTo(17_999.1);
    const table = thresholdExportTable(rows, ['a'], [{ id: 'threshold-1', column: 'a', quantile: 0.5, value: 3 }]);
    expect(table.rowCount).toBe(rows.length);
    expect(table.columns).toHaveLength(3);
    expect(table.columns[2].values).toEqual(rows.map(() => 3));
  });

  it('assigns six graph columns to independent visible axes', () => {
    const plan = thresholdAxisPlan(['a', 'b', 'c', 'd', 'e', 'f']);
    expect(new Set(plan.axes.map((axis) => axis.traceAxis)).size).toBe(6);
    expect(plan.axes.filter((axis) => axis.side === 'left')).toHaveLength(3);
    expect(plan.axes.filter((axis) => axis.side === 'right')).toHaveLength(3);
    expect(plan.xDomain[0]).toBeLessThan(plan.xDomain[1]);
    expect(() => thresholdAxisPlan(['1', '2', '3', '4', '5', '6', '7'])).toThrow(/six/);
  });
});
