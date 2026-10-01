import { describe, expect, it } from 'vitest';
import { buildInferenceExport, type ExportOptions, type InferenceExportSource, type SamplingMethod } from './export';
import { tabularTableToCsv } from '../lib/tabularExport';
import { inferenceExportPlotData } from './plot';

const original: ExportOptions = { samplingRate: 'original', method: 'mean', utc: false };
const source = (points: Array<[string, number]>, id = 1, name = 'Inference'): InferenceExportSource => ({
  id, name, points: points.map(([timestamp, value], position) => ({ timestamp, value, position })),
});
const at = (time: string) => `2025-09-16T${time}`;
const rows = (table: ReturnType<typeof buildInferenceExport>) => Array.from({ length: table.rowCount }, (_, i) => table.columns.map((column) => column.values[i]));

describe('inference CSV export', () => {
  it('joins chronologically, retains duplicates, and preserves selection order and empty cells', () => {
    const table = buildInferenceExport([
      source([[at('00:01:00'), 7], [at('00:00:00'), 1], [at('00:00:00'), 2]], 2),
      source([[at('00:00:00'), 3], [at('00:02:00'), 4]], 1),
    ], original);
    expect(table.columns.map((column) => column.name)).toEqual(['timestamp', 'Inference (#2)', 'Inference (#1)']);
    expect(rows(table)).toEqual([
      ['2025.09.16 00:00:00', 1, 3], ['2025.09.16 00:00:00', 2, null],
      ['2025.09.16 00:01:00', 7, null], ['2025.09.16 00:02:00', null, 4],
    ]);
  });

  it('does not merge distinct microseconds that format to the same second', () => {
    const table = buildInferenceExport([
      source([[at('00:00:00.000002'), 2], [at('00:00:00.000001'), 1]]),
      source([[at('00:00:00.000002'), 3]], 2),
    ], original);
    expect(rows(table)).toEqual([['2025.09.16 00:00:00', 1, null], ['2025.09.16 00:00:00', 2, 3]]);
  });

  it.each<[SamplingMethod, number]>([['first', 9], ['mean', 5], ['median', 5], ['min', 1], ['max', 9]])('aggregates using %s in half-open minute buckets', (method, expected) => {
    const input = source([[at('00:00:50'), 1], [at('00:00:00'), 9], [at('00:01:00'), 11], [at('00:00:20'), 5], [at('00:04:30'), 20]]);
    const table = buildInferenceExport([input], { ...original, samplingRate: '1min', method });
    expect(rows(table)).toEqual([['2025.09.16 00:00:00', expected], ['2025.09.16 00:01:00', 11], ['2025.09.16 00:04:00', 20]]);
  });

  it('uses stored position to break first-value ties and calculates even medians without mutating input', () => {
    const input = source([[at('00:00:00'), 10], [at('00:00:00'), 2]]);
    input.points.reverse();
    expect(buildInferenceExport([input], { ...original, samplingRate: '1min', method: 'first' }).columns[1].values).toEqual([10]);
    expect(buildInferenceExport([input], { ...original, samplingRate: '1min', method: 'median' }).columns[1].values).toEqual([6]);
    expect(input.points.map((point) => point.value)).toEqual([2, 10]);
  });

  it('aligns five-minute intervals, aggregates sources independently, and never fills gaps', () => {
    const table = buildInferenceExport([
      source([[at('00:04:59.999999'), 2], [at('00:00:01'), 4], [at('00:05:00'), 10]]),
      source([[at('00:14:00'), 8]], 2),
    ], { ...original, samplingRate: '5min' });
    expect(rows(table)).toEqual([['2025.09.16 00:00:00', 3, null], ['2025.09.16 00:05:00', 10, null], ['2025.09.16 00:10:00', null, 8]]);
  });

  it.each([
    ['2025-09-16T00:00:00', '2025.09.15 22:00:00'],
    ['2025-01-01T00:30:00', '2024.12.31 23:30:00'],
    ['2025-03-30T01:59:59', '2025.03.30 00:59:59'],
    ['2025-03-30T03:00:00', '2025.03.30 01:00:00'],
    ['2025-10-26T01:59:59', '2025.10.25 23:59:59'],
    ['2025-10-26T03:00:00', '2025.10.26 02:00:00'],
  ])('converts Berlin %s to UTC with the correct offset', (timestamp, expected) => {
    expect(buildInferenceExport([source([[timestamp, 1]])], { ...original, utc: true }).columns[0].values).toEqual([expected]);
  });

  it.each(['original', '1min', '5min'] as const)('rejects ambiguous/nonexistent Berlin times before %s sampling', (samplingRate) => {
    for (const [timestamp, reason] of [['2025-03-30T02:30:00', 'Nonexistent'], ['2025-10-26T02:30:00', 'Ambiguous']]) {
      expect(() => buildInferenceExport([source([[timestamp, 1]])], { ...original, utc: true, samplingRate })).toThrow(new RegExp(`${reason}.*${timestamp}`));
      expect(buildInferenceExport([source([[timestamp, 1]])], { ...original, samplingRate }).rowCount).toBe(1);
    }
  });

  it('converts all sources before resampling and retains microsecond matches in UTC', () => {
    const sources = [source([[at('00:00:00.000001'), 2]]), source([[at('00:00:00.000001'), 4]], 2)];
    expect(rows(buildInferenceExport(sources, { ...original, utc: true }))).toEqual([['2025.09.15 22:00:00', 2, 4]]);
    expect(rows(buildInferenceExport(sources, { ...original, utc: true, samplingRate: '5min' }))).toEqual([['2025.09.15 22:00:00', 2, 4]]);
  });

  it('writes the exact CSV format with unquoted numbers and properly escaped names', () => {
    const table = buildInferenceExport([
      source([[at('00:00:00'), 0.125], [at('00:01:00'), -3.25]], 1, 'Score, "A"'),
      source([[at('00:00:00'), 1.5]], 2, 'B'),
    ], { ...original, utc: true });
    expect(tabularTableToCsv(table)).toBe('\uFEFFtimestamp,"Score, ""A"" (#1)",B (#2)\r\n2025.09.15 22:00:00,0.125,1.5\r\n2025.09.15 22:01:00,-3.25,');
  });

  it('rejects invalid, empty and incomplete sources instead of producing a partial table', () => {
    expect(() => buildInferenceExport([], original)).toThrow('Select at least');
    expect(() => buildInferenceExport([source([])], original)).toThrow('no stored points');
    expect(() => buildInferenceExport([source([[at('00:00:00'), NaN]])], original)).toThrow('non-finite');
    expect(() => buildInferenceExport([source([['2025-02-30T12:00:00', 1]])], original)).toThrow('Invalid source timestamp');
    expect(() => buildInferenceExport([source([[at('00:00:00'), 1]]), source([])], original)).toThrow('more than once');
  });

  it('bounds the chart while retaining full export data, endpoints, and explicit gaps', () => {
    const table = buildInferenceExport([
      source(Array.from({ length: 25 }, (_, i) => [at(`00:${String(i).padStart(2, '0')}:00`), i])),
      source([[at('00:00:00'), 10], [at('00:24:00'), 20]], 2),
    ], original);
    const traces = inferenceExportPlotData(table, 3) as Array<{ x: (string | null)[]; y: (number | null)[] }>;
    expect(traces[0].y).toEqual([0, 12, 24]);
    expect(traces[1].y).toEqual([10, null, 20]);
    expect(traces[0].x[0]).toBe('2025-09-16T00:00:00');
    expect(table.rowCount).toBe(25);
    expect(table.columns[1].values).toHaveLength(25);
  });
});
