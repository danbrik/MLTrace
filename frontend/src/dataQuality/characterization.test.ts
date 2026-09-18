import { describe, expect, it } from 'vitest';
import { summaryRows, temporalTraces, histogramTrace } from './characterization';
import type { CharacterizationSummary } from '../types';
const row = (name: string, mean: number | null): CharacterizationSummary => ({ variable: name, data_type: mean == null ? 'text' : 'numeric', valid_n: 10,
  mean, median: mean, std: mean, iqr: mean, min: mean, q01: mean, q05: mean, q25: mean, q75: mean, q95: mean, q99: mean, max: mean });
describe('characterization display', () => {
  it('keeps non-applicable values last in either direction', () => {
    const rows = [row('Text', null), row('B', 2), row('A', 1)];
    expect(summaryRows(rows, '', 'all', 'mean', false).map(r => r.variable)).toEqual(['A', 'B', 'Text']);
    expect(summaryRows(rows, '', 'all', 'mean', true).map(r => r.variable)).toEqual(['B', 'A', 'Text']);
    expect(summaryRows(rows, 'a', 'numeric', 'variable', false)).toHaveLength(1);
    expect(rows[0].variable).toBe('Text');
  });
  it('inserts explicit line breaks at missing observations', () => {
    const traces = temporalTraces({ aggregated: false, interval_seconds: 60, points: [
      { time: '2026-01-01T00:00', end: '2026-01-01T00:00', value: 1, q25: 1, q75: 1, valid_n: 1, connect_previous: false, has_internal_gap: false },
      { time: '2026-01-01T00:02', end: '2026-01-01T00:02', value: 2, q25: 2, q75: 2, valid_n: 1, connect_previous: false, has_internal_gap: false },
    ] });
    expect((traces[0] as { y: unknown }).y).toEqual([1, null, 2]);
    expect((traces[0] as { connectgaps: boolean }).connectgaps).toBe(false);
  });
  it('uses server histogram counts without browser resampling', () => {
    const trace = histogramTrace({ edges: [0, 2, 4], counts: [3, 7] }, 'test')[0] as { x: number[]; y: number[] };
    expect(trace.x).toEqual([1, 3]); expect(trace.y).toEqual([3, 7]);
  });
  it('handles 80 sensor rows independently of pagination', () => {
    expect(summaryRows(Array.from({length: 80}, (_, i) => row(`Sensor_${i}`, i)), '', 'numeric', 'mean', true)).toHaveLength(80);
  });
});

it('builds separate IQR polygons on each side of a gap', () => {
  const points = [0, 1, 4, 5].map((n, i) => ({ time: `2026-01-01T00:0${n}`, end: `2026-01-01T00:0${n}`, valid_n: 3,
    value: 2, q25: 1, q75: 3, connect_previous: i === 1 || i === 3, has_internal_gap: false }));
  const band = temporalTraces({ aggregated: true, interval_seconds: 180, points })[0] as { fill: string; x: unknown[] };
  expect(band.fill).toBe('toself');
  expect(band.x.filter(x => x === null)).toHaveLength(2);
});
