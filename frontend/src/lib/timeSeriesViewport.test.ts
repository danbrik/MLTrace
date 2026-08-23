import { describe, expect, it } from 'vitest';
import type { PlotRelayoutEvent } from './plotly';
import {
  medianPositiveTimeDelta,
  paddedAxisRange,
  relayoutRange,
  visibleAxisRanges,
  type TimeSeriesTraceValues,
} from './timeSeriesViewport';

const traces: TimeSeriesTraceValues[] = [{
  x: [
    '2026-01-01T00:00:00Z',
    '2026-01-01T01:00:00Z',
    '2026-01-01T02:00:00Z',
    '2026-01-01T03:00:00Z',
  ],
  y: [0, 10, Number.NaN, 30],
  yaxis: 'y',
}];

describe('time-series viewport ranges', () => {
  it('pads the finite values in the visible x window by five percent', () => {
    const start = new Date('2026-01-01T00:30:00Z').getTime();
    const end = new Date('2026-01-01T03:00:00Z').getTime();
    expect(visibleAxisRanges(traces, [start, end])).toEqual({ y: [9, 31] });
  });

  it('uses a stable non-zero padding for constant values', () => {
    expect(paddedAxisRange(5, 5)).toEqual([4.75, 5.25]);
    expect(paddedAxisRange(0, 0)).toEqual([-1e-9, 1e-9]);
  });

  it('returns no replacement range when the viewport has no finite samples', () => {
    const start = new Date('2026-01-01T01:30:00Z').getTime();
    const end = new Date('2026-01-01T02:30:00Z').getTime();
    expect(visibleAxisRanges(traces, [start, end])).toEqual({});
  });

  it('parses Plotly zoom, range-slider, and autorange relayout shapes', () => {
    const direct = { 'xaxis.range': ['2026-01-01T03:00:00Z', '2026-01-01T01:00:00Z'] } as unknown as PlotRelayoutEvent;
    const split = { 'xaxis.range[0]': '2026-01-01T00:00:00Z', 'xaxis.range[1]': '2026-01-01T02:00:00Z' } as unknown as PlotRelayoutEvent;
    expect(relayoutRange(direct, 'xaxis')).toEqual([
      new Date('2026-01-01T01:00:00Z').getTime(),
      new Date('2026-01-01T03:00:00Z').getTime(),
    ]);
    expect(relayoutRange(split, 'xaxis')).toEqual([
      new Date('2026-01-01T00:00:00Z').getTime(),
      new Date('2026-01-01T02:00:00Z').getTime(),
    ]);
    expect(relayoutRange({ 'xaxis.autorange': true } as unknown as PlotRelayoutEvent, 'xaxis')).toBeNull();
  });

  it('preserves the Analysis cadence helper while sharing the viewport code', () => {
    expect(medianPositiveTimeDelta([
      '2026-01-01T00:00:00Z',
      '2026-01-01T00:00:02Z',
      '2026-01-01T00:00:06Z',
    ])).toBe(3000);
  });
});
