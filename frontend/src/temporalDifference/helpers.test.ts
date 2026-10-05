import { describe, expect, it } from 'vitest';
import { addDelta, defaultConfig, defaultPlot, plotData, plotLayout, validateConfig, validatePlot } from './helpers';
import type { Summary } from './types';
const period = { start: '2025-09-15T00:00:00', end: '2025-09-16T00:00:00' };
const configured = { ...defaultConfig, training_dataset_id: 1, preprocessing_pipeline_id: 2, reference: period, comparison: period };
describe('temporal selection', () => {
  it('starts with requested lags and adds positive whole seconds exactly once', () => {
    expect(defaultConfig.deltas_seconds).toEqual([1, 2, 5, 15, 30, 60]);
    expect(addDelta([60, 1], 5)).toEqual([1, 5, 60]);
    expect(addDelta([1, 5], 5)).toEqual([1, 5]);
    for (const value of [0, -1, .5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) expect(addDelta([1], value)).toEqual([1]);
  });
  it('validates periods, available bounds and nonempty unique deltas', () => {
    expect(validateConfig(configured)).toBeNull();
    expect(validateConfig(defaultConfig)).not.toBeNull();
    expect(validateConfig({ ...configured, deltas_seconds: [] })).not.toBeNull();
    expect(validateConfig({ ...configured, deltas_seconds: [2, 2] })).not.toBeNull();
    expect(validateConfig({ ...configured, comparison: { start: period.end, end: period.start } })).not.toBeNull();
    expect(validateConfig(configured, '2025-09-15T01:00:00')).not.toBeNull();
  });
  it('changes the preview signature when a preset or delta is changed', () => {
    const signature = JSON.stringify(configured);
    expect(JSON.stringify({ ...configured, reference: { ...period, start: '2025-09-15T01:00:00' } })).not.toBe(signature);
    expect(JSON.stringify({ ...configured, deltas_seconds: [1] })).not.toBe(signature);
    expect(JSON.stringify({ ...configured, preprocessing_pipeline_id: 3 })).not.toBe(signature);
  });
});
describe('persisted result plots', () => {
  const row = (delta: number, median: number | null): Summary => ({ role: 'reference', delta_seconds: delta, median, q1: median === null ? null : median - 1, q3: median === null ? null : median + 1, iqr: median === null ? null : 2, pair_count: median === null ? 0 : 4 });
  it('uses medians and asymmetric quartile endpoints without bridging unavailable lags', () => {
    const rows = [row(1, 2), row(2, null), row(5, 8)];
    const traces = plotData(rows, defaultPlot) as Array<{ x: number[]; y: (number | null)[]; name?: string; fill?: string; connectgaps: boolean }>;
    expect(traces.find(trace => trace.name === 'Referenz')?.y).toEqual([2, null, 8]);
    expect(traces.filter(trace => trace.fill === 'tonexty').map(trace => trace.x)).toEqual([[1], [5]]);
    expect(traces.every(trace => !trace.connectgaps)).toBe(true);
    expect(rows[1].median).toBeNull();
  });
  it('keeps linear numeric axes and a white export with editable ranges and titles', () => {
    const settings = { ...defaultPlot, title: '<b>Literal title</b>', x_range: { minimum: 0, maximum: 70 } };
    expect(validatePlot(settings)).toBeNull();
    const layout = plotLayout(settings);
    expect(layout.paper_bgcolor).toBe('#ffffff');
    expect(layout.xaxis).toMatchObject({ type: 'linear', range: [0, 70], autorange: false });
    expect(layout.title).toEqual({ text: '&lt;b&gt;Literal title&lt;/b&gt;' });
    for (const x_range of [{minimum: 1, maximum: 1}, {minimum: NaN, maximum: 4}]) expect(validatePlot({ ...settings, x_range })).not.toBeNull();
    expect(validatePlot({ ...settings, reference_color: 'red' })).not.toBeNull();
  });
});

describe('stratified defaults', () => {
  it('uses version 2, five minute blocks and seed 42', () => {
    expect(defaultConfig.selection_version).toBe(2);
    expect(defaultConfig.block_seconds).toBe(300);
    expect(defaultConfig.seed).toBe(42);
    expect(validateConfig({...configured,block_seconds:0})).not.toBeNull();
    expect(validateConfig({...configured,seed:NaN})).not.toBeNull();
    expect(validateConfig({...configured,seed:-1})).not.toBeNull();
    expect(validateConfig({...configured,selection_version:1,block_seconds:undefined,seed:undefined})).toBeNull();
    expect(JSON.stringify({...configured,seed:43})).not.toBe(JSON.stringify(configured));
  });
});
