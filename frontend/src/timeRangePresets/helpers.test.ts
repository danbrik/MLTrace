import { describe, expect, it } from 'vitest';
import { copyTimeRange, rangeProblem, formatRangeDuration } from './helpers';

const range = { start: '2026-01-01T10:00:00', end: '2026-01-01T11:00:00' };
describe('time range copying and wall-clock bounds', () => {
  it('copies only times into either role and leaves all other settings untouched', () => {
    const source = { ...range, id: 23, name: 'Saved' };
    const config = {
      training_dataset_id: 7, preprocessing_pipeline_id: 2,
      reference: { start: '', end: '', mode: 'random', random_count: 99, seed: 42, sampling_rate: 15 },
      anomaly: { start: '', end: '', sampling_rate: 3 },
    };
    const before = JSON.stringify(config);
    const copied = { ...config, reference: copyTimeRange(config.reference, source), anomaly: copyTimeRange(config.anomaly, source) };
    expect(copied).toEqual({ ...config, reference: { ...config.reference, ...range }, anomaly: { ...config.anomaly, ...range } });
    expect(JSON.stringify(config)).toBe(before);
    source.start = '2026-02-01T00:00:00';
    expect(copied.reference.start).toBe(range.start);
  });
  it('accepts equal inclusive bounds and normalizes fractional precision', () => {
    expect(rangeProblem(range, range.start, range.end + '.000000')).toBeNull();
    expect(rangeProblem({ start: range.start, end: range.start })).toBeNull();
    expect(rangeProblem({ start: '2026-01-01T10:00', end: range.end }, range.start)).toBeNull();
  });
  it('rejects empty, invalid, reversed, and out-of-bounds ranges without clipping', () => {
    for (const input of [{ ...range, start: '' }, { ...range, start: '2026-02-30T10:00:00' }, { start: range.end, end: range.start }]) expect(rangeProblem(input)).toBeTruthy();
    expect(rangeProblem(range, '2026-01-01T10:00:00.000001')).toContain('Außerhalb');
    expect(rangeProblem(range, undefined, '2026-01-01T10:59:59')).toContain('Außerhalb');
    expect(range.start).toBe('2026-01-01T10:00:00');
  });
});

it('shows wall-clock duration without DST conversion, including zero and microseconds', () => {
  expect(formatRangeDuration({ start: '2026-03-29T01:00:00', end: '2026-03-29T03:00:00' })).toBe('2 h 0 s');
  expect(formatRangeDuration({ start: range.start, end: range.start })).toBe('0 s');
  expect(formatRangeDuration({ start: '2026-01-01T00:00:00.000001', end: '2026-01-02T01:02:03.000003' })).toBe('1 d 1 h 2 min 3,000002 s');
  expect(formatRangeDuration({ start: '', end: '' })).toBe('—');
});
