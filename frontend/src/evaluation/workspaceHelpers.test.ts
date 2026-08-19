import { describe, expect, it } from 'vitest';
import {
  normalizeHorizontalSelection,
  separationPairsEqual,
  toDatasetLocalTimestamp,
  upsertBucketDecision,
} from './workspaceHelpers';

const pair = (overrides: Partial<Record<string, string>> = {}) => ({
  pair_key: 'p1', name: 'Fault',
  normal_start: '2026-01-01T09:00:00', normal_end: '2026-01-01T09:20:00',
  anomaly_start: '2026-01-01T10:00:00', anomaly_end: '2026-01-01T10:10:00',
  ...overrides,
});

describe('model evaluation workspace helpers', () => {
  it('normalizes a right-to-left plot drag', () => {
    expect(normalizeHorizontalSelection({ start: '2026-01-02T10:00:00.000', end: '2026-01-01T09:00:00.000' })).toEqual({
      start: '2026-01-01T09:00:00', end: '2026-01-02T10:00:00',
    });
  });

  it('converts the space separated plot format into a form the date inputs accept', () => {
    expect(toDatasetLocalTimestamp('2026-08-19 12:34:56.7891')).toBe('2026-08-19T12:34:56');
  });

  it('pads the parts plotly omits at coarse zoom levels', () => {
    expect(toDatasetLocalTimestamp('2026-08-19')).toBe('2026-08-19T00:00:00');
    expect(toDatasetLocalTimestamp('2026-08-19 12:34')).toBe('2026-08-19T12:34:00');
  });

  it('orders a reversed drag after normalizing mixed precision bounds', () => {
    expect(normalizeHorizontalSelection({ start: '2026-08-19 12:34', end: '2026-08-19' })).toEqual({
      start: '2026-08-19T00:00:00', end: '2026-08-19T12:34:00',
    });
  });

  it('returns an empty string for an unparsable value', () => {
    expect(toDatasetLocalTimestamp('')).toBe('');
    expect(toDatasetLocalTimestamp('not a timestamp')).toBe('');
  });

  it('treats a just-saved layout as clean despite the server timestamp format', () => {
    expect(separationPairsEqual(
      [pair({ normal_start: '2026-01-01 09:00:00' })],
      [pair({ normal_start: '2026-01-01T09:00:00' })],
    )).toBe(true);
  });

  it('still detects a real edit', () => {
    expect(separationPairsEqual([pair()], [pair({ anomaly_end: '2026-01-01T10:11:00' })])).toBe(false);
    expect(separationPairsEqual([pair()], [])).toBe(false);
  });

  it('changes only the selected persisted bucket decision', () => {
    expect(upsertBucketDecision([
      { bucket_key: 'a', decision: 'include' }, { bucket_key: 'b', decision: 'include' },
    ], 'b', 'filter_points')).toEqual([
      { bucket_key: 'a', decision: 'include' }, { bucket_key: 'b', decision: 'filter_points' },
    ]);
  });
});
