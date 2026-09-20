import { describe, expect, it } from 'vitest';
import { intervalError, sortIntervals, timestampKey } from './intervals';
import type { TimeInterval } from './types';
const bounds = { start: '2026-01-01T00:00:00Z', end: '2026-01-01T00:00:09Z' };
const range = (id: string, start: string, end: string, subset: TimeInterval['subset'] = 'train'): TimeInterval => ({ id, start: `2026-01-01T00:00:${start}Z`, end: `2026-01-01T00:00:${end}Z`, subset, tags: [] });
describe('inclusive temporal partitions', () => {
  it('rejects overlap across subsets, containment, and a shared boundary', () => {
    const existing = [range('a', '02', '04')];
    for (const [start, end] of [['01', '02'], ['04', '05'], ['02', '04'], ['03', '03'], ['00', '09']]) {
      expect(intervalError(range('b', start, end, 'test'), existing, bounds)).toContain('zugeordnet');
    }
  });
  it('allows self edits and disjoint intervals', () => {
    expect(intervalError(range('a', '02', '04'), [range('a', '02', '04')], bounds)).toBeNull();
    expect(intervalError(range('b', '04.000000001', '05'), [range('a', '02', '04')], bounds)).toBeNull();
  });
  it('validates boundaries and reversed ranges', () => {
    expect(intervalError(range('a', '04', '02'), [], bounds)).toContain('Beginn');
    expect(intervalError(range('a', '00', '10'), [], bounds)).toContain('innerhalb');
    expect(timestampKey('bad')).toBeNull();
    expect(timestampKey('2026-02-30T00:00:00Z')).toBeNull();
    expect(timestampKey('2026-02-29T00:00:00Z')).toBeNull();
    expect(timestampKey('2024-02-29T00:00:00Z')).not.toBeNull();
    expect(timestampKey('2026-01-01T24:00:00Z')).toBeNull();
  });
  it('normalizes timezone offsets and keeps nanosecond precision', () => {
    expect(timestampKey('2026-01-01T01:00:00+01:00')).toBe(timestampKey(bounds.start));
    expect(timestampKey('2026-01-01T00:00:00.000000001Z')! - timestampKey(bounds.start)!).toBe(1n);
    expect(timestampKey('2026-01-01T00:00')).toBe(timestampKey(bounds.start));
  });
  it('sorts without mutating saved records', () => {
    const records = [range('b', '04', '05'), range('a', '00', '01')];
    expect(sortIntervals(records).map((item) => item.id)).toEqual(['a', 'b']);
    expect(records[0].id).toBe('b');
  });
});
