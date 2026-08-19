import { describe, expect, it } from 'vitest';
import { normalizeHorizontalSelection, upsertBucketDecision } from './workspaceHelpers';

describe('model evaluation workspace helpers', () => {
  it('normalizes a right-to-left plot drag', () => {
    expect(normalizeHorizontalSelection({ start: '2026-01-02T10:00:00.000', end: '2026-01-01T09:00:00.000' })).toEqual({
      start: '2026-01-01T09:00:00', end: '2026-01-02T10:00:00',
    });
  });

  it('changes only the selected persisted bucket decision', () => {
    expect(upsertBucketDecision([
      { bucket_key: 'a', decision: 'include' }, { bucket_key: 'b', decision: 'include' },
    ], 'b', 'filter_points')).toEqual([
      { bucket_key: 'a', decision: 'include' }, { bucket_key: 'b', decision: 'filter_points' },
    ]);
  });
});
