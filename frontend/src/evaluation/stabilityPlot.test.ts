import { describe, expect, it } from 'vitest';
import type { Data } from '../lib/plotly';
import { preparePlotData, withLineGapPolicy } from '../lib/plotGaps';
import type { EvaluationDriftCalculation } from '../types';
import {
  buildDriftPlotSeries,
  SCORE_STABILITY_EXPLANATION_DEFAULT_OPEN,
  SCORE_STABILITY_HELP,
} from './stabilityPlot';

type Bucket = EvaluationDriftCalculation['buckets'][number];

function bucket(
  id: number,
  start: string,
  end: string,
  value: number | null,
  included = true,
  status = 'ready',
): Bucket {
  return {
    id,
    bucket_key: `bucket-${id}`,
    start_timestamp: start,
    end_timestamp: end,
    decision: included ? 'include' : 'drop_bucket',
    original_point_count: 10,
    used_point_count: included ? 10 : 0,
    exclusion_overlap: !included,
    status,
    reason: included ? null : 'removed',
    wasserstein_1: value,
    normalized_drift: value,
    included,
  };
}

describe('Score Stability plot preparation', () => {
  it('sorts buckets and keeps excluded or invalid intervals as null breaks', () => {
    const result = buildDriftPlotSeries([
      bucket(2, '2026-01-01T01:00:00Z', '2026-01-01T02:00:00Z', 2, false),
      bucket(1, '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 1),
      bucket(3, '2026-01-01T02:00:00Z', '2026-01-01T03:00:00Z', null, false, 'excluded'),
    ]);
    expect(result.x).toEqual([
      '2026-01-01T00:00:00Z',
      '2026-01-01T01:00:00Z',
      '2026-01-01T02:00:00Z',
    ]);
    expect(result.y).toEqual([1, null, null]);
    expect(result.continuity).toEqual([0, 0, 0]);
  });

  it('adds an explicit line break between non-adjacent bucket periods', () => {
    const series = buildDriftPlotSeries([
      bucket(1, '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', 1),
      bucket(2, '2026-01-01T03:00:00Z', '2026-01-01T04:00:00Z', 2),
    ]);
    const trace = withLineGapPolicy({ type: 'scatter', mode: 'lines+markers', x: series.x, y: series.y } as Data, {
      continuity: series.continuity,
    });
    const prepared = preparePlotData([trace])[0] as unknown as { y: Array<number | null>; connectgaps: boolean };
    expect(series.continuity).toEqual([0, 1]);
    expect(prepared.y).toEqual([1, null, 2]);
    expect(prepared.connectgaps).toBe(false);
  });

  it('keeps the requested guidance visible by default and complete', () => {
    expect(SCORE_STABILITY_EXPLANATION_DEFAULT_OPEN).toBe(true);
    expect(SCORE_STABILITY_HELP.reference).toContain('normal baseline');
    expect(SCORE_STABILITY_HELP.analysis).toContain('fixed, non-overlapping buckets');
    expect(SCORE_STABILITY_HELP.exclusion).toContain('affected points');
    expect(SCORE_STABILITY_HELP.bucketDuration).toContain('incomplete final remainder');
  });
});
