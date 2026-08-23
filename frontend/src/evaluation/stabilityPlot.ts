import type { EvaluationDriftCalculation } from '../types';

type DriftBucket = EvaluationDriftCalculation['buckets'][number];

export type DriftPlotSeries = {
  x: string[];
  y: Array<number | null>;
  continuity: number[];
};

/**
 * Keep every bucket on the time axis. A bucket without an included, valid
 * result becomes a null point so Plotly cannot bridge the omitted interval.
 */
export function buildDriftPlotSeries(buckets: DriftBucket[]): DriftPlotSeries {
  const ordered = [...buckets].sort((left, right) => {
    const delta = new Date(left.start_timestamp).getTime() - new Date(right.start_timestamp).getTime();
    return delta || left.bucket_key.localeCompare(right.bucket_key);
  });
  let segment = 0;
  let previousEnd: number | null = null;
  const continuity: number[] = [];
  const x: string[] = [];
  const y: Array<number | null> = [];

  ordered.forEach((bucket) => {
    const start = new Date(bucket.start_timestamp).getTime();
    if (previousEnd !== null && (!Number.isFinite(start) || start !== previousEnd)) segment += 1;
    const value = bucket.normalized_drift;
    x.push(bucket.start_timestamp);
    y.push(bucket.included && bucket.status === 'ready' && value !== null && Number.isFinite(value) ? value : null);
    continuity.push(segment);
    const end = new Date(bucket.end_timestamp).getTime();
    previousEnd = Number.isFinite(end) ? end : null;
  });

  return { x, y, continuity };
}

export const SCORE_STABILITY_EXPLANATION_DEFAULT_OPEN = true;

export const SCORE_STABILITY_HELP = {
  reference: 'Defines the normal baseline distribution. It is not included in D_mean or D_max.',
  analysis: 'Is split from its start into fixed, non-overlapping buckets.',
  exclusion: 'Marks maintenance or invalid periods. Choose whether overlapping buckets are removed or only affected points are filtered.',
  bucketDuration: 'Length of each half-open bucket [start, end). An incomplete final remainder is excluded.',
} as const;
