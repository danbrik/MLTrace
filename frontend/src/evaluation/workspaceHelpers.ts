import type { EvaluationSeparationPair } from '../types';

export type HorizontalSelection = { start: string; end: string };

const TIMESTAMP = /^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2})(?::(\d{2}))?(?::(\d{2}))?)?/;

/**
 * Plotly reports axis ranges as local date strings with a space separator and a
 * precision that depends on the zoom level ("2026-08-19", "2026-08-19 12:34",
 * "2026-08-19 12:34:56.789").  Persisted ranges and `DateTime24Input` both expect
 * a full `YYYY-MM-DDTHH:mm:ss`, so pad the missing parts instead of slicing.
 */
export function toDatasetLocalTimestamp(value: string): string {
  const match = TIMESTAMP.exec(value.trim());
  if (!match) return '';
  const [, date, hours = '00', minutes = '00', seconds = '00'] = match;
  return `${date}T${hours}:${minutes}:${seconds}`;
}

/** Plotly can report a right-to-left drag; persisted ranges are always ordered. */
export function normalizeHorizontalSelection(selection: HorizontalSelection): HorizontalSelection {
  const start = toDatasetLocalTimestamp(selection.start);
  const end = toDatasetLocalTimestamp(selection.end);
  return start <= end ? { start, end } : { start: end, end: start };
}

function normalizedPair(pair: EvaluationSeparationPair): EvaluationSeparationPair {
  return {
    pair_key: pair.pair_key,
    name: pair.name,
    normal_start: toDatasetLocalTimestamp(pair.normal_start),
    normal_end: toDatasetLocalTimestamp(pair.normal_end),
    anomaly_start: toDatasetLocalTimestamp(pair.anomaly_start),
    anomaly_end: toDatasetLocalTimestamp(pair.anomaly_end),
  };
}

export function normalizePairs(pairs: EvaluationSeparationPair[]): EvaluationSeparationPair[] {
  return pairs.map(normalizedPair);
}

/**
 * Compares an edited layout against its persisted counterpart.  The server echoes
 * timestamps in ISO form, so a raw JSON comparison would keep a just-saved layout
 * marked dirty and block the calculation.
 */
export function separationPairsEqual(left: EvaluationSeparationPair[], right: EvaluationSeparationPair[]): boolean {
  return JSON.stringify(normalizePairs(left)) === JSON.stringify(normalizePairs(right));
}

export function upsertBucketDecision<T extends { bucket_key: string; decision: string }>(
  buckets: T[], bucketKey: string, decision: 'include' | 'drop_bucket' | 'filter_points',
): T[] {
  return buckets.map((bucket) => bucket.bucket_key === bucketKey ? { ...bucket, decision } : bucket);
}
