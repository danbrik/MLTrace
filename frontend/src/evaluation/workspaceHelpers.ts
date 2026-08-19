export type HorizontalSelection = { start: string; end: string };

/** Plotly can report a right-to-left drag; persisted ranges are always ordered. */
export function normalizeHorizontalSelection(selection: HorizontalSelection): HorizontalSelection {
  return selection.start <= selection.end
    ? { start: selection.start.slice(0, 19), end: selection.end.slice(0, 19) }
    : { start: selection.end.slice(0, 19), end: selection.start.slice(0, 19) };
}

export function upsertBucketDecision<T extends { bucket_key: string; decision: string }>(
  buckets: T[], bucketKey: string, decision: 'include' | 'drop_bucket' | 'filter_points',
): T[] {
  return buckets.map((bucket) => bucket.bucket_key === bucketKey ? { ...bucket, decision } : bucket);
}
