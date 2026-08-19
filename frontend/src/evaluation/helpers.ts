import type { Shape } from 'plotly.js';
import type { Data } from '../lib/plotly';
import { withLineGapPolicy } from '../lib/plotGaps';
import type {
  EvaluationCalculationStatus,
  EvaluationDetectionOperatingPoint,
  EvaluationLabelEvent,
  EvaluationScorePoint,
  ModelEvaluation,
  TestingRun,
} from '../types';
import {
  aggregationKeyForRun,
  aggregationLabel,
  metricKeyForRun,
  metricLabel,
  roiKeyForRun,
  roiLabelForRun,
} from '../testing/inferenceRunMetadata';

export const EVALUATION_QUANTILES = [0.99, 0.995, 0.999, 0.9995, 0.9999] as const;
export const DEFAULT_EVALUATION_QUANTILE = 0.999;
export type TimelineSelectionMode = 'range' | 'target' | 'exclusion';

export type EvaluationRunFilters = {
  query: string;
  model: string[];
  modelTrainingDataset: string[];
  inferenceDataset: string[];
  roi: string[];
  metric: string[];
  aggregation: string[];
  preprocessing: string[];
  method: string[];
};

export function emptyEvaluationRunFilters(): EvaluationRunFilters {
  return {
    query: '',
    model: [],
    modelTrainingDataset: [],
    inferenceDataset: [],
    roi: [],
    metric: [],
    aggregation: [],
    preprocessing: [],
    method: [],
  };
}

function oneOf(selected: string[], values: string[]): boolean {
  return selected.length === 0 || values.some((value) => selected.includes(value));
}

export function evaluationRunMatches(run: TestingRun, filters: EvaluationRunFilters): boolean {
  if (run.status !== 'finished' || (run.image_count ?? 0) <= 0) return false;
  const searchable = [
    run.name,
    run.training_run_name,
    run.training_pipeline_name,
    run.training_dataset_name,
    run.preprocessing_pipeline_name,
    run.method_type,
    roiLabelForRun(run),
    metricLabel(metricKeyForRun(run)),
    aggregationLabel(aggregationKeyForRun(run)),
    ...run.model_training_dataset_names,
  ].join(' ').toLowerCase();
  if (filters.query.trim() && !searchable.includes(filters.query.trim().toLowerCase())) return false;
  return oneOf(filters.model, [String(run.training_run_id)])
    && oneOf(filters.modelTrainingDataset, run.model_training_dataset_names)
    && oneOf(filters.inferenceDataset, [String(run.training_dataset_id)])
    && oneOf(filters.roi, [roiKeyForRun(run)])
    && oneOf(filters.metric, [metricKeyForRun(run)])
    && oneOf(filters.aggregation, [aggregationKeyForRun(run)])
    && oneOf(filters.preprocessing, [run.preprocessing_pipeline_name])
    && oneOf(filters.method, [run.method_type]);
}

export function filterEvaluationRuns(runs: TestingRun[], filters: EvaluationRunFilters): TestingRun[] {
  return runs.filter((run) => evaluationRunMatches(run, filters));
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`).join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

/** Conservative client-side compatibility hint. The server repeats every check authoritatively. */
export function evaluationRunsCompatible(evaluationRun: TestingRun, candidate: TestingRun): boolean {
  if (!evaluationRun.artifact_signature || !candidate.artifact_signature) return false;
  return evaluationRun.artifact_signature === candidate.artifact_signature
    && evaluationRun.preprocessing_pipeline_name === candidate.preprocessing_pipeline_name
    && evaluationRun.method_type === candidate.method_type
    && evaluationRun.method_family === candidate.method_family
    && evaluationRun.artifact_kind === candidate.artifact_kind
    && canonicalJson(evaluationRun.roi_geometry) === canonicalJson(candidate.roi_geometry)
    && canonicalJson(evaluationRun.inference_config) === canonicalJson(candidate.inference_config);
}

export function scoreSeriesForRun(run: TestingRun | null): Array<{ value: string; label: string }> {
  if (!run) return [{ value: 'score', label: 'Score' }];
  const values = [{ value: 'score', label: 'Configured score' }, { value: 'full_mse', label: 'Full-frame MSE' }];
  if (run.roi_mse_mean !== null) values.push({ value: 'roi_mse', label: 'ROI MSE' });
  return values;
}

export function statusIsCurrent(status: EvaluationCalculationStatus): boolean {
  return ['current', 'ready', 'complete', 'completed', 'success', 'succeeded', 'fresh'].includes(status.toLowerCase());
}

export function statusIsStale(status: EvaluationCalculationStatus): boolean {
  return status.toLowerCase() === 'stale';
}

export function calculationStatusColor(status: EvaluationCalculationStatus): string {
  if (statusIsCurrent(status)) return 'green';
  if (statusIsStale(status)) return 'yellow';
  if (['failed', 'error'].includes(status.toLowerCase())) return 'red';
  return 'gray';
}

export function calculationStatusLabel(status: EvaluationCalculationStatus): string {
  if (statusIsCurrent(status)) return 'Current';
  if (statusIsStale(status)) return 'Stale';
  if (['failed', 'error'].includes(status.toLowerCase())) return 'Failed';
  return 'Not calculated';
}

export function canFinalizeEvaluation(evaluation: ModelEvaluation): boolean {
  return evaluation.status === 'draft'
    && statusIsCurrent(evaluation.separation_status)
    && statusIsCurrent(evaluation.drift_status)
    && statusIsCurrent(evaluation.detection_status);
}

export function closestOperatingPoint(
  evaluation: ModelEvaluation,
  quantile: number,
): EvaluationDetectionOperatingPoint | null {
  const points = evaluation.detection_result?.operating_points ?? [];
  if (points.length === 0) return null;
  return points.reduce((best, candidate) => (
    Math.abs(candidate.quantile - quantile) < Math.abs(best.quantile - quantile) ? candidate : best
  ));
}

export function metricValues(
  evaluation: ModelEvaluation,
  quantile: number,
): Array<{ key: string; label: string; value: number | null; unit?: string; group: 'A' | 'B' | 'C' }> {
  const operatingPoint = closestOperatingPoint(evaluation, quantile);
  return [
    { key: 'sep_median', label: 'Median separation', value: evaluation.sep_median ?? evaluation.separation_result?.sep_median ?? null, group: 'A' },
    { key: 'sep_min', label: 'Minimum separation', value: evaluation.sep_min ?? evaluation.separation_result?.sep_min ?? null, group: 'A' },
    { key: 'd_mean', label: 'Mean normalized drift', value: evaluation.drift_mean ?? evaluation.drift_result?.d_mean ?? null, group: 'B' },
    { key: 'd_max', label: 'Maximum normalized drift', value: evaluation.drift_max ?? evaluation.drift_result?.d_max ?? null, group: 'B' },
    { key: 'event_recall', label: 'Event recall', value: operatingPoint?.event_recall ?? evaluation.event_recall, unit: '%', group: 'C' },
    { key: 'median_delay', label: 'Median delay', value: operatingPoint?.median_delay_seconds ?? evaluation.median_delay_seconds, unit: 's', group: 'C' },
    { key: 'frame_fpr', label: 'Frame FPR', value: operatingPoint?.frame_fpr ?? evaluation.frame_fpr, unit: '%', group: 'C' },
    { key: 'far_t0', label: 'FAR T₀', value: operatingPoint?.far_t0 ?? evaluation.false_alarm_rate_t0, group: 'C' },
  ];
}

export function formatEvaluationMetric(value: number | null, unit?: string): string {
  if (value === null || !Number.isFinite(value)) return 'N/A';
  const displayed = unit === '%' ? value * 100 : value;
  const absolute = Math.abs(displayed);
  const formatted = absolute >= 1000 || (absolute > 0 && absolute < 0.001)
    ? displayed.toExponential(3)
    : displayed.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return `${formatted}${unit === '%' ? '%' : unit ? ` ${unit}` : ''}`;
}

export function eventShapes(events: EvaluationLabelEvent[]): Shape[] {
  return events.map((event) => ({
    type: 'rect',
    xref: 'x',
    yref: 'paper',
    x0: event.start_timestamp,
    x1: event.end_timestamp,
    y0: 0,
    y1: 1,
    fillcolor: event.type === 'target' ? 'rgba(250,82,82,0.14)' : 'rgba(134,142,150,0.16)',
    line: { color: event.type === 'target' ? '#fa5252' : '#868e96', width: 1, dash: event.type === 'target' ? 'solid' : 'dot' },
    layer: 'below',
  })) as Shape[];
}

export function scoreTrace(points: EvaluationScorePoint[], name = 'Score'): Data {
  return withLineGapPolicy({
    type: 'scatter',
    mode: 'lines',
    name,
    x: points.map((point) => point.timestamp),
    y: points.map((point) => point.value),
    customdata: points.map((point) => point.continuity_segment ?? 0),
    line: { color: '#228be6', width: 1.5 },
    hovertemplate: '%{x}<br>%{y:.6g}<extra></extra>',
  } as Data, { continuity: points.map((point) => point.continuity_segment ?? 0) });
}

export function orderedRange(start: string, end: string): { start: string; end: string } {
  return start <= end ? { start, end } : { start: end, end: start };
}

export function resolveTimelineSelection(mode: TimelineSelectionMode, start: string, end: string): {
  pendingRange: { start_timestamp: string; end_timestamp: string } | null;
  annotation: { type: 'target' | 'exclusion'; start_timestamp: string; end_timestamp: string } | null;
} {
  const ordered = orderedRange(start, end);
  const interval = { start_timestamp: ordered.start.slice(0, 19), end_timestamp: ordered.end.slice(0, 19) };
  return mode === 'range'
    ? { pendingRange: interval, annotation: null }
    : { pendingRange: null, annotation: { type: mode, ...interval } };
}

export function applyPendingTimelineSelection(
  configured: { start_timestamp: string; end_timestamp: string },
  pending: { start_timestamp: string; end_timestamp: string } | null,
): { start_timestamp: string; end_timestamp: string } {
  return pending ?? configured;
}

export function rangesOverlap(
  firstStart: string | null,
  firstEnd: string | null,
  secondStart: string | null,
  secondEnd: string | null,
): boolean {
  if (!firstStart || !firstEnd || !secondStart || !secondEnd) return false;
  return firstStart <= secondEnd && secondStart <= firstEnd;
}
