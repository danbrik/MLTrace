import type { TestingRun } from '../types';

export function normalizeMetricKey(metric: string): string {
  const normalized = metric.trim().toLowerCase();
  if (normalized === 'ssim' || normalized === 'ssim_distance') return 'ssim_distance';
  if (normalized === 'l1' || normalized === 'mae') return 'mae';
  if (normalized === 'l2' || normalized === 'mse') return 'mse';
  return normalized || 'mse';
}

export function metricKeyForRun(run: TestingRun): string {
  const configured = run.inference_config?.error_metric ?? run.inference_config?.residual_metric;
  return typeof configured === 'string' && configured.trim() ? normalizeMetricKey(configured) : 'mse';
}

export function metricLabel(metric: string): string {
  const normalized = normalizeMetricKey(metric);
  if (normalized === 'mse') return 'MSE';
  if (normalized === 'mae') return 'MAE';
  if (normalized === 'ssim_distance') return 'SSIM';
  return normalized.replaceAll('_', ' ').toUpperCase();
}

export function metricOrder(metric: string): number {
  const normalized = normalizeMetricKey(metric);
  if (normalized === 'mse') return 0;
  if (normalized === 'mae') return 1;
  if (normalized === 'ssim_distance') return 2;
  return 10;
}

export function normalizeAggregationKey(aggregation: string): string {
  return aggregation.trim().toLowerCase() || 'mean';
}

export function aggregationKeyForRun(run: TestingRun): string {
  const configured = run.inference_config?.frame_score_aggregation;
  return typeof configured === 'string' && configured.trim() ? normalizeAggregationKey(configured) : 'mean';
}

export function aggregationLabel(aggregation: string): string {
  const normalized = normalizeAggregationKey(aggregation);
  if (normalized === 'mean') return 'Mean';
  if (normalized === 'max') return 'Max';
  return normalized.toUpperCase();
}

export function aggregationOrder(aggregation: string): number {
  const normalized = normalizeAggregationKey(aggregation);
  if (normalized === 'mean') return -1;
  if (normalized === 'max') return 1000;
  const percentile = Number(normalized.replace(/^p/, ''));
  return Number.isFinite(percentile) ? percentile : 500;
}

export function roiKeyForRun(run: TestingRun): string {
  return run.roi_id === null ? 'none' : String(run.roi_id);
}

export function roiLabelForRun(run: TestingRun): string {
  return run.roi_id === null ? 'No ROI' : run.roi_name ?? `ROI #${run.roi_id}`;
}
