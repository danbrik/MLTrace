import { describe, expect, it } from 'vitest';

import type { ModelEvaluation, TestingRun } from '../types';
import {
  canFinalizeEvaluation,
  calculationStatusColor,
  calculationStatusLabel,
  closestOperatingPoint,
  emptyEvaluationRunFilters,
  evaluationRunsCompatible,
  evaluationRunMatches,
  formatEvaluationMetric,
  applyPendingTimelineSelection,
  orderedRange,
  rangesOverlap,
  resolveTimelineSelection,
  scoreSeriesForRun,
} from './helpers';

function run(overrides: Partial<TestingRun> = {}): TestingRun {
  return {
    id: 1,
    name: 'Bearing validation',
    training_run_id: 10,
    training_dataset_id: 20,
    roi_id: null,
    status: 'finished',
    image_count: 100,
    model_training_dataset_names: ['Healthy train'],
    training_dataset_name: 'Evaluation week',
    preprocessing_pipeline_name: 'Crop and scale',
    method_type: 'autoencoder',
    inference_config: { error_metric: 'mae', frame_score_aggregation: 'p95' },
    training_run_name: 'AE #10',
    training_pipeline_name: 'Bearing AE',
    roi_name: null,
    roi_geometry: null,
    enqueued_at: null,
    queue_rank: null,
    started_at: null,
    ended_at: null,
    duration_seconds: null,
    gpu_index: null,
    device: null,
    error_message: null,
    expected_image_count: 100,
    skipped_image_count: 0,
    skipped_images: null,
    score_mean: null,
    score_min: null,
    score_max: null,
    full_mse_mean: null,
    roi_mse_mean: null,
    results_path: 'results.csv',
    results_size_bytes: 10,
    checkpoint_at: null,
    checkpoint_input_count: null,
    checkpoint_result_count: null,
    restart_mode: null,
    method_family: 'reconstruction',
    training_mode: 'standard',
    artifact_kind: 'torch',
    artifact_path: 'model.pt',
    created_at: '2026-01-01T00:00:00',
    updated_at: '2026-01-01T00:00:00',
    ...overrides,
  };
}

describe('evaluation helpers', () => {
  it('filters runs using search and independent facets', () => {
    const filters = emptyEvaluationRunFilters();
    filters.query = 'bearing';
    filters.metric = ['mae'];
    filters.aggregation = ['p95'];
    expect(evaluationRunMatches(run(), filters)).toBe(true);
    filters.roi = ['4'];
    expect(evaluationRunMatches(run(), filters)).toBe(false);
  });

  it('does not offer unfinished or empty inference runs', () => {
    const filters = emptyEvaluationRunFilters();
    expect(evaluationRunMatches(run({ status: 'running' }), filters)).toBe(false);
    expect(evaluationRunMatches(run({ image_count: 0 }), filters)).toBe(false);
  });

  it('offers ROI MSE from stored score availability rather than a live ROI id', () => {
    expect(scoreSeriesForRun(run({ roi_id: null, roi_mse_mean: 0.2 })).map((option) => option.value)).toContain('roi_mse');
    expect(scoreSeriesForRun(run({ roi_id: 4, roi_mse_mean: null })).map((option) => option.value)).not.toContain('roi_mse');
  });

  it('only advertises sources whose complete model and score semantics match', () => {
    const evaluation = run({ artifact_signature: 'sha256:abc', roi_geometry: { x: 1, y: 2 } });
    expect(evaluationRunsCompatible(evaluation, run({ id: 2, artifact_signature: 'sha256:abc', roi_geometry: { y: 2, x: 1 } }))).toBe(true);
    expect(evaluationRunsCompatible(evaluation, run({ id: 3, artifact_signature: null, roi_geometry: { x: 1, y: 2 } }))).toBe(false);
    expect(evaluationRunsCompatible(evaluation, run({ id: 4, artifact_signature: 'sha256:abc', roi_geometry: { x: 2, y: 2 } }))).toBe(false);
  });

  it('treats touching inclusive role ranges as overlapping', () => {
    expect(rangesOverlap('2026-01-01T00:00:00', '2026-01-01T01:00:00', '2026-01-01T01:00:00', '2026-01-01T02:00:00')).toBe(true);
    expect(rangesOverlap('2026-01-01T00:00:00', '2026-01-01T01:00:01', '2026-01-01T01:00:00', '2026-01-01T02:00:00')).toBe(true);
    expect(orderedRange('b', 'a')).toEqual({ start: 'a', end: 'b' });
  });

  it('keeps a reversed drag pending until explicit apply', () => {
    const configured = { start_timestamp: '2026-01-01T00:00:00', end_timestamp: '2026-01-01T01:00:00' };
    const decision = resolveTimelineSelection('range', '2026-01-02T02:00:00', '2026-01-02T01:00:00');
    expect(decision).toEqual({ pendingRange: { start_timestamp: '2026-01-02T01:00:00', end_timestamp: '2026-01-02T02:00:00' }, annotation: null });
    expect(configured).toEqual({ start_timestamp: '2026-01-01T00:00:00', end_timestamp: '2026-01-01T01:00:00' });
    expect(applyPendingTimelineSelection(configured, decision.pendingRange)).toEqual(decision.pendingRange);
  });

  it('routes annotation drags without changing the configured range', () => {
    expect(resolveTimelineSelection('target', '2026-01-01T01:00:00', '2026-01-01T02:00:00')).toEqual({
      pendingRange: null,
      annotation: { type: 'target', start_timestamp: '2026-01-01T01:00:00', end_timestamp: '2026-01-01T02:00:00' },
    });
  });

  it('maps current, stale and error calculation badges', () => {
    expect([calculationStatusLabel('current'), calculationStatusColor('current')]).toEqual(['Current', 'green']);
    expect([calculationStatusLabel('stale'), calculationStatusColor('stale')]).toEqual(['Stale', 'yellow']);
    expect([calculationStatusLabel('error'), calculationStatusColor('error')]).toEqual(['Failed', 'red']);
  });

  it('selects the closest available detection operating point', () => {
    const evaluation = {
      detection_result: {
        operating_points: [
          { quantile: 0.99 },
          { quantile: 0.999 },
        ],
      },
    } as unknown as ModelEvaluation;
    expect(closestOperatingPoint(evaluation, 0.998)?.quantile).toBe(0.999);
  });

  it('requires all three independent calculations before finalization', () => {
    const evaluation = { status: 'draft', separation_status: 'ready', drift_status: 'complete', detection_status: 'success' } as ModelEvaluation;
    expect(canFinalizeEvaluation(evaluation)).toBe(true);
    expect(canFinalizeEvaluation({ ...evaluation, drift_status: 'stale' })).toBe(false);
    expect(formatEvaluationMetric(null)).toBe('N/A');
    expect(formatEvaluationMetric(0.125, '%')).toBe('12.5%');
  });
});
