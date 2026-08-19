import { describe, expect, it } from 'vitest';

import type { ModelEvaluationPayload } from '../types';
import { projectEvaluationDraft, projectedCalculationStatus } from './draftFreshness';

function payload(): ModelEvaluationPayload {
  return {
    name: 'Evaluation', evaluation_testing_run_id: 1, evaluation_start_timestamp: '2026-01-01T00:00:00', evaluation_end_timestamp: '2026-01-01T01:00:00',
    reference_testing_run_id: 2, reference_start_timestamp: '2026-01-02T00:00:00', reference_end_timestamp: '2026-01-02T01:00:00',
    calibration_testing_run_id: 3, calibration_start_timestamp: '2026-01-03T00:00:00', calibration_end_timestamp: '2026-01-03T01:00:00',
    score_series: 'score', label_set_id: 4, profile_id: 5, selected_categories: ['bearing'], normal_window_overrides: {}, profile_overrides: {}, active_quantile: 0.999,
  };
}

describe('evaluation draft freshness projection', () => {
  it('projects evaluation range changes to all stages before save', () => {
    const baseline = payload();
    const projection = projectEvaluationDraft(baseline, { ...baseline, evaluation_end_timestamp: '2026-01-01T02:00:00' });
    expect(projection.dirty).toBe(true);
    expect([...projection.changedStages].sort()).toEqual(['detection', 'drift', 'separation']);
  });

  it('projects role and local profile changes only to dependent stages', () => {
    const baseline = payload();
    expect([...projectEvaluationDraft(baseline, { ...baseline, reference_start_timestamp: '2026-01-02T00:30:00' }).changedStages]).toEqual(['drift']);
    expect([...projectEvaluationDraft(baseline, { ...baseline, profile_overrides: { anticipation_seconds: 30 } }).changedStages]).toEqual(['detection']);
    expect([...projectEvaluationDraft(baseline, { ...baseline, profile_overrides: { epsilon: 1e-9 } }).changedStages].sort()).toEqual(['drift', 'separation']);
  });

  it('keeps category ordering semantically stable and treats quantile as dirty without staling C', () => {
    const baseline = { ...payload(), selected_categories: ['bearing', 'motor'] };
    expect(projectEvaluationDraft(baseline, { ...baseline, selected_categories: ['motor', 'bearing'] }).dirty).toBe(false);
    expect(projectEvaluationDraft(baseline, { ...baseline, active_quantile: 0.9999 }).changedStages.size).toBe(0);
    expect(projectedCalculationStatus('current', true)).toBe('stale');
    expect(projectedCalculationStatus('not_calculated', true)).toBe('not_calculated');
  });
});
