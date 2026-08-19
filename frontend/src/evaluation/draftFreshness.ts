import type { EvaluationCalculationStatus, ModelEvaluationPayload } from '../types';

export type EvaluationStage = 'separation' | 'drift' | 'detection';

export type EvaluationDraftProjection = {
  dirty: boolean;
  changedStages: Set<EvaluationStage>;
};

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

function differs(left: unknown, right: unknown): boolean {
  return canonical(left) !== canonical(right);
}

export function projectEvaluationDraft(
  baseline: ModelEvaluationPayload | null,
  draft: ModelEvaluationPayload,
): EvaluationDraftProjection {
  if (!baseline) return { dirty: true, changedStages: new Set() };
  const changedStages = new Set<EvaluationStage>();
  const allStageFields: Array<keyof ModelEvaluationPayload> = [
    'evaluation_testing_run_id', 'score_series', 'evaluation_start_timestamp', 'evaluation_end_timestamp', 'label_set_id',
  ];
  if (allStageFields.some((field) => differs(baseline[field], draft[field]))
    || differs([...baseline.selected_categories].sort(), [...draft.selected_categories].sort())) {
    changedStages.add('separation'); changedStages.add('drift'); changedStages.add('detection');
  }
  if (differs(baseline.normal_window_overrides, draft.normal_window_overrides)) changedStages.add('separation');
  if (['reference_testing_run_id', 'reference_start_timestamp', 'reference_end_timestamp'].some((field) => differs(baseline[field as keyof ModelEvaluationPayload], draft[field as keyof ModelEvaluationPayload]))) changedStages.add('drift');
  if (['calibration_testing_run_id', 'calibration_start_timestamp', 'calibration_end_timestamp'].some((field) => differs(baseline[field as keyof ModelEvaluationPayload], draft[field as keyof ModelEvaluationPayload]))) changedStages.add('detection');
  if (baseline.profile_id !== draft.profile_id) {
    changedStages.add('separation'); changedStages.add('drift'); changedStages.add('detection');
  }
  const profileDependencies: Record<EvaluationStage, string[]> = {
    separation: ['normal_window_duration_seconds', 'normal_window_buffer_seconds', 'epsilon'],
    drift: ['drift_window_seconds', 'epsilon'],
    detection: ['false_alarm_horizon_seconds', 'anticipation_seconds'],
  };
  (Object.entries(profileDependencies) as Array<[EvaluationStage, string[]]>).forEach(([stage, fields]) => {
    if (fields.some((field) => differs(baseline.profile_overrides[field], draft.profile_overrides[field]))) changedStages.add(stage);
  });
  const dirty = differs(
    { ...baseline, selected_categories: [...baseline.selected_categories].sort() },
    { ...draft, selected_categories: [...draft.selected_categories].sort() },
  );
  return { dirty, changedStages };
}

export function projectedCalculationStatus(
  status: EvaluationCalculationStatus,
  changed: boolean,
): EvaluationCalculationStatus {
  if (!changed || status === 'not_calculated') return status;
  return 'stale';
}
