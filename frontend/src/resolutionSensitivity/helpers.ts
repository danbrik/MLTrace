import { compareLocalTimes, rangeProblem } from '../timeRangePresets/helpers';
import type { EvaluationLabelSet, PreprocessingPipeline, ResolutionSensitivityInterval } from '../types';

export const REQUIRED_RESOLUTIONS = [840, 512, 256, 128] as const;

export function mapSelectedPipelines(
  selectedIds: string[], pipelines: PreprocessingPipeline[],
): Map<number, PreprocessingPipeline> {
  const mapping = new Map<number, PreprocessingPipeline>();
  for (const id of selectedIds.map(Number)) {
    const pipeline = pipelines.find((item) => item.id === id);
    if (pipeline?.output_width != null) mapping.set(pipeline.output_width, pipeline);
  }
  return mapping;
}

export function hasRequiredPipelineMapping(mapping: Map<number, PreprocessingPipeline>): boolean {
  return mapping.size === REQUIRED_RESOLUTIONS.length
    && REQUIRED_RESOLUTIONS.every((resolution) => mapping.has(resolution));
}

export function targetIntervalsFromLabelSet(labelSet: EvaluationLabelSet): ResolutionSensitivityInterval[] {
  return labelSet.events.filter((event) => event.type === 'target').map((event) => ({
    id: `label-${event.event_id}`,
    name: event.name || event.event_id,
    type: 'event',
    start: event.start_timestamp.slice(0, 19),
    end: event.end_timestamp.slice(0, 19),
  }));
}

/** Inclusive interval validation shared by inline errors and the start guard. */
export function validateResolutionIntervals(intervals: ResolutionSensitivityInterval[], min?: string, max?: string) {
  const errors: Record<string, string[]> = {};
  const add = (id: string, message: string) => { (errors[id] ??= []).push(message); };
  for (const item of intervals) {
    if (!item.name.trim()) add(item.id, 'Bitte einen Namen angeben.');
    const problem = rangeProblem(item, min, max);
    if (problem) add(item.id, problem);
  }
  for (let i = 0; i < intervals.length; i++) {
    const a = intervals[i];
    if (rangeProblem(a)) continue;
    for (const b of intervals.slice(i + 1)) {
      if (rangeProblem(b)) continue;
      if (compareLocalTimes(a.start, b.end) <= 0 && compareLocalTimes(b.start, a.end) <= 0) {
        add(a.id, `Überschneidung mit „${b.name}“. Beginn und Ende zählen mit.`);
        add(b.id, `Überschneidung mit „${a.name}“. Beginn und Ende zählen mit.`);
      }
    }
  }
  const formError = !intervals.some(item => item.type === 'normal') || !intervals.some(item => item.type === 'event')
    ? 'Mindestens einen Normalzeitraum und einen Anomalie-/Ereigniszeitraum hinzufügen.'
    : intervals.length > 200 ? 'Höchstens 200 Zeiträume sind erlaubt.' : null;
  return { errors, formError, valid: !formError && Object.keys(errors).length === 0 };
}
