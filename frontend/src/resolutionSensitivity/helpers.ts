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
