import { describe, expect, it } from 'vitest';

import type { EvaluationLabelSet, PreprocessingPipeline } from '../types';
import { hasRequiredPipelineMapping, mapSelectedPipelines, targetIntervalsFromLabelSet } from './helpers';

function pipeline(id: number, resolution: number): PreprocessingPipeline {
  return {
    id, name: `p${resolution}`, description: null, graph: { nodes: [], edges: [] }, preview_folder_id: null,
    input_width: 1000, input_height: 1000, output_width: resolution, output_height: resolution,
    created_at: '', updated_at: '', is_update_locked: false, update_lock_reasons: [],
  };
}

describe('resolution sensitivity helpers', () => {
  it('recognizes exactly one selected pipeline for every required resolution', () => {
    const pipelines = [pipeline(1, 840), pipeline(2, 512), pipeline(3, 256), pipeline(4, 128)];
    expect(hasRequiredPipelineMapping(mapSelectedPipelines(['1', '2', '3', '4'], pipelines))).toBe(true);
    expect(hasRequiredPipelineMapping(mapSelectedPipelines(['1', '2', '3'], pipelines))).toBe(false);
  });

  it('imports target events only as editable intervals', () => {
    const labelSet = {
      id: 1, name: 'Known events', training_dataset_id: 4, version: 2, categories: [], created_at: '', updated_at: '',
      events: [
        { event_id: 'u1', type: 'target', name: 'U1', category: 'event', start_timestamp: '2026-01-01T01:00:00', end_timestamp: '2026-01-01T02:00:00', notes: null },
        { event_id: 'x1', type: 'exclusion', name: 'Maintenance', category: null, start_timestamp: '2026-01-02T01:00:00', end_timestamp: '2026-01-02T02:00:00', notes: null },
      ],
    } satisfies EvaluationLabelSet;
    expect(targetIntervalsFromLabelSet(labelSet)).toEqual([{
      id: 'label-u1', name: 'U1', type: 'event', start: '2026-01-01T01:00:00', end: '2026-01-01T02:00:00',
    }]);
  });
});
