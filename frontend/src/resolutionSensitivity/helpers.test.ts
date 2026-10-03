import { copyTimeRange } from '../timeRangePresets/helpers';
import { describe, expect, it } from 'vitest';

import type { EvaluationLabelSet, PreprocessingPipeline, ResolutionSensitivityInterval } from '../types';
import { hasRequiredPipelineMapping, mapSelectedPipelines, targetIntervalsFromLabelSet, validateResolutionIntervals } from './helpers';

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

const intervals = (): ResolutionSensitivityInterval[] => [
  { id: 'n', name: 'Normal', type: 'normal', start: '2026-01-01T00:00:00', end: '2026-01-01T00:00:10' },
  { id: 'a', name: 'Anomalie', type: 'event', start: '2026-01-01T00:00:11', end: '2026-01-01T00:00:20' },
];
describe('inclusive resolution intervals and presets', () => {
  it('accepts disjoint intervals and isolated single points', () => {
    const items = intervals(); items[0].end = items[0].start;
    expect(validateResolutionIntervals(items).valid).toBe(true);
  });
  it('reports overlap on every affected row, including nested ranges and equal endpoints', () => {
    const items = intervals(); items[1].start = '2026-01-01T00:00:10.000000';
    expect(Object.keys(validateResolutionIntervals(items).errors)).toEqual(['n', 'a']);
    items.push({ ...items[0], id: 'third', start: '2026-01-01T00:00:02', end: '2026-01-01T00:00:03' });
    expect(Object.keys(validateResolutionIntervals(items).errors)).toContain('third');
  });
  it('validates names, bounds, reversed times, and required roles', () => {
    expect(validateResolutionIntervals(intervals(), '2026-01-01T00:00:01').errors.n).toContain('Außerhalb des verfügbaren Datensatzzeitraums.');
    const items = intervals(); items[0].name = ' '; items[0].end = '2025-12-31T00:00:00';
    expect(validateResolutionIntervals(items).errors.n).toHaveLength(2);
    expect(validateResolutionIntervals([]).formError).toBeTruthy();
    expect(validateResolutionIntervals([intervals()[0]]).valid).toBe(false);
  });
  it('copies only timestamps, retains role, name and analysis settings, and does not link the preset', () => {
    const config = { intervals: intervals(), samples: 75, pipelineIds: ['1', '2', '3', '4'] };
    const preset = { name: 'Different name', start: '2026-01-01T00:00:05', end: '2026-01-01T00:00:08' };
    const next = { ...config, intervals: [copyTimeRange(config.intervals[0], preset), config.intervals[1]] };
    expect(next.intervals[0]).toEqual({ ...config.intervals[0], start: preset.start, end: preset.end });
    expect(next.samples).toBe(75); expect(next.pipelineIds).toEqual(config.pipelineIds);
    preset.start = '2026-01-01T00:00:09';
    expect(next.intervals[0].start).toBe('2026-01-01T00:00:05');
    expect(config.intervals[0].start).toBe('2026-01-01T00:00:00');
  });
});
