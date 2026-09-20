import { describe, expect, it } from 'vitest';
import { nextEventId, projectionTraces, validateConfig } from './helpers';
import type { RepresentationConfig, RepresentationPoint } from '../types';

const config: RepresentationConfig = {
  training_dataset_id: 1, preprocessing_pipeline_id: 2, cluster_count: 3, pca_variance: 0.95, seed: 42,
  intervals: [
    { id: 'a', name: 'Normal', label: 'normal', start: '2026-01-01T00:00:00', end: '2026-01-01T00:01:00', sampling_rate: 30, random: false, event_id: null },
    { id: 'b', name: 'Anomalie', label: 'anomaly', start: '2026-01-01T00:01:00', end: '2026-01-01T00:02:00', sampling_rate: 30, random: true, event_id: 'A7' },
  ],
};

describe('representation configuration', () => {
  it('allows touching ranges but rejects overlapping ranges and incomplete timestamps', () => {
    expect(validateConfig(config)).toBeNull();
    const changed = structuredClone(config);
    changed.intervals[1].start = '2026-01-01T00:00:30';
    expect(validateConfig(changed)).toContain('überlappen');
    changed.intervals[1].start = '2026-01-01';
    expect(validateConfig(changed)).toContain('gültigen');
  });
  it('validates sampling and analysis parameters', () => {
    expect(validateConfig({ ...config, cluster_count: 1 })).toContain('Clusterzahl');
    expect(validateConfig({ ...config, pca_variance: 1 })).toContain('PCA');
    expect(validateConfig({ ...config, seed: -1 })).toContain('Seed');
    expect(validateConfig({ ...config, intervals: [{ ...config.intervals[0], sampling_rate: 0 }] })).toContain('Samplingrate');
    expect(nextEventId(config.intervals)).toBe('A8');
  });
  it('uses the same coordinates for labels and events, including buffer points', () => {
    const points: RepresentationPoint[] = ['normal', 'anomaly', 'buffer'].map((label, index) => ({
      file_path: String(index), timestamp: '2026-01-01T00:00:00', interval_id: String(index), interval_name: `Range ${index}`,
      label: label as RepresentationPoint['label'], event_id: label === 'anomaly' ? 'A7' : null,
      pca_x: index, pca_y: index + 1, umap_x: index + 2, umap_y: index + 3, cluster: index,
    }));
    const labels = projectionTraces(points, 'pca', false) as Array<{ name: string; x: number[] }>;
    const events = projectionTraces(points, 'pca', true) as Array<{ name: string; x: number[] }>;
    expect(labels.map(row => row.name).sort()).toEqual(['Anomalie', 'Normal', 'Puffer']);
    expect(events.map(row => row.name).sort()).toEqual(['A7', 'Normal', 'Puffer']);
    expect(labels.flatMap(row => row.x).sort()).toEqual(events.flatMap(row => row.x).sort());
  });
});
