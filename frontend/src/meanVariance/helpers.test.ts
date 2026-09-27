import { describe, expect, it } from 'vitest';
import { initialConfig, validateConfig } from './helpers';
import { copyTimeRange } from '../timeRangePresets/helpers';
const valid = () => ({ ...structuredClone(initialConfig), training_dataset_id: 1, preprocessing_pipeline_id: 2,
  reference: { ...initialConfig.reference, start: '2026-01-01T10:00:00', end: '2026-01-01T11:00:00' },
  anomaly: { ...initialConfig.anomaly, start: '2026-01-01T10:00:00', end: '2026-01-01T11:00:00' } });
describe('mean/variance configuration', () => {
  it('defaults to automatic independent scales, sampling 1 and seed 42', () => {
    expect(validateConfig(valid())).toBeNull();
    expect(initialConfig.reference.seed).toBe(42);
    expect(initialConfig.reference.sampling_rate).toBe(1);
    expect(initialConfig.anomaly.sampling_rate).toBe(1);
    expect(initialConfig.mean_scale).not.toBe(initialConfig.variance_scale);
    expect(initialConfig).not.toHaveProperty('shift');
    expect(initialConfig).not.toHaveProperty('fps');
  });
  it('validates each manual scale without linking them', () => {
    for (const key of ['mean_scale', 'variance_scale'] as const) {
      const cfg = valid(); cfg[key].mode = 'manual';
      for (const limit of [null, 0, -1, Infinity, NaN]) {
        cfg[key].limit = limit; expect(validateConfig(cfg)).toContain('Grenzwert');
      }
      cfg[key].limit = .1; expect(validateConfig(cfg)).toBeNull();
    }
  });
  it('rejects invalid intervals and sampling, accepts equal and overlapping bounds', () => {
    const cfg = valid(); cfg.reference.end = cfg.reference.start;
    expect(validateConfig(cfg)).toBeNull();
    cfg.anomaly.sampling_rate = 1.5; expect(validateConfig(cfg)).toContain('Sampling');
    cfg.anomaly.sampling_rate = 1; cfg.anomaly.start = '2026-01-02T00:00:00';
    expect(validateConfig(cfg)).toContain('Beginn');
  });
  it('validates random count and seed', () => {
    const cfg = valid(); cfg.reference.mode = 'random'; cfg.reference.count = 0;
    expect(validateConfig(cfg)).toContain('Zufallsanzahl');
    cfg.reference.count = 15; cfg.reference.seed = -1;
    expect(validateConfig(cfg)).toContain('Seed');
  });
  it('copies a shared preset to either role while retaining sampling and scales', () => {
    const cfg = valid(); cfg.reference.mode = 'random'; cfg.reference.count = 9; cfg.anomaly.sampling_rate = 15;
    const range = { start: '2026-01-01T10:01:00', end: '2026-01-01T10:02:00' };
    const next = { ...cfg, reference: copyTimeRange(cfg.reference, range), anomaly: copyTimeRange(cfg.anomaly, range) };
    expect(next.reference).toEqual({ ...cfg.reference, ...range });
    expect(next.anomaly).toEqual({ ...cfg.anomaly, ...range });
    expect(next.mean_scale).toEqual(cfg.mean_scale);
    expect(JSON.stringify(next)).not.toBe(JSON.stringify(cfg));
  });
});
