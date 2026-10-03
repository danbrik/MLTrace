import { describe, expect, it } from 'vitest';
import { emptyPair, initialConfig, templateConfig, validateConfig } from './helpers';
import type { LegacyMeanVarianceConfig } from './types';
const valid = () => ({ ...structuredClone(initialConfig), training_dataset_id: 1, preprocessing_pipeline_id: 2,
  pairs: [{ normal: { start: '2026-01-01T10:00:00', end: '2026-01-01T11:00:00' }, anomaly: { start: '2026-01-01T10:00:00', end: '2026-01-01T11:00:00' } }] });
describe('variance pairs configuration', () => {
  it('uses shared sampling and independent automatic scales, with no mean or random configuration', () => {
    expect(validateConfig(valid())).toBeNull(); expect(initialConfig.sampling_rate).toBe(1);
    expect(initialConfig.pairs).toHaveLength(1); expect(initialConfig.variance_scale).not.toBe(initialConfig.difference_scale);
    expect(initialConfig).not.toHaveProperty('mean_scale'); expect(initialConfig).not.toHaveProperty('reference');
  });
  it('validates pair count, completion, inclusive bounds, sampling and independent limits', () => {
    const cfg = valid(); cfg.pairs[0].normal.end = cfg.pairs[0].normal.start;
    expect(validateConfig(cfg)).toBeNull(); cfg.pairs = Array.from({ length: 6 }, () => structuredClone(cfg.pairs[0]));
    expect(validateConfig(cfg)).toBeNull(); cfg.pairs.push(emptyPair()); expect(validateConfig(cfg)).toContain('sechs');
    cfg.pairs = []; expect(validateConfig(cfg)).toContain('sechs'); cfg.pairs = [emptyPair()]; expect(validateConfig(cfg)).toContain('u1');
    for (const n of [0, -1, 1.5, NaN]) { const c = valid(); c.sampling_rate = n; expect(validateConfig(c)).toContain('Sampling'); }
    expect(validateConfig(valid(), '2026-01-02T00:00:00')).toContain('Datensatz');
    for (const key of ['variance_scale', 'difference_scale'] as const) {
      const c = valid(); c[key].mode = 'manual';
      for (const limit of [null, 0, -1, Infinity, NaN]) { c[key].limit = limit; expect(validateConfig(c)).toContain('Grenzwert'); }
      c[key].limit = .1; expect(validateConfig(c)).toBeNull();
    }
  });
  it('copies stored settings without linking them to new edits', () => {
    const stored = valid(), draft = templateConfig(stored);
    draft.pairs[0].normal.start = 'changed'; draft.variance_scale.limit = 5;
    expect(stored.pairs[0].normal.start).not.toBe('changed'); expect(stored.variance_scale.limit).toBeNull();
  });
  it('converts legacy periods and difference scale, requiring sampling choice when incompatible', () => {
    const old: LegacyMeanVarianceConfig = { training_dataset_id: 1, preprocessing_pipeline_id: 2,
      reference: { ...valid().pairs[0].normal, sampling_rate: 3, mode: 'regular', count: 2, seed: 42 },
      anomaly: { ...valid().pairs[0].anomaly, sampling_rate: 3 }, mean_scale: { mode: 'auto', limit: null }, variance_scale: { mode: 'manual', limit: 5 } };
    expect(templateConfig(old).sampling_rate).toBe(3); expect(templateConfig(old).difference_scale.limit).toBe(5);
    expect(templateConfig(old).pairs[0].normal).toEqual(valid().pairs[0].normal);
    old.reference.mode = 'random'; expect(templateConfig(old).sampling_rate).toBe(0);
    old.reference.mode = 'regular'; old.anomaly.sampling_rate = 4; expect(templateConfig(old).sampling_rate).toBe(0);
  });
});
