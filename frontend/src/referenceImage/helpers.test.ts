import { describe, expect, it } from 'vitest';
import { initialConfig, validateConfig, frameFilename } from './helpers';
import type { ReferenceImageConfig } from '../types';

const valid = (): ReferenceImageConfig => ({ ...structuredClone(initialConfig), training_dataset_id: 1, preprocessing_pipeline_id: 2,
  reference: { ...initialConfig.reference, start: '2026-01-01T00:00:00', end: '2026-01-01T01:00:00' },
  anomaly: { start: '2026-01-01T02:00:00', end: '2026-01-01T03:00:00', sampling_rate: 15 },
});
describe('reference image configuration', () => {
  it('defaults the start offset to zero and accepts only nonnegative whole minutes', () => {
    const config = valid();
    expect(config.start_offset_minutes).toBe(0);
    config.start_offset_minutes = 30;
    expect(validateConfig(config)).toBeNull();
    for (const value of [-1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      config.start_offset_minutes = value;
      expect(validateConfig(config)).toContain('Offset Start');
    }
  });
  it('requires a dataset, pipeline and complete intervals', () => {
    expect(validateConfig(initialConfig)).toBeTruthy();
    expect(validateConfig(valid())).toBeNull();
    const config = valid(); config.anomaly.start = '';
    expect(validateConfig(config)).toBeTruthy();
  });
  it('allows a single timestamp and overlap but rejects reversed ranges', () => {
    const config = valid(); config.anomaly = { ...config.reference };
    expect(validateConfig(config)).toBeNull();
    config.anomaly.end = config.anomaly.start;
    expect(validateConfig(config)).toBeNull();
    config.anomaly.end = '2025-01-01T00:00:00';
    expect(validateConfig(config)).toBeTruthy();
  });
  it('validates random count, seed, integer sampling, FPS and manual contrast', () => {
    const config = valid(); config.reference.mode = 'random'; config.reference.count = 0;
    expect(validateConfig(config)).toBeTruthy();
    config.reference.count = 15; config.reference.seed = -1;
    expect(validateConfig(config)).toBeTruthy();
    config.reference.seed = 42; config.anomaly.sampling_rate = 1.5;
    expect(validateConfig(config)).toBeTruthy();
    config.anomaly.sampling_rate = 1; config.processing_mode = 'signed'; config.scale_mode = 'manual';
    expect(validateConfig(config)).toBeTruthy();
    config.scale_limit = 100;
    expect(validateConfig(config)).toBeNull();
    config.fps = 0;
    expect(validateConfig(config)).toBeTruthy();
  });
  it('addresses the exact saved frame, including large indices', () => {
    expect(frameFilename(0)).toBe('frame_000000.png');
    expect(frameFilename(1234567)).toBe('frame_1234567.png');
  });
});

it('defaults to shift/clipping and validates the uint16 limits', () => {
  const config = valid();
  expect(config.processing_mode).toBe('shift_clip');
  expect([config.shift, config.clip_min, config.clip_max]).toEqual([10000, 0, 12000]);
  config.clip_max = 65536;
  expect(validateConfig(config)).toBeTruthy();
  config.clip_max = 12000; config.clip_min = 12000;
  expect(validateConfig(config)).toBeTruthy();
  config.clip_min = -1;
  expect(validateConfig(config)).toBeTruthy();
  config.clip_min = 0; config.shift = NaN;
  expect(validateConfig(config)).toBeTruthy();
  config.shift = -100;
  expect(validateConfig(config)).toBeNull();
});
