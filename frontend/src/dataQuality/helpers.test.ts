import { describe, expect, it } from 'vitest';
import { qualityFlags, visibleSensors } from './helpers';
import type { SensorQuality } from '../types';
const sensor = (changes: Partial<SensorQuality> = {}): SensorQuality => ({ sensor: 'A', data_type: 'numeric', valid_n: 100,
  missing_percent: 0, longest_gap_minutes: 0, unique: 5, constant: false, invalid_n: 0, conflict_n: 0,
  min: 0, q01: 1, median: 2, q99: 4, max: 5, iqr: 2, std: 1, ...changes });
describe('quality display', () => {
  it('uses strict thresholds and does not flag low unique counts', () => {
    expect(qualityFlags(sensor({ missing_percent: 1, longest_gap_minutes: 60, unique: 2 }), 1, 60)).toEqual([]);
    expect(qualityFlags(sensor({ missing_percent: 1.1, longest_gap_minutes: 61 }), 1, 60)).toEqual(['High missingness', 'Long gap']);
    expect(qualityFlags(sensor({ constant: true }), 1, 60)).toEqual(['Constant']);
    expect(qualityFlags(sensor({ valid_n: 0 }), 100, 999)).toEqual(['No valid values']);
  });
  it('searches, sorts and filters 80 sensors without changing source results', () => {
    const sensors = Array.from({ length: 80 }, (_, i) => sensor({ sensor: `Sensor ${i}`, missing_percent: i }));
    const shown = visibleSensors(sensors, 'sensor', 'missing', true, 60, 60);
    expect(shown).toHaveLength(19); expect(shown[0].missing_percent).toBe(79);
    expect(sensors[0].missing_percent).toBe(0);
  });
});
