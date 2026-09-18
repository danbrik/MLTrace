import type { SensorQuality } from '../types';

export function qualityFlags(sensor: SensorQuality, missingThreshold: number, gapThreshold: number): string[] {
  return [sensor.missing_percent > missingThreshold ? 'High missingness' : '',
    sensor.longest_gap_minutes > gapThreshold ? 'Long gap' : '',
    sensor.constant ? 'Constant' : '', sensor.valid_n === 0 ? 'No valid values' : ''].filter(Boolean);
}

export function visibleSensors(sensors: SensorQuality[], search: string, sort: string, flaggedOnly: boolean, missing: number, gap: number) {
  return sensors.filter(s => s.sensor.toLowerCase().includes(search.toLowerCase()) && (!flaggedOnly || qualityFlags(s, missing, gap).length > 0))
    .sort((a, b) => sort === 'missing' ? b.missing_percent - a.missing_percent || a.sensor.localeCompare(b.sensor)
      : sort === 'gap' ? b.longest_gap_minutes - a.longest_gap_minutes || a.sensor.localeCompare(b.sensor)
        : a.sensor.localeCompare(b.sensor));
}
