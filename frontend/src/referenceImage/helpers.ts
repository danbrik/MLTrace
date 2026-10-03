import type { ReferenceImageConfig } from '../types';

export const PHASES: Record<string, string> = {
  queued: 'Wartend', running: 'In Bearbeitung', reference: 'Referenzbild berechnen',
  difference: 'Abweichungen berechnen', rendering: 'Frames erstellen', encoding: 'MP4 exportieren',
  finished: 'Fertig', failed: 'Fehlgeschlagen', aborted: 'Abgebrochen',
};
export const initialConfig: ReferenceImageConfig = {
  training_dataset_id: 0, preprocessing_pipeline_id: 0,
  reference: { start: '', end: '', sampling_rate: 1, mode: 'regular', count: 1, seed: 42 },
  anomaly: { start: '', end: '', sampling_rate: 1 },
  start_offset_minutes: 0,
  processing_mode: 'shift_clip', shift: 10000, clip_min: 0, clip_max: 12000,
  scale_mode: 'auto', scale_limit: null, fps: 10,
};
export const frameFilename = (index: number) => `frame_${String(index).padStart(6, '0')}.png`;
export const displayTime = (value: string) => value.replace('T', ' ');
export function validateConfig(config: ReferenceImageConfig): string | null {
  if (!config.training_dataset_id || !config.preprocessing_pipeline_id) return 'Datensatz und Preprocessing auswählen.';
  if (!Number.isSafeInteger(config.start_offset_minutes) || config.start_offset_minutes < 0) return 'Offset Start muss eine ganze Zahl ab 0 Minuten sein.';
  for (const role of ['reference', 'anomaly'] as const) {
    const interval = config[role];
    if (!interval.start || !interval.end || !Number.isFinite(Date.parse(interval.start)) || !Number.isFinite(Date.parse(interval.end))) return 'Beide Zeiträume vollständig angeben.';
    if (interval.end < interval.start) return 'Das Ende darf nicht vor dem Beginn liegen.';
    if (!Number.isInteger(interval.sampling_rate) || interval.sampling_rate < 1) return 'Sampling muss eine positive ganze Zahl sein.';
  }
  if (config.reference.mode === 'random') {
    if (!Number.isInteger(config.reference.count) || config.reference.count < 1) return 'Die Zufallsanzahl muss eine positive ganze Zahl sein.';
    if (!Number.isInteger(config.reference.seed) || config.reference.seed < 0 || config.reference.seed > 4294967295) return 'Seed muss zwischen 0 und 4294967295 liegen.';
  }
  if (!Number.isFinite(config.shift)) return 'Shift muss eine endliche Zahl sein.';
  if (![config.clip_min, config.clip_max].every(value => Number.isInteger(value) && value >= 0 && value <= 65535) || config.clip_min >= config.clip_max) return 'Clip-Grenzen müssen ganze Zahlen zwischen 0 und 65535 sein; Minimum muss kleiner als Maximum sein.';
  if (config.processing_mode === 'signed' && config.scale_mode === 'manual' && (!Number.isFinite(config.scale_limit) || (config.scale_limit ?? 0) <= 0)) return 'Einen positiven Kontrast-Grenzwert angeben.';
  if (!Number.isInteger(config.fps) || config.fps < 1 || config.fps > 120) return 'Videogeschwindigkeit muss zwischen 1 und 120 FPS liegen.';
  return null;
}
