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
  scale_mode: 'auto', scale_limit: null, fps: 10,
};
export const frameFilename = (index: number) => `frame_${String(index).padStart(6, '0')}.png`;
export const displayTime = (value: string) => value.replace('T', ' ');
export function validateConfig(config: ReferenceImageConfig): string | null {
  if (!config.training_dataset_id || !config.preprocessing_pipeline_id) return 'Datensatz und Preprocessing auswählen.';
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
  if (config.scale_mode === 'manual' && (!Number.isFinite(config.scale_limit) || (config.scale_limit ?? 0) <= 0)) return 'Einen positiven Kontrast-Grenzwert angeben.';
  if (!Number.isInteger(config.fps) || config.fps < 1 || config.fps > 120) return 'Videogeschwindigkeit muss zwischen 1 und 120 FPS liegen.';
  return null;
}
