import { rangeProblem } from '../timeRangePresets/helpers';
import type { MeanVarianceConfig } from './types';

export { displayTime } from '../referenceImage/helpers';
export const PHASES: Record<string, string> = {
  queued: 'Wartend', running: 'In Bearbeitung', reference: 'Normalphase berechnen',
  anomaly: 'Anomaliephase berechnen', difference: 'Differenzen berechnen',
  rendering: 'Heatmaps erstellen', exporting: 'PNG exportieren',
  finished: 'Fertig', failed: 'Fehlgeschlagen', aborted: 'Abgebrochen',
};
export const initialConfig: MeanVarianceConfig = {
  training_dataset_id: 0, preprocessing_pipeline_id: 0,
  reference: { start: '', end: '', sampling_rate: 1, mode: 'regular', count: 1, seed: 42 },
  anomaly: { start: '', end: '', sampling_rate: 1 },
  mean_scale: { mode: 'auto', limit: null }, variance_scale: { mode: 'auto', limit: null },
};
export function validateConfig(config: MeanVarianceConfig): string | null {
  if (!config.training_dataset_id || !config.preprocessing_pipeline_id) return 'Datensatz und Preprocessing auswählen.';
  for (const role of ['reference', 'anomaly'] as const) {
    const interval = config[role], problem = rangeProblem(interval);
    if (problem) return problem;
    if (!Number.isInteger(interval.sampling_rate) || interval.sampling_rate < 1) return 'Sampling muss eine positive ganze Zahl sein.';
  }
  if (config.reference.mode === 'random') {
    if (!Number.isInteger(config.reference.count) || config.reference.count < 1) return 'Die Zufallsanzahl muss eine positive ganze Zahl sein.';
    if (!Number.isInteger(config.reference.seed) || config.reference.seed < 0 || config.reference.seed > 4294967295) return 'Seed muss zwischen 0 und 4294967295 liegen.';
  }
  for (const scale of [config.mean_scale, config.variance_scale]) {
    if (scale.mode === 'manual' && (!Number.isFinite(scale.limit) || (scale.limit ?? 0) <= 0)) return 'Für jede manuelle Farbskala einen positiven Grenzwert angeben.';
  }
  return null;
}
