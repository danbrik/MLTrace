import { rangeProblem } from '../timeRangePresets/helpers';
import { isPairConfig, type MeanVarianceConfig, type StoredConfig, type VariancePair } from './types';
export { displayTime } from '../referenceImage/helpers';
export const PHASES: Record<string, string> = {
  queued: 'Wartend', running: 'In Bearbeitung', reference: 'Normalphase berechnen',
  anomaly: 'Anomaliephase berechnen', difference: 'Differenzen berechnen',
  rendering: 'Vergleichsplot erstellen', exporting: 'PNG exportieren',
  finished: 'Fertig', failed: 'Fehlgeschlagen', aborted: 'Abgebrochen',
};
export function phaseLabel(phase: string) {
  const match = /^u(\d+)_(normal|anomaly)$/.exec(phase);
  return match ? `u${match[1]} · ${match[2] === 'normal' ? 'Normalzustand' : 'Anomaliephase'} berechnen` : PHASES[phase] ?? phase;
}
export const emptyPair = (): VariancePair => ({ normal: { start: '', end: '' }, anomaly: { start: '', end: '' } });
export const initialConfig: MeanVarianceConfig = {
  version: 2, training_dataset_id: 0, preprocessing_pipeline_id: 0, sampling_rate: 1,
  pairs: [emptyPair()], variance_scale: { mode: 'auto', limit: null }, difference_scale: { mode: 'auto', limit: null },
};
export function templateConfig(stored: StoredConfig): MeanVarianceConfig {
  if (isPairConfig(stored)) return structuredClone(stored);
  const compatible = stored.reference.mode === 'regular' && stored.reference.sampling_rate === stored.anomaly.sampling_rate;
  return {
    ...structuredClone(initialConfig), training_dataset_id: stored.training_dataset_id,
    preprocessing_pipeline_id: stored.preprocessing_pipeline_id,
    sampling_rate: compatible ? stored.reference.sampling_rate : 0,
    pairs: [{ normal: { start: stored.reference.start, end: stored.reference.end }, anomaly: { start: stored.anomaly.start, end: stored.anomaly.end } }],
    difference_scale: structuredClone(stored.variance_scale),
  };
}
export function validateConfig(config: MeanVarianceConfig, min?: string, max?: string): string | null {
  if (!config.training_dataset_id || !config.preprocessing_pipeline_id) return 'Datensatz und Preprocessing auswählen.';
  if (!Number.isInteger(config.sampling_rate) || config.sampling_rate < 1) return 'Bitte ein gemeinsames Sampling als positive ganze Zahl auswählen.';
  if (!config.pairs.length || config.pairs.length > 6) return 'Ein bis sechs vollständige Paare sind erforderlich.';
  for (const [index, pair] of config.pairs.entries()) {
    for (const role of ['normal', 'anomaly'] as const) {
      const problem = rangeProblem(pair[role], min, max);
      if (problem) return `u${index + 1} ${role === 'normal' ? 'Normalzustand' : 'Anomaliephase'}: ${problem}`;
    }
  }
  for (const scale of [config.variance_scale, config.difference_scale]) {
    if (scale.mode === 'manual' && (!Number.isFinite(scale.limit) || (scale.limit ?? 0) <= 0)) return 'Für jede manuelle Farbskala einen positiven Grenzwert angeben.';
  }
  return null;
}
