import type { ReferenceImageConfig, ReferenceImagePreview, ReferenceImageRun } from '../types';
import type { TimeRangeValue } from '../timeRangePresets/types';
export type HeatmapScale = { mode: 'auto' | 'manual'; limit: number | null };
export type LegacyMeanVarianceConfig = Pick<ReferenceImageConfig, 'training_dataset_id' | 'preprocessing_pipeline_id' | 'reference' | 'anomaly'> & {
  mean_scale: HeatmapScale; variance_scale: HeatmapScale;
};
export type VariancePair = { normal: TimeRangeValue; anomaly: TimeRangeValue };
export type MeanVarianceConfig = {
  version: 2; training_dataset_id: number; preprocessing_pipeline_id: number; sampling_rate: number;
  pairs: VariancePair[]; variance_scale: HeatmapScale; difference_scale: HeatmapScale;
};
export type StoredConfig = MeanVarianceConfig | LegacyMeanVarianceConfig;
export const isPairConfig = (config: StoredConfig): config is MeanVarianceConfig => 'version' in config && config.version === 2;
export type MeanVariancePreview = { version: 2; pairs: Pick<ReferenceImagePreview, 'reference' | 'anomaly' | 'errors'>[]; errors: string[] };
export type MapStatistics = { minimum: number; maximum: number; maximum_absolute: number; all_zero: boolean };
export type DifferenceHeatmap = MapStatistics & { filename: string; title: string; unit: string; scale_limit: number };
export type LegacyResults = {
  total_images: number; reference_count: number; anomaly_count: number;
  width: number; height: number; ddof: 0; warnings: string[];
  maps: { mean: DifferenceHeatmap; variance: DifferenceHeatmap };
};
export type VarianceResults = {
  version: 2; filename: string; unit: string; total_images: number; width: number; height: number; ddof: 0;
  dataset_name: string; pipeline_name: string; warnings: string[];
  variance_scale_limit: number; difference_scale_limit: number;
  pairs: { label: string; periods: VariancePair; counts: { normal: number; anomaly: number }; maps: Record<'normal' | 'anomaly' | 'difference', MapStatistics> }[];
};
export type MeanVarianceResults = VarianceResults | LegacyResults;
export const isPairResults = (result: MeanVarianceResults): result is VarianceResults => 'version' in result && result.version === 2;
export type MeanVarianceRun = Omit<ReferenceImageRun, 'config' | 'result'> & {
  config: StoredConfig; result: MeanVarianceResults | null;
};
