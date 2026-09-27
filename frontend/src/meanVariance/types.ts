import type { ReferenceImageConfig, ReferenceImagePreview, ReferenceImageRun } from '../types';

export type HeatmapScale = { mode: 'auto' | 'manual'; limit: number | null };
export type MeanVarianceConfig = Pick<ReferenceImageConfig, 'training_dataset_id' | 'preprocessing_pipeline_id' | 'reference' | 'anomaly'> & {
  mean_scale: HeatmapScale;
  variance_scale: HeatmapScale;
};
export type MeanVariancePreview = ReferenceImagePreview;
export type DifferenceHeatmap = {
  filename: string; title: string; unit: string; scale_limit: number;
  minimum: number; maximum: number; maximum_absolute: number; all_zero: boolean;
};
export type MeanVarianceResults = {
  total_images: number; reference_count: number; anomaly_count: number;
  width: number; height: number; ddof: 0; warnings: string[];
  maps: { mean: DifferenceHeatmap; variance: DifferenceHeatmap };
};
export type MeanVarianceRun = Omit<ReferenceImageRun, 'config' | 'result'> & {
  config: MeanVarianceConfig; result: MeanVarianceResults | null;
};
