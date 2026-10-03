export type RoiRectangle = { x: number; y: number; width: number; height: number };
export type RoiConfig = { roi: RoiRectangle; opacity: number };
export type RoiBasis = {
  version: number; width: number; height: number; difference_scale_limit: number;
  background_min: number; background_max: number; dataset_name: string; pipeline_name: string;
  pairs: { label: string; periods: { normal: {start: string; end: string}; anomaly: {start: string; end: string} }; counts: {normal: number; anomaly: number} }[];
};
export type RoiResult = RoiBasis & RoiConfig & {
  rows: {label: string; area_percent: number; increase_percent: number | null; positive_roi: number; positive_total: number}[];
  area_percent: number; mean_increase_percent: number | null; valid_pairs: number; warnings: string[];
  plot: string; table: string;
};
type RoiJobBase = {
  id: number; parent_run_id: number; training_dataset_name: string;
  status: string; current_step: string;
  processed_images: number; total_images: number | null; cancel_requested: boolean; error_message: string | null;
  queue_rank: number | null; enqueued_at: string | null; started_at: string | null; ended_at: string | null;
  duration_seconds: number | null; heartbeat_at: string | null; device: string | null; gpu_index: number | null;
  pid: number | null; log_path: string | null; created_at: string; updated_at: string;
};
export type RoiEvaluationJob = RoiJobBase & {operation: 'evaluate'; config: RoiConfig; result: RoiResult | null};
export type VarianceRoiJob = RoiEvaluationJob | (RoiJobBase & {operation: 'prepare'; config: Record<string, never>; result: {total_images: number} | null});
export type RoiState = {ready: boolean; basis: RoiBasis | null; job: VarianceRoiJob | null; saved: RoiEvaluationJob | null};
