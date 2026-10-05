import type { TimeRangeValue } from '../timeRangePresets/types';
export type Role = 'reference' | 'comparison';
export type Config = {
  selection_version?: 1 | 2; block_seconds?: number; seed?: number;
  training_dataset_id: number; preprocessing_pipeline_id: number;
  reference: TimeRangeValue; comparison: TimeRangeValue; deltas_seconds: number[];
};
export type AxisRange = { minimum: number; maximum: number };
export type PlotSettings = {
  title: string; x_title: string; y_title: string;
  x_range: AxisRange | null; y_range: AxisRange | null;
  reference_color: string; comparison_color: string;
};
export type Preview = {
  periods: Record<Role, { image_count: number; start_range_start?: string; start_range_end?: string | null; block_count?: number; candidate_count?: number; valid_candidates?: number; selected_start_count?: number; empty_blocks?: number; deltas: { delta_seconds: number; pair_count: number; missing_targets: number }[] }>;
  errors: string[];
};
export type Summary = { role: Role; delta_seconds: number; pair_count: number; median: number | null; q1: number | null; q3: number | null; iqr: number | null };
export type PairValue = { id: number; role: Role; delta_seconds: number; first_file: string; second_file: string;
  first_timestamp: string; second_timestamp: string; first_utc: string; second_utc: string; value: number };
export type PairPage = { total: number; items: PairValue[] };
export type Run = {
  id: number; training_dataset_id: number; training_dataset_name: string; config: Config;
  pipeline_snapshot: { id: number; name: string; graph: unknown }; dataset_snapshot: Record<string, unknown>;
  plot_settings: PlotSettings; result: { total_images: number; total_pairs: number; width: number; height: number; selection: Preview } | null;
  status: string; current_step: string; processed_images: number; total_images: number | null;
  cancel_requested: boolean; error_message: string | null; queue_rank: number | null;
  enqueued_at: string | null; started_at: string | null; ended_at: string | null; heartbeat_at: string | null;
  duration_seconds: number | null; device: string | null; gpu_index: number | null; created_at: string;
};

export type MatrixConfig = { role: Role; deltas_seconds: number[]; start_times: string[]; top_percent: number | null };
export type MatrixState = { available_deltas?: number[]; config: MatrixConfig | null; artifact: string | null; warnings: string[] };
export type MatrixCandidates = { start_times: string[]; suggested: string[] };
