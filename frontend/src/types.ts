export type DatasetFolder = {
  id: number;
  relative_path: string;
  image_count: number;
  first_timestamp: string | null;
  last_timestamp: string | null;
  extension_summary: Record<string, number> | null;
  resolution_summary: Record<string, number> | null;
  image_metadata: Record<string, unknown> | null;
  filename_template: Record<string, unknown> | null;
  cadence_summary: {
    min_seconds: number | null;
    median_seconds: number | null;
    mean_seconds?: number | null;
    max_seconds: number | null;
    sampled_adjacent_pairs?: number | null;
  } | null;
};

export type Dataset = {
  id: number;
  name: string;
  root_path: string;
  status: string;
  timestamp_regex: string | null;
  timestamp_format: string | null;
  timestamp_example: string | null;
  scan_error: string | null;
  scan_summary: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
  folders: DatasetFolder[];
  is_update_locked: boolean;
  update_lock_reasons: string[];
};

export type DatasetConnectionTest = {
  root_path: string;
  exists: boolean;
  is_directory: boolean;
  supported_file_found: boolean;
  sample_file_path: string | null;
  message: string;
};

export type CacheRevisions = {
  revisions: Record<string, string>;
};

export type TrainingDatasetRuleInput = {
  folder_id: number;
  start_timestamp: string;
  end_timestamp: string;
  stride: number;
};

export type TrainingDatasetPreview = {
  total_matching_images: number;
  total_selected_images: number;
  rules: Array<{
    folder_id: number;
    start_timestamp: string;
    end_timestamp: string;
    stride: number;
    matching_images: number;
    selected_images: number;
  }>;
};

export type TrainingDataset = {
  id: number;
  name: string;
  usage_label: 'train' | 'test' | 'validation' | 'mixed' | string;
  notes: string | null;
  created_at: string;
  start_timestamp: string | null;
  end_timestamp: string | null;
  dataset_names: string[];
  image_resolutions: string[];
  image_signatures: string[];
  total_matching_images: number;
  total_selected_images: number;
  rules: Array<{
    id: number;
    folder_id: number;
    dataset_id: number;
    dataset_name: string;
    dataset_root_path: string;
    folder_relative_path: string;
    folder_first_timestamp: string | null;
    folder_last_timestamp: string | null;
    folder_extension_summary: Record<string, number> | null;
    folder_resolution_summary: Record<string, number> | null;
    folder_image_metadata: Record<string, unknown> | null;
    folder_image_signature: string | null;
    start_timestamp: string;
    end_timestamp: string;
    stride: number;
    matching_images: number | null;
    selected_images: number | null;
  }>;
  is_update_locked: boolean;
  update_lock_reasons: string[];
  invalid_rule_count: number;
  integrity_warnings: string[];
  counts_missing: boolean;
};

export type PreprocessingGraphNode = {
  id: string;
  type: string;
  config: Record<string, unknown>;
  position?: { x: number; y: number } | null;
};

export type PreprocessingGraphEdge = {
  id?: string | null;
  source: string;
  target: string;
};

export type PreprocessingGraph = {
  nodes: PreprocessingGraphNode[];
  edges: PreprocessingGraphEdge[];
};

export type PreprocessingStepDefinition = {
  type: string;
  label: string;
  category: string;
  input_kind: string;
  output_kind: string;
  config_schema: {
    type: string;
    ui_control?: string;
    properties: Record<
      string,
      {
        type: string;
        label?: string;
        enum?: string[];
        default?: unknown;
        minimum?: number;
        maximum?: number;
        ui_control?: string;
        default_from?: 'input_width' | 'input_height';
        visible_when?: Record<string, unknown>;
      }
    >;
  };
  default_config: Record<string, unknown>;
};

export type PreprocessingPipeline = {
  id: number;
  name: string;
  description: string | null;
  graph: PreprocessingGraph;
  preview_folder_id: number | null;
  input_width: number | null;
  input_height: number | null;
  output_width: number | null;
  output_height: number | null;
  created_at: string;
  updated_at: string;
  is_update_locked: boolean;
  update_lock_reasons: string[];
  step_count?: number | null;
  step_types?: string[];
};

export type PreprocessingPreviewImage = {
  node_id: string;
  step_type: string;
  label: string;
  width: number;
  height: number;
  channels: number;
  dtype: string;
  value_min: number;
  value_max: number;
  image_data_url: string;
};

export type PreprocessingPreview = {
  source_image_id: number;
  source_image_path: string;
  source_timestamp: string;
  previews: PreprocessingPreviewImage[];
};

export type SchemaProperty = {
  type: 'string' | 'integer' | 'number' | 'boolean';
  label?: string;
  enum?: string[];
  default?: unknown;
  minimum?: number;
  maximum?: number;
  description?: string;
  help_text?: string;
};

export type ConfigSchema = {
  type: string;
  required?: string[];
  properties: Record<string, SchemaProperty>;
};

export type MethodDefinition = {
  type: string;
  label: string;
  category: string;
  description: string;
  framework: string;
  method_family: string;
  method_version: string;
  training_mode: 'gradient' | 'fit' | 'none' | string;
  architecture_version: string;
  requires_training: boolean;
  supports_training_pipeline: boolean;
  artifact_kind: string;
  builder_kind: 'sequential_autoencoder' | 'sequential_variational_autoencoder' | 'form' | string;
  capabilities: Record<string, unknown>;
  method_schema: ConfigSchema;
  model_schema: ConfigSchema;
  training_schema: ConfigSchema;
  inference_schema: ConfigSchema;
  default_method_config: Record<string, unknown>;
  default_model_config: Record<string, unknown>;
  default_training_config: Record<string, unknown>;
  default_inference_config: Record<string, unknown>;
};

export type ModelLayerDefinition = {
  type: string;
  label: string;
  category: string;
  config_schema: ConfigSchema;
  default_config: Record<string, unknown>;
  input_rank: number | null;
  output_rank: number | null;
  shape_notes: string | null;
};

export type ModelLayerInstance = {
  id: string;
  type: string;
  config: Record<string, unknown>;
};

export type ModelGraph = {
  builder_kind?: string;
  encoder?: ModelLayerInstance[];
  latent?: Record<string, unknown>;
  decoder?: ModelLayerInstance[];
  [key: string]: unknown;
};

export type ModelDiagram = {
  method_type?: string;
  architecture_type: string;
  builder_kind: string;
  nodes: Array<{
    id: string;
    label: string;
    section: string;
    detail: string;
  }>;
  edges: Array<{
    source: string;
    target: string;
  }>;
};

export type MethodConfigurationParameter = {
  path: string;
  value_type: string;
  value_text: string | null;
  value_number: number | null;
  value_bool: boolean | null;
};

export type MethodValidation = {
  valid: boolean;
  errors: string[];
  warnings: string[];
  layer_specs: Array<{
    section: string;
    index: number;
    layer_id: string | null;
    layer_type: string;
    input_label: string;
    output_label: string;
  }>;
  torch_check: {
    status: 'available' | 'missing' | 'failed' | string;
    message: string;
    input_shape?: unknown;
    output_shape?: unknown;
    elapsed_ms?: number;
  } | null;
  diagram: ModelDiagram;
};

export type MethodTorchCheckResponse = {
  valid: boolean;
  status: 'available' | 'missing' | 'failed' | 'not_applicable' | string;
  errors: string[];
  warnings: string[];
  logs: string[];
  torch_check: {
    status: 'available' | 'missing' | 'failed' | 'not_applicable' | string;
    message: string;
    input_shape?: unknown;
    output_shape?: unknown;
    elapsed_ms?: number;
  } | null;
};

export type MethodConfiguration = {
  id: number;
  name: string;
  description: string | null;
  method_type: string;
  method_family: string;
  method_version: string;
  training_mode: string;
  architecture_type: string;
  architecture_version: string;
  requires_training: boolean;
  supports_training_pipeline: boolean;
  artifact_kind: string;
  builder_kind: string;
  method_graph: ModelGraph;
  model_graph: ModelGraph;
  method_config: Record<string, unknown>;
  model_config: Record<string, unknown>;
  training_config: Record<string, unknown>;
  inference_config: Record<string, unknown>;
  diagram: ModelDiagram;
  created_at: string;
  updated_at: string;
  validation: MethodValidation | null;
  parameters: MethodConfigurationParameter[];
  is_update_locked: boolean;
  update_lock_reasons: string[];
};

export type MethodConfigurationPayload = {
  method_type: string;
  method_graph: ModelGraph;
  method_config: Record<string, unknown>;
  training_config: Record<string, unknown>;
  inference_config: Record<string, unknown>;
};

export type MethodConfigurationSavePayload = MethodConfigurationPayload & {
  name: string;
  description?: string | null;
};

export type MethodValidationResponse = MethodValidation;

export type TrainingPipelineDatasetSummary = {
  training_dataset_id: number;
  position: number;
  name: string;
  start_timestamp: string | null;
  end_timestamp: string | null;
  total_selected_images: number;
  dataset_names: string[];
};

export type TrainingPipeline = {
  id: number;
  name: string;
  description: string | null;
  shuffle: boolean;
  training_parameters: Record<string, unknown>;
  preprocessing_pipeline_id: number;
  preprocessing_pipeline_name: string;
  preprocessing_input_width: number | null;
  preprocessing_input_height: number | null;
  preprocessing_output_width: number | null;
  preprocessing_output_height: number | null;
  method_configuration_id: number;
  method_configuration_name: string;
  method_type: string;
  training_mode: string;
  builder_kind: string;
  total_selected_images: number;
  training_datasets: TrainingPipelineDatasetSummary[];
  created_at: string;
  updated_at: string;
  is_update_locked: boolean;
  update_lock_reasons: string[];
};

export type TrainingPipelinePayload = {
  training_dataset_ids: number[];
  preprocessing_pipeline_id: number;
  method_configuration_id: number;
  shuffle: boolean;
  training_parameters: Record<string, unknown>;
};

export type TrainingPipelineSavePayload = TrainingPipelinePayload & {
  name: string;
  description?: string | null;
};

export type TrainingPipelineModelOutput = {
  input_shape: number[];
  output_shape: number[];
  width: number;
  height: number;
  channels: number;
  dtype: string;
  value_min: number;
  value_max: number;
  image_data_url: string;
  elapsed_ms: number;
};

export type TrainingPipelineDryRun = {
  valid: boolean;
  mode: 'forward_pass' | 'fit_contribution' | 'failed' | string;
  errors: string[];
  warnings: string[];
  logs: string[];
  training_dataset_name: string | null;
  source_image_path: string | null;
  source_timestamp: string | null;
  preprocessing_previews: PreprocessingPreviewImage[];
  model_output: TrainingPipelineModelOutput | null;
  note: string | null;
};

export type TrainingRunStatus = 'queued' | 'running' | 'finished' | 'failed' | 'aborted';

export type TrainingRunMetric = {
  epoch: number;
  train_loss: number | null;
  val_loss: number | null;
};

export type TrainingRun = {
  id: number;
  training_pipeline_id: number;
  status: TrainingRunStatus;
  enqueued_at: string | null;
  queue_rank: number | null;
  started_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
  gpu_index: number | null;
  device: string | null;
  epochs_total: number | null;
  epochs_completed: number;
  train_loss: number | null;
  val_loss: number | null;
  best_val_loss: number | null;
  image_count: number | null;
  artifact_kind: string | null;
  artifact_path: string | null;
  artifact_size_bytes: number | null;
  error_message: string | null;
  training_pipeline_name: string;
  method_type: string;
  method_family: string;
  training_mode: string;
  builder_kind: string;
  preprocessing_pipeline_name: string;
  dataset_names: string[];
  shuffle: boolean;
  input_resolution: string | null;
  epochs: number | null;
  learning_rate: number | null;
  training_parameters: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  metrics: TrainingRunMetric[];
};

export type TrainingRunFilters = {
  status?: string | null;
  method_type?: string | null;
  training_mode?: string | null;
  search?: string | null;
  max_val_loss?: number | null;
  max_train_loss?: number | null;
  max_duration?: number | null;
};

export type RoiDefinition = {
  id: number;
  name: string;
  description: string | null;
  image_width: number;
  image_height: number;
  x: number;
  y: number;
  width: number;
  height: number;
  geometry_type: string;
  points: Array<{ x: number; y: number }> | null;
  tile_rows: number;
  tile_cols: number;
  created_at: string;
  updated_at: string;
};

export type RoiDefinitionPayload = {
  name: string;
  description?: string | null;
  image_width: number;
  image_height: number;
  x: number;
  y: number;
  width: number;
  height: number;
  geometry_type?: string;
  points?: Array<{ x: number; y: number }> | null;
  tile_rows?: number;
  tile_cols?: number;
};

export type RoiPreview = {
  training_run_id: number;
  training_dataset_id: number;
  preprocessing_pipeline_id: number;
  source_image_path: string;
  source_timestamp: string;
  width: number;
  height: number;
  channels: number;
  dtype: string;
  image_data_url: string;
};

export type TestingRunStatus = 'queued' | 'running' | 'finished' | 'failed' | 'aborted';

export type TestingRun = {
  id: number;
  name: string;
  training_run_id: number;
  training_dataset_id: number;
  roi_id: number | null;
  status: TestingRunStatus | string;
  enqueued_at: string | null;
  queue_rank: number | null;
  started_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
  gpu_index: number | null;
  device: string | null;
  error_message: string | null;
  image_count: number | null;
  expected_image_count: number | null;
  score_mean: number | null;
  score_min: number | null;
  score_max: number | null;
  full_mse_mean: number | null;
  roi_mse_mean: number | null;
  results_path: string | null;
  results_size_bytes: number | null;
  training_run_name: string;
  training_pipeline_name: string;
  training_dataset_name: string;
  preprocessing_pipeline_name: string;
  method_type: string;
  method_family: string;
  training_mode: string;
  artifact_kind: string;
  artifact_path: string;
  roi_name: string | null;
  roi_geometry: Record<string, unknown> | null;
  inference_config: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
};

export type TestingRunBulkSkipped = {
  training_run_id: number;
  training_dataset_id: number;
  roi_id: number | null;
  existing_testing_run_id: number;
  existing_name: string;
  reason: string;
};

export type TestingRunBulkError = {
  training_run_id: number | null;
  training_dataset_id: number | null;
  message: string;
};

export type TestingRunBulkResponse = {
  created: TestingRun[];
  skipped: TestingRunBulkSkipped[];
  errors: TestingRunBulkError[];
};

export type TestingRunResult = {
  id: number;
  position: number;
  image_path: string;
  timestamp: string;
  score: number;
  full_mse: number;
  roi_mse: number | null;
  tile_scores: Array<Record<string, unknown>> | null;
  result_metadata: Record<string, unknown> | null;
  width: number;
  height: number;
};

export type TestingRunResults = {
  testing_run: TestingRun;
  results: TestingRunResult[];
  total: number;
  decimated: boolean;
};

export type TestingRunResultImage = {
  testing_run_id: number;
  result_id: number;
  image_path: string;
  timestamp: string;
  width: number;
  height: number;
  channels: number;
  dtype: string;
  image_data_url: string;
};

export type HeatmapVisualizationConfig = {
  residual_source: 'pixel_residual' | 'ssim_residual';
  error_mode: 'squared' | 'absolute';
  threshold_enabled: boolean;
  threshold: number;
  max_clip_enabled: boolean;
  max_clip: number;
  max_opacity: number;
  fixed_ceiling_enabled: boolean;
  fixed_ceiling: number;
  signed_deviations: boolean;
  positive_weight: number;
  negative_weight: number;
  ssim_window_size: number;
  ssim_alpha: number;
  ssim_beta: number;
  ssim_gamma: number;
  ssim_k1: number;
  ssim_k2: number;
  ssim_data_range: number;
};

export type HeatmapRun = {
  id: number;
  testing_run_id: number;
  testing_result_id: number | null;
  status: string;
  error_message: string | null;
  image_path: string;
  timestamp: string;
  width: number;
  height: number;
  channels: number;
  dtype: string;
  max_error: number;
  mean_error: number;
  max_x: number;
  max_y: number;
  source_image_data_url: string;
  reconstruction_image_data_url: string;
  heatmap_image_data_url: string;
  error_matrix: number[][] | null;
  visualization_config: HeatmapVisualizationConfig;
  config_signature: string;
  render_version: number;
  created_at: string;
  updated_at: string;
};

/** Lightweight heatmap metadata from the list endpoint (no image data URLs or
 *  error matrix — those are fetched per heatmap). */
export type HeatmapRunSummary = {
  id: number;
  testing_run_id: number;
  testing_result_id: number | null;
  status: string;
  error_message: string | null;
  image_path: string;
  timestamp: string;
  width: number;
  height: number;
  max_error: number;
  mean_error: number;
  max_x: number;
  max_y: number;
  config_signature: string;
  render_version: number;
  created_at: string;
  updated_at: string;
};

export type HeatmapRangeRun = {
  id: number;
  testing_run_id: number;
  testing_run_name: string;
  status: string;
  error_message: string | null;
  enqueued_at: string | null;
  queue_rank: number | null;
  started_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
  gpu_index: number | null;
  device: string | null;
  start_timestamp: string;
  end_timestamp: string;
  stride: number;
  fps: number;
  scale_mode: 'per_frame' | 'shared';
  global_vmax: number | null;
  frame_max_errors: number[] | null;
  visualization_config: HeatmapVisualizationConfig;
  render_version: number;
  frame_count: number | null;
  done_count: number;
  video_path: string | null;
  config_signature: string;
  created_at: string;
  updated_at: string;
};

export type InspectPreview = {
  training_dataset_id: number;
  preprocessing_pipeline_id: number;
  start_timestamp: string;
  end_timestamp: string;
  stride: number;
  matching_images: number;
  selected_images: number;
  first_image_path: string;
  first_timestamp: string;
  width: number;
  height: number;
  channels: number;
  dtype: string;
  value_min: number;
  value_max: number;
  image_data_url: string;
  preview_frame_count: number;
  preview_frames: Array<{
    index: number;
    timestamp: string;
    image_path: string;
    image_data_url: string;
  }>;
  analysis_mode: string;
  analysis_config: Record<string, unknown> | null;
  roi_id: number | null;
  roi_name: string | null;
  generate_video: boolean;
  diagnostic_columns: string[];
  diagnostic_series: Array<Record<string, unknown>>;
  plot_image_data_url: string | null;
  preview_video_url: string | null;
  contrast_enabled?: boolean;
  contrast_reference_frames_used?: number | null;
  contrast_diff_min?: number | null;
  contrast_diff_max?: number | null;
};

export type InspectRun = {
  id: number;
  training_dataset_id: number;
  preprocessing_pipeline_id: number;
  status: string;
  enqueued_at: string | null;
  started_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
  error_message: string | null;
  device: string | null;
  start_timestamp: string;
  end_timestamp: string;
  stride: number;
  fps: number;
  content_mode: string;
  analysis_mode: string;
  analysis_config: Record<string, unknown> | null;
  roi_id: number | null;
  generate_video: boolean;
  contrast_enabled: boolean;
  contrast_reference_frames: number | null;
  contrast_shift: number | null;
  contrast_vmax: number | null;
  contrast_ma_radius: number | null;
  frame_count: number | null;
  done_count: number;
  frames_dir: string | null;
  video_path: string | null;
  csv_path: string | null;
  summary_json_path: string | null;
  plot_preview_path: string | null;
  overlay_video_path: string | null;
  training_dataset_name: string;
  preprocessing_pipeline_name: string;
  created_at: string;
  updated_at: string;
};

export type InspectArtifactRun = {
  kind: 'inspect' | 'heatmap';
  id: number;
  mode: string;
  status: string;
  error_message: string | null;
  training_dataset_id: number;
  training_dataset_name: string;
  preprocessing_pipeline_id: number;
  preprocessing_pipeline_name: string;
  start_timestamp: string;
  end_timestamp: string;
  stride: number;
  fps: number;
  frame_count: number | null;
  done_count: number;
  has_video: boolean;
  has_csv: boolean;
  has_summary: boolean;
  created_at: string;
  updated_at: string;
};

export type InspectArtifactRunPage = {
  items: InspectArtifactRun[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
  active_total: number;
};

export type InspectCsvData = {
  columns: Array<{ name: string; kind: 'number' | 'datetime' | 'text' }>;
  rows: Array<Record<string, string | number | null>>;
};

export type AnalysisLayout = {
  id: number;
  name: string;
  description: string | null;
  layout: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type OptimizationParameterSpec = {
  path: string;
  kind: 'int' | 'float' | 'categorical';
  low?: number | null;
  high?: number | null;
  step?: number | null;
  log?: boolean;
  choices?: Array<number | string | boolean> | null;
};

export type OptimizationStudyPayload = {
  name: string;
  description?: string | null;
  preprocessing_pipeline_id: number;
  method_configuration_ids: number[];
  normal_train_dataset_id: number;
  normal_validation_dataset_id: number;
  anomaly_validation_dataset_id: number;
  normal_holdout_dataset_id?: number | null;
  anomaly_holdout_dataset_id?: number | null;
  search_space: OptimizationParameterSpec[];
  objective_name: string;
  direction: 'maximize' | 'minimize';
  n_trials: number;
  max_parallel_trials: number;
  sampler: 'tpe' | 'random';
  split_config?: Record<string, unknown>;
  objective_config?: Record<string, unknown>;
};

export type OptimizationTrial = {
  id: number;
  study_id: number;
  number: number;
  status: string;
  phase: string;
  sampled_params: Record<string, unknown>;
  method_configuration_id: number | null;
  training_pipeline_id: number | null;
  training_run_id: number | null;
  normal_testing_run_id: number | null;
  anomaly_testing_run_id: number | null;
  normal_holdout_testing_run_id: number | null;
  anomaly_holdout_testing_run_id: number | null;
  objective_value: number | null;
  metrics: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
};

export type OptimizationStudy = {
  id: number;
  name: string;
  description: string | null;
  status: string;
  objective_name: string;
  direction: string;
  n_trials: number;
  max_parallel_trials: number;
  sampler: string;
  preprocessing_pipeline_id: number;
  preprocessing_pipeline_name: string;
  method_configuration_ids: number[];
  normal_train_dataset_id: number;
  normal_train_dataset_name: string;
  normal_validation_dataset_id: number;
  normal_validation_dataset_name: string;
  anomaly_validation_dataset_id: number;
  anomaly_validation_dataset_name: string;
  normal_holdout_dataset_id: number | null;
  normal_holdout_dataset_name: string | null;
  anomaly_holdout_dataset_id: number | null;
  anomaly_holdout_dataset_name: string | null;
  search_space: OptimizationParameterSpec[];
  split_config: Record<string, unknown>;
  objective_config: Record<string, unknown>;
  best_trial_id: number | null;
  best_value: number | null;
  error_message: string | null;
  started_at: string | null;
  ended_at: string | null;
  created_at: string;
  updated_at: string;
  trials: OptimizationTrial[];
};

export type OptimizationSplitPayload = {
  name_prefix: string;
  normal_source_dataset_id: number;
  anomaly_source_dataset_id: number;
  normal_train_fraction: number;
  normal_validation_fraction: number;
  anomaly_validation_fraction: number;
};

export type OptimizationSplit = {
  normal_train_dataset: TrainingDataset;
  normal_validation_dataset: TrainingDataset;
  normal_holdout_dataset: TrainingDataset;
  anomaly_validation_dataset: TrainingDataset;
  anomaly_holdout_dataset: TrainingDataset;
};

export type SchedulerSettings = {
  detected_gpu_count: number;
  max_gpu_slots: number;
  only_gpu: boolean;
};

export type ModelArchitecture = MethodDefinition;
export type ModelConfigurationParameter = MethodConfigurationParameter;
export type ModelConfiguration = MethodConfiguration;
export type ModelConfigurationPayload = MethodConfigurationPayload;
export type ModelConfigurationSavePayload = MethodConfigurationSavePayload;
export type ModelValidationResponse = MethodValidationResponse;

// -- Data Manager (registry) --------------------------------------------------

export type RegistryFilterDef = {
  key: string;
  label: string;
  kind: 'select' | 'daterange' | 'usage';
  options: string[] | null;
};

export type RegistryTypeSummary = {
  key: string;
  label: string;
  count: number;
  filters: RegistryFilterDef[];
};

export type RegistrySummary = { types: RegistryTypeSummary[] };

export type RegistryRow = {
  id: number;
  name: string;
  usage_count: number;
  disk_size_bytes?: number | null;
  status?: string;
  created_at?: string;
  [key: string]: unknown;
};

export type RegistryList = { total: number; rows: RegistryRow[] };

export type RegistryArtifact = {
  path: string;
  exists: boolean;
  is_dir: boolean;
  size_bytes: number;
};

export type RegistryDependent = { entity_type: string; id: number; name: string };

export type RegistryDetail = {
  entity_type: string;
  id: number;
  name: string;
  fields: Record<string, unknown>;
  artifacts: RegistryArtifact[];
  dependents: RegistryDependent[];
  blockers: string[];
};

export type RegistryItemRef = { entity_type: string; id: number };

export type RegistryDeletePreview = {
  groups: Array<{
    entity_type: string;
    label: string;
    items: Array<{ id: number; name: string; selected: boolean }>;
  }>;
  total_objects: number;
  dependent_objects: number;
  files: RegistryArtifact[];
  total_bytes: number;
  blockers: string[];
  notes: string[];
};

export type RegistryDeleteResult = { deleted: Record<string, number>; freed_bytes: number };
