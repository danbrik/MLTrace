export type DatasetFolder = {
  id: number;
  relative_path: string;
  image_count: number;
  first_timestamp: string | null;
  last_timestamp: string | null;
  extension_summary: Record<string, number> | null;
  resolution_summary: Record<string, number> | null;
  image_metadata: Record<string, unknown> | null;
  cadence_summary: {
    min_seconds: number | null;
    median_seconds: number | null;
    max_seconds: number | null;
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
};

export type DatasetConnectionTest = {
  root_path: string;
  exists: boolean;
  is_directory: boolean;
  supported_file_found: boolean;
  sample_file_path: string | null;
  message: string;
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
  notes: string | null;
  created_at: string;
  dataset_names: string[];
  image_resolutions: string[];
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
    start_timestamp: string;
    end_timestamp: string;
    stride: number;
    matching_images: number;
    selected_images: number;
  }>;
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

export type ModelArchitecture = MethodDefinition;
export type ModelConfigurationParameter = MethodConfigurationParameter;
export type ModelConfiguration = MethodConfiguration;
export type ModelConfigurationPayload = MethodConfigurationPayload;
export type ModelConfigurationSavePayload = MethodConfigurationSavePayload;
export type ModelValidationResponse = MethodValidationResponse;
