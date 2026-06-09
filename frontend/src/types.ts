export type DatasetFolder = {
  id: number;
  relative_path: string;
  image_count: number;
  first_timestamp: string | null;
  last_timestamp: string | null;
  extension_summary: Record<string, number> | null;
  resolution_summary: Record<string, number> | null;
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
