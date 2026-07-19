from datetime import datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    root_path: str = Field(min_length=1)


class DatasetConnectionTestRequest(BaseModel):
    root_path: str = Field(min_length=1)


class DatasetConnectionTestResponse(BaseModel):
    root_path: str
    exists: bool
    is_directory: bool
    supported_file_found: bool
    sample_file_path: str | None
    message: str


class TimestampFormatConfirm(BaseModel):
    timestamp_regex: str = Field(min_length=1)
    timestamp_format: str = Field(min_length=1)


class DatasetFolderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    relative_path: str
    image_count: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    extension_summary: dict | None
    resolution_summary: dict | None
    image_metadata: dict | None
    cadence_summary: dict | None
    filename_template: dict | None = None


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    root_path: str
    status: str
    timestamp_regex: str | None
    timestamp_format: str | None
    timestamp_example: str | None
    scan_error: str | None
    scan_summary: dict | None
    created_at: datetime
    updated_at: datetime
    folders: list[DatasetFolderRead] = []
    is_update_locked: bool = False
    update_lock_reasons: list[str] = []


class TrainingDatasetRuleInput(BaseModel):
    folder_id: int
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int = Field(default=1, ge=1)

    @field_validator("end_timestamp")
    @classmethod
    def validate_range(cls, value: datetime, info):
        start = info.data.get("start_timestamp")
        if start and value < start:
            raise ValueError("end_timestamp must be after start_timestamp")
        return value


class TrainingDatasetPreviewRequest(BaseModel):
    rules: list[TrainingDatasetRuleInput]


class TrainingDatasetRulePreview(BaseModel):
    folder_id: int
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int
    matching_images: int
    selected_images: int


class TrainingDatasetPreviewResponse(BaseModel):
    total_matching_images: int
    total_selected_images: int
    rules: list[TrainingDatasetRulePreview]


class TrainingDatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    usage_label: str = Field(default="train", pattern="^(train|test|validation|mixed)$")
    notes: str | None = None
    rules: list[TrainingDatasetRuleInput] = Field(min_length=1)


class TrainingDatasetRuleRead(BaseModel):
    id: int
    folder_id: int
    dataset_id: int
    dataset_name: str
    dataset_root_path: str
    folder_relative_path: str
    folder_first_timestamp: datetime | None
    folder_last_timestamp: datetime | None
    folder_extension_summary: dict | None = None
    folder_resolution_summary: dict | None = None
    folder_image_metadata: dict | None = None
    folder_image_signature: str | None = None
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int
    matching_images: int | None = None
    selected_images: int | None = None


class TrainingDatasetRead(BaseModel):
    id: int
    name: str
    usage_label: str = "train"
    notes: str | None
    created_at: datetime
    updated_at: datetime | None = None
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    dataset_names: list[str]
    # Sorted unique "WxH" image resolutions across all rule folders. Drives the
    # size column and the size-compatibility cross-filtering on the UI.
    image_resolutions: list[str] = []
    image_signatures: list[str] = []
    total_matching_images: int
    total_selected_images: int
    rules: list[TrainingDatasetRuleRead] = []
    is_update_locked: bool = False
    update_lock_reasons: list[str] = []
    invalid_rule_count: int = 0
    integrity_warnings: list[str] = []
    counts_missing: bool = False


class TrainingDatasetSummaryRead(BaseModel):
    id: int
    name: str
    usage_label: str = "train"
    notes: str | None
    created_at: datetime
    updated_at: datetime | None = None
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    dataset_names: list[str]
    image_resolutions: list[str] = []
    image_signatures: list[str] = []
    total_matching_images: int
    total_selected_images: int
    is_update_locked: bool = False
    update_lock_reasons: list[str] = []
    invalid_rule_count: int = 0
    integrity_warnings: list[str] = []
    counts_missing: bool = False


class PreprocessingGraphNode(BaseModel):
    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    config: dict = Field(default_factory=dict)
    position: dict | None = None


class PreprocessingGraphEdge(BaseModel):
    id: str | None = None
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


class PreprocessingGraph(BaseModel):
    nodes: list[PreprocessingGraphNode] = Field(min_length=1)
    edges: list[PreprocessingGraphEdge] = []


class PreprocessingStepRead(BaseModel):
    type: str
    label: str
    category: str
    input_kind: str
    output_kind: str
    config_schema: dict
    default_config: dict


class PreprocessingPipelineCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    graph: PreprocessingGraph
    preview_folder_id: int | None = None
    input_width: int | None = None
    input_height: int | None = None
    output_width: int | None = None
    output_height: int | None = None


class PreprocessingPipelineRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    graph: dict
    preview_folder_id: int | None
    input_width: int | None
    input_height: int | None
    output_width: int | None
    output_height: int | None
    created_at: datetime
    updated_at: datetime
    is_update_locked: bool = False
    update_lock_reasons: list[str] = []
    step_count: int | None = None
    step_types: list[str] = []


class PreprocessingPipelineSummaryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    preview_folder_id: int | None
    input_width: int | None
    input_height: int | None
    output_width: int | None
    output_height: int | None
    created_at: datetime
    updated_at: datetime
    is_update_locked: bool = False
    update_lock_reasons: list[str] = []
    step_count: int
    step_types: list[str] = []


class PreprocessingPreviewRequest(BaseModel):
    folder_id: int
    graph: PreprocessingGraph


class PreprocessingPreviewImage(BaseModel):
    node_id: str
    step_type: str
    label: str
    width: int
    height: int
    channels: int
    dtype: str
    value_min: float
    value_max: float
    image_data_url: str


class PreprocessingPreviewResponse(BaseModel):
    source_image_id: int
    source_image_path: str
    source_timestamp: datetime
    previews: list[PreprocessingPreviewImage]


class InspectPreviewRequest(BaseModel):
    training_dataset_id: int
    preprocessing_pipeline_id: int
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int = Field(default=1, ge=1)
    content_mode: Literal["final_preprocessed_output"] = "final_preprocessed_output"
    analysis_mode: Literal["preprocessed_video", "contrast_enhanced", "energy", "optical_flow"] = "preprocessed_video"
    analysis_config: dict | None = None
    roi_id: int | None = None
    generate_video: bool = True
    fps: int = Field(default=12, ge=1, le=60)

    contrast_enabled: bool = False
    contrast_reference_frames: int = Field(default=100, ge=1)
    contrast_shift: float = 10000.0
    contrast_vmax: float = Field(default=12000.0, gt=0)
    contrast_ma_radius: int = Field(default=3, ge=0)

    @field_validator("end_timestamp")
    @classmethod
    def validate_range(cls, value: datetime, info):
        start = info.data.get("start_timestamp")
        if start and value < start:
            raise ValueError("end_timestamp must be after start_timestamp")
        return value


class InspectPreviewResponse(BaseModel):
    training_dataset_id: int
    preprocessing_pipeline_id: int
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int
    matching_images: int
    selected_images: int
    first_image_path: str
    first_timestamp: datetime
    width: int
    height: int
    channels: int
    dtype: str
    value_min: float
    value_max: float
    image_data_url: str
    preview_frame_count: int = 1
    preview_frames: list[dict] = []
    analysis_mode: str = "preprocessed_video"
    analysis_config: dict | None = None
    roi_id: int | None = None
    roi_name: str | None = None
    generate_video: bool = True
    diagnostic_columns: list[str] = []
    diagnostic_series: list[dict] = []
    plot_image_data_url: str | None = None
    preview_video_url: str | None = None
    contrast_enabled: bool = False
    contrast_reference_frames_used: int | None = None
    contrast_diff_min: float | None = None
    contrast_diff_max: float | None = None


class InspectRunCreate(InspectPreviewRequest):
    fps: int = Field(default=12, ge=1, le=60)


class InspectRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    training_dataset_id: int
    preprocessing_pipeline_id: int
    status: str
    enqueued_at: datetime | None
    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: float | None
    error_message: str | None
    device: str | None
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int
    fps: int
    content_mode: str
    analysis_mode: str
    analysis_config: dict | None
    roi_id: int | None
    generate_video: bool
    contrast_enabled: bool
    contrast_reference_frames: int | None
    contrast_shift: float | None
    contrast_vmax: float | None
    contrast_ma_radius: int | None
    frame_count: int | None
    done_count: int
    frames_dir: str | None
    video_path: str | None
    csv_path: str | None
    summary_json_path: str | None
    plot_preview_path: str | None
    overlay_video_path: str | None
    training_dataset_name: str
    preprocessing_pipeline_name: str
    created_at: datetime
    updated_at: datetime


class AnalysisLayoutCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    layout: dict = Field(default_factory=dict)


class AnalysisLayoutRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    layout: dict
    created_at: datetime
    updated_at: datetime


class OptimizationParameterSpec(BaseModel):
    path: str = Field(min_length=1)
    kind: Literal["int", "float", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list[float | int | str | bool] | None = None


class OptimizationStudyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    preprocessing_pipeline_id: int
    method_configuration_ids: list[int] = Field(min_length=1)
    normal_train_dataset_id: int
    normal_validation_dataset_id: int
    anomaly_validation_dataset_id: int
    normal_holdout_dataset_id: int | None = None
    anomaly_holdout_dataset_id: int | None = None
    search_space: list[OptimizationParameterSpec] = Field(default_factory=list)
    objective_name: Literal[
        "median_anomaly_minus_p95_normal",
        "mean_gap",
        "roc_auc",
        "pr_auc",
        "normal_validation_loss",
    ] = "median_anomaly_minus_p95_normal"
    direction: Literal["maximize", "minimize"] = "maximize"
    n_trials: int = Field(default=10, ge=1, le=1000)
    max_parallel_trials: int = Field(default=1, ge=1, le=64)
    sampler: Literal["tpe", "random"] = "tpe"
    split_config: dict = Field(default_factory=dict)
    objective_config: dict = Field(default_factory=dict)

    @field_validator("method_configuration_ids")
    @classmethod
    def validate_method_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("method_configuration_ids must not contain duplicates")
        return value


class OptimizationStudyUpdate(OptimizationStudyCreate):
    pass


class OptimizationSplitCreate(BaseModel):
    name_prefix: str = Field(min_length=1, max_length=160)
    normal_source_dataset_id: int
    anomaly_source_dataset_id: int
    normal_train_fraction: float = Field(default=0.75, gt=0.0, lt=1.0)
    normal_validation_fraction: float = Field(default=0.125, gt=0.0, lt=1.0)
    anomaly_validation_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_fractions(self):
        if self.normal_train_fraction + self.normal_validation_fraction >= 1.0:
            raise ValueError("normal_train_fraction + normal_validation_fraction must be < 1.0")
        return self


class OptimizationSplitRead(BaseModel):
    normal_train_dataset: TrainingDatasetRead
    normal_validation_dataset: TrainingDatasetRead
    normal_holdout_dataset: TrainingDatasetRead
    anomaly_validation_dataset: TrainingDatasetRead
    anomaly_holdout_dataset: TrainingDatasetRead


class OptimizationTrialRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    study_id: int
    number: int
    status: str
    phase: str
    sampled_params: dict
    method_configuration_id: int | None
    training_pipeline_id: int | None
    training_run_id: int | None
    normal_testing_run_id: int | None
    anomaly_testing_run_id: int | None
    normal_holdout_testing_run_id: int | None
    anomaly_holdout_testing_run_id: int | None
    objective_value: float | None
    metrics: dict | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class OptimizationStudyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    status: str
    objective_name: str
    direction: str
    n_trials: int
    max_parallel_trials: int
    sampler: str
    preprocessing_pipeline_id: int
    preprocessing_pipeline_name: str
    method_configuration_ids: list[int]
    normal_train_dataset_id: int
    normal_train_dataset_name: str
    normal_validation_dataset_id: int
    normal_validation_dataset_name: str
    anomaly_validation_dataset_id: int
    anomaly_validation_dataset_name: str
    normal_holdout_dataset_id: int | None
    normal_holdout_dataset_name: str | None
    anomaly_holdout_dataset_id: int | None
    anomaly_holdout_dataset_name: str | None
    search_space: list[dict]
    split_config: dict
    objective_config: dict
    best_trial_id: int | None
    best_value: float | None
    error_message: str | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime
    updated_at: datetime
    trials: list[OptimizationTrialRead] = []


class OptimizationPromoteRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class MethodDefinitionRead(BaseModel):
    type: str
    label: str
    category: str
    description: str
    framework: str
    method_family: str
    method_version: str
    training_mode: str
    architecture_version: str
    requires_training: bool
    supports_training_pipeline: bool
    artifact_kind: str
    builder_kind: str
    capabilities: dict
    method_schema: dict
    model_schema: dict
    training_schema: dict
    inference_schema: dict
    default_method_config: dict
    default_model_config: dict
    default_training_config: dict
    default_inference_config: dict


class ModelLayerRead(BaseModel):
    type: str
    label: str
    category: str
    config_schema: dict
    default_config: dict
    input_rank: int | None
    output_rank: int | None
    shape_notes: str | None = None


class MethodConfigurationParameterRead(BaseModel):
    path: str
    value_type: str
    value_text: str | None = None
    value_number: float | None = None
    value_bool: bool | None = None


class MethodConfigurationPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    method_type: str | None = Field(default=None, min_length=1, max_length=128)
    architecture_type: str | None = Field(default=None, min_length=1, max_length=128)
    method_graph: dict = Field(default_factory=dict)
    model_graph: dict = Field(default_factory=dict)
    method_config: dict = Field(default_factory=dict)
    model_params: dict = Field(default_factory=dict, alias="model_config")
    training_config: dict = Field(default_factory=dict)
    inference_config: dict = Field(default_factory=dict)


class MethodConfigurationCreate(MethodConfigurationPayload):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class MethodConfigurationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    name: str
    description: str | None
    method_type: str
    method_family: str
    method_version: str
    training_mode: str
    architecture_type: str
    architecture_version: str
    requires_training: bool
    supports_training_pipeline: bool
    artifact_kind: str
    builder_kind: str
    method_graph: dict
    model_graph: dict
    method_config: dict
    model_params: dict = Field(alias="model_config")
    training_config: dict
    inference_config: dict
    diagram: dict
    created_at: datetime
    updated_at: datetime
    validation: dict | None = None
    parameters: list[MethodConfigurationParameterRead] = []
    is_update_locked: bool = False
    update_lock_reasons: list[str] = []


class MethodConfigurationSummaryRead(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    name: str
    description: str | None
    method_type: str
    method_family: str
    method_version: str
    training_mode: str
    architecture_type: str
    architecture_version: str
    requires_training: bool
    supports_training_pipeline: bool
    artifact_kind: str
    builder_kind: str
    method_config: dict
    model_params: dict = Field(alias="model_config")
    training_config: dict
    inference_config: dict
    created_at: datetime
    updated_at: datetime
    validation: dict | None = None
    is_update_locked: bool = False
    update_lock_reasons: list[str] = []


class MethodConfigurationValidationResponse(BaseModel):
    valid: bool
    errors: list[str] = []
    warnings: list[str] = []
    layer_specs: list[dict] = []
    torch_check: dict | None = None
    diagram: dict


class MethodTorchCheckResponse(BaseModel):
    valid: bool
    status: str
    errors: list[str] = []
    warnings: list[str] = []
    logs: list[str] = []
    torch_check: dict | None = None


class TrainingPipelinePayload(BaseModel):
    """The composition of a training pipeline, shared by save and dry-run requests."""

    training_dataset_ids: list[int] = Field(min_length=1)
    preprocessing_pipeline_id: int
    method_configuration_id: int
    shuffle: bool = True
    training_parameters: dict = Field(default_factory=dict)

    @field_validator("training_dataset_ids")
    @classmethod
    def validate_unique_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("training_dataset_ids must not contain duplicates")
        return value


class TrainingPipelineCreate(TrainingPipelinePayload):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class TrainingPipelineDatasetRead(BaseModel):
    training_dataset_id: int
    position: int
    name: str
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    total_selected_images: int
    dataset_names: list[str]


class TrainingPipelineRead(BaseModel):
    id: int
    name: str
    description: str | None
    shuffle: bool
    training_parameters: dict
    preprocessing_pipeline_id: int
    preprocessing_pipeline_name: str
    preprocessing_input_width: int | None
    preprocessing_input_height: int | None
    preprocessing_output_width: int | None
    preprocessing_output_height: int | None
    method_configuration_id: int
    method_configuration_name: str
    method_type: str
    training_mode: str
    builder_kind: str
    total_selected_images: int
    training_datasets: list[TrainingPipelineDatasetRead] = []
    created_at: datetime
    updated_at: datetime
    is_update_locked: bool = False
    update_lock_reasons: list[str] = []


class TrainingPipelineSummaryRead(BaseModel):
    id: int
    name: str
    description: str | None
    shuffle: bool
    training_parameters: dict
    preprocessing_pipeline_id: int
    preprocessing_pipeline_name: str
    preprocessing_input_width: int | None
    preprocessing_input_height: int | None
    preprocessing_output_width: int | None
    preprocessing_output_height: int | None
    method_configuration_id: int
    method_configuration_name: str
    method_type: str
    training_mode: str
    builder_kind: str
    total_selected_images: int
    training_datasets: list[TrainingPipelineDatasetRead] = []
    created_at: datetime
    updated_at: datetime
    is_update_locked: bool = False
    update_lock_reasons: list[str] = []


class TrainingPipelineDryRunRequest(TrainingPipelinePayload):
    """Dry-run works on saved and unsaved compositions; the client always sends the full composition."""


class TrainingPipelineModelOutput(BaseModel):
    input_shape: list[int]
    output_shape: list[int]
    width: int
    height: int
    channels: int
    dtype: str
    value_min: float
    value_max: float
    image_data_url: str
    elapsed_ms: float


class TrainingPipelineDryRunResponse(BaseModel):
    """Result of pushing the first training image through preprocessing and the model.

    Composition-level findings (shape mismatch, missing images) are reported
    in-band via valid/errors instead of HTTP errors so the UI can still render
    the stages that did succeed.
    """

    valid: bool
    mode: str  # "forward_pass" | "fit_contribution" | "failed"
    errors: list[str] = []
    warnings: list[str] = []
    logs: list[str] = []
    training_dataset_name: str | None = None
    source_image_path: str | None = None
    source_timestamp: datetime | None = None
    # Step previews in chain order: index 0 is the loaded image before any
    # processing, the last entry is the final preprocessing output.
    preprocessing_previews: list[PreprocessingPreviewImage] = []
    model_output: TrainingPipelineModelOutput | None = None
    note: str | None = None


class TrainingPipelineDuplicateResponse(BaseModel):
    """Result of checking whether an identical pipeline configuration exists."""

    existing_pipeline: TrainingPipelineRead | None = None


class TrainingRunEnqueueRequest(BaseModel):
    training_pipeline_id: int


class SchedulerJobMoveRequest(BaseModel):
    direction: Literal["up", "down"]


class SchedulerJobMoveResponse(BaseModel):
    kind: Literal["train", "test", "heatmap"]
    run_id: int
    queue_rank: int | None


class TrainingRunMetricRead(BaseModel):
    epoch: int
    train_loss: float | None
    val_loss: float | None


class TrainingRunRead(BaseModel):
    id: int
    training_pipeline_id: int
    status: str
    enqueued_at: datetime | None
    queue_rank: int | None = None
    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: float | None
    gpu_index: int | None
    device: str | None
    epochs_total: int | None
    epochs_completed: int
    train_loss: float | None
    val_loss: float | None
    best_val_loss: float | None
    image_count: int | None
    skipped_image_count: int | None = None
    skipped_images: list[str] | None = None
    artifact_kind: str | None
    artifact_path: str | None
    artifact_size_bytes: int | None
    error_message: str | None
    # Denormalized pipeline snapshot (for display + filtering).
    training_pipeline_name: str
    method_type: str
    method_family: str
    training_mode: str
    builder_kind: str
    preprocessing_pipeline_name: str
    dataset_names: list[str]
    shuffle: bool
    input_resolution: str | None
    epochs: int | None
    learning_rate: float | None
    training_parameters: dict
    created_at: datetime
    updated_at: datetime
    metrics: list[TrainingRunMetricRead] = []


class TrainingRunLogResponse(BaseModel):
    log: str


class RoiPoint(BaseModel):
    x: float
    y: float


class RoiDefinitionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    geometry_type: str = "polygon"
    points: list[RoiPoint] | None = None
    tile_rows: int = Field(default=1, ge=1, le=20)
    tile_cols: int = Field(default=1, ge=1, le=20)

    @field_validator("width")
    @classmethod
    def validate_width(cls, value: int, info):
        x = info.data.get("x")
        image_width = info.data.get("image_width")
        if x is not None and image_width is not None and x + value > image_width:
            raise ValueError("ROI width extends beyond image_width")
        return value

    @field_validator("height")
    @classmethod
    def validate_height(cls, value: int, info):
        y = info.data.get("y")
        image_height = info.data.get("image_height")
        if y is not None and image_height is not None and y + value > image_height:
            raise ValueError("ROI height extends beyond image_height")
        return value


class RoiDefinitionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    image_width: int
    image_height: int
    x: int
    y: int
    width: int
    height: int
    geometry_type: str = "polygon"
    points: list[dict] | None = None
    tile_rows: int = 1
    tile_cols: int = 1
    created_at: datetime
    updated_at: datetime


class RoiPreviewRequest(BaseModel):
    training_run_id: int
    training_dataset_id: int


class RoiPreviewResponse(BaseModel):
    training_run_id: int
    training_dataset_id: int
    preprocessing_pipeline_id: int
    source_image_path: str
    source_timestamp: datetime
    width: int
    height: int
    channels: int
    dtype: str
    image_data_url: str


class TestingRunCreate(BaseModel):
    __test__: ClassVar[bool] = False

    training_run_id: int
    training_dataset_id: int
    roi_id: int | None = None
    name: str | None = Field(default=None, max_length=255)
    inference_config: dict | None = None


class TestingRunBulkCreate(BaseModel):
    __test__: ClassVar[bool] = False

    training_run_ids: list[int] = Field(min_length=1)
    training_dataset_ids: list[int] = Field(min_length=1)
    roi_id: int | None = None
    name_prefix: str | None = Field(default=None, max_length=255)
    inference_config: dict | None = None


class TestingRunBulkSkipped(BaseModel):
    __test__: ClassVar[bool] = False

    training_run_id: int
    training_dataset_id: int
    roi_id: int | None = None
    existing_testing_run_id: int
    existing_name: str
    reason: str


class TestingRunBulkError(BaseModel):
    __test__: ClassVar[bool] = False

    training_run_id: int | None = None
    training_dataset_id: int | None = None
    message: str


class TestingRunResultRead(BaseModel):
    __test__: ClassVar[bool] = False

    id: int
    position: int
    image_path: str
    timestamp: datetime
    score: float
    full_mse: float
    roi_mse: float | None
    tile_scores: list[dict] | None = None
    result_metadata: dict | None = None
    width: int
    height: int


class TestingRunRead(BaseModel):
    __test__: ClassVar[bool] = False

    id: int
    name: str
    training_run_id: int
    training_dataset_id: int
    roi_id: int | None
    status: str
    enqueued_at: datetime | None = None
    queue_rank: int | None = None
    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: float | None
    gpu_index: int | None = None
    device: str | None = None
    error_message: str | None
    image_count: int | None
    expected_image_count: int | None = None
    skipped_image_count: int | None = None
    skipped_images: list[str] | None = None
    score_mean: float | None
    score_min: float | None
    score_max: float | None
    full_mse_mean: float | None
    roi_mse_mean: float | None
    results_path: str | None
    results_size_bytes: int | None
    training_run_name: str
    training_pipeline_name: str
    training_dataset_name: str
    preprocessing_pipeline_name: str
    method_type: str
    method_family: str
    training_mode: str
    artifact_kind: str
    artifact_path: str
    roi_name: str | None
    roi_geometry: dict | None
    inference_config: dict | None = None
    created_at: datetime
    updated_at: datetime


class TestingRunBulkResponse(BaseModel):
    __test__: ClassVar[bool] = False

    created: list[TestingRunRead]
    skipped: list[TestingRunBulkSkipped] = []
    errors: list[TestingRunBulkError] = []


class TestingRunResultsResponse(BaseModel):
    __test__: ClassVar[bool] = False

    testing_run: TestingRunRead
    results: list[TestingRunResultRead]
    # Total stored rows; ``results`` may be decimated to <= max_points for charts.
    total: int = 0
    decimated: bool = False


class TestingRunResultImageResponse(BaseModel):
    __test__: ClassVar[bool] = False

    testing_run_id: int
    result_id: int
    image_path: str
    timestamp: datetime
    width: int
    height: int
    channels: int
    dtype: str
    image_data_url: str


class HeatmapVisualizationConfig(BaseModel):
    residual_source: Literal["pixel_residual", "ssim_residual"] = "pixel_residual"
    error_mode: Literal["squared", "absolute"] = "squared"
    threshold_enabled: bool = False
    threshold: float = Field(default=0.0, ge=0.0)
    max_clip_enabled: bool = False
    max_clip: float = Field(default=0.33, gt=0.0, le=1.0)
    max_opacity: float = Field(default=0.55, ge=0.0, le=1.0)
    fixed_ceiling_enabled: bool = False
    fixed_ceiling: float = Field(default=1.0, gt=0.0)
    signed_deviations: bool = False
    positive_weight: float = Field(default=1.0, ge=0.0)
    negative_weight: float = Field(default=1.0, ge=0.0)
    ssim_window_size: int = Field(default=11, ge=3)
    ssim_alpha: float = Field(default=1.0, ge=0.0)
    ssim_beta: float = Field(default=1.0, ge=0.0)
    ssim_gamma: float = Field(default=1.0, ge=0.0)
    ssim_k1: float = Field(default=0.01, ge=0.0)
    ssim_k2: float = Field(default=0.03, ge=0.0)
    ssim_data_range: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def validate_normalization_mode(self):
        if self.fixed_ceiling_enabled and self.max_clip_enabled:
            raise ValueError("Fixed ceiling and max clip cannot be enabled at the same time.")
        return self


class HeatmapRunCreate(BaseModel):
    testing_run_id: int
    testing_result_id: int | None = None
    timestamp: datetime | None = None
    force_recompute: bool = False
    stae_view: Literal["reconstruction", "prediction"] = "reconstruction"
    prediction_horizon: int = Field(default=1, ge=1)
    visualization_config: HeatmapVisualizationConfig = Field(default_factory=HeatmapVisualizationConfig)

    @model_validator(mode="after")
    def validate_result_or_timestamp(self):
        if self.testing_result_id is None and self.timestamp is None:
            raise ValueError("Either testing_result_id or timestamp is required.")
        return self


class HeatmapRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    testing_run_id: int
    testing_result_id: int | None
    status: str
    error_message: str | None
    image_path: str
    timestamp: datetime
    width: int
    height: int
    channels: int
    dtype: str
    max_error: float
    mean_error: float
    max_x: int
    max_y: int
    source_image_data_url: str
    reconstruction_image_data_url: str = ""
    heatmap_image_data_url: str
    error_matrix: list[list[float]] | None = None
    visualization_config: HeatmapVisualizationConfig
    config_signature: str
    render_version: int
    created_at: datetime
    updated_at: datetime


class HeatmapRunSummary(BaseModel):
    """Lightweight heatmap row for list endpoints: metadata only, no image data
    URLs or error matrix (those live on disk and are fetched per heatmap)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    testing_run_id: int
    testing_result_id: int | None
    status: str
    error_message: str | None
    image_path: str
    timestamp: datetime
    width: int
    height: int
    max_error: float
    mean_error: float
    max_x: int
    max_y: int
    visualization_config: HeatmapVisualizationConfig
    config_signature: str
    render_version: int
    created_at: datetime


class CacheRevisionsRead(BaseModel):
    revisions: dict[str, str]


class HeatmapRangeRunCreate(BaseModel):
    testing_run_id: int
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int = Field(default=1, ge=1)
    fps: int = Field(default=8, ge=1, le=60)
    scale_mode: Literal["per_frame", "shared"] = "per_frame"
    visualization_config: HeatmapVisualizationConfig = Field(default_factory=HeatmapVisualizationConfig)
    force_recompute: bool = False


class HeatmapRangeRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    testing_run_id: int
    testing_run_name: str
    status: str
    error_message: str | None
    enqueued_at: datetime | None
    queue_rank: int | None = None
    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: float | None
    gpu_index: int | None
    device: str | None
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int
    fps: int
    scale_mode: str
    global_vmax: float | None
    frame_max_errors: list[float] | None
    visualization_config: HeatmapVisualizationConfig
    render_version: int
    frame_count: int | None
    done_count: int
    video_path: str | None
    config_signature: str
    created_at: datetime
    updated_at: datetime


class SchedulerJobWithProjectRead(BaseModel):
    project_id: str
    project_name: str
    kind: Literal["train", "test", "heatmap"]
    queue_rank: int | None = None
    run: TrainingRunRead | TestingRunRead | HeatmapRangeRunRead


class InspectArtifactRunRead(BaseModel):
    kind: Literal["inspect", "heatmap"]
    id: int
    mode: str
    status: str
    error_message: str | None
    training_dataset_id: int
    training_dataset_name: str
    preprocessing_pipeline_id: int
    preprocessing_pipeline_name: str
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int
    fps: int
    frame_count: int | None
    done_count: int
    has_video: bool
    has_csv: bool
    has_summary: bool
    created_at: datetime
    updated_at: datetime


class InspectArtifactRunPage(BaseModel):
    items: list[InspectArtifactRunRead]
    total: int
    page: int
    page_size: int
    pages: int
    active_total: int


class InspectCsvColumn(BaseModel):
    name: str
    kind: Literal["number", "datetime", "text"]


class InspectCsvData(BaseModel):
    columns: list[InspectCsvColumn]
    rows: list[dict]


class SchedulerSettingsRead(BaseModel):
    detected_gpu_count: int
    max_gpu_slots: int
    only_gpu: bool


class SchedulerSettingsUpdate(BaseModel):
    max_gpu_slots: int = Field(ge=1)
    only_gpu: bool = False


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)


class ProjectRead(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    last_opened_at: datetime | None = None


class ProjectGpuUsageRead(BaseModel):
    project_id: str
    project_name: str
    gpu_memory_mb: int = 0
    running_jobs: int = 0
    queued_jobs: int = 0
    gpu_slots: int = 0


class GpuDeviceUsageRead(BaseModel):
    index: int
    uuid: str
    name: str
    utilization_percent: float
    memory_used_mb: int
    memory_total_mb: int
    temperature_c: float | None = None
    mltrace_memory_mb: int = 0
    projects: list[ProjectGpuUsageRead] = Field(default_factory=list)


class GpuSnapshotRead(BaseModel):
    captured_at: datetime
    available: bool
    error: str | None = None
    devices: list[GpuDeviceUsageRead] = Field(default_factory=list)
    mltrace_memory_mb: int = 0
    running_jobs: int = 0
    queued_jobs: int = 0
    gpu_slots: int = 0
    projects: list[ProjectGpuUsageRead] = Field(default_factory=list)


class RegistryItemRef(BaseModel):
    entity_type: str
    id: int


class RegistryDeleteRequest(BaseModel):
    items: list[RegistryItemRef] = Field(min_length=1)
    cascade: bool = False


ModelArchitectureRead = MethodDefinitionRead
ModelConfigurationParameterRead = MethodConfigurationParameterRead
ModelConfigurationPayload = MethodConfigurationPayload
ModelConfigurationCreate = MethodConfigurationCreate
ModelConfigurationRead = MethodConfigurationRead
ModelConfigurationValidationResponse = MethodConfigurationValidationResponse
