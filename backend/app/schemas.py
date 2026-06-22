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


class TrainingRunMetricRead(BaseModel):
    epoch: int
    train_loss: float | None
    val_loss: float | None


class TrainingRunRead(BaseModel):
    id: int
    training_pipeline_id: int
    status: str
    enqueued_at: datetime | None
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
    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: float | None
    gpu_index: int | None = None
    device: str | None = None
    error_message: str | None
    image_count: int | None
    expected_image_count: int | None = None
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
    created_at: datetime
    updated_at: datetime


class TestingRunResultsResponse(BaseModel):
    __test__: ClassVar[bool] = False

    testing_run: TestingRunRead
    results: list[TestingRunResultRead]


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


class HeatmapRangeRunCreate(BaseModel):
    testing_run_id: int
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int = Field(default=1, ge=1)
    scale_mode: Literal["per_frame", "shared"] = "per_frame"
    visualization_config: HeatmapVisualizationConfig = Field(default_factory=HeatmapVisualizationConfig)


class HeatmapRangeRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    testing_run_id: int
    testing_run_name: str
    status: str
    error_message: str | None
    enqueued_at: datetime | None
    started_at: datetime | None
    ended_at: datetime | None
    duration_seconds: float | None
    gpu_index: int | None
    device: str | None
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int
    scale_mode: str
    global_vmax: float | None
    frame_max_errors: list[float] | None
    visualization_config: HeatmapVisualizationConfig
    render_version: int
    frame_count: int | None
    done_count: int
    config_signature: str
    created_at: datetime
    updated_at: datetime


class SchedulerSettingsRead(BaseModel):
    detected_gpu_count: int
    max_gpu_slots: int
    only_gpu: bool


class SchedulerSettingsUpdate(BaseModel):
    max_gpu_slots: int = Field(ge=1)
    only_gpu: bool = False


ModelArchitectureRead = MethodDefinitionRead
ModelConfigurationParameterRead = MethodConfigurationParameterRead
ModelConfigurationPayload = MethodConfigurationPayload
ModelConfigurationCreate = MethodConfigurationCreate
ModelConfigurationRead = MethodConfigurationRead
ModelConfigurationValidationResponse = MethodConfigurationValidationResponse
