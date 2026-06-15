from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    start_timestamp: datetime
    end_timestamp: datetime
    stride: int
    matching_images: int
    selected_images: int


class TrainingDatasetRead(BaseModel):
    id: int
    name: str
    notes: str | None
    created_at: datetime
    dataset_names: list[str]
    # Sorted unique "WxH" image resolutions across all rule folders. Drives the
    # size column and the size-compatibility cross-filtering on the UI.
    image_resolutions: list[str] = []
    total_matching_images: int
    total_selected_images: int
    rules: list[TrainingDatasetRuleRead] = []


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


ModelArchitectureRead = MethodDefinitionRead
ModelConfigurationParameterRead = MethodConfigurationParameterRead
ModelConfigurationPayload = MethodConfigurationPayload
ModelConfigurationCreate = MethodConfigurationCreate
ModelConfigurationRead = MethodConfigurationRead
ModelConfigurationValidationResponse = MethodConfigurationValidationResponse
