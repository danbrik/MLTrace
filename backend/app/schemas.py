from datetime import datetime
import math
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
    analysis_mode: Literal["preprocessed_video", "contrast_enhanced", "energy", "optical_flow", "temporal_dynamics"] = "preprocessed_video"
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


class TemporalDynamicsRequest(BaseModel):
    training_dataset_id: int
    preprocessing_pipeline_id: int
    reference_timestamp: datetime
    analysis_window_seconds: int = Field(default=1800, ge=2, le=86400)
    stride: int = Field(default=1, ge=1)
    lags_seconds: list[int] = Field(default=[1, 2, 4, 8, 16, 32, 64, 128], min_length=1, max_length=32)
    distance_metric: Literal["mae", "mse", "ssim"] = "mae"
    roi_id: int | None = None
    autocorrelation_max_lag_seconds: int = Field(default=128, ge=1, le=3600)
    autocorrelation_threshold: float = Field(default=0.2, ge=-1.0, le=1.0)

    @field_validator("lags_seconds")
    @classmethod
    def validate_lags(cls, value: list[int]):
        parsed = sorted(set(int(lag) for lag in value))
        if not parsed or parsed[0] < 1:
            raise ValueError("Lag values must be positive whole seconds.")
        return parsed


class TemporalLagStatistics(BaseModel):
    lag_seconds: int
    pair_count: int
    mean: float | None
    median: float | None
    std: float | None
    p25: float | None
    p75: float | None


class TemporalMotionPoint(BaseModel):
    timestamp: datetime
    difference: float
    interval_seconds: float
    segment_id: int


class TemporalAutocorrelationPoint(BaseModel):
    lag_seconds: int
    autocorrelation: float | None
    pair_count: int


class TemporalComparisonExample(BaseModel):
    lag_seconds: int
    reference_timestamp: datetime
    comparison_timestamp: datetime
    actual_lag_seconds: float
    difference: float
    reference_image_data_url: str
    comparison_image_data_url: str
    difference_image_data_url: str


class TemporalDynamicsResponse(BaseModel):
    training_dataset_id: int
    preprocessing_pipeline_id: int
    training_dataset_name: str
    preprocessing_pipeline_name: str
    roi_id: int | None
    roi_name: str | None
    reference_timestamp: datetime
    start_timestamp: datetime
    end_timestamp: datetime
    distance_metric: str
    distance_label: str
    image_width: int
    image_height: int
    stride: int
    loaded_frame_count: int
    skipped_frame_count: int
    contiguous_segment_count: int
    lag_statistics: list[TemporalLagStatistics]
    motion_signal: list[TemporalMotionPoint]
    autocorrelation: list[TemporalAutocorrelationPoint]
    autocorrelation_threshold: float
    estimated_correlation_length_seconds: int | None
    estimated_lag_plateau_seconds: int | None
    estimated_relevant_time_scale_seconds: int
    recommended_sequence_length: int
    recommended_temporal_stride: int
    covered_time_window_seconds: int
    comparison_examples: list[TemporalComparisonExample]
    cached: bool = False


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


class BaselineAnalysisRegion(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_range(self):
        if self.end < self.start:
            self.start, self.end = self.end, self.start
        return self


class BaselineAnalysisMethod(BaseModel):
    kind: Literal[
        "raw", "ewma", "derivative", "smoothed_derivative", "second_derivative", "rolling_slope",
        "rolling_median", "rolling_mad", "robust_z", "positive_exceedance", "rolling_area",
        "rolling_mean", "rolling_max", "drawdown", "positive_slope_count", "positive_slope_fraction",
        "rising_streak", "cusum", "page_hinkley", "evidence_score", "slope_height_ratio", "energy_ratio",
        "snr_db", "snr_ratio", "rolling_std", "rolling_cv", "time_since_onset", "state_machine",
    ]
    params: dict[str, float | int | str | bool] = Field(default_factory=dict)


class BaselineAnalysisTraceRequest(BaseModel):
    testing_run_id: int = Field(gt=0)
    label: str = Field(min_length=1, max_length=512)
    color: str = Field(default="#1c7ed6", max_length=32)
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_range(self):
        if self.end < self.start:
            self.start, self.end = self.end, self.start
        return self


class BaselineNormalizationRequest(BaseModel):
    traces: list[BaselineAnalysisTraceRequest] = Field(min_length=1, max_length=32)
    score_series: str = Field(default="score", min_length=1, max_length=64)
    moving_average: int = Field(default=1, ge=1, le=1_000_000)
    analytics_pipeline: list[BaselineAnalysisMethod] = Field(default_factory=list, max_length=64)
    stage_index: int = Field(default=-1, ge=-1)
    sampling: int = Field(default=1, ge=1, le=1_000_000)
    baseline_regions: list[BaselineAnalysisRegion] = Field(min_length=1, max_length=256)
    analysis_regions: list[BaselineAnalysisRegion] = Field(min_length=1, max_length=256)
    normalization: Literal["classic", "robust"] = "classic"
    thresholds: list[float] = Field(default_factory=lambda: [3.0, 5.0], min_length=1, max_length=32)
    persistence_samples: int = Field(default=1, ge=1, le=100_000)
    max_points: int = Field(default=8000, ge=100, le=50_000)

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.stage_index >= len(self.analytics_pipeline):
            raise ValueError("Selected analytics stage does not exist in the supplied pipeline.")
        cleaned = sorted({float(value) for value in self.thresholds if math.isfinite(value) and value > 0})
        if not cleaned:
            raise ValueError("At least one positive finite threshold is required.")
        if self.max_points < len(self.analysis_regions):
            raise ValueError("max_points must allow at least one plot point per analysis region.")
        self.thresholds = cleaned
        return self


class BaselineStatisticsRead(BaseModel):
    sample_count: int
    mean: float
    std: float
    median: float
    mad: float
    center: float
    scale: float


class BaselineThresholdStatisticsRead(BaseModel):
    threshold: float
    sample_count: int
    sample_fraction: float
    longest_seconds: float


class BaselineSeriesPointRead(BaseModel):
    timestamp: datetime
    raw: float | None
    signal: float | None
    z: float | None
    continuity_segment: int = 0


class BaselineAnomalyEventRead(BaseModel):
    threshold: float
    start: datetime
    end: datetime
    sample_count: int


class BaselineRegionStatisticsRead(BaseModel):
    region_id: str
    region_name: str
    start: datetime
    end: datetime
    sample_count: int
    raw_mean: float | None
    raw_max: float | None
    signal_mean: float | None
    signal_max: float | None
    signal_std: float | None
    z_mean: float | None
    z_median: float | None
    z_max: float | None
    thresholds: list[BaselineThresholdStatisticsRead]
    series: list[BaselineSeriesPointRead] = Field(default_factory=list)
    events: list[BaselineAnomalyEventRead] = Field(default_factory=list)
    total_points: int = 0
    decimated: bool = False


class BaselineTraceResultRead(BaseModel):
    testing_run_id: int
    label: str
    color: str
    fingerprint: str
    baseline: BaselineStatisticsRead
    regions: list[BaselineRegionStatisticsRead]
    # Legacy trace-wide plot fields. New responses keep plot data on each
    # region so detection and presentation cannot escape its selected range.
    series: list[BaselineSeriesPointRead] = Field(default_factory=list)
    events: list[BaselineAnomalyEventRead] = Field(default_factory=list)
    total_points: int = 0
    decimated: bool = False


class BaselineNormalizationResponse(BaseModel):
    computed_at: datetime
    normalization: Literal["classic", "robust"]
    thresholds: list[float]
    persistence_samples: int
    traces: list[BaselineTraceResultRead]


class AnalysisImageComparisonRequest(BaseModel):
    testing_run_id: int = Field(gt=0)
    reference_result_id: int = Field(gt=0)
    comparison_result_ids: list[int] = Field(min_length=1, max_length=50)
    image_source: Literal["input", "reconstruction"] = "input"

    @model_validator(mode="after")
    def validate_result_ids(self):
        self.comparison_result_ids = list(dict.fromkeys(self.comparison_result_ids))
        if self.reference_result_id in self.comparison_result_ids:
            raise ValueError("The reference image cannot also be a comparison image.")
        return self


class AnalysisImageComparisonItemRead(BaseModel):
    result_id: int
    timestamp: datetime
    image_data_url: str
    heatmap_image_data_url: str
    max_difference: float
    mean_difference: float


class AnalysisImageComparisonResponse(BaseModel):
    testing_run_id: int
    image_source: Literal["input", "reconstruction"]
    reference_result_id: int
    reference_timestamp: datetime
    reference_image_data_url: str
    width: int
    height: int
    shared_max_difference: float
    comparisons: list[AnalysisImageComparisonItemRead]


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
    artifact_signature: str | None = None
    artifact_size_bytes: int | None
    checkpoint_at: datetime | None = None
    checkpoint_epoch: int | None = None
    checkpoint_phase: str | None = None
    checkpoint_iteration: int | None = None
    checkpoint_path: str | None = None
    checkpoint_size_bytes: int | None = None
    checkpoint_signature: str | None = None
    checkpoint_warning: str | None = None
    restart_mode: Literal["checkpoint"] | None = None
    resume_count: int = 0
    auto_retry_count: int = 0
    next_retry_at: datetime | None = None
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
    continuity_segment: int = 0


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
    checkpoint_at: datetime | None = None
    checkpoint_input_count: int | None = None
    checkpoint_result_count: int | None = None
    restart_mode: Literal["checkpoint"] | None = None
    training_run_name: str
    training_pipeline_name: str
    model_training_dataset_names: list[str] = []
    training_dataset_name: str
    preprocessing_pipeline_name: str
    method_type: str
    method_family: str
    training_mode: str
    artifact_kind: str
    artifact_path: str
    artifact_signature: str | None = None
    result_revision: int = 0
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


class AnomalyDetectionTimeRange(BaseModel):
    start_timestamp: datetime
    end_timestamp: datetime

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("Normal-range end must be after its start.")
        return self


class AnomalyDetectionConfig(BaseModel):
    algorithm: Literal[
        "robust_zscore", "robust_cusum", "event_threshold", "rolling_sigma", "simple_threshold"
    ] = "robust_cusum"
    smoothing_half_life_minutes: float = Field(default=5.0, gt=0.0, le=1440.0)
    baseline_window_minutes: float = Field(default=120.0, gt=0.0, le=43200.0)
    warmup_minutes: float = Field(default=30.0, ge=0.0, le=43200.0)
    minimum_warmup_points: int = Field(default=30, ge=3, le=100000)
    warning_z: float = Field(default=3.0, gt=0.0, le=1000.0)
    high_z: float = Field(default=5.0, gt=0.0, le=1000.0)
    minimum_score_for_detection: float = Field(default=0.0, ge=0.0, le=1000000.0)
    minimum_delta_for_detection: float | None = Field(default=None, ge=0.0, le=1000000.0)
    robust_zscore_baseline_rebuild_mode: Literal[
        "disabled", "after_warning", "after_confirmed"
    ] = "disabled"
    minimum_scale_relative: float = Field(default=1e-3, ge=0.0, le=1.0)
    minimum_scale_absolute: float = Field(default=1e-9, ge=0.0, le=1000000.0)
    cusum_drift: float = Field(default=1.0, ge=0.0, le=1000.0)
    cusum_threshold: float = Field(default=10.0, gt=0.0, le=1000000.0)
    cusum_z_cap: float = Field(default=20.0, gt=0.0, le=1000000.0)
    confirmation_mode: Literal["minutes", "samples"] = "minutes"
    confirmation_minutes: float = Field(default=5.0, ge=0.0, le=43200.0)
    confirmation_samples: int = Field(default=1, ge=1, le=100000)
    recovery_z: float = Field(default=1.0, ge=-1000.0, le=1000.0)
    recovery_minutes: float = Field(default=15.0, ge=0.0, le=43200.0)
    fallback_recovery_minutes: float = Field(default=60.0, ge=0.0, le=43200.0)
    preroll_minutes: float = Field(default=120.0, ge=0.0, le=43200.0)
    gap_multiplier: float = Field(default=5.0, gt=1.0, le=1000.0)
    minimum_gap_minutes: float = Field(default=15.0, gt=0.0, le=43200.0)
    event_smoothing_enabled: bool = True
    event_smoothing_method: Literal["median", "moving_average"] = "median"
    event_smoothing_window_seconds: float = Field(default=5.0, gt=0.0, le=86400.0)
    threshold_mode: Literal["manual", "quantile"] = "quantile"
    manual_threshold: float | None = Field(default=None, ge=0.0)
    threshold_quantile: float = Field(default=0.9999, gt=0.0, lt=1.0)
    persistence_k: int = Field(default=10, ge=1, le=100000)
    persistence_n: int = Field(default=15, ge=1, le=100000)
    threshold_off_factor: float = Field(default=0.8, gt=0.0, le=1.0)
    normal_close_seconds: float = Field(default=30.0, ge=0.0, le=86400.0)
    merge_gap_seconds: float = Field(default=60.0, ge=0.0, le=86400.0)
    event_minimum_gap_seconds: float = Field(default=15.0, gt=0.0, le=86400.0)
    sigma_threshold: float = Field(default=3.0, gt=0.0, le=1000.0)
    simple_threshold_mode: Literal["manual", "quantile"] = "quantile"
    simple_threshold_signal: Literal["raw", "ewma"] = "raw"
    simple_threshold_value: float | None = Field(default=None, ge=0.0)
    simple_threshold_quantile: float = Field(default=0.999, gt=0.0, lt=1.0)
    simple_threshold_quantile_source: Literal["normal_ranges", "full_range"] = "normal_ranges"
    simple_threshold_ewma_half_life_minutes: float = Field(default=5.0, gt=0.0, le=1440.0)
    simple_threshold_normal_ranges: list[AnomalyDetectionTimeRange] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_threshold_order(self):
        # Rolling Sigma historically confirmed the first above-threshold
        # sample immediately. Preserve that behavior for old stored configs
        # and API payloads that predate the explicit persistence selector.
        if self.algorithm == "rolling_sigma" and "confirmation_mode" not in self.model_fields_set:
            self.confirmation_mode = "samples"
        if self.algorithm in {"robust_zscore", "robust_cusum"}:
            if self.high_z < self.warning_z:
                raise ValueError("high_z must be greater than or equal to warning_z.")
            if self.recovery_z >= self.warning_z:
                raise ValueError("recovery_z must be lower than warning_z.")
            if self.algorithm == "robust_cusum" and self.cusum_z_cap < self.high_z:
                raise ValueError("cusum_z_cap must be greater than or equal to high_z.")
        if self.algorithm == "event_threshold":
            if self.persistence_k > self.persistence_n:
                raise ValueError("persistence_k must be lower than or equal to persistence_n.")
            if self.threshold_mode == "manual" and self.manual_threshold is None:
                raise ValueError("manual_threshold is required for a manual event threshold.")
            if self.manual_threshold is not None and not math.isfinite(self.manual_threshold):
                raise ValueError("manual_threshold must be finite.")
        if self.algorithm == "simple_threshold":
            if self.simple_threshold_mode == "manual" and self.simple_threshold_value is None:
                raise ValueError("simple_threshold_value is required in manual mode.")
            if self.simple_threshold_value is not None and not math.isfinite(self.simple_threshold_value):
                raise ValueError("simple_threshold_value must be finite.")
            if (
                self.simple_threshold_mode == "quantile"
                and self.simple_threshold_quantile_source == "normal_ranges"
                and not self.simple_threshold_normal_ranges
            ):
                raise ValueError("At least one normal range is required for a normal-range quantile.")
        return self


class AnomalyDetectionRunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    testing_run_id: int
    score_series: Literal["score", "full_mse", "roi_mse"] = "score"
    start_timestamp: datetime
    end_timestamp: datetime
    config: AnomalyDetectionConfig = Field(default_factory=AnomalyDetectionConfig)
    threshold_testing_run_id: int | None = None
    threshold_start_timestamp: datetime | None = None
    threshold_end_timestamp: datetime | None = None
    progress_token: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_timestamp < self.start_timestamp:
            raise ValueError("end_timestamp must be after start_timestamp.")
        threshold_fields = (
            self.threshold_testing_run_id,
            self.threshold_start_timestamp,
            self.threshold_end_timestamp,
        )
        if self.config.algorithm == "event_threshold" and self.config.threshold_mode == "quantile":
            if any(value is None for value in threshold_fields):
                raise ValueError("A validation inference and time range are required for a quantile threshold.")
            if self.threshold_end_timestamp < self.threshold_start_timestamp:
                raise ValueError("threshold_end_timestamp must be after threshold_start_timestamp.")
            if self.threshold_testing_run_id == self.testing_run_id:
                raise ValueError("The validation inference must differ from the analyzed inference.")
        if self.config.algorithm == "simple_threshold":
            for normal_range in self.config.simple_threshold_normal_ranges:
                if (
                    normal_range.start_timestamp < self.start_timestamp
                    or normal_range.end_timestamp > self.end_timestamp
                ):
                    raise ValueError("Simple-threshold normal ranges must lie inside the analysis range.")
        return self


class AnomalyDetectionThresholdPreviewRequest(BaseModel):
    testing_run_id: int
    score_series: Literal["score", "full_mse", "roi_mse"] = "score"
    start_timestamp: datetime
    end_timestamp: datetime
    smoothing_enabled: bool = True
    smoothing_method: Literal["median", "moving_average"] = "median"
    smoothing_window_seconds: float = Field(default=5.0, gt=0.0, le=86400.0)
    gap_multiplier: float = Field(default=5.0, gt=1.0, le=1000.0)
    minimum_gap_seconds: float = Field(default=15.0, gt=0.0, le=86400.0)
    quantile: float = Field(default=0.9999, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_timestamp < self.start_timestamp:
            raise ValueError("end_timestamp must be after start_timestamp.")
        return self


class AnomalyDetectionThresholdPreviewRead(BaseModel):
    testing_run_id: int
    testing_run_name: str
    score_series: str
    start_timestamp: datetime
    end_timestamp: datetime
    point_count: int
    quantile: float
    threshold: float


class AnomalyDetectionSimpleThresholdPreviewRequest(BaseModel):
    testing_run_id: int
    score_series: Literal["score", "full_mse", "roi_mse"] = "score"
    start_timestamp: datetime
    end_timestamp: datetime
    signal: Literal["raw", "ewma"] = "raw"
    ewma_half_life_minutes: float = Field(default=5.0, gt=0.0, le=1440.0)
    preroll_minutes: float = Field(default=120.0, ge=0.0, le=43200.0)
    gap_multiplier: float = Field(default=5.0, gt=1.0, le=1000.0)
    minimum_gap_minutes: float = Field(default=15.0, gt=0.0, le=43200.0)
    quantile: float = Field(default=0.999, gt=0.0, lt=1.0)
    quantile_source: Literal["normal_ranges", "full_range"] = "normal_ranges"
    normal_ranges: list[AnomalyDetectionTimeRange] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("Analysis end must be after its start.")
        if self.quantile_source == "normal_ranges" and not self.normal_ranges:
            raise ValueError("At least one normal range is required.")
        for normal_range in self.normal_ranges:
            if (
                normal_range.start_timestamp < self.start_timestamp
                or normal_range.end_timestamp > self.end_timestamp
            ):
                raise ValueError("Normal ranges must lie inside the analysis range.")
        return self


class AnomalyDetectionSimpleThresholdPreviewRead(BaseModel):
    testing_run_id: int
    testing_run_name: str
    score_series: str
    start_timestamp: datetime
    end_timestamp: datetime
    signal: Literal["raw", "ewma"]
    quantile: float
    quantile_source: Literal["normal_ranges", "full_range"]
    normal_ranges: list[AnomalyDetectionTimeRange]
    point_count: int
    threshold: float
    warnings: list[str] = Field(default_factory=list)


class AnomalyDetectionCalibrationRequest(BaseModel):
    testing_run_id: int
    score_series: Literal["score", "full_mse", "roi_mse"] = "score"
    start_timestamp: datetime
    end_timestamp: datetime
    algorithm: Literal["robust_zscore", "robust_cusum"]
    profile: Literal["sensitive", "balanced", "conservative"] = "balanced"
    config: AnomalyDetectionConfig

    @model_validator(mode="after")
    def validate_calibration(self):
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("end_timestamp must be after start_timestamp.")
        if self.config.algorithm != self.algorithm:
            raise ValueError("config.algorithm must match algorithm.")
        return self


class AnomalyDetectionCalibrationRecommendation(BaseModel):
    minimum_scale_relative: float
    minimum_scale_absolute: float
    warning_z: float
    high_z: float
    cusum_drift: float | None = None
    cusum_threshold: float | None = None


class AnomalyDetectionCalibrationMetrics(BaseModel):
    point_count: int
    ready_point_count: int
    duration_minutes: float
    gap_count: int
    warning_quantile: float
    high_quantile: float
    observed_warning_z: float
    observed_high_z: float
    warning_rate: float
    confirmed_event_count: int
    max_cusum: float


class AnomalyDetectionCalibrationRead(BaseModel):
    testing_run_id: int
    testing_run_name: str
    score_series: str
    start_timestamp: datetime
    end_timestamp: datetime
    algorithm: Literal["robust_zscore", "robust_cusum"]
    profile: Literal["sensitive", "balanced", "conservative"]
    confidence: Literal["low", "medium", "high"]
    recommendation: AnomalyDetectionCalibrationRecommendation
    metrics: AnomalyDetectionCalibrationMetrics
    warnings: list[str] = Field(default_factory=list)


class AnomalyDetectionEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    warning_start: datetime
    confirmed_at: datetime | None
    end_timestamp: datetime
    end_reason: str
    peak_timestamp: datetime
    max_score: float
    max_robust_z: float | None
    duration_seconds: float | None
    max_smoothed_score: float | None
    mean_smoothed_score: float | None
    threshold: float | None


class AnomalyDetectionRunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    testing_run_id: int
    testing_run_name: str
    threshold_testing_run_id: int | None
    threshold_testing_run_name: str | None
    threshold_start_timestamp: datetime | None
    threshold_end_timestamp: datetime | None
    resolved_threshold: float | None
    score_series: str
    start_timestamp: datetime
    end_timestamp: datetime
    algorithm_version: str
    config: AnomalyDetectionConfig
    point_count: int
    warning_count: int
    anomaly_count: int
    created_at: datetime
    updated_at: datetime


class AnomalyDetectionSeriesPoint(BaseModel):
    timestamp: datetime
    score: float
    smoothed: float
    baseline: float | None
    mad: float | None = None
    scale: float | None = None
    warning_threshold: float | None
    high_threshold: float | None
    robust_z: float | None
    cusum_increment: float | None = None
    cusum: float
    threshold_on: float | None = None
    threshold_off: float | None = None
    candidate: bool = False
    persistence_count: int = 0
    baseline_std: float | None = None
    state: Literal["warmup", "normal", "warning", "confirmed"]
    continuity_segment: int = 0


class AnomalyDetectionBaselineTransition(BaseModel):
    timestamp: datetime
    kind: Literal["frozen", "rebuilding", "ready"]


class AnomalyDetectionRunRead(AnomalyDetectionRunSummary):
    events: list[AnomalyDetectionEventRead]
    series: list[AnomalyDetectionSeriesPoint]
    baseline_transitions: list[AnomalyDetectionBaselineTransition] = Field(default_factory=list)
    total: int
    decimated: bool


class AnomalyDetectionProgressRead(BaseModel):
    progress_token: str
    phase: Literal["loading", "smoothing", "detecting", "saving", "plotting", "complete"]
    status: Literal["running", "complete", "error"]
    completed: int
    total: int
    percent: float
    message: str
    error: str | None = None
    updated_at: datetime


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
    stae_view: Literal["reconstruction", "prediction"] = "reconstruction"
    prediction_horizon: int = Field(default=1, ge=1)
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
    stae_view: Literal["reconstruction", "prediction"]
    prediction_horizon: int
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
    started_at: datetime | None
    duration_seconds: float | None
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


class EvaluationProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    normal_window_duration_seconds: float = Field(default=3600.0, gt=0.0)
    normal_window_buffer_seconds: float = Field(default=0.0, ge=0.0)
    drift_window_seconds: float = Field(default=3600.0, gt=0.0)
    false_alarm_horizon_seconds: float = Field(default=3600.0, gt=0.0)
    anticipation_seconds: float = Field(default=0.0, ge=0.0)
    epsilon: float = Field(default=1e-12, gt=0.0)


class EvaluationProfileRead(EvaluationProfileCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


def _dataset_local_naive(value: datetime) -> datetime:
    """Keep MLTrace wall-clock values while discarding any supplied UTC offset."""

    return value.replace(tzinfo=None)


class EvaluationLabelEventInput(BaseModel):
    event_id: str | None = Field(default=None, min_length=1, max_length=64)
    type: Literal["target", "exclusion"]
    name: str | None = Field(default=None, max_length=255)
    category: str | None = Field(default=None, max_length=128)
    start_timestamp: datetime
    end_timestamp: datetime
    notes: str | None = None

    @model_validator(mode="after")
    def validate_interval(self):
        self.start_timestamp = _dataset_local_naive(self.start_timestamp)
        self.end_timestamp = _dataset_local_naive(self.end_timestamp)
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("Event end_timestamp must be after start_timestamp.")
        if self.type == "target" and not (self.name or "").strip():
            raise ValueError("Target events require a name.")
        if self.type == "target" and not (self.category or "").strip():
            raise ValueError("Target events require a category.")
        return self


class EvaluationLabelEventRead(EvaluationLabelEventInput):
    model_config = ConfigDict(from_attributes=True)

    event_id: str


class EvaluationLabelSetCreate(BaseModel):
    training_dataset_id: int
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    events: list[EvaluationLabelEventInput] = Field(default_factory=list)


class EvaluationLabelSetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    training_dataset_id: int
    name: str
    description: str | None
    version: int
    events: list[EvaluationLabelEventRead] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class EvaluationLabelCsvPreviewRequest(BaseModel):
    training_dataset_id: int
    csv_text: str = Field(min_length=1)


class EvaluationLabelCsvImportRequest(EvaluationLabelCsvPreviewRequest):
    mode: Literal["replace", "append"] = "replace"


class EvaluationLabelCsvError(BaseModel):
    row: int
    message: str


class EvaluationLabelCsvPreviewRead(BaseModel):
    valid: bool
    events: list[EvaluationLabelEventRead] = Field(default_factory=list)
    errors: list[EvaluationLabelCsvError] = Field(default_factory=list)


EvaluationScoreSeries = Literal["score", "full_mse", "roi_mse"]


class ModelEvaluationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    evaluation_testing_run_id: int | None = None
    reference_testing_run_id: int | None = None
    calibration_testing_run_id: int | None = None
    profile_id: int | None = None
    label_set_id: int | None = None
    score_series: EvaluationScoreSeries = "score"
    evaluation_start_timestamp: datetime | None = None
    evaluation_end_timestamp: datetime | None = None
    reference_start_timestamp: datetime | None = None
    reference_end_timestamp: datetime | None = None
    calibration_start_timestamp: datetime | None = None
    calibration_end_timestamp: datetime | None = None
    selected_categories: list[str] = Field(default_factory=list)
    normal_window_overrides: dict[str, dict[str, datetime]] = Field(default_factory=dict)
    profile_overrides: dict[str, float] = Field(default_factory=dict)
    active_quantile: Literal[0.99, 0.995, 0.999, 0.9995, 0.9999] = 0.999

    @model_validator(mode="after")
    def validate_ranges(self):
        for role in ("evaluation", "reference", "calibration"):
            start = getattr(self, f"{role}_start_timestamp")
            end = getattr(self, f"{role}_end_timestamp")
            if start is not None:
                start = _dataset_local_naive(start)
                setattr(self, f"{role}_start_timestamp", start)
            if end is not None:
                end = _dataset_local_naive(end)
                setattr(self, f"{role}_end_timestamp", end)
            if start is not None and end is not None and end <= start:
                raise ValueError(f"{role}_end_timestamp must be after {role}_start_timestamp.")
        return self


class ModelEvaluationUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    evaluation_testing_run_id: int | None = None
    reference_testing_run_id: int | None = None
    calibration_testing_run_id: int | None = None
    profile_id: int | None = None
    label_set_id: int | None = None
    score_series: EvaluationScoreSeries | None = None
    evaluation_start_timestamp: datetime | None = None
    evaluation_end_timestamp: datetime | None = None
    reference_start_timestamp: datetime | None = None
    reference_end_timestamp: datetime | None = None
    calibration_start_timestamp: datetime | None = None
    calibration_end_timestamp: datetime | None = None
    selected_categories: list[str] | None = None
    normal_window_overrides: dict[str, dict[str, datetime]] | None = None
    profile_overrides: dict[str, float] | None = None
    active_quantile: Literal[0.99, 0.995, 0.999, 0.9995, 0.9999] | None = None

    @model_validator(mode="after")
    def reject_null_required_fields(self):
        for field in ("name", "score_series", "active_quantile"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} may not be null.")
        return self


class ModelEvaluationDuplicateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)


class ModelEvaluationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    status: Literal["draft", "finalized"]
    evaluation_testing_run_id: int | None
    reference_testing_run_id: int | None
    calibration_testing_run_id: int | None
    profile_id: int | None
    label_set_id: int | None
    score_series: EvaluationScoreSeries
    evaluation_start_timestamp: datetime | None
    evaluation_end_timestamp: datetime | None
    reference_start_timestamp: datetime | None
    reference_end_timestamp: datetime | None
    calibration_start_timestamp: datetime | None
    calibration_end_timestamp: datetime | None
    selected_categories: list[str] = Field(default_factory=list)
    normal_window_overrides: dict = Field(default_factory=dict)
    profile_overrides: dict = Field(default_factory=dict)
    profile_snapshot: dict | None
    label_snapshot: dict | None
    source_snapshot: dict | None
    config_signature: str | None
    separation_status: Literal["not_calculated", "current", "stale", "error"]
    separation_config_signature: str | None
    separation_result: dict | None
    separation_error: str | None
    drift_status: Literal["not_calculated", "current", "stale", "error"]
    drift_config_signature: str | None
    drift_result: dict | None
    drift_error: str | None
    detection_status: Literal["not_calculated", "current", "stale", "error"]
    detection_config_signature: str | None
    detection_result: dict | None
    detection_error: str | None
    warnings: list = Field(default_factory=list)
    sep_median: float | None
    sep_min: float | None
    drift_mean: float | None
    drift_max: float | None
    event_recall: float | None
    median_delay_seconds: float | None
    frame_fpr: float | None
    false_alarm_rate_t0: float | None
    active_quantile: float
    finalized_at: datetime | None
    created_at: datetime
    updated_at: datetime


class EvaluationScorePreviewPoint(BaseModel):
    result_id: int
    position: int
    timestamp: datetime
    value: float
    continuity_segment: int


class EvaluationScorePreviewRead(BaseModel):
    testing_run_id: int
    score_series: EvaluationScoreSeries
    start_timestamp: datetime | None
    end_timestamp: datetime | None
    total: int
    decimated: bool
    points: list[EvaluationScorePreviewPoint]


# Model-centred evaluation workspace (migration 0047).
class EvaluationSeparationPairInput(BaseModel):
    pair_key: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    normal_start: datetime
    normal_end: datetime
    anomaly_start: datetime
    anomaly_end: datetime

    @model_validator(mode="after")
    def validate_ranges(self):
        for field in ("normal_start", "normal_end", "anomaly_start", "anomaly_end"):
            setattr(self, field, _dataset_local_naive(getattr(self, field)))
        if self.normal_end <= self.normal_start:
            raise ValueError("Normal range end must be after start.")
        if self.anomaly_end < self.anomaly_start:
            raise ValueError("Anomaly range end must not be before start.")
        return self


class EvaluationSeparationLayoutInput(BaseModel):
    training_dataset_id: int
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    pairs: list[EvaluationSeparationPairInput] = Field(min_length=1)


class EvaluationDriftExclusionInput(BaseModel):
    exclusion_key: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    start_timestamp: datetime
    end_timestamp: datetime

    @model_validator(mode="after")
    def validate_range(self):
        self.start_timestamp = _dataset_local_naive(self.start_timestamp)
        self.end_timestamp = _dataset_local_naive(self.end_timestamp)
        if self.end_timestamp < self.start_timestamp:
            raise ValueError("Exclusion end must not be before start.")
        return self


class EvaluationDriftBucketInput(BaseModel):
    bucket_key: str = Field(min_length=1, max_length=64)
    start_timestamp: datetime
    end_timestamp: datetime
    decision: Literal["include", "drop_bucket", "filter_points"] = "include"


class EvaluationDriftLayoutInput(BaseModel):
    training_dataset_id: int
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    reference_start: datetime
    reference_end: datetime
    analysis_start: datetime
    analysis_end: datetime
    bucket_seconds: float = Field(gt=0)
    reference_exclusion_action: Literal["filter_points", "drop_reference"] = "filter_points"
    exclusions: list[EvaluationDriftExclusionInput] = Field(default_factory=list)
    buckets: list[EvaluationDriftBucketInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranges(self):
        for field in ("reference_start", "reference_end", "analysis_start", "analysis_end"):
            setattr(self, field, _dataset_local_naive(getattr(self, field)))
        if self.reference_end <= self.reference_start or self.analysis_end <= self.analysis_start:
            raise ValueError("Reference and analysis ranges require end after start.")
        return self


class EvaluationSeparationCalculateRequest(BaseModel):
    testing_run_id: int
    layout_id: int
    pair_keys: list[str] = Field(min_length=1)
    score_series: EvaluationScoreSeries = "score"


class EvaluationDriftPreviewRequest(BaseModel):
    testing_run_id: int
    score_series: EvaluationScoreSeries = "score"
    layout: EvaluationDriftLayoutInput


class EvaluationDriftCalculateRequest(BaseModel):
    testing_run_id: int
    layout_id: int
    score_series: EvaluationScoreSeries = "score"


class EvaluationIncludeUpdate(BaseModel):
    included: bool


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
