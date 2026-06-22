import type {
  Dataset,
  DatasetConnectionTest,
  HeatmapRangeRun,
  HeatmapRun,
  MethodConfiguration,
  MethodConfigurationPayload,
  MethodConfigurationSavePayload,
  MethodTorchCheckResponse,
  MethodDefinition,
  MethodValidationResponse,
  ModelLayerDefinition,
  PreprocessingGraph,
  PreprocessingPipeline,
  PreprocessingPreview,
  PreprocessingStepDefinition,
  RoiDefinition,
  RoiDefinitionPayload,
  RoiPreview,
  SchedulerSettings,
  TestingRun,
  TestingRunResultImage,
  TestingRunResults,
  TrainingDataset,
  TrainingDatasetPreview,
  TrainingDatasetRuleInput,
  TrainingPipeline,
  TrainingPipelineDryRun,
  TrainingPipelinePayload,
  TrainingPipelineSavePayload,
  TrainingRun,
  TrainingRunFilters,
} from './types';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').trim().replace(/\/+$/, '');

/** Error carrying the HTTP status and the raw `detail` payload (string or object). */
export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, options?: RequestInit, timeoutMs?: number): Promise<T> {
  const controller = timeoutMs ? new AbortController() : null;
  const timeout = controller
    ? window.setTimeout(() => controller.abort(), timeoutMs)
    : null;

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
      signal: controller?.signal ?? options?.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(`Request timed out after ${Math.round((timeoutMs ?? 0) / 1000)} seconds.`);
    }
    throw error;
  } finally {
    if (timeout !== null) {
      window.clearTimeout(timeout);
    }
  }

  if (!response.ok) {
    const body = await response.json().catch(() => undefined);
    const detail = body?.detail;
    // `detail` can be a string (most errors) or a structured object (e.g. the
    // duplicate-pipeline 409 carrying the existing pipeline id).
    const message =
      typeof detail === 'string'
        ? detail
        : (detail && typeof detail === 'object' && 'message' in detail
            ? String((detail as { message: unknown }).message)
            : `Request failed with status ${response.status}`);
    throw new ApiError(message, response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export function listDatasets(): Promise<Dataset[]> {
  return request<Dataset[]>('/api/datasets');
}

export function getDataset(datasetId: number): Promise<Dataset> {
  return request<Dataset>(`/api/datasets/${datasetId}`);
}

export function createDataset(payload: { name: string; root_path: string }): Promise<Dataset> {
  return request<Dataset>('/api/datasets', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 60_000);
}

export function testDatasetConnection(payload: { root_path: string }): Promise<DatasetConnectionTest> {
  return request<DatasetConnectionTest>('/api/datasets/test-connection', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 60_000);
}

export async function deleteDataset(datasetId: number): Promise<void> {
  await request<void>(`/api/datasets/${datasetId}`, {
    method: 'DELETE',
  });
}

export function confirmTimestampFormat(
  datasetId: number,
  payload: { timestamp_regex: string; timestamp_format: string },
): Promise<Dataset> {
  return request<Dataset>(`/api/datasets/${datasetId}/confirm-timestamp-format`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function rescanDataset(datasetId: number): Promise<Dataset> {
  return request<Dataset>(`/api/datasets/${datasetId}/rescan`, {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export function previewTrainingDataset(payload: {
  rules: TrainingDatasetRuleInput[];
}): Promise<TrainingDatasetPreview> {
  return request<TrainingDatasetPreview>('/api/training-datasets/preview', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function createTrainingDataset(payload: {
  name: string;
  usage_label?: string;
  notes?: string;
  rules: TrainingDatasetRuleInput[];
}): Promise<TrainingDataset> {
  return request<TrainingDataset>('/api/training-datasets', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function listTrainingDatasets(): Promise<TrainingDataset[]> {
  return request<TrainingDataset[]>('/api/training-datasets');
}

export function getTrainingDataset(trainingDatasetId: number): Promise<TrainingDataset> {
  return request<TrainingDataset>(`/api/training-datasets/${trainingDatasetId}`);
}

export function updateTrainingDataset(
  trainingDatasetId: number,
  payload: {
    name: string;
    usage_label?: string;
    notes?: string;
    rules: TrainingDatasetRuleInput[];
  },
): Promise<TrainingDataset> {
  return request<TrainingDataset>(`/api/training-datasets/${trainingDatasetId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  });
}

export function cleanupTrainingDatasetInvalidRules(trainingDatasetId: number): Promise<TrainingDataset> {
  return request<TrainingDataset>(`/api/training-datasets/${trainingDatasetId}/cleanup-invalid-rules`, {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export function refreshTrainingDatasetCounts(trainingDatasetId: number): Promise<TrainingDataset> {
  return request<TrainingDataset>(`/api/training-datasets/${trainingDatasetId}/refresh-counts`, {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export async function deleteTrainingDataset(trainingDatasetId: number): Promise<void> {
  await request<void>(`/api/training-datasets/${trainingDatasetId}`, {
    method: 'DELETE',
  });
}

export function listPreprocessingSteps(): Promise<PreprocessingStepDefinition[]> {
  return request<PreprocessingStepDefinition[]>('/api/preprocessing/steps');
}

export function listPreprocessingPipelines(): Promise<PreprocessingPipeline[]> {
  return request<PreprocessingPipeline[]>('/api/preprocessing/pipelines');
}

export function getPreprocessingPipeline(pipelineId: number): Promise<PreprocessingPipeline> {
  return request<PreprocessingPipeline>(`/api/preprocessing/pipelines/${pipelineId}`);
}

export function createPreprocessingPipeline(payload: {
  name: string;
  description?: string;
  graph: PreprocessingGraph;
  preview_folder_id?: number | null;
  input_width?: number | null;
  input_height?: number | null;
  output_width?: number | null;
  output_height?: number | null;
}): Promise<PreprocessingPipeline> {
  return request<PreprocessingPipeline>('/api/preprocessing/pipelines', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function updatePreprocessingPipeline(
  pipelineId: number,
  payload: {
    name: string;
    description?: string;
    graph: PreprocessingGraph;
    preview_folder_id?: number | null;
    input_width?: number | null;
    input_height?: number | null;
    output_width?: number | null;
    output_height?: number | null;
  },
): Promise<PreprocessingPipeline> {
  return request<PreprocessingPipeline>(`/api/preprocessing/pipelines/${pipelineId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  });
}

export async function deletePreprocessingPipeline(pipelineId: number): Promise<void> {
  await request<void>(`/api/preprocessing/pipelines/${pipelineId}`, {
    method: 'DELETE',
  });
}

export function previewPreprocessingPipeline(payload: {
  folder_id: number;
  graph: PreprocessingGraph;
}): Promise<PreprocessingPreview> {
  return request<PreprocessingPreview>('/api/preprocessing/pipelines/preview', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function listMethodDefinitions(): Promise<MethodDefinition[]> {
  return request<MethodDefinition[]>('/api/methods/definitions');
}

export function getMethodDefinition(methodType: string): Promise<MethodDefinition> {
  return request<MethodDefinition>(`/api/methods/definitions/${methodType}`);
}

export function listModelLayers(): Promise<ModelLayerDefinition[]> {
  return request<ModelLayerDefinition[]>('/api/methods/layers');
}

export function listMethodConfigurations(): Promise<MethodConfiguration[]> {
  return request<MethodConfiguration[]>('/api/methods/configurations');
}

export function getMethodConfiguration(configurationId: number): Promise<MethodConfiguration> {
  return request<MethodConfiguration>(`/api/methods/configurations/${configurationId}`);
}

export function createMethodConfiguration(payload: MethodConfigurationSavePayload): Promise<MethodConfiguration> {
  return request<MethodConfiguration>('/api/methods/configurations', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function updateMethodConfiguration(
  configurationId: number,
  payload: MethodConfigurationSavePayload,
): Promise<MethodConfiguration> {
  return request<MethodConfiguration>(`/api/methods/configurations/${configurationId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  });
}

export async function deleteMethodConfiguration(configurationId: number): Promise<void> {
  await request<void>(`/api/methods/configurations/${configurationId}`, {
    method: 'DELETE',
  });
}

export function validateMethodConfiguration(payload: MethodConfigurationPayload): Promise<MethodValidationResponse> {
  return request<MethodValidationResponse>('/api/methods/configurations/validate', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function buildMethodDiagram(payload: MethodConfigurationPayload): Promise<MethodValidationResponse> {
  return request<MethodValidationResponse>('/api/methods/configurations/diagram', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function runMethodTorchCheck(payload: MethodConfigurationPayload): Promise<MethodTorchCheckResponse> {
  return request<MethodTorchCheckResponse>('/api/methods/configurations/torch-check', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 120_000);
}

export function listTrainingPipelines(): Promise<TrainingPipeline[]> {
  return request<TrainingPipeline[]>('/api/training-pipelines');
}

export function getTrainingPipeline(pipelineId: number): Promise<TrainingPipeline> {
  return request<TrainingPipeline>(`/api/training-pipelines/${pipelineId}`);
}

export function createTrainingPipeline(payload: TrainingPipelineSavePayload): Promise<TrainingPipeline> {
  return request<TrainingPipeline>('/api/training-pipelines', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function updateTrainingPipeline(
  pipelineId: number,
  payload: TrainingPipelineSavePayload,
): Promise<TrainingPipeline> {
  return request<TrainingPipeline>(`/api/training-pipelines/${pipelineId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  });
}

export async function deleteTrainingPipeline(pipelineId: number): Promise<void> {
  await request<void>(`/api/training-pipelines/${pipelineId}`, {
    method: 'DELETE',
  });
}

export function dryRunTrainingPipeline(payload: TrainingPipelinePayload): Promise<TrainingPipelineDryRun> {
  return request<TrainingPipelineDryRun>('/api/training-pipelines/dry-run', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 120_000);
}

export function resolveDuplicateTrainingPipeline(
  payload: TrainingPipelinePayload,
): Promise<{ existing_pipeline: TrainingPipeline | null }> {
  return request<{ existing_pipeline: TrainingPipeline | null }>('/api/training-pipelines/resolve-duplicate', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function listTrainingRuns(filters: TrainingRunFilters = {}): Promise<TrainingRun[]> {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value !== null && value !== undefined && value !== '') {
      params.set(key, String(value));
    }
  }
  const query = params.toString();
  return request<TrainingRun[]>(`/api/training-runs${query ? `?${query}` : ''}`);
}

export function getTrainingRun(runId: number): Promise<TrainingRun> {
  return request<TrainingRun>(`/api/training-runs/${runId}`);
}

export function getTrainingRunLog(runId: number): Promise<{ log: string }> {
  return request<{ log: string }>(`/api/training-runs/${runId}/log`);
}

export function enqueueTrainingRun(trainingPipelineId: number): Promise<TrainingRun> {
  return request<TrainingRun>('/api/training-runs', {
    method: 'POST',
    body: JSON.stringify({ training_pipeline_id: trainingPipelineId }),
  });
}

export function abortTrainingRun(runId: number): Promise<TrainingRun> {
  return request<TrainingRun>(`/api/training-runs/${runId}/abort`, { method: 'POST' });
}

export function restartTrainingRun(runId: number): Promise<TrainingRun> {
  return request<TrainingRun>(`/api/training-runs/${runId}/restart`, { method: 'POST' });
}

export async function deleteTrainingRun(runId: number): Promise<void> {
  await request<void>(`/api/training-runs/${runId}`, { method: 'DELETE' });
}

export function listRois(): Promise<RoiDefinition[]> {
  return request<RoiDefinition[]>('/api/rois');
}

export function createRoi(payload: RoiDefinitionPayload): Promise<RoiDefinition> {
  return request<RoiDefinition>('/api/rois', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function deleteRoi(roiId: number): Promise<void> {
  await request<void>(`/api/rois/${roiId}`, { method: 'DELETE' });
}

export function previewRoi(payload: { training_run_id: number; training_dataset_id: number }): Promise<RoiPreview> {
  return request<RoiPreview>('/api/rois/preview', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 120_000);
}

export function listTestingRuns(): Promise<TestingRun[]> {
  return request<TestingRun[]>('/api/testing-runs');
}

export function enqueueTestingRun(payload: {
  training_run_id: number;
  training_dataset_id: number;
  roi_id?: number | null;
  name?: string | null;
}): Promise<TestingRun> {
  return request<TestingRun>('/api/testing-runs', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getTestingRunResults(runId: number): Promise<TestingRunResults> {
  return request<TestingRunResults>(`/api/testing-runs/${runId}/results`);
}

export function getTestingRunResultImage(runId: number, resultId: number): Promise<TestingRunResultImage> {
  return request<TestingRunResultImage>(`/api/testing-runs/${runId}/results/${resultId}/image`, undefined, 120_000);
}

export function listHeatmaps(): Promise<HeatmapRun[]> {
  return request<HeatmapRun[]>('/api/heatmaps');
}

export function createHeatmap(payload: {
  testing_run_id: number;
  testing_result_id?: number | null;
  timestamp?: string | null;
  force_recompute?: boolean;
}): Promise<HeatmapRun> {
  return request<HeatmapRun>('/api/heatmaps', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 120_000);
}

export async function clearHeatmaps(): Promise<void> {
  await request<void>('/api/heatmaps', { method: 'DELETE' });
}

export function listHeatmapRanges(): Promise<HeatmapRangeRun[]> {
  return request<HeatmapRangeRun[]>('/api/heatmap-ranges');
}

export function createHeatmapRange(payload: {
  testing_run_id: number;
  start_timestamp: string;
  end_timestamp: string;
  stride?: number;
  scale_mode?: 'per_frame' | 'shared';
}): Promise<HeatmapRangeRun> {
  return request<HeatmapRangeRun>('/api/heatmap-ranges', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getHeatmapRange(runId: number): Promise<HeatmapRangeRun> {
  return request<HeatmapRangeRun>(`/api/heatmap-ranges/${runId}`);
}

export function abortHeatmapRange(runId: number): Promise<HeatmapRangeRun> {
  return request<HeatmapRangeRun>(`/api/heatmap-ranges/${runId}/abort`, { method: 'POST' });
}

export async function deleteHeatmapRange(runId: number): Promise<void> {
  await request<void>(`/api/heatmap-ranges/${runId}`, { method: 'DELETE' });
}

export function getHeatmapRangeLog(runId: number): Promise<{ log: string }> {
  return request<{ log: string }>(`/api/heatmap-ranges/${runId}/log`);
}

/** URL for one rendered overlay frame PNG (served directly, not via fetch). */
export function heatmapRangeFrameUrl(runId: number, index: number): string {
  return `${API_BASE_URL}/api/heatmap-ranges/${runId}/frames/${index}.png`;
}

export function getSchedulerSettings(): Promise<SchedulerSettings> {
  return request<SchedulerSettings>('/api/scheduler/settings');
}

export function updateSchedulerSettings(payload: { max_gpu_slots: number; only_gpu: boolean }): Promise<SchedulerSettings> {
  return request<SchedulerSettings>('/api/scheduler/settings', {
    method: 'PUT',
    body: JSON.stringify(payload),
  });
}

export function getTestingRunLog(runId: number): Promise<{ log: string }> {
  return request<{ log: string }>(`/api/testing-runs/${runId}/log`);
}

export function abortTestingRun(runId: number): Promise<TestingRun> {
  return request<TestingRun>(`/api/testing-runs/${runId}/abort`, { method: 'POST' });
}

export function restartTestingRun(runId: number): Promise<TestingRun> {
  return request<TestingRun>(`/api/testing-runs/${runId}/restart`, { method: 'POST' });
}

export async function deleteTestingRun(runId: number): Promise<void> {
  await request<void>(`/api/testing-runs/${runId}`, { method: 'DELETE' });
}

export const listModelArchitectures = listMethodDefinitions;
export const getModelArchitecture = getMethodDefinition;
export const listModelConfigurations = listMethodConfigurations;
export const getModelConfiguration = getMethodConfiguration;
export const createModelConfiguration = createMethodConfiguration;
export const updateModelConfiguration = updateMethodConfiguration;
export const deleteModelConfiguration = deleteMethodConfiguration;
export const validateModelConfiguration = validateMethodConfiguration;
export const buildModelDiagram = buildMethodDiagram;
export const runModelTorchCheck = runMethodTorchCheck;
