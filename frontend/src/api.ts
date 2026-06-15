import type {
  Dataset,
  DatasetConnectionTest,
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

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

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
    throw new Error(body?.detail ?? `Request failed with status ${response.status}`);
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
