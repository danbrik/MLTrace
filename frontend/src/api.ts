import type {
  Dataset,
  DatasetConnectionTest,
  PreprocessingGraph,
  PreprocessingPipeline,
  PreprocessingPreview,
  PreprocessingStepDefinition,
  TrainingDataset,
  TrainingDatasetPreview,
  TrainingDatasetRuleInput,
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
