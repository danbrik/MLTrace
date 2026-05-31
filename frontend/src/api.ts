import type {
  Dataset,
  TrainingDataset,
  TrainingDatasetPreview,
  TrainingDatasetRuleInput,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

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
