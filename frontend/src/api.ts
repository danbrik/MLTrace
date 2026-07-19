import type {
  CacheRevisions,
  Dataset,
  DatasetConnectionTest,
  AnalysisLayout,
  HeatmapRangeRun,
  HeatmapRun,
  HeatmapRunSummary,
  HeatmapVisualizationConfig,
  InspectPreview,
  InspectRun,
  InspectArtifactRunPage,
  InspectCsvData,
  MethodConfiguration,
  MethodConfigurationPayload,
  MethodConfigurationSavePayload,
  MethodTorchCheckResponse,
  MethodDefinition,
  MethodValidationResponse,
  Project,
  GpuSnapshot,
  SchedulerJobWithProject,
  ModelLayerDefinition,
  OptimizationSplit,
  OptimizationSplitPayload,
  OptimizationStudy,
  OptimizationStudyPayload,
  PreprocessingGraph,
  PreprocessingPipeline,
  PreprocessingPreview,
  PreprocessingStepDefinition,
  RegistryDeletePreview,
  RegistryDeleteResult,
  RegistryDetail,
  RegistryItemRef,
  RegistryList,
  RegistrySummary,
  RoiDefinition,
  RoiDefinitionPayload,
  RoiPreview,
  SchedulerSettings,
  TestingRun,
  TestingRunBulkResponse,
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
import { cachedResource, invalidateAllResources, invalidateResources, setResourceRevision, type ResourceKey } from './resourceCache';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').trim().replace(/\/+$/, '');
const STATIC_TTL_MS = 24 * 60 * 60 * 1000;
const REFERENCE_TTL_MS = 5 * 60 * 1000;
const RUN_TTL_MS = 1500;

let revisionsCache: { value?: Record<string, string>; expiresAt: number; inFlight?: Promise<Record<string, string>> } = {
  expiresAt: 0,
};

let activeProjectId: string | null = null;

export function setActiveProject(projectId: string | null): void {
  if (activeProjectId === projectId) return;
  activeProjectId = projectId;
  revisionsCache = { expiresAt: 0 };
  invalidateAllResources();
}

export function getActiveProject(): string | null {
  return activeProjectId;
}

function projectMediaUrl(path: string): string {
  const separator = path.includes('?') ? '&' : '?';
  return `${API_BASE_URL}${path}${activeProjectId ? `${separator}project_id=${encodeURIComponent(activeProjectId)}` : ''}`;
}

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

async function request<T>(path: string, options?: RequestInit, timeoutMs?: number, projectId?: string | null): Promise<T> {
  const controller = timeoutMs ? new AbortController() : null;
  const timeout = controller
    ? window.setTimeout(() => controller.abort(), timeoutMs)
    : null;

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...((projectId ?? activeProjectId) ? { 'X-MLTrace-Project-ID': projectId ?? activeProjectId ?? '' } : {}),
        ...options?.headers,
      },
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

export function listProjects(): Promise<Project[]> {
  return request<Project[]>('/api/projects', undefined, undefined, null);
}

export function getProject(projectId: string): Promise<Project> {
  return request<Project>(`/api/projects/${projectId}`, undefined, undefined, null);
}

export function createProject(payload: { name: string; description: string }): Promise<Project> {
  return request<Project>('/api/projects', { method: 'POST', body: JSON.stringify(payload) }, undefined, null);
}

export function markProjectOpened(projectId: string): Promise<Project> {
  return request<Project>(`/api/projects/${projectId}/opened`, { method: 'POST' }, undefined, null);
}

export function getGpuUsage(refresh = false): Promise<GpuSnapshot> {
  return request<GpuSnapshot>(`/api/system/gpu-usage?refresh=${refresh ? 'true' : 'false'}`, undefined, undefined, null);
}

export function listSchedulerJobs(scope: 'project' | 'all'): Promise<SchedulerJobWithProject[]> {
  return request<SchedulerJobWithProject[]>(`/api/scheduler/jobs?scope=${scope}`);
}

async function cacheRevisions(): Promise<Record<string, string>> {
  const now = Date.now();
  if (revisionsCache.value && revisionsCache.expiresAt > now) {
    return revisionsCache.value;
  }
  if (revisionsCache.inFlight) {
    return revisionsCache.inFlight;
  }
  const inFlight = request<CacheRevisions>('/api/cache/revisions')
    .then((result) => {
      revisionsCache = { value: result.revisions, expiresAt: Date.now() + 2000 };
      return result.revisions;
    })
    .finally(() => {
      if (revisionsCache.inFlight === inFlight) {
        delete revisionsCache.inFlight;
      }
    });
  revisionsCache.inFlight = inFlight;
  return inFlight;
}

function revisionFor(key: ResourceKey): Promise<string | undefined> {
  return cacheRevisions().then((revisions) => revisions[key]);
}

function cachedList<T>(key: ResourceKey, path: string, ttlMs: number): Promise<T> {
  return cachedResource<T>(
    key,
    () =>
      request<T>(path).then((value) => {
        const revision = revisionsCache.value?.[key];
        if (revision) setResourceRevision(key, revision);
        return value;
      }),
    { ttlMs, revision: () => revisionFor(key) },
  );
}

function invalidate(keys: ResourceKey[]): void {
  revisionsCache = { expiresAt: 0 };
  invalidateResources(keys);
}

function normalizeTrainingDataset(dataset: TrainingDataset): TrainingDataset {
  return { ...dataset, rules: dataset.rules ?? [] };
}

function normalizePreprocessingPipeline(pipeline: PreprocessingPipeline): PreprocessingPipeline {
  return {
    ...pipeline,
    graph: pipeline.graph ?? { nodes: [], edges: [] },
    step_count: pipeline.step_count ?? pipeline.graph?.nodes?.length ?? 0,
    step_types: pipeline.step_types ?? pipeline.graph?.nodes?.map((node) => node.type) ?? [],
  };
}

function emptyDiagram(method: MethodConfiguration) {
  return {
    architecture_type: method.method_type,
    builder_kind: method.builder_kind,
    nodes: [],
    edges: [],
  };
}

function normalizeMethodConfiguration(method: MethodConfiguration): MethodConfiguration {
  return {
    ...method,
    method_graph: method.method_graph ?? {},
    model_graph: method.model_graph ?? method.method_graph ?? {},
    model_config: method.model_config ?? method.method_config ?? {},
    diagram: method.diagram ?? emptyDiagram(method),
    parameters: method.parameters ?? [],
  };
}

export function listDatasets(): Promise<Dataset[]> {
  return cachedList<Dataset[]>('datasets', '/api/datasets', REFERENCE_TTL_MS);
}

export function getDataset(datasetId: number): Promise<Dataset> {
  return request<Dataset>(`/api/datasets/${datasetId}`);
}

export function createDataset(payload: { name: string; root_path: string }): Promise<Dataset> {
  return request<Dataset>('/api/datasets', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 60_000).then((dataset) => {
    invalidate(['datasets']);
    return dataset;
  });
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
  invalidate(['datasets', 'trainingDatasets', 'preprocessingPipelines']);
}

export function confirmTimestampFormat(
  datasetId: number,
  payload: { timestamp_regex: string; timestamp_format: string },
): Promise<Dataset> {
  return request<Dataset>(`/api/datasets/${datasetId}/confirm-timestamp-format`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }).then((dataset) => {
    invalidate(['datasets']);
    return dataset;
  });
}

export function rescanDataset(datasetId: number): Promise<Dataset> {
  return request<Dataset>(`/api/datasets/${datasetId}/rescan`, {
    method: 'POST',
    body: JSON.stringify({}),
  }).then((dataset) => {
    invalidate(['datasets', 'trainingDatasets', 'preprocessingPipelines']);
    return dataset;
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
  }).then((dataset) => {
    invalidate(['trainingDatasets', 'trainingPipelines']);
    return normalizeTrainingDataset(dataset);
  });
}

export function listTrainingDatasets(): Promise<TrainingDataset[]> {
  return cachedList<TrainingDataset[]>('trainingDatasets', '/api/training-datasets?summary=true', REFERENCE_TTL_MS).then((datasets) =>
    datasets.map(normalizeTrainingDataset),
  );
}

export function getTrainingDataset(trainingDatasetId: number): Promise<TrainingDataset> {
  return request<TrainingDataset>(`/api/training-datasets/${trainingDatasetId}`).then(normalizeTrainingDataset);
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
  }).then((dataset) => {
    invalidate(['trainingDatasets', 'trainingPipelines']);
    return normalizeTrainingDataset(dataset);
  });
}

export function cleanupTrainingDatasetInvalidRules(trainingDatasetId: number): Promise<TrainingDataset> {
  return request<TrainingDataset>(`/api/training-datasets/${trainingDatasetId}/cleanup-invalid-rules`, {
    method: 'POST',
    body: JSON.stringify({}),
  }).then((dataset) => {
    invalidate(['trainingDatasets', 'trainingPipelines']);
    return normalizeTrainingDataset(dataset);
  });
}

export function refreshTrainingDatasetCounts(trainingDatasetId: number): Promise<TrainingDataset> {
  return request<TrainingDataset>(`/api/training-datasets/${trainingDatasetId}/refresh-counts`, {
    method: 'POST',
    body: JSON.stringify({}),
  }).then((dataset) => {
    invalidate(['trainingDatasets', 'trainingPipelines']);
    return normalizeTrainingDataset(dataset);
  });
}

export async function deleteTrainingDataset(trainingDatasetId: number): Promise<void> {
  await request<void>(`/api/training-datasets/${trainingDatasetId}`, {
    method: 'DELETE',
  });
  invalidate(['trainingDatasets', 'trainingPipelines']);
}

export function listPreprocessingSteps(): Promise<PreprocessingStepDefinition[]> {
  return cachedList<PreprocessingStepDefinition[]>('preprocessingSteps', '/api/preprocessing/steps', STATIC_TTL_MS);
}

export function listPreprocessingPipelines(): Promise<PreprocessingPipeline[]> {
  return cachedList<PreprocessingPipeline[]>(
    'preprocessingPipelines',
    '/api/preprocessing/pipelines?summary=true',
    REFERENCE_TTL_MS,
  ).then((pipelines) => pipelines.map(normalizePreprocessingPipeline));
}

export function getPreprocessingPipeline(pipelineId: number): Promise<PreprocessingPipeline> {
  return request<PreprocessingPipeline>(`/api/preprocessing/pipelines/${pipelineId}`).then(normalizePreprocessingPipeline);
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
  }).then((pipeline) => {
    invalidate(['preprocessingPipelines', 'trainingPipelines']);
    return normalizePreprocessingPipeline(pipeline);
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
  }).then((pipeline) => {
    invalidate(['preprocessingPipelines', 'trainingPipelines']);
    return normalizePreprocessingPipeline(pipeline);
  });
}

export async function deletePreprocessingPipeline(pipelineId: number): Promise<void> {
  await request<void>(`/api/preprocessing/pipelines/${pipelineId}`, {
    method: 'DELETE',
  });
  invalidate(['preprocessingPipelines', 'trainingPipelines']);
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
  return cachedList<MethodDefinition[]>('methodDefinitions', '/api/methods/definitions', STATIC_TTL_MS);
}

export function getMethodDefinition(methodType: string): Promise<MethodDefinition> {
  return request<MethodDefinition>(`/api/methods/definitions/${methodType}`);
}

export function listModelLayers(): Promise<ModelLayerDefinition[]> {
  return cachedList<ModelLayerDefinition[]>('methodLayers', '/api/methods/layers', STATIC_TTL_MS);
}

export function listMethodConfigurations(): Promise<MethodConfiguration[]> {
  return cachedList<MethodConfiguration[]>(
    'methodConfigurations',
    '/api/methods/configurations?summary=true',
    REFERENCE_TTL_MS,
  ).then((methods) => methods.map(normalizeMethodConfiguration));
}

export function getMethodConfiguration(configurationId: number): Promise<MethodConfiguration> {
  return request<MethodConfiguration>(`/api/methods/configurations/${configurationId}`).then(normalizeMethodConfiguration);
}

export function createMethodConfiguration(payload: MethodConfigurationSavePayload): Promise<MethodConfiguration> {
  return request<MethodConfiguration>('/api/methods/configurations', {
    method: 'POST',
    body: JSON.stringify(payload),
  }).then((method) => {
    invalidate(['methodConfigurations', 'trainingPipelines']);
    return normalizeMethodConfiguration(method);
  });
}

export function updateMethodConfiguration(
  configurationId: number,
  payload: MethodConfigurationSavePayload,
): Promise<MethodConfiguration> {
  return request<MethodConfiguration>(`/api/methods/configurations/${configurationId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  }).then((method) => {
    invalidate(['methodConfigurations', 'trainingPipelines']);
    return normalizeMethodConfiguration(method);
  });
}

export async function deleteMethodConfiguration(configurationId: number): Promise<void> {
  await request<void>(`/api/methods/configurations/${configurationId}`, {
    method: 'DELETE',
  });
  invalidate(['methodConfigurations', 'trainingPipelines']);
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
  return cachedList<TrainingPipeline[]>('trainingPipelines', '/api/training-pipelines?summary=true', REFERENCE_TTL_MS);
}

export function getTrainingPipeline(pipelineId: number): Promise<TrainingPipeline> {
  return request<TrainingPipeline>(`/api/training-pipelines/${pipelineId}`);
}

export function createTrainingPipeline(payload: TrainingPipelineSavePayload): Promise<TrainingPipeline> {
  return request<TrainingPipeline>('/api/training-pipelines', {
    method: 'POST',
    body: JSON.stringify(payload),
  }).then((pipeline) => {
    invalidate(['trainingPipelines', 'trainingDatasets', 'preprocessingPipelines', 'methodConfigurations']);
    return pipeline;
  });
}

export function updateTrainingPipeline(
  pipelineId: number,
  payload: TrainingPipelineSavePayload,
): Promise<TrainingPipeline> {
  return request<TrainingPipeline>(`/api/training-pipelines/${pipelineId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  }).then((pipeline) => {
    invalidate(['trainingPipelines', 'trainingDatasets', 'preprocessingPipelines', 'methodConfigurations']);
    return pipeline;
  });
}

export async function deleteTrainingPipeline(pipelineId: number): Promise<void> {
  await request<void>(`/api/training-pipelines/${pipelineId}`, {
    method: 'DELETE',
  });
  invalidate(['trainingPipelines', 'trainingDatasets', 'preprocessingPipelines', 'methodConfigurations']);
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
  if (query) return request<TrainingRun[]>(`/api/training-runs?${query}`);
  return cachedList<TrainingRun[]>('trainingRuns', '/api/training-runs', RUN_TTL_MS);
}

export function getTrainingRun(runId: number): Promise<TrainingRun> {
  return request<TrainingRun>(`/api/training-runs/${runId}`);
}

export function getTrainingRunLog(runId: number, projectId?: string): Promise<{ log: string }> {
  return request<{ log: string }>(`/api/training-runs/${runId}/log`, undefined, undefined, projectId);
}

export function enqueueTrainingRun(trainingPipelineId: number): Promise<TrainingRun> {
  return request<TrainingRun>('/api/training-runs', {
    method: 'POST',
    body: JSON.stringify({ training_pipeline_id: trainingPipelineId }),
  }).then((run) => {
    invalidate(['trainingRuns', 'trainingPipelines']);
    return run;
  });
}

export function abortTrainingRun(runId: number, projectId?: string): Promise<TrainingRun> {
  return request<TrainingRun>(`/api/training-runs/${runId}/abort`, { method: 'POST' }, undefined, projectId).then((run) => {
    invalidate(['trainingRuns']);
    return run;
  });
}

export function restartTrainingRun(runId: number, projectId?: string): Promise<TrainingRun> {
  return request<TrainingRun>(`/api/training-runs/${runId}/restart`, { method: 'POST' }, undefined, projectId).then((run) => {
    invalidate(['trainingRuns']);
    return run;
  });
}

export async function deleteTrainingRun(runId: number, projectId?: string): Promise<void> {
  await request<void>(`/api/training-runs/${runId}`, { method: 'DELETE' }, undefined, projectId);
  invalidate(['trainingRuns', 'trainingPipelines']);
}

export function listRois(): Promise<RoiDefinition[]> {
  return cachedList<RoiDefinition[]>('rois', '/api/rois', REFERENCE_TTL_MS);
}

export function createRoi(payload: RoiDefinitionPayload): Promise<RoiDefinition> {
  return request<RoiDefinition>('/api/rois', {
    method: 'POST',
    body: JSON.stringify(payload),
  }).then((roi) => {
    invalidate(['rois']);
    return roi;
  });
}

export async function deleteRoi(roiId: number): Promise<void> {
  await request<void>(`/api/rois/${roiId}`, { method: 'DELETE' });
  invalidate(['rois']);
}

export function previewRoi(payload: { training_run_id: number; training_dataset_id: number }): Promise<RoiPreview> {
  return request<RoiPreview>('/api/rois/preview', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 120_000);
}

export function listTestingRuns(): Promise<TestingRun[]> {
  return cachedList<TestingRun[]>('testingRuns', '/api/testing-runs', RUN_TTL_MS);
}

export function enqueueTestingRun(payload: {
  training_run_id: number;
  training_dataset_id: number;
  roi_id?: number | null;
  name?: string | null;
  inference_config?: Record<string, unknown> | null;
}): Promise<TestingRun> {
  return request<TestingRun>('/api/testing-runs', {
    method: 'POST',
    body: JSON.stringify(payload),
  }).then((run) => {
    invalidate(['testingRuns']);
    return run;
  });
}

export function bulkEnqueueTestingRuns(payload: {
  training_run_ids: number[];
  training_dataset_ids: number[];
  roi_id?: number | null;
  name_prefix?: string | null;
  inference_config?: Record<string, unknown> | null;
}): Promise<TestingRunBulkResponse> {
  return request<TestingRunBulkResponse>('/api/testing-runs/bulk', {
    method: 'POST',
    body: JSON.stringify(payload),
  }).then((response) => {
    invalidate(['testingRuns']);
    return response;
  });
}

/** Fetch a run's per-image results. ``maxPoints`` decimates large runs server-side
 *  so charts stay bounded (the full set is always available in the CSV export). */
export function getTestingRunResults(runId: number, maxPoints?: number): Promise<TestingRunResults> {
  const query = maxPoints ? `?max_points=${maxPoints}` : '';
  return request<TestingRunResults>(`/api/testing-runs/${runId}/results${query}`);
}

export function getTestingRunResultImage(runId: number, resultId: number): Promise<TestingRunResultImage> {
  return request<TestingRunResultImage>(`/api/testing-runs/${runId}/results/${resultId}/image`, undefined, 120_000);
}

export function listHeatmaps(): Promise<HeatmapRunSummary[]> {
  return cachedList<HeatmapRunSummary[]>('heatmaps', '/api/heatmaps', RUN_TTL_MS);
}

export function getHeatmap(runId: number): Promise<HeatmapRun> {
  return request<HeatmapRun>(`/api/heatmaps/${runId}`);
}

export function createHeatmap(payload: {
  testing_run_id: number;
  testing_result_id?: number | null;
  timestamp?: string | null;
  force_recompute?: boolean;
  stae_view?: 'reconstruction' | 'prediction';
  prediction_horizon?: number;
  visualization_config?: HeatmapVisualizationConfig;
}): Promise<HeatmapRun> {
  return request<HeatmapRun>('/api/heatmaps', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 120_000).then((run) => {
    invalidate(['heatmaps']);
    return run;
  });
}

export async function clearHeatmaps(): Promise<void> {
  await request<void>('/api/heatmaps', { method: 'DELETE' });
  invalidate(['heatmaps']);
}

export function listHeatmapRanges(): Promise<HeatmapRangeRun[]> {
  return cachedList<HeatmapRangeRun[]>('heatmapRanges', '/api/heatmap-ranges', RUN_TTL_MS);
}

export function createHeatmapRange(payload: {
  testing_run_id: number;
  start_timestamp: string;
  end_timestamp: string;
  stride?: number;
  fps?: number;
  scale_mode?: 'per_frame' | 'shared';
  visualization_config?: HeatmapVisualizationConfig;
  force_recompute?: boolean;
}): Promise<HeatmapRangeRun> {
  return request<HeatmapRangeRun>('/api/heatmap-ranges', {
    method: 'POST',
    body: JSON.stringify(payload),
  }).then((run) => {
    invalidate(['heatmapRanges']);
    return run;
  });
}

export function getHeatmapRange(runId: number): Promise<HeatmapRangeRun> {
  return request<HeatmapRangeRun>(`/api/heatmap-ranges/${runId}`);
}

export function abortHeatmapRange(runId: number, projectId?: string): Promise<HeatmapRangeRun> {
  return request<HeatmapRangeRun>(`/api/heatmap-ranges/${runId}/abort`, { method: 'POST' }, undefined, projectId).then((run) => {
    invalidate(['heatmapRanges']);
    return run;
  });
}

export async function deleteHeatmapRange(runId: number, projectId?: string): Promise<void> {
  await request<void>(`/api/heatmap-ranges/${runId}`, { method: 'DELETE' }, undefined, projectId);
  invalidate(['heatmapRanges']);
}

export function getHeatmapRangeLog(runId: number, projectId?: string): Promise<{ log: string }> {
  return request<{ log: string }>(`/api/heatmap-ranges/${runId}/log`, undefined, undefined, projectId);
}

export function listAnalysisLayouts(): Promise<AnalysisLayout[]> {
  return cachedList<AnalysisLayout[]>('analysisLayouts', '/api/analysis/layouts', REFERENCE_TTL_MS);
}

export function getAnalysisLayout(layoutId: number): Promise<AnalysisLayout> {
  return request<AnalysisLayout>(`/api/analysis/layouts/${layoutId}`);
}

export function createAnalysisLayout(payload: {
  name: string;
  description?: string | null;
  layout: Record<string, unknown>;
}): Promise<AnalysisLayout> {
  return request<AnalysisLayout>('/api/analysis/layouts', {
    method: 'POST',
    body: JSON.stringify(payload),
  }).then((layout) => {
    invalidate(['analysisLayouts']);
    return layout;
  });
}

export function updateAnalysisLayout(
  layoutId: number,
  payload: {
    name: string;
    description?: string | null;
    layout: Record<string, unknown>;
  },
): Promise<AnalysisLayout> {
  return request<AnalysisLayout>(`/api/analysis/layouts/${layoutId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  }).then((layout) => {
    invalidate(['analysisLayouts']);
    return layout;
  });
}

export async function deleteAnalysisLayout(layoutId: number): Promise<void> {
  await request<void>(`/api/analysis/layouts/${layoutId}`, { method: 'DELETE' });
  invalidate(['analysisLayouts']);
}

export function listOptimizationStudies(): Promise<OptimizationStudy[]> {
  return cachedList<OptimizationStudy[]>('optimizationStudies', '/api/optimization/studies', RUN_TTL_MS);
}

export function getOptimizationStudy(studyId: number): Promise<OptimizationStudy> {
  return request<OptimizationStudy>(`/api/optimization/studies/${studyId}`);
}

export function createOptimizationStudy(payload: OptimizationStudyPayload): Promise<OptimizationStudy> {
  return request<OptimizationStudy>('/api/optimization/studies', {
    method: 'POST',
    body: JSON.stringify(payload),
  }).then((study) => {
    invalidate(['optimizationStudies']);
    return study;
  });
}

export function updateOptimizationStudy(studyId: number, payload: OptimizationStudyPayload): Promise<OptimizationStudy> {
  return request<OptimizationStudy>(`/api/optimization/studies/${studyId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  }).then((study) => {
    invalidate(['optimizationStudies']);
    return study;
  });
}

export async function deleteOptimizationStudy(studyId: number): Promise<void> {
  await request<void>(`/api/optimization/studies/${studyId}`, { method: 'DELETE' });
  invalidate(['optimizationStudies']);
}

export function startOptimizationStudy(studyId: number): Promise<OptimizationStudy> {
  return request<OptimizationStudy>(`/api/optimization/studies/${studyId}/start`, { method: 'POST' }).then((study) => {
    invalidate(['optimizationStudies', 'methodConfigurations', 'trainingPipelines', 'trainingRuns']);
    return study;
  });
}

export function pauseOptimizationStudy(studyId: number): Promise<OptimizationStudy> {
  return request<OptimizationStudy>(`/api/optimization/studies/${studyId}/pause`, { method: 'POST' }).then((study) => {
    invalidate(['optimizationStudies']);
    return study;
  });
}

export function resumeOptimizationStudy(studyId: number): Promise<OptimizationStudy> {
  return request<OptimizationStudy>(`/api/optimization/studies/${studyId}/resume`, { method: 'POST' }).then((study) => {
    invalidate(['optimizationStudies']);
    return study;
  });
}

export function abortOptimizationStudy(studyId: number): Promise<OptimizationStudy> {
  return request<OptimizationStudy>(`/api/optimization/studies/${studyId}/abort`, { method: 'POST' }).then((study) => {
    invalidate(['optimizationStudies', 'trainingRuns', 'testingRuns']);
    return study;
  });
}

export function createOptimizationSplit(payload: OptimizationSplitPayload): Promise<OptimizationSplit> {
  return request<OptimizationSplit>('/api/optimization/splits', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 120_000).then((split) => {
    invalidate(['trainingDatasets']);
    return split;
  });
}

export function promoteOptimizationTrial(
  trialId: number,
  payload: { name: string; description?: string | null },
): Promise<TrainingPipeline> {
  return request<TrainingPipeline>(`/api/optimization/trials/${trialId}/promote`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }).then((pipeline) => {
    invalidate(['optimizationStudies', 'trainingPipelines']);
    return pipeline;
  });
}

/** URL for one rendered overlay frame PNG (served directly, not via fetch). */
export function heatmapRangeFrameUrl(runId: number, index: number): string {
  return projectMediaUrl(`/api/heatmap-ranges/${runId}/frames/${index}.png`);
}

export function heatmapRangeVideoUrl(runId: number): string {
  return projectMediaUrl(`/api/heatmap-ranges/${runId}/video.mp4`);
}

export function previewInspect(payload: {
  training_dataset_id: number;
  preprocessing_pipeline_id: number;
  start_timestamp: string;
  end_timestamp: string;
  stride: number;
  content_mode?: 'final_preprocessed_output';
  analysis_mode?: 'preprocessed_video' | 'contrast_enhanced' | 'energy' | 'optical_flow';
  analysis_config?: Record<string, unknown> | null;
  roi_id?: number | null;
  generate_video?: boolean;
  fps?: number;
  contrast_enabled?: boolean;
  contrast_reference_frames?: number;
  contrast_shift?: number;
  contrast_vmax?: number;
  contrast_ma_radius?: number;
}): Promise<InspectPreview> {
  return request<InspectPreview>('/api/inspect/preview', {
    method: 'POST',
    body: JSON.stringify({ content_mode: 'final_preprocessed_output', ...payload }),
  });
}

export function createInspectRun(payload: {
  training_dataset_id: number;
  preprocessing_pipeline_id: number;
  start_timestamp: string;
  end_timestamp: string;
  stride: number;
  fps: number;
  content_mode?: 'final_preprocessed_output';
  analysis_mode?: 'preprocessed_video' | 'contrast_enhanced' | 'energy' | 'optical_flow';
  analysis_config?: Record<string, unknown> | null;
  roi_id?: number | null;
  generate_video?: boolean;
  contrast_enabled?: boolean;
  contrast_reference_frames?: number;
  contrast_shift?: number;
  contrast_vmax?: number;
  contrast_ma_radius?: number;
}): Promise<InspectRun> {
  return request<InspectRun>('/api/inspect/runs', {
    method: 'POST',
    body: JSON.stringify({ content_mode: 'final_preprocessed_output', ...payload }),
  }).then((run) => {
    invalidate(['inspectRuns']);
    return run;
  });
}

export function listInspectRuns(): Promise<InspectRun[]> {
  return cachedList<InspectRun[]>('inspectRuns', '/api/inspect/runs', RUN_TTL_MS);
}

export function getInspectRun(runId: number): Promise<InspectRun> {
  return request<InspectRun>(`/api/inspect/runs/${runId}`);
}

export function abortInspectRun(runId: number): Promise<InspectRun> {
  return request<InspectRun>(`/api/inspect/runs/${runId}/abort`, {
    method: 'POST',
    body: JSON.stringify({}),
  }).then((run) => {
    invalidate(['inspectRuns']);
    return run;
  });
}

export async function deleteInspectRun(runId: number): Promise<void> {
  await request<void>(`/api/inspect/runs/${runId}`, {
    method: 'DELETE',
  });
  invalidate(['inspectRuns']);
}

export function getInspectRunLog(runId: number): Promise<{ log: string }> {
  return request<{ log: string }>(`/api/inspect/runs/${runId}/log`);
}

export function inspectRunFrameUrl(runId: number, index: number): string {
  return projectMediaUrl(`/api/inspect/runs/${runId}/frames/${index}.png`);
}

export function inspectRunVideoUrl(runId: number): string {
  return projectMediaUrl(`/api/inspect/runs/${runId}/video.mp4`);
}

export function inspectRunCsvUrl(runId: number): string {
  return projectMediaUrl(`/api/inspect/runs/${runId}/results.csv`);
}

export function inspectRunSummaryUrl(runId: number): string {
  return projectMediaUrl(`/api/inspect/runs/${runId}/summary.json`);
}

export function inspectRunPlotPreviewUrl(runId: number): string {
  return projectMediaUrl(`/api/inspect/runs/${runId}/plot-preview.png`);
}

export function inspectPreviewVideoUrl(relativeUrl: string): string {
  return projectMediaUrl(relativeUrl);
}

export function listInspectArtifacts(filters: {
  page?: number;
  training_dataset_id?: number | null;
  preprocessing_pipeline_id?: number | null;
  mode?: string | null;
  status?: string | null;
} = {}): Promise<InspectArtifactRunPage> {
  const params = new URLSearchParams();
  if (filters.page) params.set('page', String(filters.page));
  if (filters.training_dataset_id != null) params.set('training_dataset_id', String(filters.training_dataset_id));
  if (filters.preprocessing_pipeline_id != null) params.set('preprocessing_pipeline_id', String(filters.preprocessing_pipeline_id));
  if (filters.mode) params.set('mode', filters.mode);
  if (filters.status) params.set('status', filters.status);
  return request<InspectArtifactRunPage>(`/api/inspect/artifacts?${params.toString()}`);
}

export function getInspectCsvData(runId: number): Promise<InspectCsvData> {
  return request<InspectCsvData>(`/api/inspect/runs/${runId}/csv-data`);
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

export function moveSchedulerJob(kind: 'train' | 'test' | 'heatmap', runId: number, direction: 'up' | 'down', projectId?: string): Promise<{
  kind: 'train' | 'test' | 'heatmap';
  run_id: number;
  queue_rank: number | null;
}> {
  return request<{ kind: 'train' | 'test' | 'heatmap'; run_id: number; queue_rank: number | null }>(`/api/scheduler/jobs/${kind}/${runId}/move`, {
    method: 'POST',
    body: JSON.stringify({ direction }),
  }, undefined, projectId).then((response) => {
    invalidate(['trainingRuns', 'testingRuns', 'heatmapRanges']);
    return response;
  });
}

export function getTestingRunLog(runId: number, projectId?: string): Promise<{ log: string }> {
  return request<{ log: string }>(`/api/testing-runs/${runId}/log`, undefined, undefined, projectId);
}

export function abortTestingRun(runId: number, projectId?: string): Promise<TestingRun> {
  return request<TestingRun>(`/api/testing-runs/${runId}/abort`, { method: 'POST' }, undefined, projectId).then((run) => {
    invalidate(['testingRuns']);
    return run;
  });
}

export function restartTestingRun(runId: number, projectId?: string): Promise<TestingRun> {
  return request<TestingRun>(`/api/testing-runs/${runId}/restart`, { method: 'POST' }, undefined, projectId).then((run) => {
    invalidate(['testingRuns']);
    return run;
  });
}

export async function deleteTestingRun(runId: number, projectId?: string): Promise<void> {
  await request<void>(`/api/testing-runs/${runId}`, { method: 'DELETE' }, undefined, projectId);
  invalidate(['testingRuns']);
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

// -- Data Manager (registry) --------------------------------------------------

export function getRegistrySummary(): Promise<RegistrySummary> {
  return request<RegistrySummary>('/api/registry/summary');
}

export function listRegistry(
  entityType: string,
  params: {
    search?: string;
    filters?: Record<string, string>;
    sort?: string;
    order?: 'asc' | 'desc';
    limit?: number;
    offset?: number;
  } = {},
): Promise<RegistryList> {
  const query = new URLSearchParams();
  if (params.search) query.set('search', params.search);
  if (params.sort) query.set('sort', params.sort);
  if (params.order) query.set('order', params.order);
  if (params.limit != null) query.set('limit', String(params.limit));
  if (params.offset != null) query.set('offset', String(params.offset));
  for (const [key, value] of Object.entries(params.filters ?? {})) {
    if (value) query.set(key, value);
  }
  const suffix = query.toString() ? `?${query.toString()}` : '';
  return request<RegistryList>(`/api/registry/${entityType}${suffix}`);
}

export function getRegistryDetail(entityType: string, id: number): Promise<RegistryDetail> {
  return request<RegistryDetail>(`/api/registry/${entityType}/${id}`);
}

export function registryDeletePreview(items: RegistryItemRef[]): Promise<RegistryDeletePreview> {
  return request<RegistryDeletePreview>('/api/registry/delete-preview', {
    method: 'POST',
    body: JSON.stringify({ items, cascade: false }),
  });
}

export function registryDelete(items: RegistryItemRef[], cascade: boolean): Promise<RegistryDeleteResult> {
  return request<RegistryDeleteResult>('/api/registry/delete', {
    method: 'POST',
    body: JSON.stringify({ items, cascade }),
  }, 120_000);
}
