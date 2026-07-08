import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Group,
  Modal,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Switch,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Check, Image as ImageIcon, Info, Play, Search, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent } from 'react';
import type React from 'react';

import {
  bulkEnqueueTestingRuns,
  createRoi,
  deleteRoi,
  listMethodConfigurations,
  listPreprocessingPipelines,
  listRois,
  listTrainingDatasets,
  listTrainingPipelines,
  listTrainingRuns,
  previewRoi,
} from '../api';
import { StepCard } from '../components/StepCard';
import { usePendingIds } from '../hooks/usePendingIds';
import { formatValue, methodLabel } from '../methods/utils';
import { datasetResolutions, formatResolution, orderedGraphNodes, stepDetail } from '../training/graph';
import type {
  MethodConfiguration,
  MethodDefinition,
  ModelLayerInstance,
  PreprocessingPipeline,
  RoiDefinition,
  RoiPreview,
  TrainingDataset,
  TrainingPipeline,
  TrainingRun,
} from '../types';
import { listMethodDefinitions } from '../api';

type Rect = { x: number; y: number; width: number; height: number };
type RoiPoint = { x: number; y: number };

const TILE_OPTIONS = [1, 2, 3, 4, 5].map((value) => ({ value: String(value), label: String(value) }));
const SCORE_METRIC_OPTIONS = [
  { value: 'mse', label: 'MSE' },
  { value: 'mae', label: 'MAE' },
  { value: 'ssim_distance', label: 'SSIM distance' },
];

function notifyError(title: string, error: unknown) {
  notifications.show({ color: 'red', title, message: error instanceof Error ? error.message : 'Unknown error' });
}

function roiLabel(roi: RoiDefinition): string {
  const tiles = `${roi.tile_rows ?? 1}x${roi.tile_cols ?? 1}`;
  return `${roi.name} (${roi.image_width}x${roi.image_height}, ${tiles} tiles)`;
}

function pointFromClient(container: HTMLDivElement, clientX: number, clientY: number, preview: RoiPreview): RoiPoint {
  const rect = container.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * preview.width;
  const y = ((clientY - rect.top) / rect.height) * preview.height;
  return {
    x: Math.round(Math.max(0, Math.min(preview.width, x))),
    y: Math.round(Math.max(0, Math.min(preview.height, y))),
  };
}

function defaultRect(preview: RoiPreview): Rect {
  const width = Math.max(1, Math.round(preview.width * 0.5));
  const height = Math.max(1, Math.round(preview.height * 0.5));
  return {
    x: Math.round((preview.width - width) / 2),
    y: Math.round((preview.height - height) / 2),
    width,
    height,
  };
}

function defaultRoiPoints(preview: RoiPreview): RoiPoint[] {
  const rect = defaultRect(preview);
  return [
    { x: rect.x, y: rect.y },
    { x: rect.x + rect.width, y: rect.y },
    { x: rect.x + rect.width, y: rect.y + rect.height },
    { x: rect.x, y: rect.y + rect.height },
  ];
}

function roiPoints(roi: RoiDefinition): RoiPoint[] {
  if (roi.points?.length === 4) {
    return roi.points.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
  }
  return [
    { x: roi.x, y: roi.y },
    { x: roi.x + roi.width, y: roi.y },
    { x: roi.x + roi.width, y: roi.y + roi.height },
    { x: roi.x, y: roi.y + roi.height },
  ];
}

function boundingRect(points: RoiPoint[]): Rect {
  const minX = Math.floor(Math.min(...points.map((point) => point.x)));
  const maxX = Math.ceil(Math.max(...points.map((point) => point.x)));
  const minY = Math.floor(Math.min(...points.map((point) => point.y)));
  const maxY = Math.ceil(Math.max(...points.map((point) => point.y)));
  return { x: minX, y: minY, width: Math.max(1, maxX - minX), height: Math.max(1, maxY - minY) };
}

function interp(points: RoiPoint[], u: number, v: number): RoiPoint {
  const [tl, tr, br, bl] = points;
  const top = { x: tl.x + (tr.x - tl.x) * u, y: tl.y + (tr.y - tl.y) * u };
  const bottom = { x: bl.x + (br.x - bl.x) * u, y: bl.y + (br.y - bl.y) * u };
  return { x: top.x + (bottom.x - top.x) * v, y: top.y + (bottom.y - top.y) * v };
}

function PolygonRoiPicker({
  preview,
  points,
  tileRows,
  tileCols,
  onChange,
}: {
  preview: RoiPreview;
  points: RoiPoint[];
  tileRows: number;
  tileCols: number;
  onChange: (points: RoiPoint[]) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<number | null>(null);
  const polygonPoints = points.map((point) => `${(point.x / preview.width) * 100},${(point.y / preview.height) * 100}`).join(' ');

  function startDrag(index: number, event: PointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    event.stopPropagation();
    dragRef.current = index;
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }

  function onMove(event: PointerEvent<HTMLDivElement>) {
    const index = dragRef.current;
    const container = containerRef.current;
    if (index === null || !container) return;
    const point = pointFromClient(container, event.clientX, event.clientY, preview);
    onChange(points.map((current, currentIndex) => (currentIndex === index ? point : current)));
  }

  return (
    <Stack gap="xs">
      <div
        ref={containerRef}
        className="warp-picker"
        onPointerMove={onMove}
        onPointerUp={() => (dragRef.current = null)}
        onPointerLeave={() => (dragRef.current = null)}
      >
        <img src={preview.image_data_url} alt="ROI preview" className="warp-picker-image" />
        <svg className="roi-overlay" viewBox="0 0 100 100" preserveAspectRatio="none">
          <polygon points={polygonPoints} className="roi-polygon" />
          {Array.from({ length: Math.max(0, tileCols - 1) }, (_, index) => {
            const u = (index + 1) / tileCols;
            const a = interp(points, u, 0);
            const b = interp(points, u, 1);
            return (
              <line
                key={`col-${index}`}
                x1={(a.x / preview.width) * 100}
                y1={(a.y / preview.height) * 100}
                x2={(b.x / preview.width) * 100}
                y2={(b.y / preview.height) * 100}
                className="roi-grid-line"
              />
            );
          })}
          {Array.from({ length: Math.max(0, tileRows - 1) }, (_, index) => {
            const v = (index + 1) / tileRows;
            const a = interp(points, 0, v);
            const b = interp(points, 1, v);
            return (
              <line
                key={`row-${index}`}
                x1={(a.x / preview.width) * 100}
                y1={(a.y / preview.height) * 100}
                x2={(b.x / preview.width) * 100}
                y2={(b.y / preview.height) * 100}
                className="roi-grid-line"
              />
            );
          })}
        </svg>
        {points.map((point, index) => (
          <button
            key={index}
            type="button"
            className="warp-point"
            style={{ left: `${(point.x / preview.width) * 100}%`, top: `${(point.y / preview.height) * 100}%` }}
            onPointerDown={(event) => startDrag(index, event)}
            aria-label={`Move ROI point ${index + 1}`}
          >
            {index + 1}
          </button>
        ))}
      </div>
      <Group gap="xs">
        {points.map((point, index) => (
          <Badge key={index} variant="light">
            p{index + 1}: {Math.round(point.x)}, {Math.round(point.y)}
          </Badge>
        ))}
      </Group>
    </Stack>
  );
}

function renderPipelineDatasets(pipeline: TrainingPipeline | null, datasets: TrainingDataset[]) {
  if (!pipeline) return <Alert color="yellow">Training pipeline details are not available.</Alert>;
  const datasetById = new Map(datasets.map((dataset) => [dataset.id, dataset]));
  return (
    <Stack gap="md">
      {pipeline.training_datasets.map((entry) => {
        const dataset = datasetById.get(entry.training_dataset_id);
        return (
          <Paper key={entry.training_dataset_id} withBorder p="sm" radius="sm">
            <Stack gap="xs">
              <Group justify="space-between">
                <Text fw={700}>{entry.name}</Text>
                <Badge variant="light">
                  {dataset?.counts_missing ? 'Counts need refresh' : `${entry.total_selected_images} images`}
                </Badge>
              </Group>
              {dataset ? (
                <>
                  <Text size="xs" c="dimmed">
                    Label {dataset.usage_label ?? 'train'} · Sizes {datasetResolutions(dataset).join(', ') || 'n/a'} · Sources{' '}
                    {dataset.dataset_names.join(', ')}
                  </Text>
                  <Table striped verticalSpacing="xs">
                    <Table.Thead>
                      <Table.Tr>
                        <Table.Th>Dataset</Table.Th>
                        <Table.Th>Folder</Table.Th>
                        <Table.Th>Start</Table.Th>
                        <Table.Th>End</Table.Th>
                        <Table.Th>Stride</Table.Th>
                        <Table.Th>Images</Table.Th>
                      </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                      {dataset.rules.map((rule) => (
                        <Table.Tr key={rule.id}>
                          <Table.Td>{rule.dataset_name}</Table.Td>
                          <Table.Td>{rule.folder_relative_path}</Table.Td>
                          <Table.Td>{new Date(rule.start_timestamp).toLocaleString()}</Table.Td>
                          <Table.Td>{new Date(rule.end_timestamp).toLocaleString()}</Table.Td>
                          <Table.Td>{rule.stride}</Table.Td>
                          <Table.Td>{rule.selected_images == null ? 'Needs refresh' : rule.selected_images}</Table.Td>
                        </Table.Tr>
                      ))}
                    </Table.Tbody>
                  </Table>
                </>
              ) : (
                <Text size="sm" c="dimmed">
                  Dataset snapshot only: {entry.dataset_names.join(', ')}
                </Text>
              )}
            </Stack>
          </Paper>
        );
      })}
    </Stack>
  );
}

function renderPreprocessingDetails(pipeline: PreprocessingPipeline | null) {
  if (!pipeline) return <Alert color="yellow">Preprocessing pipeline details are not available.</Alert>;
  return (
    <Stack gap="sm">
      <Group>
        <Badge variant="light">Input {pipeline.input_width && pipeline.input_height ? `${pipeline.input_width}x${pipeline.input_height}` : 'n/a'}</Badge>
        <Badge variant="light" color="yellow">
          Output {pipeline.output_width && pipeline.output_height ? `${pipeline.output_width}x${pipeline.output_height}` : 'n/a'}
        </Badge>
      </Group>
      {orderedGraphNodes(pipeline).map((node, index) => (
        <Paper key={node.id} withBorder p="sm" radius="sm">
          <Text fw={700} size="sm">
            {index + 1}. {node.type}
          </Text>
          <Text size="xs" c="dimmed">
            {stepDetail(node)}
          </Text>
          <Group gap={6} mt={6}>
            {Object.entries(node.config ?? {}).map(([key, value]) => (
              <Badge key={key} size="sm" variant="light" color="gray">
                {key}={formatValue(value)}
              </Badge>
            ))}
          </Group>
        </Paper>
      ))}
    </Stack>
  );
}

function renderLayerSection(title: string, layers: ModelLayerInstance[] | undefined) {
  if (!layers || layers.length === 0) return null;
  return (
    <Stack gap="xs">
      <Text fw={700}>{title}</Text>
      {layers.map((layer, index) => (
        <Paper key={layer.id} withBorder p="sm" radius="sm">
          <Text size="sm" fw={600}>
            {index + 1}. {layer.type}
          </Text>
          <Group gap={6} mt={6}>
            {Object.entries(layer.config ?? {}).map(([key, value]) => (
              <Badge key={key} size="sm" variant="light" color="gray">
                {key}={formatValue(value)}
              </Badge>
            ))}
          </Group>
        </Paper>
      ))}
    </Stack>
  );
}

function renderMethodDetails(configuration: MethodConfiguration | null) {
  if (!configuration) return <Alert color="yellow">Method details are not available.</Alert>;
  return (
    <Stack gap="md">
      <div>
        <Text fw={700}>{configuration.name}</Text>
        {configuration.description && (
          <Text size="sm" c="dimmed">
            {configuration.description}
          </Text>
        )}
      </div>
      <Group gap={6}>
        {Object.entries(configuration.method_config ?? {}).map(([key, value]) => (
          <Badge key={key} size="sm" variant="light" color="gray">
            {key}={formatValue(value)}
          </Badge>
        ))}
      </Group>
      {renderLayerSection('Encoder', configuration.method_graph.encoder)}
      {renderLayerSection('Decoder', configuration.method_graph.decoder)}
      {configuration.method_graph.latent && (
        <Paper withBorder p="sm" radius="sm">
          <Text fw={700} size="sm">
            Latent
          </Text>
          <Group gap={6} mt={6}>
            {Object.entries(configuration.method_graph.latent).map(([key, value]) => (
              <Badge key={key} size="sm" variant="light" color="gray">
                {key}={formatValue(value)}
              </Badge>
            ))}
          </Group>
        </Paper>
      )}
    </Stack>
  );
}

export function TestingRunsPage({ active = true, onRunQueued }: { active?: boolean; onRunQueued?: () => void }) {
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [trainingPipelines, setTrainingPipelines] = useState<TrainingPipeline[]>([]);
  const [preprocessingPipelines, setPreprocessingPipelines] = useState<PreprocessingPipeline[]>([]);
  const [methodConfigurations, setMethodConfigurations] = useState<MethodConfiguration[]>([]);
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [methodDefinitions, setMethodDefinitions] = useState<MethodDefinition[]>([]);
  const [rois, setRois] = useState<RoiDefinition[]>([]);

  // Model selection + filters.
  const [modelSearch, setModelSearch] = useState('');
  const [modelMethodFilter, setModelMethodFilter] = useState<string | null>(null);
  const [selectedModelIds, setSelectedModelIds] = useState<number[]>([]);

  // Dataset + ROI selection.
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<number[]>([]);
  const [roiEnabled, setRoiEnabled] = useState(false);
  const [selectedRoiId, setSelectedRoiId] = useState<string | null>(null);
  const [runName, setRunName] = useState('');
  const [, setRunNameTouched] = useState(false);
  const [scoreMetric, setScoreMetric] = useState('mse');
  const [roiName, setRoiName] = useState('');
  const [preview, setPreview] = useState<RoiPreview | null>(null);
  const [points, setPoints] = useState<RoiPoint[] | null>(null);
  const [tileRows, setTileRows] = useState(1);
  const [tileCols, setTileCols] = useState(1);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [savingRoi, setSavingRoi] = useState(false);
  const [running, setRunning] = useState(false);
  const rowActions = usePendingIds();
  const [detailModal, setDetailModal] = useState<{ title: string; body: React.ReactNode } | null>(null);

  async function refresh() {
    const [nextRuns, nextPipelines, nextPreprocessing, nextMethods, nextDatasets, nextDefs, nextRois] = await Promise.all([
      listTrainingRuns(),
      listTrainingPipelines(),
      listPreprocessingPipelines(),
      listMethodConfigurations(),
      listTrainingDatasets(),
      listMethodDefinitions(),
      listRois(),
    ]);
    setTrainingRuns(nextRuns);
    setTrainingPipelines(nextPipelines);
    setPreprocessingPipelines(nextPreprocessing);
    setMethodConfigurations(nextMethods);
    setTrainingDatasets(nextDatasets);
    setMethodDefinitions(nextDefs);
    setRois(nextRois);
  }

  useEffect(() => {
    if (!active) return;
    refresh().catch((error) => notifyError('Could not load testing data', error));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  const methodByType = useMemo(
    () => new Map(methodDefinitions.map((definition) => [definition.type, definition])),
    [methodDefinitions],
  );
  const pipelineById = useMemo(
    () => new Map(trainingPipelines.map((pipeline) => [pipeline.id, pipeline])),
    [trainingPipelines],
  );
  const preprocessingById = useMemo(
    () => new Map(preprocessingPipelines.map((pipeline) => [pipeline.id, pipeline])),
    [preprocessingPipelines],
  );
  const methodConfigById = useMemo(
    () => new Map(methodConfigurations.map((configuration) => [configuration.id, configuration])),
    [methodConfigurations],
  );

  // Trained models = finished training runs that produced an artifact.
  const models = useMemo(
    () => trainingRuns.filter((run) => run.status === 'finished' && run.artifact_path),
    [trainingRuns],
  );
  const methodOptions = useMemo(
    () => [...new Set(models.map((run) => run.method_type))].map((type) => ({ value: type, label: type })),
    [models],
  );
  const selectedModels = useMemo(
    () => selectedModelIds.map((id) => models.find((run) => run.id === id)).filter((run): run is TrainingRun => Boolean(run)),
    [models, selectedModelIds],
  );
  const primaryModel = selectedModels[0] ?? null;
  // The image size a test dataset must have = the selected models' preprocessing input size.
  const requiredInputResolution = useMemo(() => {
    if (!primaryModel) return null;
    const pipeline = pipelineById.get(primaryModel.training_pipeline_id);
    return pipeline ? formatResolution(pipeline.preprocessing_input_width, pipeline.preprocessing_input_height) : null;
  }, [primaryModel, pipelineById]);

  const requiredOutputResolution = useMemo(() => {
    if (!primaryModel) return null;
    const pipeline = pipelineById.get(primaryModel.training_pipeline_id);
    return pipeline ? formatResolution(pipeline.preprocessing_output_width, pipeline.preprocessing_output_height) : null;
  }, [primaryModel, pipelineById]);
  const selectedOutputResolutions = useMemo(() => {
    const values = new Set<string>();
    selectedModels.forEach((run) => {
      const pipeline = pipelineById.get(run.training_pipeline_id);
      const output = pipeline ? formatResolution(pipeline.preprocessing_output_width, pipeline.preprocessing_output_height) : null;
      if (output) values.add(output);
    });
    return [...values].sort();
  }, [selectedModels, pipelineById]);

  const filteredModels = useMemo(() => {
    const query = modelSearch.trim().toLowerCase();
    return models.filter((run) => {
      if (modelMethodFilter && run.method_type !== modelMethodFilter) return false;
      if (requiredInputResolution) {
        const pipeline = pipelineById.get(run.training_pipeline_id);
        const inputRes = pipeline ? formatResolution(pipeline.preprocessing_input_width, pipeline.preprocessing_input_height) : null;
        if (inputRes !== requiredInputResolution && !selectedModelIds.includes(run.id)) return false;
      }
      if (!query) return true;
      return run.training_pipeline_name.toLowerCase().includes(query) || run.method_type.toLowerCase().includes(query);
    });
  }, [models, modelSearch, modelMethodFilter, pipelineById, requiredInputResolution, selectedModelIds]);

  const compatibleDatasets = useMemo(() => {
    if (selectedModels.length === 0) return [];
    if (!requiredInputResolution) return trainingDatasets;
    return trainingDatasets.filter((dataset) => datasetResolutions(dataset).includes(requiredInputResolution));
  }, [selectedModels.length, requiredInputResolution, trainingDatasets]);

  const selectedDatasets = selectedDatasetIds
    .map((id) => trainingDatasets.find((dataset) => dataset.id === id))
    .filter((dataset): dataset is TrainingDataset => Boolean(dataset));
  const previewDatasetId = selectedDatasetIds[0] ?? null;
  const selectedRoi = rois.find((roi) => String(roi.id) === selectedRoiId) ?? null;
  const selectedRoiMismatch =
    roiEnabled &&
    preview &&
    selectedRoi &&
    (preview.width !== selectedRoi.image_width || preview.height !== selectedRoi.image_height);
  const roiModelOutputMismatch = roiEnabled && selectedOutputResolutions.length > 1;

  const selectedTrainingPipeline = primaryModel ? pipelineById.get(primaryModel.training_pipeline_id) ?? null : null;
  const selectedPreprocessing = selectedTrainingPipeline
    ? preprocessingById.get(selectedTrainingPipeline.preprocessing_pipeline_id) ?? null
    : null;
  const selectedMethodConfiguration = selectedTrainingPipeline
    ? methodConfigById.get(selectedTrainingPipeline.method_configuration_id) ?? null
    : null;
  const selectedVaeSampleCount =
    selectedMethodConfiguration?.method_type === 'cnn_vae'
      ? Number(selectedMethodConfiguration.inference_config?.sample_count ?? 1)
      : null;

  useEffect(() => {
    const defaultMetric = selectedMethodConfiguration?.inference_config?.error_metric;
    setScoreMetric(typeof defaultMetric === 'string' ? defaultMetric : 'mse');
  }, [selectedMethodConfiguration?.id]);

  function toggleModel(runId: number) {
    setSelectedModelIds((current) => {
      const next = current.includes(runId) ? current.filter((id) => id !== runId) : [...current, runId];
      if (next.length === 0) {
        setSelectedDatasetIds([]);
        setSelectedRoiId(null);
        setPreview(null);
        setPoints(null);
        setRoiEnabled(false);
      }
      return next;
    });
    setRunNameTouched(false);
  }

  function clearModels() {
    setSelectedModelIds([]);
    setSelectedDatasetIds([]);
    setSelectedRoiId(null);
    setPreview(null);
    setPoints(null);
    setRoiEnabled(false);
    setRunNameTouched(false);
  }

  async function handlePreview() {
    if (!primaryModel || !previewDatasetId) return;
    if (loadingPreview) return;
    setLoadingPreview(true);
    try {
      const nextPreview = await previewRoi({ training_run_id: primaryModel.id, training_dataset_id: previewDatasetId });
      setPreview(nextPreview);
      setPoints(defaultRoiPoints(nextPreview));
      if (!roiName) setRoiName(`ROI ${nextPreview.width}x${nextPreview.height}`);
    } catch (error) {
      notifyError('Could not load ROI preview', error);
    } finally {
      setLoadingPreview(false);
    }
  }

  async function handleSaveRoi() {
    if (!preview || !points || !roiName.trim()) return;
    if (savingRoi) return;
    setSavingRoi(true);
    try {
      const rect = boundingRect(points);
      const roi = await createRoi({
        name: roiName.trim(),
        image_width: preview.width,
        image_height: preview.height,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        geometry_type: 'polygon',
        points,
        tile_rows: tileRows,
        tile_cols: tileCols,
      });
      setSelectedRoiId(String(roi.id));
      setRoiEnabled(true);
      setRoiName('');
      await refresh();
      notifications.show({ color: 'green', title: 'ROI saved', message: roi.name });
    } catch (error) {
      notifyError('Could not save ROI', error);
    } finally {
      setSavingRoi(false);
    }
  }

  async function handleDeleteRoi(roi: RoiDefinition) {
    if (!window.confirm(`Delete ROI "${roi.name}"?`)) return;
    await rowActions.runPending(`delete-roi:${roi.id}`, async () => {
      await deleteRoi(roi.id);
      if (selectedRoiId === String(roi.id)) setSelectedRoiId(null);
      await refresh();
      notifications.show({ color: 'green', title: 'ROI deleted', message: roi.name });
    }).catch((error) => {
      notifyError('Could not delete ROI', error);
    });
  }

  async function handleRun() {
    if (selectedModelIds.length === 0 || selectedDatasetIds.length === 0 || selectedRoiMismatch) return;
    if (running) return;
    setRunning(true);
    try {
      const response = await bulkEnqueueTestingRuns({
        training_run_ids: selectedModelIds,
        training_dataset_ids: selectedDatasetIds,
        roi_id: roiEnabled && selectedRoiId ? Number(selectedRoiId) : null,
        name_prefix: runName.trim() || null,
        inference_config: { error_metric: scoreMetric },
      });
      setRunName('');
      setRunNameTouched(false);
      notifications.show({
        color: response.created.length > 0 ? 'green' : 'blue',
        title: 'Inference queue updated',
        message: `${response.created.length} queued, ${response.skipped.length} duplicate${response.skipped.length === 1 ? '' : 's'} skipped.`,
      });
      onRunQueued?.();
    } catch (error) {
      notifyError('Could not queue inference batch', error);
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
    <Stack gap="lg">
      <div>
        <Title order={2}>Testing</Title>
        <Text c="dimmed" size="sm">
          Pick a trained model, choose a size-compatible inference dataset, optionally focus on an ROI, then queue an
          inference run. Progress and results appear on the Scheduler page.
        </Text>
      </div>

      {/* Step 1: model selection */}
      <StepCard
        index={1}
        title="Trained models"
        color="blue"
        complete={selectedModels.length > 0}
        action={selectedModels.length > 0 ? <Button variant="subtle" onClick={clearModels}>Clear</Button> : undefined}
      >
          {selectedModels.length > 0 && (
            <Alert color="green" icon={<Check size={16} />}>
              <Stack gap="xs">
                <Group gap="xs">
                  <Badge variant="light" color="blue">{selectedModels.length} selected</Badge>
                  <Text size="sm" c="dimmed">
                    Raw input locked to {requiredInputResolution ?? 'n/a'}
                    {roiEnabled ? ` · ROI output ${requiredOutputResolution ?? 'n/a'}` : ''}
                  </Text>
                  {selectedVaeSampleCount != null && selectedModels.length === 1 && (
                    <Badge variant="light" color={selectedVaeSampleCount > 1 ? 'violet' : 'gray'}>
                      VAE samples {selectedVaeSampleCount}
                    </Badge>
                  )}
                </Group>
                <Group gap="xs">
                  {selectedModels.map((run) => (
                    <Badge key={run.id} variant="filled" color="blue" rightSection={
                      <ActionIcon
                        size="xs"
                        variant="transparent"
                        color="white"
                        aria-label={`Remove ${run.training_pipeline_name}`}
                        onClick={() => toggleModel(run.id)}
                      >
                        <Trash2 size={10} />
                      </ActionIcon>
                    }>
                      {run.training_pipeline_name}
                    </Badge>
                  ))}
                </Group>
              </Stack>
            </Alert>
          )}
              <TextInput
                placeholder="Search by pipeline or method"
                leftSection={<Search size={16} />}
                value={modelSearch}
                onChange={(event) => setModelSearch(event.currentTarget.value)}
              />
              <Group grow>
                <Select placeholder="Method type" data={methodOptions} value={modelMethodFilter} onChange={setModelMethodFilter} clearable />
              </Group>
              <ScrollArea h={240}>
                <Table striped highlightOnHover verticalSpacing="sm">
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Pipeline</Table.Th>
                      <Table.Th>Trainsets</Table.Th>
                      <Table.Th>Preprocessing</Table.Th>
                      <Table.Th>Method</Table.Th>
                      <Table.Th>Artifact</Table.Th>
                      <Table.Th>Input size</Table.Th>
                      <Table.Th />
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {filteredModels.map((run) => {
                      const pipeline = pipelineById.get(run.training_pipeline_id);
                      const preprocessing = pipeline ? preprocessingById.get(pipeline.preprocessing_pipeline_id) ?? null : null;
                      const configuration = pipeline ? methodConfigById.get(pipeline.method_configuration_id) ?? null : null;
                      const inputRes = pipeline
                        ? formatResolution(pipeline.preprocessing_input_width, pipeline.preprocessing_input_height)
                        : null;
                      const outputRes = pipeline
                        ? formatResolution(pipeline.preprocessing_output_width, pipeline.preprocessing_output_height)
                        : null;
                      const selected = selectedModelIds.includes(run.id);
                      const disabledByRoi = roiEnabled && requiredOutputResolution !== null && outputRes !== requiredOutputResolution && !selected;
                      return (
                        <Table.Tr key={run.id} className={selected ? 'pipeline-step selected' : 'pipeline-step'}>
                          <Table.Td>{run.training_pipeline_name}</Table.Td>
                          <Table.Td>
                            <Group gap={4} wrap="nowrap">
                              <Text size="sm">
                                {pipeline?.training_datasets.map((entry) => entry.name).join(', ') ?? run.dataset_names.join(', ')}
                              </Text>
                              <Tooltip label="Inspect trainsets">
                                <ActionIcon
                                  size="sm"
                                  variant="subtle"
                                  aria-label={`Inspect trainsets for ${run.training_pipeline_name}`}
                                  onClick={() => setDetailModal({ title: 'Trainsets', body: renderPipelineDatasets(pipeline ?? null, trainingDatasets) })}
                                >
                                  <Info size={14} />
                                </ActionIcon>
                              </Tooltip>
                            </Group>
                          </Table.Td>
                          <Table.Td>
                            <Group gap={4} wrap="nowrap">
                              <Text size="sm">{run.preprocessing_pipeline_name}</Text>
                              <Tooltip label="Inspect preprocessing pipeline">
                                <ActionIcon
                                  size="sm"
                                  variant="subtle"
                                  aria-label={`Inspect preprocessing for ${run.training_pipeline_name}`}
                                  onClick={() => setDetailModal({ title: 'Preprocessing pipeline', body: renderPreprocessingDetails(preprocessing) })}
                                >
                                  <Info size={14} />
                                </ActionIcon>
                              </Tooltip>
                            </Group>
                          </Table.Td>
                          <Table.Td>
                            <Group gap={4} wrap="nowrap">
                              <Text size="sm">{configuration?.name ?? run.method_type}</Text>
                              <Tooltip label="Inspect method architecture">
                                <ActionIcon
                                  size="sm"
                                  variant="subtle"
                                  aria-label={`Inspect method for ${run.training_pipeline_name}`}
                                  onClick={() => setDetailModal({ title: 'Method architecture', body: renderMethodDetails(configuration) })}
                                >
                                  <Info size={14} />
                                </ActionIcon>
                              </Tooltip>
                            </Group>
                          </Table.Td>
                          <Table.Td>{run.artifact_kind}</Table.Td>
                          <Table.Td>
                            <Badge size="xs" variant="light" color="blue">
                              {inputRes ?? 'n/a'}
                            </Badge>
                          </Table.Td>
                          <Table.Td>
                            <Group justify="flex-end">
                              <Tooltip label={disabledByRoi ? `ROI requires output ${requiredOutputResolution}` : selected ? 'Remove model' : 'Select model'}>
                                <Button
                                  size="compact-sm"
                                  variant={selected ? 'filled' : 'light'}
                                  leftSection={<Check size={14} />}
                                  disabled={disabledByRoi}
                                  onClick={() => toggleModel(run.id)}
                                >
                                  {selected ? 'Selected' : 'Select'}
                                </Button>
                              </Tooltip>
                            </Group>
                          </Table.Td>
                        </Table.Tr>
                      );
                    })}
                  </Table.Tbody>
                </Table>
              </ScrollArea>
              {models.length === 0 && <Alert color="blue">No trained models yet. Run a training pipeline first.</Alert>}
      </StepCard>

      {/* Step 2: compatible dataset + ROI + run */}
      {selectedModels.length > 0 && (
        <StepCard index={2} title="Inference datasets & run" color="violet" complete={selectedDatasetIds.length > 0}>
            <Text size="xs" c="dimmed">
              Showing datasets whose image size matches the selected models' preprocessing input ({requiredInputResolution ?? 'any'}).
            </Text>
            <ScrollArea h={200}>
              <Table striped highlightOnHover verticalSpacing="xs">
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Name</Table.Th>
                    <Table.Th>Label</Table.Th>
                    <Table.Th>Image size</Table.Th>
                    <Table.Th>Images</Table.Th>
                    <Table.Th />
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {compatibleDatasets.map((dataset) => {
                    const selected = selectedDatasetIds.includes(dataset.id);
                    return (
                      <Table.Tr
                        key={dataset.id}
                        className={selected ? 'pipeline-step selected' : 'pipeline-step'}
                        style={{ cursor: 'pointer' }}
                        onClick={() => {
                          setSelectedDatasetIds((current) =>
                            current.includes(dataset.id) ? current.filter((id) => id !== dataset.id) : [...current, dataset.id],
                          );
                          setPreview(null);
                          setPoints(null);
                          setRunNameTouched(false);
                        }}
                      >
                        <Table.Td>{dataset.name}</Table.Td>
                        <Table.Td>{dataset.usage_label ?? 'train'}</Table.Td>
                        <Table.Td>
                          <Group gap={4}>
                            {datasetResolutions(dataset).map((res) => (
                              <Badge key={res} size="xs" variant="light" color="teal">
                                {res}
                              </Badge>
                            ))}
                          </Group>
                        </Table.Td>
                        <Table.Td>{dataset.counts_missing ? 'Needs refresh' : dataset.total_selected_images}</Table.Td>
                        <Table.Td>{selected && <Check size={16} />}</Table.Td>
                      </Table.Tr>
                    );
                  })}
                </Table.Tbody>
              </Table>
            </ScrollArea>
            {compatibleDatasets.length === 0 && (
              <Alert color="yellow">No size-compatible inference datasets found for input {requiredInputResolution ?? 'n/a'}.</Alert>
            )}
            {selectedDatasets.length > 0 && (
              <Alert color="blue">
                {selectedDatasets.length} dataset{selectedDatasets.length === 1 ? '' : 's'} selected ·{' '}
                {selectedModelIds.length * selectedDatasets.length} inference run
                {selectedModelIds.length * selectedDatasets.length === 1 ? '' : 's'} will be queued.
              </Alert>
            )}

            <Group align="flex-end">
              <TextInput
                label="Name prefix"
                description="Optional prefix added to each queued inference run."
                placeholder="Auto"
                value={runName}
                onChange={(event) => {
                  setRunNameTouched(true);
                  setRunName(event.currentTarget.value);
                }}
                style={{ flex: 1 }}
              />
              <Select
                label={
                  <Group gap={4}>
                    <Text span>Score metric</Text>
                    <Tooltip
                      label="MSE and MAE score pixel-wise residuals. SSIM distance scores local structural dissimilarity as 1 - SSIM using standard K constants."
                      multiline
                      w={320}
                      withArrow
                    >
                      <Info size={14} />
                    </Tooltip>
                  </Group>
                }
                data={SCORE_METRIC_OPTIONS}
                value={scoreMetric}
                onChange={(value) => setScoreMetric(value ?? 'mse')}
                allowDeselect={false}
                w={190}
              />
              <Button
                leftSection={<Play size={16} />}
                onClick={handleRun}
                loading={running}
                disabled={selectedDatasetIds.length === 0 || Boolean(selectedRoiMismatch) || roiModelOutputMismatch}
              >
                Run {selectedModelIds.length * selectedDatasetIds.length || ''}
              </Button>
            </Group>

            <Paper withBorder p="sm" radius="sm">
              <Stack gap="sm">
                <Switch
                  label="Use ROI scoring"
                  description="Off = full-image MSE. On = choose or define a reusable quadrilateral ROI."
                  checked={roiEnabled}
                  onChange={(event) => {
                    const checked = event.currentTarget.checked;
                    setRoiEnabled(checked);
                    if (!checked) setSelectedRoiId(null);
                    setRunNameTouched(false);
                  }}
                />
                {roiEnabled && (
                  <>
                    <Group align="flex-end">
                      <Select
                        label="Saved ROI"
                        placeholder="Choose saved ROI or define a new one"
                        data={rois.map((roi) => ({ value: String(roi.id), label: roiLabel(roi) }))}
                        value={selectedRoiId}
                        onChange={(value) => {
                          setSelectedRoiId(value);
                          setRunNameTouched(false);
                          const roi = rois.find((item) => String(item.id) === value);
                          if (roi) {
                            setTileRows(roi.tile_rows ?? 1);
                            setTileCols(roi.tile_cols ?? 1);
                            setPoints(roiPoints(roi));
                          }
                        }}
                        clearable
                        searchable
                        style={{ flex: 1 }}
                      />
                      <Button
                        variant="light"
                        leftSection={<ImageIcon size={16} />}
                        onClick={handlePreview}
                        loading={loadingPreview}
                        disabled={!previewDatasetId || roiModelOutputMismatch}
                      >
                        Load ROI Preview
                      </Button>
                    </Group>
                    <Group grow>
                      <Select
                        label="Tile rows"
                        data={TILE_OPTIONS}
                        value={String(tileRows)}
                        onChange={(value) => setTileRows(Number(value ?? 1))}
                      />
                      <Select
                        label="Tile columns"
                        data={TILE_OPTIONS}
                        value={String(tileCols)}
                        onChange={(value) => setTileCols(Number(value ?? 1))}
                      />
                    </Group>
                  </>
                )}
              </Stack>
            </Paper>

            {selectedRoiMismatch && (
              <Alert color="red">
                Selected ROI is tuned for {selectedRoi.image_width}x{selectedRoi.image_height}, but the preview is{' '}
                {preview.width}x{preview.height}.
              </Alert>
            )}
            {roiModelOutputMismatch && (
              <Alert color="red">
                ROI scoring requires all selected models to produce the same output size. Current outputs:{' '}
                {selectedOutputResolutions.join(', ')}.
              </Alert>
            )}

            {roiEnabled && preview && points && (
              <Stack gap="sm">
                <Group justify="space-between" align="center">
                  <Text size="sm" c="dimmed" className="path-text">
                    Preview: {preview.width}x{preview.height}, {preview.dtype}
                  </Text>
                  <Group>
                    <TextInput placeholder="ROI name" value={roiName} onChange={(event) => setRoiName(event.currentTarget.value)} />
                    <Button variant="light" onClick={handleSaveRoi} loading={savingRoi} disabled={!roiName.trim()}>
                      Save ROI
                    </Button>
                  </Group>
                </Group>
                <PolygonRoiPicker
                  preview={preview}
                  points={points}
                  tileRows={tileRows}
                  tileCols={tileCols}
                  onChange={setPoints}
                />
              </Stack>
            )}
        </StepCard>
      )}

      {/* Saved ROIs management */}
      <StepCard title="Saved ROIs" color="cyan">
          <ScrollArea>
            <Table striped highlightOnHover verticalSpacing="sm">
              <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Name</Table.Th>
                    <Table.Th>Image size</Table.Th>
                    <Table.Th>Tiles</Table.Th>
                    <Table.Th>Points</Table.Th>
                    <Table.Th />
                  </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {rois.map((roi) => (
                  <Table.Tr key={roi.id}>
                    <Table.Td>{roi.name}</Table.Td>
                    <Table.Td>{roi.image_width}x{roi.image_height}</Table.Td>
                    <Table.Td>{roi.tile_rows ?? 1}x{roi.tile_cols ?? 1}</Table.Td>
                    <Table.Td>
                      <Text size="xs" c="dimmed">
                        {roiPoints(roi).map((point) => `${Math.round(point.x)},${Math.round(point.y)}`).join(' · ')}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Group justify="flex-end">
                        <Tooltip label="Delete ROI">
                          <ActionIcon
                            aria-label={`Delete ROI ${roi.name}`}
                            color="red"
                            variant="subtle"
                            loading={rowActions.isPending(`delete-roi:${roi.id}`)}
                            disabled={rowActions.isPending(`delete-roi:${roi.id}`)}
                            onClick={() => handleDeleteRoi(roi)}
                          >
                            <Trash2 size={18} />
                          </ActionIcon>
                        </Tooltip>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
          {rois.length === 0 && <Alert color="blue">No ROIs saved yet.</Alert>}
      </StepCard>
    </Stack>
    <Modal
      opened={detailModal !== null}
      onClose={() => setDetailModal(null)}
      title={detailModal?.title}
      size="xl"
      scrollAreaComponent={ScrollArea.Autosize}
    >
      {detailModal?.body}
    </Modal>
    </>
  );
}
