import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Collapse,
  Group,
  NumberInput,
  MultiSelect,
  Pagination,
  Paper,
  Progress,
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
import { ChevronDown, ChevronRight, Download, Eye, FileVideo, Info, RefreshCw, Save, Square, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent } from 'react';

import {
  abortInspectRun,
  abortHeatmapRange,
  createRoi,
  createInspectRun,
  deleteInspectRun,
  deleteHeatmapRange,
  getInspectCsvData,
  heatmapRangeVideoUrl,
  inspectRunCsvUrl,
  inspectRunVideoUrl,
  inspectPreviewVideoUrl,
  listInspectArtifacts,
  listRois,
  listInspectRuns,
  listPreprocessingPipelines,
  listTrainingDatasets,
  previewInspect,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { StepCard } from '../components/StepCard';
import { PlotlyChart } from '../components/PlotlyChart';
import { usePendingIds } from '../hooks/usePendingIds';
import type { InspectArtifactRun, InspectCsvData, InspectPreview, InspectRun, PreprocessingPipeline, RoiDefinition, TrainingDataset } from '../types';

type InspectAnalysisMode = 'preprocessed_video' | 'contrast_enhanced' | 'energy' | 'optical_flow';
type RoiPoint = { x: number; y: number };

const TILE_OPTIONS = [1, 2, 3, 4, 5].map((value) => ({ value: String(value), label: String(value) }));

function toInputDateTime(value: string | null): string {
  return value ? value.slice(0, 19) : '';
}

function formatTimestamp(value: string | null): string {
  return value ? value.replace('T', ' ').slice(0, 19) : 'n/a';
}

function statusColor(status: string): string {
  if (status === 'finished') return 'green';
  if (status === 'failed') return 'red';
  if (status === 'aborted') return 'yellow';
  if (status === 'running') return 'blue';
  return 'gray';
}

function progressLabel(run: InspectRun): string {
  return `${run.done_count}${run.frame_count ? ` / ${run.frame_count}` : ''}`;
}

function InfoLabel({ label, info }: { label: string; info: string }) {
  return (
    <Group gap={5} wrap="nowrap">
      <Text size="sm">{label}</Text>
      <Tooltip label={info} multiline w={320} withArrow>
        <Info size={14} aria-label={`${label} information`} tabIndex={0} />
      </Tooltip>
    </Group>
  );
}

function roiLabel(roi: RoiDefinition): string {
  return `${roi.name} (${roi.image_width}x${roi.image_height}, ${roi.tile_rows ?? 1}x${roi.tile_cols ?? 1} tiles)`;
}

function roiPoints(roi: RoiDefinition): RoiPoint[] {
  if (roi.points?.length === 4) return roi.points.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
  return [
    { x: roi.x, y: roi.y },
    { x: roi.x + roi.width, y: roi.y },
    { x: roi.x + roi.width, y: roi.y + roi.height },
    { x: roi.x, y: roi.y + roi.height },
  ];
}

function defaultRoiPoints(preview: InspectPreview): RoiPoint[] {
  const width = Math.max(1, Math.round(preview.width * 0.5));
  const height = Math.max(1, Math.round(preview.height * 0.5));
  const x = Math.round((preview.width - width) / 2);
  const y = Math.round((preview.height - height) / 2);
  return [
    { x, y },
    { x: x + width, y },
    { x: x + width, y: y + height },
    { x, y: y + height },
  ];
}

function boundingRect(points: RoiPoint[]) {
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

function pointFromClient(container: HTMLDivElement, clientX: number, clientY: number, preview: InspectPreview): RoiPoint {
  const rect = container.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * preview.width;
  const y = ((clientY - rect.top) / rect.height) * preview.height;
  return { x: Math.round(Math.max(0, Math.min(preview.width, x))), y: Math.round(Math.max(0, Math.min(preview.height, y))) };
}

function PolygonRoiPicker({
  preview,
  points,
  tileRows,
  tileCols,
  onChange,
}: {
  preview: InspectPreview;
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
      <div ref={containerRef} className="warp-picker" onPointerMove={onMove} onPointerUp={() => (dragRef.current = null)} onPointerLeave={() => (dragRef.current = null)}>
        <img src={preview.image_data_url} alt="ROI preview" className="warp-picker-image" />
        <svg className="roi-overlay" viewBox="0 0 100 100" preserveAspectRatio="none">
          <polygon points={polygonPoints} className="roi-polygon" />
          {Array.from({ length: Math.max(0, tileCols - 1) }, (_, index) => {
            const u = (index + 1) / tileCols;
            const a = interp(points, u, 0);
            const b = interp(points, u, 1);
            return <line key={`col-${index}`} x1={(a.x / preview.width) * 100} y1={(a.y / preview.height) * 100} x2={(b.x / preview.width) * 100} y2={(b.y / preview.height) * 100} className="roi-grid-line" />;
          })}
          {Array.from({ length: Math.max(0, tileRows - 1) }, (_, index) => {
            const v = (index + 1) / tileRows;
            const a = interp(points, 0, v);
            const b = interp(points, 1, v);
            return <line key={`row-${index}`} x1={(a.x / preview.width) * 100} y1={(a.y / preview.height) * 100} x2={(b.x / preview.width) * 100} y2={(b.y / preview.height) * 100} className="roi-grid-line" />;
          })}
        </svg>
        {points.map((point, index) => (
          <button key={index} type="button" className="warp-point" style={{ left: `${(point.x / preview.width) * 100}%`, top: `${(point.y / preview.height) * 100}%` }} onPointerDown={(event) => startDrag(index, event)}>
            {index + 1}
          </button>
        ))}
      </div>
    </Stack>
  );
}

function selectionSignature(values: {
  trainingDatasetId: number | null;
  preprocessingPipelineId: number | null;
  start: string;
  end: string;
  stride: number;
  analysisMode: InspectAnalysisMode;
  analysisConfig: Record<string, unknown>;
  roiId: number | null;
  generateVideo: boolean;
  contrastEnabled: boolean;
  contrastReferenceFrames: number;
  contrastShift: number;
  contrastVmax: number;
  contrastMaRadius: number;
}): string {
  return JSON.stringify(values);
}

function artifactVideoUrl(artifact: InspectArtifactRun): string {
  return artifact.kind === 'heatmap' ? heatmapRangeVideoUrl(artifact.id) : inspectRunVideoUrl(artifact.id);
}

function ArtifactViewer({ artifact }: { artifact: InspectArtifactRun | null }) {
  const [view, setView] = useState<'video' | 'csv'>('video');
  const [csvData, setCsvData] = useState<InspectCsvData | null>(null);
  const [csvLoading, setCsvLoading] = useState(false);
  const [xColumn, setXColumn] = useState<string | null>(null);
  const [yColumns, setYColumns] = useState<string[]>([]);

  useEffect(() => {
    setView(artifact?.has_video ? 'video' : 'csv');
    setCsvData(null);
    setXColumn(null);
    setYColumns([]);
  }, [artifact?.kind, artifact?.id]);

  useEffect(() => {
    if (!artifact || view !== 'csv' || !artifact.has_csv || csvData || csvLoading) return;
    setCsvLoading(true);
    getInspectCsvData(artifact.id)
      .then(setCsvData)
      .catch((error) => notifications.show({ color: 'red', title: 'CSV load failed', message: error instanceof Error ? error.message : 'Unknown error' }))
      .finally(() => setCsvLoading(false));
  }, [artifact, view, csvData, csvLoading]);

  if (!artifact) return null;
  const viewOptions = [
    ...(artifact.has_video ? [{ value: 'video', label: 'MP4' }] : []),
    ...(artifact.has_csv ? [{ value: 'csv', label: 'CSV plot' }] : []),
  ];
  const numericColumns = csvData?.columns.filter((column) => column.kind === 'number') ?? [];
  const plotData = yColumns.map((column) => ({
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: column,
    x: csvData?.rows.map((row, index) => xColumn ? row[xColumn] : index) ?? [],
    y: csvData?.rows.map((row) => row[column]) ?? [],
  }));

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="sm">
        <Group justify="space-between" align="flex-end" wrap="wrap">
          <Group gap="xs">
            <Text fw={700}>{artifact.training_dataset_name}</Text>
            <Badge variant="light">{artifact.preprocessing_pipeline_name}</Badge>
            <Badge variant="light" color="cyan">{artifact.mode.replaceAll('_', ' ')}</Badge>
          </Group>
          <Group gap="xs">
            {viewOptions.length > 1 && <Select size="xs" data={viewOptions} value={view} onChange={(value) => setView((value ?? 'video') as 'video' | 'csv')} />}
            {artifact.has_video && <Button size="compact-sm" component="a" href={artifactVideoUrl(artifact)} download leftSection={<Download size={14} />}>MP4</Button>}
            {artifact.has_csv && <Button size="compact-sm" variant="light" component="a" href={inspectRunCsvUrl(artifact.id)} download leftSection={<Download size={14} />}>CSV</Button>}
          </Group>
        </Group>
        {view === 'video' && artifact.has_video && (
          <video src={artifactVideoUrl(artifact)} controls muted playsInline style={{ display: 'block', maxWidth: '100%', maxHeight: 'min(70vh, 720px)', margin: '0 auto', borderRadius: 6 }} />
        )}
        {view === 'csv' && artifact.has_csv && (
          <Stack gap="sm">
            <Group grow align="flex-end">
              <Select label="X column" data={(csvData?.columns ?? []).map((column) => ({ value: column.name, label: `${column.name} (${column.kind})` }))} value={xColumn} onChange={setXColumn} searchable disabled={csvLoading} />
              <MultiSelect label="Y columns" data={numericColumns.map((column) => ({ value: column.name, label: column.name }))} value={yColumns} onChange={setYColumns} searchable disabled={csvLoading} />
            </Group>
            {csvLoading && <Progress value={100} animated />}
            {csvData && (!xColumn || yColumns.length === 0) && <Alert color="blue">Choose one X column and at least one numeric Y column.</Alert>}
            {csvData && xColumn && yColumns.length > 0 && <PlotlyChart data={plotData} layout={{ title: { text: `${artifact.mode.replaceAll('_', ' ')} · run ${artifact.id}` }, xaxis: { title: { text: xColumn } }, yaxis: { title: { text: 'Value' } }, hovermode: 'x unified' }} height={480} />}
          </Stack>
        )}
      </Stack>
    </Paper>
  );
}

export function InspectPage({ active = true }: { active?: boolean }) {
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [preprocessingPipelines, setPreprocessingPipelines] = useState<PreprocessingPipeline[]>([]);
  const [rois, setRois] = useState<RoiDefinition[]>([]);
  const [runs, setRuns] = useState<InspectRun[]>([]);
  const [artifactItems, setArtifactItems] = useState<InspectArtifactRun[]>([]);
  const [artifactTotal, setArtifactTotal] = useState(0);
  const [artifactPages, setArtifactPages] = useState(1);
  const [artifactActiveTotal, setArtifactActiveTotal] = useState(0);
  const [artifactPage, setArtifactPage] = useState(1);
  const [artifactModeFilter, setArtifactModeFilter] = useState<string | null>(null);
  const [artifactDatasetFilter, setArtifactDatasetFilter] = useState<number | null>(null);
  const [artifactPipelineFilter, setArtifactPipelineFilter] = useState<number | null>(null);
  const [artifactStatusFilter, setArtifactStatusFilter] = useState<string | null>(null);
  const [artifactsOpen, setArtifactsOpen] = useState(true);
  const [selectedArtifact, setSelectedArtifact] = useState<InspectArtifactRun | null>(null);
  const [matchingArtifacts, setMatchingArtifacts] = useState<InspectArtifactRun[]>([]);
  const [trainingDatasetId, setTrainingDatasetId] = useState<number | null>(null);
  const [preprocessingPipelineId, setPreprocessingPipelineId] = useState<number | null>(null);
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [stride, setStride] = useState(1);
  const [fps, setFps] = useState(12);
  const [analysisMode, setAnalysisMode] = useState<InspectAnalysisMode>('preprocessed_video');
  const [generateVideo, setGenerateVideo] = useState(true);
  const [contrastEnabled, setContrastEnabled] = useState(false);
  const [contrastReferenceFrames, setContrastReferenceFrames] = useState(100);
  const [contrastShift, setContrastShift] = useState(10000);
  const [contrastVmax, setContrastVmax] = useState(12000);
  const [contrastMaRadius, setContrastMaRadius] = useState(3);
  const [energyVariant, setEnergyVariant] = useState<'pairwise' | 'window'>('pairwise');
  const [energyAggregation, setEnergyAggregation] = useState<'sum' | 'mean' | 'p95'>('sum');
  const [energyWindowSize, setEnergyWindowSize] = useState(5);
  const [energyNormalize, setEnergyNormalize] = useState(false);
  const [flowAggregation, setFlowAggregation] = useState<'mean_magnitude' | 'p95_magnitude' | 'max_magnitude'>('mean_magnitude');
  const [flowPyrScale, setFlowPyrScale] = useState(0.5);
  const [flowLevels, setFlowLevels] = useState(3);
  const [flowWinSize, setFlowWinSize] = useState(15);
  const [flowIterations, setFlowIterations] = useState(3);
  const [flowPolyN, setFlowPolyN] = useState(5);
  const [flowPolySigma, setFlowPolySigma] = useState(1.2);
  const [flowNormalize, setFlowNormalize] = useState(true);
  const [roiEnabled, setRoiEnabled] = useState(false);
  const [selectedRoiId, setSelectedRoiId] = useState<number | null>(null);
  const [roiName, setRoiName] = useState('');
  const [roiPointsDraft, setRoiPointsDraft] = useState<RoiPoint[] | null>(null);
  const [tileRows, setTileRows] = useState(1);
  const [tileCols, setTileCols] = useState(1);
  const [savingRoi, setSavingRoi] = useState(false);
  const [preview, setPreview] = useState<InspectPreview | null>(null);
  const [previewSignature, setPreviewSignature] = useState('');
  const [previewLoading, setPreviewLoading] = useState(false);
  const [runLoading, setRunLoading] = useState(false);
  const rowActions = usePendingIds();

  async function refreshArtifacts() {
    const result = await listInspectArtifacts({
      page: artifactPage,
      training_dataset_id: artifactDatasetFilter,
      preprocessing_pipeline_id: artifactPipelineFilter,
      mode: artifactModeFilter,
      status: artifactStatusFilter,
    });
    setArtifactItems(result.items);
    setArtifactTotal(result.total);
    setArtifactPages(result.pages);
    setArtifactActiveTotal(result.active_total);
    if (result.page !== artifactPage) setArtifactPage(result.page);
  }

  async function refresh() {
    const [nextDatasets, nextPipelines, nextRois, nextRuns] = await Promise.all([
      listTrainingDatasets(),
      listPreprocessingPipelines(),
      listRois(),
      listInspectRuns(),
    ]);
    setTrainingDatasets(nextDatasets);
    setPreprocessingPipelines(nextPipelines);
    setRois(nextRois);
    setRuns(nextRuns);
    await refreshArtifacts();
  }

  useEffect(() => {
    if (!active) return;
    refresh().catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Could not load Inspect data',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    });
  }, [active]);

  useEffect(() => {
    if (!active) return;
    refreshArtifacts().catch(() => undefined);
  }, [active, artifactPage, artifactModeFilter, artifactDatasetFilter, artifactPipelineFilter, artifactStatusFilter]);

  useEffect(() => {
    if (!active || trainingDatasetId == null || preprocessingPipelineId == null) {
      setMatchingArtifacts([]);
      return;
    }
    let cancelled = false;
    (async () => {
      const all: InspectArtifactRun[] = [];
      let page = 1;
      let pages = 1;
      do {
        const result = await listInspectArtifacts({ page, training_dataset_id: trainingDatasetId, preprocessing_pipeline_id: preprocessingPipelineId });
        all.push(...result.items);
        pages = result.pages;
        page += 1;
      } while (page <= pages);
      if (!cancelled) setMatchingArtifacts(all);
    })().catch(() => { if (!cancelled) setMatchingArtifacts([]); });
    return () => { cancelled = true; };
  }, [active, trainingDatasetId, preprocessingPipelineId, artifactItems]);

  useEffect(() => {
    if (!active) return;
    const hasActive = artifactActiveTotal > 0 || runs.some((run) => run.status === 'queued' || run.status === 'running');
    if (!hasActive) return;
    const timer = window.setInterval(() => {
      Promise.all([listInspectRuns().then(setRuns), refreshArtifacts()]).catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [active, runs, artifactActiveTotal, artifactPage, artifactModeFilter, artifactDatasetFilter, artifactPipelineFilter, artifactStatusFilter]);

  const selectedDataset = trainingDatasets.find((dataset) => dataset.id === trainingDatasetId) ?? null;
  const selectedPipeline = preprocessingPipelines.find((pipeline) => pipeline.id === preprocessingPipelineId) ?? null;
  const selectedRoi = rois.find((roi) => roi.id === selectedRoiId) ?? null;
  const minDate = toInputDateTime(selectedDataset?.start_timestamp ?? null);
  const maxDate = toInputDateTime(selectedDataset?.end_timestamp ?? null);
  const analysisConfig = useMemo<Record<string, unknown>>(() => {
    if (analysisMode === 'energy') {
      return {
        energy_variant: energyVariant,
        aggregation: energyAggregation,
        window_size: energyWindowSize,
        window_aggregation: 'sum',
        normalize_by_pixels: energyNormalize,
      };
    }
    if (analysisMode === 'optical_flow') {
      return {
        flow_method: 'farneback',
        aggregation: flowAggregation,
        normalize_by_pixels: flowNormalize,
        pyr_scale: flowPyrScale,
        levels: flowLevels,
        winsize: flowWinSize,
        iterations: flowIterations,
        poly_n: flowPolyN,
        poly_sigma: flowPolySigma,
      };
    }
    if (analysisMode === 'contrast_enhanced') {
      return {
        reference_frames: contrastReferenceFrames,
        shift: contrastShift,
        vmax: contrastVmax,
        ma_radius: contrastMaRadius,
      };
    }
    return {};
  }, [
    analysisMode,
    contrastMaRadius,
    contrastReferenceFrames,
    contrastShift,
    contrastVmax,
    energyAggregation,
    energyNormalize,
    energyVariant,
    energyWindowSize,
    flowAggregation,
    flowIterations,
    flowLevels,
    flowNormalize,
    flowPolyN,
    flowPolySigma,
    flowPyrScale,
    flowWinSize,
  ]);
  const currentSignature = selectionSignature({
    trainingDatasetId,
    preprocessingPipelineId,
    start,
    end,
    stride,
    analysisMode,
    analysisConfig,
    roiId: roiEnabled ? selectedRoiId : null,
    generateVideo,
    contrastEnabled: analysisMode === 'contrast_enhanced',
    contrastReferenceFrames,
    contrastShift,
    contrastVmax,
    contrastMaRadius,
  });
  const previewFresh = Boolean(preview && previewSignature === currentSignature);
  const invalidRange = Boolean(start && end && end < start);
  const canPreview = Boolean(trainingDatasetId && preprocessingPipelineId && start && end && !invalidRange);
  const roiReadyForRun = !roiEnabled || selectedRoiId !== null;
  const canRun = canPreview && !runLoading && roiReadyForRun && (analysisMode !== 'contrast_enhanced' || contrastVmax > 0);

  function handleDatasetChange(value: string | null) {
    const id = value ? Number(value) : null;
    const dataset = trainingDatasets.find((item) => item.id === id) ?? null;
    setTrainingDatasetId(id);
    setStart(toInputDateTime(dataset?.start_timestamp ?? null));
    setEnd(toInputDateTime(dataset?.end_timestamp ?? null));
    setPreview(null);
    setPreviewSignature('');
  }

  function markPreviewStale() {
    setPreviewSignature('');
  }

  function handleAutoFitContrast() {
    if (
      !preview?.contrast_enabled ||
      preview.contrast_diff_min == null ||
      preview.contrast_diff_max == null
    ) {
      return;
    }
    const diffMin = preview.contrast_diff_min;
    const diffMax = preview.contrast_diff_max;
    setContrastShift(Math.round(-diffMin));
    setContrastVmax(Math.max(1, Math.round(diffMax - diffMin)));
    markPreviewStale();
  }

  async function handleSaveRoi() {
    if (!preview || !roiPointsDraft || !roiName.trim()) return;
    const rect = boundingRect(roiPointsDraft);
    setSavingRoi(true);
    try {
      const roi = await createRoi({
        name: roiName.trim(),
        description: 'Created from Inspect',
        image_width: preview.width,
        image_height: preview.height,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        geometry_type: 'polygon',
        points: roiPointsDraft,
        tile_rows: tileRows,
        tile_cols: tileCols,
      });
      const nextRois = await listRois();
      setRois(nextRois);
      setSelectedRoiId(roi.id);
      notifications.show({ color: 'green', title: 'ROI saved', message: roi.name });
      markPreviewStale();
    } catch (error) {
      notifications.show({ color: 'red', title: 'ROI save failed', message: error instanceof Error ? error.message : 'Unknown error' });
    } finally {
      setSavingRoi(false);
    }
  }

  async function handlePreview() {
    if (!canPreview || trainingDatasetId == null || preprocessingPipelineId == null) return;
    setPreviewLoading(true);
    try {
      const result = await previewInspect({
        training_dataset_id: trainingDatasetId,
        preprocessing_pipeline_id: preprocessingPipelineId,
        start_timestamp: start,
        end_timestamp: end,
        stride,
        analysis_mode: analysisMode,
        analysis_config: analysisConfig,
        roi_id: roiEnabled ? selectedRoiId : null,
        generate_video: generateVideo,
        fps,
        contrast_enabled: analysisMode === 'contrast_enhanced',
        contrast_reference_frames: contrastReferenceFrames,
        contrast_shift: contrastShift,
        contrast_vmax: contrastVmax,
        contrast_ma_radius: contrastMaRadius,
      });
      setPreview(result);
      if (roiEnabled && !selectedRoiId && !roiPointsDraft) {
        setRoiPointsDraft(defaultRoiPoints(result));
      }
      setPreviewSignature(currentSignature);
    } catch (error) {
      setPreview(null);
      setPreviewSignature('');
      notifications.show({
        color: 'red',
        title: 'Inspect preview failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setPreviewLoading(false);
    }
  }

  async function handleRun() {
    if (!canRun || trainingDatasetId == null || preprocessingPipelineId == null) return;
    setRunLoading(true);
    try {
      const created = await createInspectRun({
        training_dataset_id: trainingDatasetId,
        preprocessing_pipeline_id: preprocessingPipelineId,
        start_timestamp: start,
        end_timestamp: end,
        stride,
        fps,
        analysis_mode: analysisMode,
        analysis_config: analysisConfig,
        roi_id: roiEnabled ? selectedRoiId : null,
        generate_video: generateVideo,
        contrast_enabled: analysisMode === 'contrast_enhanced',
        contrast_reference_frames: contrastReferenceFrames,
        contrast_shift: contrastShift,
        contrast_vmax: contrastVmax,
        contrast_ma_radius: contrastMaRadius,
      });
      setRuns((current) => [created, ...current.filter((run) => run.id !== created.id)]);
      await refreshArtifacts();
      notifications.show({ color: 'green', title: 'Inspect run queued', message: created.training_dataset_name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Inspect run failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setRunLoading(false);
    }
  }

  async function handleArtifactAbort(run: InspectArtifactRun) {
    await rowActions.runPending(`artifact-abort:${run.kind}:${run.id}`, async () => {
      if (run.kind === 'heatmap') await abortHeatmapRange(run.id);
      else await abortInspectRun(run.id);
      await refreshArtifacts();
    }).catch((error) => notifications.show({ color: 'red', title: 'Abort failed', message: error instanceof Error ? error.message : 'Unknown error' }));
  }

  async function handleArtifactDelete(run: InspectArtifactRun) {
    if (!window.confirm(`Delete ${run.kind} run ${run.id}?`)) return;
    await rowActions.runPending(`artifact-delete:${run.kind}:${run.id}`, async () => {
      if (run.kind === 'heatmap') await deleteHeatmapRange(run.id);
      else await deleteInspectRun(run.id);
      if (selectedArtifact?.kind === run.kind && selectedArtifact.id === run.id) setSelectedArtifact(null);
      await refreshArtifacts();
    }).catch((error) => notifications.show({ color: 'red', title: 'Delete failed', message: error instanceof Error ? error.message : 'Unknown error' }));
  }

  return (
    <Stack gap="lg">
      <div>
        <Title order={2}>Inspect</Title>
        <Text c="dimmed" size="sm">
          Render selected Train/Test dataset images through a preprocessing pipeline into a playable inspection video.
        </Text>
      </div>

      <StepCard title="Inspect source" color="blue">
        <Stack gap="md">
          <Group grow align="flex-end">
            <Select
              label="Train/Test Dataset"
              placeholder="Select dataset rules"
              searchable
              data={trainingDatasets.map((dataset) => ({
                value: String(dataset.id),
                label: `${dataset.name} (${dataset.total_selected_images} images)`,
              }))}
              value={trainingDatasetId == null ? null : String(trainingDatasetId)}
              onChange={handleDatasetChange}
            />
            <Select
              label="Preprocessing Pipeline"
              placeholder="Select preprocessing"
              searchable
              data={preprocessingPipelines.map((pipeline) => ({
                value: String(pipeline.id),
                label: `${pipeline.name} (${pipeline.output_width ?? '?'}x${pipeline.output_height ?? '?'})`,
              }))}
              value={preprocessingPipelineId == null ? null : String(preprocessingPipelineId)}
              onChange={(value) => {
                setPreprocessingPipelineId(value ? Number(value) : null);
                markPreviewStale();
              }}
            />
          </Group>

          {trainingDatasetId != null && preprocessingPipelineId != null && (
            <Paper withBorder p="sm" radius="sm">
              <Stack gap="xs">
                <Group justify="space-between">
                  <Text fw={600} size="sm">Available artifacts for this combination</Text>
                  <Badge variant="light">{matchingArtifacts.length}</Badge>
                </Group>
                {matchingArtifacts.length === 0 ? (
                  <Text size="sm" c="dimmed">No Inspect or heatmap artifacts yet.</Text>
                ) : (
                  <Group gap="xs">
                    {matchingArtifacts.map((artifact) => (
                      <Button
                        key={`${artifact.kind}:${artifact.id}`}
                        size="compact-sm"
                        variant="light"
                        disabled={artifact.status !== 'finished' || (!artifact.has_video && !artifact.has_csv)}
                        onClick={() => setSelectedArtifact(artifact)}
                      >
                        {artifact.mode.replaceAll('_', ' ')} · #{artifact.id} · {artifact.status}
                      </Button>
                    ))}
                  </Group>
                )}
              </Stack>
            </Paper>
          )}

          {selectedDataset && (
            <Alert color="blue" title="Dataset bounds">
              {selectedDataset.name}: {formatTimestamp(selectedDataset.start_timestamp)} to{' '}
              {formatTimestamp(selectedDataset.end_timestamp)}
            </Alert>
          )}

          <Group grow align="flex-start">
            <DateTime24Input
              label="Start"
              value={start}
              min={minDate}
              max={maxDate}
              error={invalidRange ? 'Invalid range' : undefined}
              onChange={(value) => {
                setStart(value);
                markPreviewStale();
              }}
            />
            <DateTime24Input
              label="End"
              value={end}
              min={minDate}
              max={maxDate}
              error={invalidRange ? 'Invalid range' : undefined}
              onChange={(value) => {
                setEnd(value);
                markPreviewStale();
              }}
            />
            <NumberInput
              label="Inspect stride"
              min={1}
              value={stride}
              onChange={(value) => {
                setStride(Math.max(1, Number(value) || 1));
                markPreviewStale();
              }}
            />
            <NumberInput
              label="Video fps"
              min={1}
              max={60}
              value={fps}
              onChange={(value) => setFps(Math.max(1, Math.min(60, Number(value) || 1)))}
            />
          </Group>

          <Paper withBorder p="md" radius="sm">
            <Stack gap="md">
              <Select
                label={<InfoLabel label="Inspect method" info="Choose the diagnostic generated from the selected dataset and preprocessing pipeline. Energy and optical flow primarily produce CSV/plot outputs; video is optional." />}
                data={[
                  { value: 'preprocessed_video', label: 'Preprocessed video' },
                  { value: 'contrast_enhanced', label: 'Contrast enhance' },
                  { value: 'energy', label: 'Energy' },
                  { value: 'optical_flow', label: 'Optical flow' },
                ]}
                value={analysisMode}
                onChange={(value) => {
                  const next = (value ?? 'preprocessed_video') as InspectAnalysisMode;
                  setAnalysisMode(next);
                  setContrastEnabled(next === 'contrast_enhanced');
                  setGenerateVideo(next === 'preprocessed_video' || next === 'contrast_enhanced');
                  markPreviewStale();
                }}
              />

              {(analysisMode === 'energy' || analysisMode === 'optical_flow') && (
                <Switch
                  label={<InfoLabel label="Generate video" info="Off stores CSV, summary JSON and plot preview only. On additionally writes an MP4 overlay/flow video." />}
                  checked={generateVideo}
                  onChange={(event) => {
                    setGenerateVideo(event.currentTarget.checked);
                    markPreviewStale();
                  }}
                />
              )}

              {analysisMode === 'contrast_enhanced' && (
                <>
                  <Group grow align="flex-start">
                    <NumberInput label={<InfoLabel label="Reference frames" info="First N frames averaged into the mean reference that is subtracted from each frame." />} min={1} value={contrastReferenceFrames} onChange={(value) => { setContrastReferenceFrames(Math.max(1, Number(value) || 1)); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Shift" info="Value added after frame minus reference. Useful to move negative differences into visible range." />} value={contrastShift} onChange={(value) => { setContrastShift(Number(value) || 0); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="vmax" info="Upper clip value mapped to white. Must be greater than zero." />} min={1} value={contrastVmax} error={contrastVmax > 0 ? undefined : 'Must be > 0'} onChange={(value) => { setContrastVmax(Math.max(1, Number(value) || 1)); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Moving average radius" info="Centered temporal smoothing radius. 0 disables smoothing." />} min={0} value={contrastMaRadius} onChange={(value) => { setContrastMaRadius(Math.max(0, Number(value) || 0)); markPreviewStale(); }} />
                  </Group>
                  {preview?.contrast_enabled && preview.contrast_diff_min != null && (
                    <Alert color="grape" title="Preview diff range">
                      <Text size="sm">
                        Frame − reference spans {Math.round(preview.contrast_diff_min)} to {Math.round(preview.contrast_diff_max ?? 0)}.
                      </Text>
                      <Button mt="xs" size="compact-sm" variant="light" color="grape" onClick={handleAutoFitContrast}>
                        Auto-fit shift &amp; vmax
                      </Button>
                    </Alert>
                  )}
                </>
              )}

              {analysisMode === 'energy' && (
                <Group grow align="flex-start">
                  <Select label={<InfoLabel label="Energy variant" info="Pairwise compares each consecutive frame pair. Window aggregates pairwise differences over a causal time window." />} data={[{ value: 'pairwise', label: 'Pairwise frame energy' }, { value: 'window', label: 'Window activity energy' }]} value={energyVariant} onChange={(value) => { setEnergyVariant((value ?? 'pairwise') as 'pairwise' | 'window'); markPreviewStale(); }} />
                  <Select label={<InfoLabel label="Aggregation" info="Sum measures total changed intensity, mean normalizes by area, p95 focuses on strong local changes." />} data={[{ value: 'sum', label: 'Sum' }, { value: 'mean', label: 'Mean' }, { value: 'p95', label: 'P95' }]} value={energyAggregation} onChange={(value) => { setEnergyAggregation((value ?? 'sum') as 'sum' | 'mean' | 'p95'); markPreviewStale(); }} />
                  <NumberInput label={<InfoLabel label="Window size" info="Number of consecutive pairwise energies aggregated when variant is Window." />} min={1} disabled={energyVariant !== 'window'} value={energyWindowSize} onChange={(value) => { setEnergyWindowSize(Math.max(1, Number(value) || 1)); markPreviewStale(); }} />
                  <Switch label={<InfoLabel label="Normalize by pixels" info="For sum aggregation, divide by pixel count so tiles of different size are comparable." />} checked={energyNormalize} onChange={(event) => { setEnergyNormalize(event.currentTarget.checked); markPreviewStale(); }} />
                </Group>
              )}

              {analysisMode === 'optical_flow' && (
                <Stack gap="sm">
                  <Group grow align="flex-start">
                    <Select label={<InfoLabel label="Flow aggregation" info="How motion magnitude is reduced to one score per frame or tile." />} data={[{ value: 'mean_magnitude', label: 'Mean magnitude' }, { value: 'p95_magnitude', label: 'P95 magnitude' }, { value: 'max_magnitude', label: 'Max magnitude' }]} value={flowAggregation} onChange={(value) => { setFlowAggregation((value ?? 'mean_magnitude') as 'mean_magnitude' | 'p95_magnitude' | 'max_magnitude'); markPreviewStale(); }} />
                    <Switch label={<InfoLabel label="Normalize by pixels" info="Keeps scores comparable between full image and ROI/tile masks." />} checked={flowNormalize} onChange={(event) => { setFlowNormalize(event.currentTarget.checked); markPreviewStale(); }} />
                  </Group>
                  <Group grow align="flex-start">
                    <NumberInput label={<InfoLabel label="Pyramid scale" info="Farneback scale between pyramid levels. 0.5 is the common OpenCV default." />} min={0.1} max={0.99} step={0.05} value={flowPyrScale} onChange={(value) => { setFlowPyrScale(Number(value) || 0.5); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Levels" info="Number of pyramid levels used by Farneback optical flow." />} min={1} value={flowLevels} onChange={(value) => { setFlowLevels(Math.max(1, Number(value) || 1)); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Window size" info="Averaging window size for Farneback; larger windows smooth motion more." />} min={3} value={flowWinSize} onChange={(value) => { setFlowWinSize(Math.max(3, Number(value) || 15)); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Iterations" info="Solver iterations at each pyramid level." />} min={1} value={flowIterations} onChange={(value) => { setFlowIterations(Math.max(1, Number(value) || 1)); markPreviewStale(); }} />
                  </Group>
                  <Group grow align="flex-start">
                    <NumberInput label={<InfoLabel label="Poly N" info="Pixel neighborhood size for polynomial expansion. Typical values are 5 or 7." />} min={5} value={flowPolyN} onChange={(value) => { setFlowPolyN(Math.max(5, Number(value) || 5)); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Poly sigma" info="Gaussian sigma for polynomial expansion. 1.1–1.5 is typical." />} min={0.1} step={0.1} value={flowPolySigma} onChange={(value) => { setFlowPolySigma(Number(value) || 1.2); markPreviewStale(); }} />
                  </Group>
                </Stack>
              )}

              {(analysisMode === 'energy' || analysisMode === 'optical_flow') && (
                <Paper withBorder p="sm" radius="sm">
                  <Stack gap="sm">
                    <Switch label={<InfoLabel label="Use ROI / tiles" info="Restrict diagnostics to a reusable quadrilateral ROI. If the ROI has tiles, a separate time series is calculated per tile." />} checked={roiEnabled} onChange={(event) => { setRoiEnabled(event.currentTarget.checked); markPreviewStale(); }} />
                    {roiEnabled && (
                      <>
                        <Group grow align="flex-end">
                          <Select
                            label="Saved ROI"
                            placeholder="Choose saved ROI or define a new one"
                            data={rois.map((roi) => ({ value: String(roi.id), label: roiLabel(roi) }))}
                            value={selectedRoiId == null ? null : String(selectedRoiId)}
                            clearable
                            searchable
                            onChange={(value) => {
                              const roi = value ? rois.find((item) => item.id === Number(value)) ?? null : null;
                              setSelectedRoiId(roi?.id ?? null);
                              if (roi) {
                                setTileRows(roi.tile_rows ?? 1);
                                setTileCols(roi.tile_cols ?? 1);
                                setRoiPointsDraft(roiPoints(roi));
                              }
                              markPreviewStale();
                            }}
                          />
                          <TextInput label="New ROI name" placeholder="Optional new ROI" value={roiName} onChange={(event) => setRoiName(event.currentTarget.value)} />
                          <Button leftSection={<Save size={16} />} variant="light" loading={savingRoi} disabled={!preview || !roiPointsDraft || !roiName.trim()} onClick={handleSaveRoi}>
                            Save ROI
                          </Button>
                        </Group>
                        <Group grow>
                          <Select label="Tile rows" data={TILE_OPTIONS} value={String(tileRows)} onChange={(value) => { setTileRows(Number(value ?? 1)); markPreviewStale(); }} />
                          <Select label="Tile columns" data={TILE_OPTIONS} value={String(tileCols)} onChange={(value) => { setTileCols(Number(value ?? 1)); markPreviewStale(); }} />
                        </Group>
                        {preview && (
                          <PolygonRoiPicker
                            preview={preview}
                            points={roiPointsDraft ?? defaultRoiPoints(preview)}
                            tileRows={tileRows}
                            tileCols={tileCols}
                            onChange={(points) => {
                              setRoiPointsDraft(points);
                              setSelectedRoiId(null);
                              markPreviewStale();
                            }}
                          />
                        )}
                        {selectedRoi && preview && (selectedRoi.image_width !== preview.width || selectedRoi.image_height !== preview.height) && (
                          <Alert color="red">
                            Selected ROI is tuned for {selectedRoi.image_width}x{selectedRoi.image_height}, but preview output is {preview.width}x{preview.height}.
                          </Alert>
                        )}
                        {roiEnabled && roiPointsDraft && selectedRoiId === null && (
                          <Alert color="yellow">
                            Save the drawn ROI before running so Inspect can reuse the global ROI definition.
                          </Alert>
                        )}
                      </>
                    )}
                  </Stack>
                </Paper>
              )}
            </Stack>
          </Paper>

          <Group justify="flex-end">
            <Button
              leftSection={<Eye size={18} />}
              variant="light"
              loading={previewLoading}
              disabled={!canPreview || previewLoading}
              onClick={handlePreview}
            >
              Load preview
            </Button>
            <Button
              leftSection={<FileVideo size={18} />}
              loading={runLoading}
              disabled={!canRun}
              onClick={handleRun}
            >
              Run
            </Button>
          </Group>

          {preview && (
            <Stack gap="md">
              <Paper withBorder p="md" radius="sm">
                <Stack gap="sm">
                  <Group gap="xs">
                    <Text fw={700}>Preview MP4</Text>
                    <Badge variant="light" color={previewFresh ? 'green' : 'yellow'}>{previewFresh ? 'Preview current' : 'Preview stale'}</Badge>
                    <Badge variant="light">{preview.selected_images} selected images</Badge>
                    <Badge variant="light" color="gray">{preview.preview_frame_count} preview frames</Badge>
                    <Badge variant="light" color="gray">{preview.width}x{preview.height}, {preview.channels} ch</Badge>
                  </Group>
                  {preview.preview_video_url && <video src={inspectPreviewVideoUrl(preview.preview_video_url)} controls muted playsInline style={{ display: 'block', maxWidth: '100%', maxHeight: 'min(70vh, 680px)', margin: '0 auto', borderRadius: 6 }} />}
                </Stack>
              </Paper>
              {preview.plot_image_data_url && (
                <Paper withBorder p="md" radius="sm">
                  <Stack gap="sm">
                    <Group gap="xs">
                      <Text fw={700}>Diagnostic preview</Text>
                      <Badge variant="light">{preview.diagnostic_series.length} samples</Badge>
                    </Group>
                    <img src={preview.plot_image_data_url} alt="Inspect diagnostic plot preview" style={{ maxWidth: '100%', borderRadius: 6 }} />
                  </Stack>
                </Paper>
              )}
            </Stack>
          )}
        </Stack>
      </StepCard>

      <ArtifactViewer artifact={selectedArtifact} />

      <StepCard
        title={`Inspect runs (${artifactTotal})`}
        color="cyan"
        action={
          <Group gap="xs">
            <Button size="compact-sm" variant="subtle" leftSection={<RefreshCw size={14} />} onClick={() => refreshArtifacts()}>Refresh</Button>
            <ActionIcon variant="subtle" aria-label={artifactsOpen ? 'Collapse Inspect runs' : 'Expand Inspect runs'} onClick={() => setArtifactsOpen((current) => !current)}>
              {artifactsOpen ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
            </ActionIcon>
          </Group>
        }
      >
        <Collapse in={artifactsOpen}>
          <Stack gap="md">
            <Group grow align="flex-end">
              <Select label="Mode" clearable data={[
                { value: 'preprocessed_video', label: 'Preprocessed video' },
                { value: 'contrast_enhanced', label: 'Contrast enhanced' },
                { value: 'energy', label: 'Energy' },
                { value: 'optical_flow', label: 'Optical flow' },
                { value: 'heatmap', label: 'Heatmap' },
              ]} value={artifactModeFilter} onChange={(value) => { setArtifactModeFilter(value); setArtifactPage(1); }} />
              <Select label="Train/Test Dataset" clearable searchable data={trainingDatasets.map((item) => ({ value: String(item.id), label: item.name }))} value={artifactDatasetFilter == null ? null : String(artifactDatasetFilter)} onChange={(value) => { setArtifactDatasetFilter(value ? Number(value) : null); setArtifactPage(1); }} />
              <Select label="Preprocessing" clearable searchable data={preprocessingPipelines.map((item) => ({ value: String(item.id), label: item.name }))} value={artifactPipelineFilter == null ? null : String(artifactPipelineFilter)} onChange={(value) => { setArtifactPipelineFilter(value ? Number(value) : null); setArtifactPage(1); }} />
              <Select label="Status" clearable data={['queued', 'running', 'finished', 'failed', 'aborted']} value={artifactStatusFilter} onChange={(value) => { setArtifactStatusFilter(value); setArtifactPage(1); }} />
            </Group>
            <ScrollArea>
              <Table striped verticalSpacing="sm" miw={1050}>
                <Table.Thead><Table.Tr>
                  <Table.Th>Status</Table.Th><Table.Th>Mode</Table.Th><Table.Th>Dataset</Table.Th><Table.Th>Preprocessing</Table.Th><Table.Th>Range</Table.Th><Table.Th>Progress</Table.Th><Table.Th>Artifacts</Table.Th><Table.Th />
                </Table.Tr></Table.Thead>
                <Table.Tbody>
                  {artifactItems.map((run) => {
                    const busy = run.status === 'queued' || run.status === 'running';
                    return <Table.Tr key={`${run.kind}:${run.id}`}>
                      <Table.Td><Badge color={statusColor(run.status)} variant="light">{run.status}</Badge>{run.error_message && <Text size="xs" c="red">{run.error_message}</Text>}</Table.Td>
                      <Table.Td><Badge variant="light" color={run.kind === 'heatmap' ? 'red' : 'cyan'}>{run.mode.replaceAll('_', ' ')}</Badge></Table.Td>
                      <Table.Td>{run.training_dataset_name}</Table.Td>
                      <Table.Td>{run.preprocessing_pipeline_name}</Table.Td>
                      <Table.Td><Text size="xs">{formatTimestamp(run.start_timestamp)}</Text><Text size="xs">{formatTimestamp(run.end_timestamp)}</Text></Table.Td>
                      <Table.Td><Text size="xs">{run.done_count}{run.frame_count ? ` / ${run.frame_count}` : ''}</Text>{run.frame_count ? <Progress value={(run.done_count / run.frame_count) * 100} size="xs" /> : null}</Table.Td>
                      <Table.Td><Group gap={4} wrap="nowrap">
                        {run.has_video && <Button size="compact-xs" variant="light" component="a" href={artifactVideoUrl(run)} download>MP4</Button>}
                        {run.has_csv && <Button size="compact-xs" variant="light" component="a" href={inspectRunCsvUrl(run.id)} download>CSV</Button>}
                        {run.status === 'finished' && !run.has_video && run.kind === 'heatmap' && <Text size="xs" c="orange">Re-render MP4 in Analysis</Text>}
                      </Group></Table.Td>
                      <Table.Td><Group gap={4} justify="flex-end" wrap="nowrap">
                        {run.status === 'finished' && (run.has_video || run.has_csv) && <Button size="compact-xs" onClick={() => setSelectedArtifact(run)}>Load</Button>}
                        {busy && <ActionIcon color="yellow" variant="subtle" loading={rowActions.isPending(`artifact-abort:${run.kind}:${run.id}`)} onClick={() => handleArtifactAbort(run)}><Square size={16} /></ActionIcon>}
                        <ActionIcon color="red" variant="subtle" disabled={busy} loading={rowActions.isPending(`artifact-delete:${run.kind}:${run.id}`)} onClick={() => handleArtifactDelete(run)}><Trash2 size={16} /></ActionIcon>
                      </Group></Table.Td>
                    </Table.Tr>;
                  })}
                </Table.Tbody>
              </Table>
            </ScrollArea>
            {artifactItems.length === 0 && <Alert color="blue">No runs match the combined filters.</Alert>}
            {artifactPages > 1 && <Group justify="center"><Pagination total={artifactPages} value={artifactPage} onChange={setArtifactPage} /></Group>}
          </Stack>
        </Collapse>
      </StepCard>
    </Stack>
  );
}
