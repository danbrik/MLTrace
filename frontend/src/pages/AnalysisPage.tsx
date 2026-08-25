import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Checkbox,
  Collapse,
  ColorInput,
  Group,
  Loader,
  MultiSelect,
  Modal,
  NumberInput,
  Paper,
  Progress,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Table,
  TagsInput,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Activity, ArrowDown, ArrowUp, Check, ChevronDown, ChevronRight, Download, Film, Flame, Image, Info, Pause, Pencil, Play, Plus, RefreshCw, RotateCcw, Save, Search, SlidersHorizontal, Trash2, Upload } from 'lucide-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type React from 'react';

import {
  abortHeatmapRange,
  calculateAnalysisImageComparison,
  calculateBaselineNormalization,
  createAnalysisLayout,
  createHeatmap,
  createHeatmapRange,
  deleteAnalysisLayout,
  getAnalysisLayout,
  getHeatmap,
  getHeatmapRange,
  getFullTestingRunPlotSeries,
  getTestingRunResults,
  heatmapRangeVideoUrl,
  listAnalysisLayouts,
  listHeatmapRanges,
  listHeatmaps,
  listMethodConfigurations,
  listPreprocessingPipelines,
  listTestingRuns,
  listTrainingDatasets,
  listTrainingPipelines,
  listTrainingRuns,
  updateAnalysisLayout,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart, type PlotlyChartClick, type PlotlyChartDoubleClick, type PlotlyChartSelection } from '../components/PlotlyChart';
import { StepCard } from '../components/StepCard';
import { DEFAULT_TABLE_PAGE_SIZE, TablePagination } from '../components/TablePagination';
import type { Data, Layout, PlotRelayoutEvent } from '../lib/plotly';
import { buildPlotExportTable } from '../lib/plotExport';
import { withLineGapPolicy } from '../lib/plotGaps';
import {
  medianPositiveTimeDelta,
  relayoutRange,
  useVisibleAutomaticYRanges,
  type TimeSeriesAxisRange,
  type TimeSeriesTraceValues,
} from '../lib/timeSeriesViewport';
import { formatValue } from '../methods/utils';
import { datasetResolutions, formatResolution, orderedGraphNodes, stepDetail } from '../training/graph';
import {
  countFacetValues,
  facetOption,
  matchingFacetGroupIds,
  type FacetFilterState,
  type FacetRecord,
} from '../testing/facetFilters';
import {
  aggregationKeyForRun,
  aggregationLabel,
  aggregationOrder,
  metricKeyForRun,
  metricLabel,
  metricOrder,
  normalizeAggregationKey,
  normalizeMetricKey,
  roiKeyForRun,
  roiLabelForRun,
} from '../testing/inferenceRunMetadata';
import type {
  AnalysisLayout,
  AnalysisImageComparisonResult,
  BaselineAnomalyEvent,
  BaselineAnalysisRegion,
  BaselineNormalizationResult,
  BaselineRegionStatistics,
  BaselineSeriesPoint,
  HeatmapRangeRun,
  HeatmapRun,
  HeatmapRunSummary,
  HeatmapVisualizationConfig,
  MethodConfiguration,
  PreprocessingPipeline,
  TestingRun,
  TestingRunResult,
  TestingRunResults,
  TrainingDataset,
  TrainingPipeline,
  TrainingRun,
} from '../types';

// Cap on result rows fetched per inference run; large runs are decimated
// server-side so the Analysis page stays responsive regardless of run size.
const ANALYSIS_MAX_POINTS = 8000;

type PlotType = 'timeseries' | 'heatmap';
type HeatmapMode = 'single' | 'range';
type AnalyticsDisplayMode = 'multi_panel';
type AnalyticsKind =
  | 'raw'
  | 'ewma'
  | 'derivative'
  | 'smoothed_derivative'
  | 'second_derivative'
  | 'rolling_slope'
  | 'rolling_median'
  | 'rolling_mad'
  | 'robust_z'
  | 'positive_exceedance'
  | 'rolling_area'
  | 'rolling_mean'
  | 'rolling_max'
  | 'drawdown'
  | 'positive_slope_count'
  | 'positive_slope_fraction'
  | 'rising_streak'
  | 'cusum'
  | 'page_hinkley'
  | 'evidence_score'
  | 'slope_height_ratio'
  | 'energy_ratio'
  | 'snr_db'
  | 'snr_ratio'
  | 'rolling_std'
  | 'rolling_cv'
  | 'time_since_onset'
  | 'state_machine';

type AnalyticsMethodConfig = {
  kind: AnalyticsKind;
  params: Record<string, number | string | boolean>;
};

type PlotDraft = {
  plotType: PlotType;
  testingRunId: string | null;
  title: string;
  subtitle: string;
  scoreSeries: string;
  start: string;
  end: string;
  sampling: number;
  movingAverage: number;
  timeseriesAnalytics: AnalyticsMethodConfig[];
  analyticsDisplayMode: AnalyticsDisplayMode;
  showIntermediateAnalyticsPanels: boolean;
  panelHeightPx: number;
  heatmapMode: HeatmapMode;
  timestamp: string | null;
  includeReference: boolean;
  staeHeatmapView: 'reconstruction' | 'prediction';
  predictionHorizon: number;
  heatmapConfig: HeatmapVisualizationConfig;
};

type AnalysisPlot = PlotDraft & {
  id: string;
  sources: PlotSourceConfig[];
  traces?: PlotTraceConfig[];
  derivedFromPlotId?: string;
  autoStartHeatmap?: boolean;
  heatmapFps?: number;
  heatmapScaleMode?: 'per_frame' | 'shared';
  heatmapDisplaySize?: number;
  baselineAnalysis?: BaselineAnalysisConfig;
  imageComparison?: AnalysisImageComparisonConfig;
  detailSubtitle?: string;
};

type ImageComparisonPoint = {
  resultId: number;
  timestamp: string;
};

type AnalysisImageComparisonConfig = {
  testingRunId: string;
  imageSource: 'input' | 'reconstruction';
  reference?: ImageComparisonPoint;
  comparisons: ImageComparisonPoint[];
  result?: AnalysisImageComparisonResult;
  stale?: boolean;
};

type BaselineAnalysisConfig = {
  baselineRegions: BaselineAnalysisRegion[];
  analysisRegions: BaselineAnalysisRegion[];
  selectedRunIds: string[];
  stageIndex: number;
  normalization: 'classic' | 'robust';
  thresholds: number[];
  persistenceSamples: number;
  result?: BaselineNormalizationResult;
  stale?: boolean;
};

type TimeSeriesHeatmapDraft = {
  sourcePlotId: string;
  mode: HeatmapMode;
  start: string;
  end: string;
  testingRunId: string;
  sampling: number;
  fps: number;
  scaleMode: 'per_frame' | 'shared';
  heatmapConfig: HeatmapVisualizationConfig;
  includeReference: boolean;
  displaySize: number;
  advancedOpen: boolean;
};

type HeatmapSelectionDefaults = {
  sampling: number;
  fps: number;
  scaleMode: 'per_frame' | 'shared';
  includeReference: boolean;
  displaySize: number;
  heatmapConfig: HeatmapVisualizationConfig;
};

const HEATMAP_SELECTION_DEFAULTS_KEY = 'mltrace.analysis.heatmap-selection-defaults.v1';
const DEFAULT_HEATMAP_DISPLAY_SIZE = 400;

type DetailModalState = {
  title: string;
  body: React.ReactNode;
} | null;

type PlotSourceConfig = {
  testingRunId: string;
  start: string;
  end: string;
  sampling: number;
  timestamp: string | null;
};

type PlotTraceConfig = PlotSourceConfig & {
  metric: string;
  aggregation?: string;
  modelLabel: string;
  legendLabel: string;
  color: string;
};

type PlotPreview = {
  title: string;
  subtitle: string;
  traces: PlotTraceConfig[];
  duplicateNotes: string[];
  plot: AnalysisPlot;
};

type AutomaticPlotMetadata = {
  title: string;
  summaryLine: string;
  detailLines: string[];
};

type EditingPlotState = {
  plot: AnalysisPlot;
  index: number;
} | null;

type CombinedResult = TestingRunResult & {
  testingRunId: number;
  testingRunName: string;
  heatmapTimestampOnly?: boolean;
};

function notifyError(title: string, error: unknown) {
  notifications.show({ color: 'red', title, message: error instanceof Error ? error.message : 'Unknown error' });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown error';
}

function valueAsNumber(value: string | number, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function defaultHeatmapConfig(): HeatmapVisualizationConfig {
  return {
    residual_source: 'pixel_residual',
    error_mode: 'squared',
    threshold_enabled: false,
    threshold: 0,
    max_clip_enabled: false,
    max_clip: 0.33,
    max_opacity: 0.55,
    fixed_ceiling_enabled: false,
    fixed_ceiling: 1,
    signed_deviations: false,
    positive_weight: 1,
    negative_weight: 1,
    ssim_window_size: 11,
    ssim_alpha: 1,
    ssim_beta: 1,
    ssim_gamma: 1,
    ssim_k1: 0.01,
    ssim_k2: 0.03,
    ssim_data_range: 1,
  };
}

function defaultHeatmapSelectionDefaults(): HeatmapSelectionDefaults {
  return {
    sampling: 1,
    fps: 8,
    scaleMode: 'per_frame',
    includeReference: false,
    displaySize: DEFAULT_HEATMAP_DISPLAY_SIZE,
    heatmapConfig: defaultHeatmapConfig(),
  };
}

function loadHeatmapSelectionDefaults(): HeatmapSelectionDefaults {
  const defaults = defaultHeatmapSelectionDefaults();
  try {
    const raw = window.localStorage.getItem(HEATMAP_SELECTION_DEFAULTS_KEY);
    if (!raw) return defaults;
    const saved = JSON.parse(raw) as Partial<HeatmapSelectionDefaults>;
    return {
      sampling: typeof saved.sampling === 'number' && saved.sampling >= 1 ? saved.sampling : defaults.sampling,
      fps: typeof saved.fps === 'number' && saved.fps >= 1 && saved.fps <= 60 ? saved.fps : defaults.fps,
      scaleMode: saved.scaleMode === 'shared' ? 'shared' : 'per_frame',
      includeReference: typeof saved.includeReference === 'boolean' ? saved.includeReference : defaults.includeReference,
      displaySize: typeof saved.displaySize === 'number' && saved.displaySize >= 240 && saved.displaySize <= 1200
        ? saved.displaySize
        : defaults.displaySize,
      heatmapConfig: {
        ...defaults.heatmapConfig,
        ...(saved.heatmapConfig && typeof saved.heatmapConfig === 'object' ? saved.heatmapConfig : {}),
      },
    };
  } catch {
    return defaults;
  }
}

function saveHeatmapSelectionDefaults(defaults: HeatmapSelectionDefaults) {
  window.localStorage.setItem(HEATMAP_SELECTION_DEFAULTS_KEY, JSON.stringify(defaults));
}

function heatmapConfigKey(config: HeatmapVisualizationConfig, staeView = 'reconstruction', predictionHorizon = 1): string {
  return [
    config.residual_source,
    config.error_mode,
    Number(config.threshold_enabled),
    config.threshold,
    Number(config.max_clip_enabled),
    config.max_clip,
    config.max_opacity,
    Number(config.fixed_ceiling_enabled),
    config.fixed_ceiling,
    Number(config.signed_deviations),
    config.positive_weight,
    config.negative_weight,
    config.ssim_window_size,
    config.ssim_alpha,
    config.ssim_beta,
    config.ssim_gamma,
    config.ssim_k1,
    config.ssim_k2,
    config.ssim_data_range,
    staeView,
    predictionHorizon,
  ].join(':');
}

function heatmapCacheKey(frame: CombinedResult, config: HeatmapVisualizationConfig, staeView = 'reconstruction', predictionHorizon = 1): string {
  const source = frame.heatmapTimestampOnly ? frame.timestamp : frame.id;
  return `${frame.testingRunId}:${source}:${heatmapConfigKey(config, staeView, predictionHorizon)}`;
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

function scoreValue(result: TestingRunResult, series = 'score'): number {
  const metadata = result.result_metadata ?? {};
  const fastAnogan = metadata.fast_anogan;
  if (typeof fastAnogan === 'object' && fastAnogan !== null) {
    const values = fastAnogan as { residual_score?: unknown; feature_score?: unknown; combined_score?: unknown };
    if (series === 'fast_residual' && typeof values.residual_score === 'number') return values.residual_score;
    if (series === 'fast_feature' && typeof values.feature_score === 'number') return values.feature_score;
    if (series === 'fast_combined' && typeof values.combined_score === 'number') return values.combined_score;
  }
  if (series === 'reconstruction') {
    const value = metadata.reconstruction_score;
    return typeof value === 'number' ? value : result.full_mse;
  }
  if (series === 'prediction') {
    const value = metadata.prediction_score;
    return typeof value === 'number' ? value : (result.roi_mse ?? result.score);
  }
  if (series.startsWith('future+')) {
    const horizon = Number(series.slice('future+'.length));
    const futureScores = Array.isArray(metadata.future_scores) ? metadata.future_scores : [];
    const match = futureScores.find((item) => typeof item === 'object' && item !== null && Number((item as { horizon?: unknown }).horizon) === horizon);
    const value = match && typeof (match as { score?: unknown }).score === 'number' ? (match as { score: number }).score : undefined;
    return value ?? result.score;
  }
  return result.score ?? result.roi_mse ?? result.full_mse;
}

function scoreSeriesOptions(results: CombinedResult[]) {
  const options = [
    { value: 'score', label: 'Combined / primary score' },
    { value: 'reconstruction', label: 'Reconstruction score' },
  ];
  if (results.some((result) => typeof result.result_metadata?.prediction_score === 'number')) {
    options.push({ value: 'prediction', label: 'Prediction score' });
  }
  if (results.some((result) => typeof result.result_metadata?.fast_anogan === 'object' && result.result_metadata?.fast_anogan !== null)) {
    options.push(
      { value: 'fast_combined', label: 'fastAnoGAN combined score' },
      { value: 'fast_residual', label: 'fastAnoGAN pixel residual' },
      { value: 'fast_feature', label: 'fastAnoGAN critic feature score' },
    );
  }
  const horizons = new Set<number>();
  for (const result of results) {
    const futureScores = result.result_metadata?.future_scores;
    if (!Array.isArray(futureScores)) continue;
    for (const item of futureScores) {
      if (typeof item === 'object' && item !== null) {
        const horizon = Number((item as { horizon?: unknown }).horizon);
        if (Number.isFinite(horizon)) horizons.add(horizon);
      }
    }
  }
  for (const horizon of [...horizons].sort((left, right) => left - right)) {
    options.push({ value: `future+${horizon}`, label: `Future +${horizon}` });
  }
  return options;
}

function resultLabel(result: TestingRunResult): string {
  return new Date(result.timestamp).toLocaleString();
}

function pad(value: number): string {
  return String(value).padStart(2, '0');
}

function toDateTimeLocal(value: string | null | undefined): string {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function sourceBounds(dataset: TrainingDataset | null | undefined): { start: string; end: string } {
  if (dataset?.start_timestamp && dataset?.end_timestamp) {
    return { start: toDateTimeLocal(dataset.start_timestamp), end: toDateTimeLocal(dataset.end_timestamp) };
  }
  const starts = (dataset?.rules ?? []).map((rule) => rule.start_timestamp).filter(Boolean).sort();
  const ends = (dataset?.rules ?? []).map((rule) => rule.end_timestamp).filter(Boolean).sort();
  return {
    start: toDateTimeLocal(starts[0]),
    end: toDateTimeLocal(ends.at(-1)),
  };
}

function traceToSource(trace: PlotTraceConfig): PlotSourceConfig {
  return {
    testingRunId: trace.testingRunId,
    start: trace.start,
    end: trace.end,
    sampling: trace.sampling,
    timestamp: trace.timestamp,
  };
}

function plotSources(plot: AnalysisPlot): PlotSourceConfig[] {
  if (plot.traces?.length) return plot.traces.map(traceToSource);
  if (plot.sources.length > 0) return plot.sources;
  return plot.testingRunId
    ? [{
        testingRunId: plot.testingRunId,
        start: plot.start,
        end: plot.end,
        sampling: plot.sampling,
        timestamp: plot.timestamp,
      }]
    : [];
}

function sourceTraceKey(source: PlotSourceConfig): string {
  return `${source.testingRunId}`;
}

function traceLabelForRun(
  run: TestingRun,
  metric: string,
  multipleMetrics: boolean,
  aggregation: string,
  multipleAggregations: boolean,
): string {
  const modelLabel = run.training_pipeline_name || run.training_run_name || `Training run #${run.training_run_id}`;
  const parts = [modelLabel];
  if (multipleMetrics) parts.push(metricLabel(metric));
  if (multipleAggregations) parts.push(aggregationLabel(aggregation));
  return parts.join(' · ');
}

function filterAndSampleResults(
  results: TestingRunResult[],
  start: string,
  end: string,
  sampling: number,
): TestingRunResult[] {
  const startMs = start ? new Date(start).getTime() : Number.NEGATIVE_INFINITY;
  const endMs = end ? new Date(end).getTime() : Number.POSITIVE_INFINITY;
  const step = Math.max(1, Math.floor(sampling));
  return results
    .filter((result) => {
      const timestamp = new Date(result.timestamp).getTime();
      return timestamp >= startMs && timestamp <= endMs;
    })
    .filter((_, index) => index % step === 0);
}

function movingAverage(values: number[], windowSize: number): number[] {
  const size = Math.max(1, Math.floor(windowSize));
  if (size <= 1) return values;
  return values.map((_, index) => {
    const start = Math.max(0, index - size + 1);
    const slice = values.slice(start, index + 1);
    return slice.reduce((sum, value) => sum + value, 0) / slice.length;
  });
}

function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a';
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(3);
  return value.toFixed(5);
}

function DetailButton({ title, body, onOpen }: { title: string; body: React.ReactNode; onOpen: (detail: DetailModalState) => void }) {
  return (
    <Tooltip label={`Inspect ${title.toLowerCase()}`}>
      <ActionIcon size="sm" variant="subtle" onClick={() => onOpen({ title, body })}>
        <Info size={15} />
      </ActionIcon>
    </Tooltip>
  );
}

function DetailModal({ detail, onClose }: { detail: DetailModalState; onClose: () => void }) {
  return (
    <Modal opened={detail !== null} onClose={onClose} title={detail?.title ?? ''} size="xl">
      <Paper withBorder p="sm" radius="sm">
        <ScrollArea h={460}>
          {detail?.body}
        </ScrollArea>
      </Paper>
    </Modal>
  );
}

const USAGE_LABELS: Record<string, string> = { train: 'Train', test: 'Test', validation: 'Validation', mixed: 'Mixed' };

function usageLabel(value: string | undefined): string {
  const key = value ?? 'train';
  return USAGE_LABELS[key] ?? key;
}

function usageColor(value: string | undefined): string {
  if (value === 'test') return 'orange';
  if (value === 'validation') return 'violet';
  if (value === 'mixed') return 'gray';
  return 'teal';
}

function datasetStrides(dataset: TrainingDataset | null): string {
  if (!dataset) return '—';
  const strides = [...new Set(dataset.rules.map((rule) => rule.stride))].sort((a, b) => a - b);
  return strides.length > 0 ? strides.join(', ') : '—';
}

function renderTrainsetDetails(dataset: TrainingDataset | null) {
  if (!dataset) return <Alert color="yellow">Trainset details are not available.</Alert>;
  return (
    <Stack gap="md">
      <Group justify="space-between">
        <div>
          <Text fw={700}>{dataset.name}</Text>
          <Text size="sm" c="dimmed">
            Label {dataset.usage_label} · {dataset.counts_missing ? 'Counts need refresh' : `${dataset.total_selected_images} selected images`} · Sources {dataset.dataset_names.join(', ')}
          </Text>
        </div>
        <Badge variant="light">{dataset.image_resolutions.join(', ') || 'n/a'}</Badge>
      </Group>
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
    </Stack>
  );
}

function renderPreprocessingDetails(pipeline: PreprocessingPipeline | null) {
  if (!pipeline) return <Alert color="yellow">Preprocessing details are not available.</Alert>;
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

function renderMethodDetails(configuration: MethodConfiguration | null) {
  if (!configuration) return <Alert color="yellow">Method details are not available.</Alert>;
  return (
    <Stack gap="md">
      <div>
        <Text fw={700}>{configuration.name}</Text>
        {configuration.description && <Text size="sm" c="dimmed">{configuration.description}</Text>}
      </div>
      <Group gap={6}>
        {Object.entries(configuration.method_config ?? {}).map(([key, value]) => (
          <Badge key={key} size="sm" variant="light" color="gray">
            {key}={formatValue(value)}
          </Badge>
        ))}
      </Group>
      {(['encoder', 'decoder'] as const).map((section) => {
        const layers = configuration.method_graph[section] ?? [];
        if (!Array.isArray(layers) || layers.length === 0) return null;
        return (
          <Stack key={section} gap="xs">
            <Text fw={700}>{section}</Text>
            {layers.map((layer, index) => (
              <Paper key={layer.id} withBorder p="sm" radius="sm">
                <Text size="sm" fw={600}>{index + 1}. {layer.type}</Text>
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
      })}
    </Stack>
  );
}

function renderPipelineDetails(
  pipeline: TrainingPipeline | null,
  trainsets: TrainingDataset[],
  preprocessing: PreprocessingPipeline | null,
  method: MethodConfiguration | null,
) {
  if (!pipeline) return <Alert color="yellow">Training pipeline details are not available.</Alert>;
  return (
    <Stack gap="md">
      <div>
        <Text fw={700}>{pipeline.name}</Text>
        {pipeline.description && <Text size="sm" c="dimmed">{pipeline.description}</Text>}
      </div>
      <Group>
        <Badge variant={pipeline.shuffle ? 'filled' : 'outline'} color="teal">{pipeline.shuffle ? 'shuffled' : 'in order'}</Badge>
        <Badge variant="light">{pipeline.total_selected_images} images</Badge>
      </Group>
      {renderTrainsetPipelineSummary(pipeline, trainsets)}
      <Title order={5}>Preprocessing</Title>
      {renderPreprocessingDetails(preprocessing)}
      <Title order={5}>Method</Title>
      {renderMethodDetails(method)}
    </Stack>
  );
}

function renderTrainsetPipelineSummary(pipeline: TrainingPipeline, datasets: TrainingDataset[]) {
  const byId = new Map(datasets.map((dataset) => [dataset.id, dataset]));
  return (
    <Stack gap="sm">
      <Title order={5}>Trainsets</Title>
      {pipeline.training_datasets.map((entry) => (
        <Paper key={entry.training_dataset_id} withBorder p="sm" radius="sm">
          {renderTrainsetDetails(byId.get(entry.training_dataset_id) ?? null)}
        </Paper>
      ))}
    </Stack>
  );
}

const TRACE_COLORS = ['#1c7ed6', '#e8590c', '#2f9e44', '#9c36b5', '#0c8599', '#e03131', '#5f3dc4', '#66a80f'];

type AnalyticsDefinition = {
  kind: AnalyticsKind;
  label: string;
  description: string;
  defaultParams: Record<string, number | string | boolean>;
};

const ANALYTICS_DEFINITIONS: AnalyticsDefinition[] = [
  { kind: 'raw', label: 'Raw score', description: 'Original anomaly score without additional transformation.', defaultParams: {} },
  { kind: 'ewma', label: 'EWMA', description: 'Causal exponential moving average.', defaultParams: { alpha: 0.2 } },
  { kind: 'derivative', label: 'First derivative', description: 'Point-to-point slope of raw or smoothed score.', defaultParams: { source: 'smoothed', alpha: 0.2, timeNormalized: false } },
  { kind: 'smoothed_derivative', label: 'Smoothed derivative', description: 'EWMA-smoothed first derivative.', defaultParams: { source: 'smoothed', alpha: 0.2, beta: 0.2, timeNormalized: false } },
  { kind: 'second_derivative', label: 'Second derivative', description: 'Change of the derivative.', defaultParams: { source: 'smoothed', alpha: 0.2, beta: 0.2, timeNormalized: false } },
  { kind: 'rolling_slope', label: 'Rolling slope', description: 'Causal slope over a past window.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, alpha: 0.2, timeNormalized: false } },
  { kind: 'rolling_median', label: 'Rolling median baseline', description: 'Causal rolling median baseline.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2 } },
  { kind: 'rolling_mad', label: 'Rolling MAD', description: 'Robust local spread around rolling median.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2 } },
  { kind: 'robust_z', label: 'Robust z-score', description: 'Score relative to rolling median and MAD.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, minimumScaleRelative: 1e-3, minimumScaleAbsolute: 1e-9 } },
  { kind: 'positive_exceedance', label: 'Positive exceedance', description: 'Positive part above a z-score threshold.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, threshold: 1, epsilon: 1e-12 } },
  { kind: 'rolling_area', label: 'Rolling area', description: 'Accumulated positive exceedance in a causal window.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, baselineWindowSamples: 60, baselineWindowMinutes: 60, alpha: 0.2, threshold: 1, epsilon: 1e-12 } },
  { kind: 'rolling_mean', label: 'Rolling mean', description: 'Causal local average of the smoothed score.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, alpha: 0.2 } },
  { kind: 'rolling_max', label: 'Rolling maximum', description: 'Causal local maximum.', defaultParams: { windowMode: 'samples', windowSamples: 20, windowMinutes: 5, alpha: 0.2 } },
  { kind: 'drawdown', label: 'Drawdown', description: 'Drop from causal rolling maximum.', defaultParams: { windowMode: 'samples', windowSamples: 20, windowMinutes: 5, alpha: 0.2, mode: 'relative', epsilon: 1e-12 } },
  { kind: 'positive_slope_count', label: 'Positive slope count', description: 'Number of positive slopes in the causal window.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, alpha: 0.2, slopeThreshold: 0 } },
  { kind: 'positive_slope_fraction', label: 'Positive slope fraction', description: 'Fraction of slopes above threshold in the causal window.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, alpha: 0.2, slopeThreshold: 0 } },
  { kind: 'rising_streak', label: 'Rising streak', description: 'Current consecutive count of positive slopes.', defaultParams: { alpha: 0.2, slopeThreshold: 0 } },
  { kind: 'cusum', label: 'CUSUM', description: 'Positive evidence accumulator on robust z-score.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, k: 1, h: 8, zCap: 20, minimumScaleRelative: 1e-3, minimumScaleAbsolute: 1e-9 } },
  { kind: 'page_hinkley', label: 'Page-Hinkley', description: 'Online mean-shift accumulator.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, delta: 0.2, lambda: 8, epsilon: 1e-12 } },
  { kind: 'evidence_score', label: 'Evidence score', description: 'Online positive/negative evidence score.', defaultParams: { windowMode: 'samples', windowSamples: 20, windowMinutes: 5, alpha: 0.2, zThreshold: 1, slopeThreshold: 0, w1: 1, w2: 1, w3: 0.2, v1: 1, v2: 0.5, v3: 1, epsilon: 1e-12 } },
  { kind: 'slope_height_ratio', label: 'Slope / height ratio', description: 'Current slope relative to robust z-score height.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, epsilon: 1e-12 } },
  { kind: 'energy_ratio', label: 'Short / long energy ratio', description: 'Short rolling area divided by long rolling area.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, longWindowSamples: 40, longWindowMinutes: 10, baselineWindowSamples: 60, baselineWindowMinutes: 60, alpha: 0.2, threshold: 1, epsilon: 1e-12 } },
  { kind: 'snr_db', label: 'Signal-to-noise ratio (dB)', description: 'Causal rolling signal-to-noise ratio in decibels. Signal is the absolute local mean; noise is local standard deviation.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, epsilon: 1e-12 } },
  { kind: 'snr_ratio', label: 'Signal-to-noise ratio (ratio)', description: 'Causal rolling signal-to-noise ratio as a unitless quotient. Signal is the absolute local mean; noise is local standard deviation.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, epsilon: 1e-12 } },
  { kind: 'rolling_std', label: 'Rolling std', description: 'Causal local standard deviation.', defaultParams: { windowMode: 'samples', windowSamples: 20, windowMinutes: 5, alpha: 0.2 } },
  { kind: 'rolling_cv', label: 'Rolling coefficient of variation', description: 'Rolling std divided by rolling mean.', defaultParams: { windowMode: 'samples', windowSamples: 20, windowMinutes: 5, alpha: 0.2, epsilon: 1e-12 } },
  { kind: 'time_since_onset', label: 'Time since onset', description: 'Elapsed time since z and slope crossed onset thresholds.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, onsetThreshold: 1, slopeThreshold: 0, resetThreshold: 0.5, epsilon: 1e-12 } },
  { kind: 'state_machine', label: 'State machine', description: 'Visual state band from z-score, slope and CUSUM thresholds.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, lowThreshold: 1, slopeThreshold: 0, hLow: 5, hHigh: 10, offThreshold: 0.5, k: 1, zCap: 20, minimumScaleRelative: 1e-3, minimumScaleAbsolute: 1e-9 } },
];

function analyticsDefinition(kind: AnalyticsKind): AnalyticsDefinition {
  return ANALYTICS_DEFINITIONS.find((definition) => definition.kind === kind) ?? ANALYTICS_DEFINITIONS[0];
}

function defaultAnalyticsConfig(kind: AnalyticsKind): AnalyticsMethodConfig {
  const definition = analyticsDefinition(kind);
  return { kind, params: { ...definition.defaultParams } };
}

function analyticsParamLabel(key: string): string {
  const labels: Record<string, string> = {
    alpha: 'Alpha',
    beta: 'Beta',
    windowMode: 'Window unit',
    windowSamples: 'Window samples',
    windowMinutes: 'Window minutes',
    baselineWindowSamples: 'Baseline samples',
    baselineWindowMinutes: 'Baseline minutes',
    longWindowSamples: 'Long window samples',
    longWindowMinutes: 'Long window minutes',
    threshold: 'Threshold',
    slopeThreshold: 'Slope threshold',
    onsetThreshold: 'Onset threshold',
    resetThreshold: 'Reset threshold',
    lowThreshold: 'Low threshold',
    offThreshold: 'Off threshold',
    zThreshold: 'Z threshold',
    epsilon: 'Epsilon',
    minimumScaleRelative: 'Relative scale floor',
    minimumScaleAbsolute: 'Absolute scale floor',
    zCap: 'CUSUM z cap',
    k: 'CUSUM k',
    h: 'CUSUM h',
    hLow: 'Low evidence threshold',
    hHigh: 'High evidence threshold',
    delta: 'Delta',
    lambda: 'Lambda',
    source: 'Signal basis',
    timeNormalized: 'Normalize by time',
    mode: 'Mode',
    w1: 'Positive z weight',
    w2: 'Positive slope weight',
    w3: 'Positive slope flag weight',
    v1: 'Negative slope weight',
    v2: 'Below-threshold weight',
    v3: 'Drawdown weight',
    snr_db: 'Signal-to-noise ratio (dB)',
    snr_ratio: 'Signal-to-noise ratio (ratio)',
  };
  return labels[key] ?? key;
}

function analyticsParamInfo(key: string): string {
  const infos: Record<string, string> = {
    alpha: 'EWMA smoothing factor. Lower values smooth more strongly and react later; higher values react faster but are noisier.',
    beta: 'EWMA smoothing factor for the derivative.',
    windowMode: 'Choose whether rolling windows are counted in samples or in elapsed minutes.',
    windowSamples: 'Number of past samples included in the causal rolling window.',
    windowMinutes: 'Past time span included in the causal rolling window.',
    baselineWindowSamples: 'Sample window used for robust median/MAD baseline estimation.',
    baselineWindowMinutes: 'Minute window used for robust median/MAD baseline estimation.',
    longWindowSamples: 'Long sample window for ratios such as short/long rolling energy.',
    longWindowMinutes: 'Long minute window for ratios such as short/long rolling energy.',
    threshold: 'Value above which positive evidence is counted.',
    slopeThreshold: 'Minimum slope considered a meaningful positive rise.',
    onsetThreshold: 'z-score threshold that starts an onset candidate.',
    resetThreshold: 'z-score level below which an onset candidate is reset.',
    lowThreshold: 'Lower z-score threshold for early state transitions.',
    offThreshold: 'Level below which the state machine can return to normal.',
    zThreshold: 'Robust z-score threshold used for positive evidence.',
    epsilon: 'Small value added to denominators for numerical stability.',
    minimumScaleRelative: 'Minimum robust scale as a fraction of the rolling median magnitude.',
    minimumScaleAbsolute: 'Absolute lower bound for the robust scale in score units.',
    zCap: 'Maximum robust z-score contribution added to CUSUM per sample.',
    k: 'CUSUM drift allowance. Larger values ignore more weak evidence.',
    h: 'CUSUM alarm threshold shown for interpretation.',
    hLow: 'Low CUSUM/evidence threshold for likely anomaly state.',
    hHigh: 'High CUSUM/evidence threshold for confirmed anomaly state.',
    delta: 'Page-Hinkley tolerance for mean-shift accumulation.',
    lambda: 'Page-Hinkley alarm threshold shown for interpretation.',
    source: 'Input signal used by this stage: raw previous output or EWMA-smoothed previous output.',
    timeNormalized: 'Divide changes by elapsed seconds, useful for irregular sampling.',
    mode: 'Relative drawdown normalizes by rolling maximum; absolute drawdown keeps score units.',
    w1: 'Weight of positive z-score evidence.',
    w2: 'Weight of positive slope magnitude evidence.',
    w3: 'Weight of positive slope indicator evidence.',
    v1: 'Weight of negative slope evidence.',
    v2: 'Penalty when z-score is below threshold.',
    v3: 'Penalty from relative drawdown.',
    snr_db: 'Causal rolling signal-to-noise ratio on a logarithmic dB scale. +20 dB means the local signal amplitude is 10x the local noise amplitude.',
    snr_ratio: 'Causal rolling signal-to-noise ratio as a unitless quotient: absolute local mean divided by local standard deviation.',
  };
  return infos[key] ?? 'Parameter used by this causal time-series analytics stage.';
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  return [...new Set(values.map((value) => value?.trim()).filter((value): value is string => Boolean(value)))];
}

function automaticValue(value: number | string | boolean): string {
  if (typeof value === 'boolean') return value ? 'on' : 'off';
  if (typeof value === 'number') return formatValue(value);
  return value;
}

function scoreSeriesLabel(value: string): string {
  const labels: Record<string, string> = {
    score: 'Combined / primary score',
    reconstruction: 'Reconstruction score',
    prediction: 'Prediction score',
    fast_combined: 'fastAnoGAN combined score',
    fast_residual: 'fastAnoGAN pixel residual',
    fast_feature: 'fastAnoGAN critic feature score',
  };
  if (value.startsWith('future+')) return `Future +${value.slice('future+'.length)}`;
  return labels[value] ?? value.replaceAll('_', ' ');
}

function analyticsDetailLines(configs: AnalyticsMethodConfig[]): string[] {
  if (configs.length === 0) return ['Analytics: none'];
  return configs.map((config, index) => {
    const params = Object.entries(config.params)
      .map(([key, value]) => `${analyticsParamLabel(key)}=${automaticValue(value)}`)
      .join(', ');
    return `Analytics ${index + 1}: ${analyticsDefinition(config.kind).label}${params ? ` · ${params}` : ''}`;
  });
}

function automaticPlotMetadata(plot: AnalysisPlot, testingRuns: TestingRun[]): AutomaticPlotMetadata {
  const sources = plotSources(plot);
  const runById = new Map(testingRuns.map((run) => [run.id, run]));
  const runs = sources.map((source) => runById.get(Number(source.testingRunId)) ?? null);
  const traces = plot.traces ?? [];

  const modelLabels = uniqueStrings(sources.map((source) => {
    const run = runById.get(Number(source.testingRunId));
    const trace = traces.find((item) => item.testingRunId === source.testingRunId);
    return run?.training_pipeline_name || run?.training_run_name || trace?.modelLabel || `Source run #${source.testingRunId}`;
  }));
  const datasetLabels = uniqueStrings(runs.map((run, index) => run?.training_dataset_name ?? `Inference dataset (source #${sources[index].testingRunId})`));
  const roiLabels = uniqueStrings(runs.map((run, index) => run
    ? (run.roi_name ?? (run.roi_id === null ? 'No ROI' : `ROI #${run.roi_id}`))
    : `ROI (source #${sources[index].testingRunId})`));
  const modelPart = modelLabels.length === 1 ? modelLabels[0] : modelLabels.length > 1 ? `${modelLabels.length} models` : 'Model unavailable';
  const datasetPart = datasetLabels.length === 1
    ? datasetLabels[0]
    : datasetLabels.length > 1 ? `${datasetLabels.length} inference datasets` : 'Inference dataset unavailable';
  const roiPart = roiLabels.length === 1 ? roiLabels[0] : roiLabels.length > 1 ? `${roiLabels.length} ROIs` : 'ROI unavailable';

  const metrics = uniqueStrings(traces.length > 0
    ? traces.map((trace) => metricLabel(trace.metric))
    : runs.map((run) => run ? metricLabel(metricKeyForRun(run)) : null));
  const aggregations = uniqueStrings(traces.length > 0
    ? traces.map((trace) => aggregationLabel(trace.aggregation ?? 'mean'))
    : runs.map((run) => run ? aggregationLabel(aggregationKeyForRun(run)) : null));
  const sourceNames = uniqueStrings(sources.map((source) => runById.get(Number(source.testingRunId))?.name ?? `Testing run #${source.testingRunId}`));
  const detailLines = [
    `Sources: ${sourceNames.join(', ') || 'unavailable'} · Metrics: ${metrics.join(', ') || 'unavailable'} · Aggregations: ${aggregations.join(', ') || 'unavailable'}`,
  ];

  if (plot.plotType === 'heatmap' && plot.heatmapMode === 'single') {
    detailLines.push(`Timestamp: ${artifactTimestamp(plot.timestamp ?? plot.start)} · Sampling: every 1 frame`);
  } else {
    detailLines.push(`Range: ${artifactTimestamp(plot.start)} – ${artifactTimestamp(plot.end)} · Sampling: every ${Math.max(1, plot.sampling)} frame${Math.max(1, plot.sampling) === 1 ? '' : 's'}`);
  }

  if (plot.plotType === 'timeseries') {
    detailLines.push(`Score series: ${scoreSeriesLabel(plot.scoreSeries)} · Moving average: ${Math.max(1, plot.movingAverage)} · Panel height: ${plot.panelHeightPx}px · Intermediate panels: ${plot.showIntermediateAnalyticsPanels ? 'shown' : 'hidden'}`);
    detailLines.push(...analyticsDetailLines(plot.timeseriesAnalytics));
  } else {
    const config = plot.heatmapConfig;
    const residual = config.residual_source === 'ssim_residual' ? 'SSIM residual' : 'Pixel residual';
    const error = config.residual_source === 'ssim_residual' ? '1 - SSIM' : `${config.error_mode} error`;
    detailLines.push(`Heatmap: ${plot.heatmapMode === 'single' ? 'single image' : 'range video'} · Residual: ${residual} · Error: ${error}`);
    detailLines.push(`Visualization: threshold ${config.threshold_enabled ? 'on' : 'off'} (${formatValue(config.threshold)}) · Max clip ${config.max_clip_enabled ? 'on' : 'off'} (${formatValue(config.max_clip * 100)}%) · Fixed ceiling ${config.fixed_ceiling_enabled ? 'on' : 'off'} (${formatValue(config.fixed_ceiling)}) · Max opacity ${formatValue(config.max_opacity * 100)}%`);
    detailLines.push(`Signed deviations: ${config.signed_deviations ? 'on' : 'off'} · Positive weight ${formatValue(config.positive_weight)} · Negative weight ${formatValue(config.negative_weight)}`);
    if (config.residual_source === 'ssim_residual') {
      detailLines.push(`SSIM: window ${config.ssim_window_size} · alpha ${formatValue(config.ssim_alpha)} · beta ${formatValue(config.ssim_beta)} · gamma ${formatValue(config.ssim_gamma)} · K1 ${formatValue(config.ssim_k1)} · K2 ${formatValue(config.ssim_k2)} · data range ${formatValue(config.ssim_data_range)}`);
    }
    detailLines.push(`Reference: ${plot.includeReference ? 'shown' : 'hidden'} · Image size: ${plot.heatmapDisplaySize ?? DEFAULT_HEATMAP_DISPLAY_SIZE}px`);
    if (plot.heatmapMode === 'range') {
      detailLines.push(`Video: ${plot.heatmapFps ?? 8} fps · Color scale: ${config.fixed_ceiling_enabled ? 'fixed ceiling' : plot.heatmapScaleMode === 'shared' ? 'shared' : 'per-frame'}`);
    }
    const stae = runs.some((run) => run?.method_type === 'spatiotemporal_autoencoder' || run?.method_family === 'spatiotemporal_autoencoder');
    if (stae || plot.staeHeatmapView === 'prediction') {
      detailLines.push(`STAE: ${plot.staeHeatmapView === 'prediction' ? 'future prediction' : 'reconstruction'} · Prediction horizon: ${plot.predictionHorizon}`);
    }
  }

  const summaryLine = plot.plotType === 'timeseries'
    ? `${scoreSeriesLabel(plot.scoreSeries)} · ${artifactTimestamp(plot.start)} – ${artifactTimestamp(plot.end)}`
    : plot.heatmapMode === 'single'
      ? `Single heatmap · ${artifactTimestamp(plot.timestamp ?? plot.start)}`
      : `Heatmap video · ${artifactTimestamp(plot.start)} – ${artifactTimestamp(plot.end)}`;
  return { title: `${modelPart} · ${datasetPart} · ${roiPart}`, summaryLine, detailLines };
}

function withAutomaticPlotMetadata(plot: AnalysisPlot, testingRuns: TestingRun[]): AnalysisPlot {
  const metadata = automaticPlotMetadata(plot, testingRuns);
  return { ...plot, title: metadata.title, subtitle: metadata.summaryLine, detailSubtitle: metadata.detailLines.join('\n') };
}

function withBaselineFreshness(plot: AnalysisPlot, testingRuns: TestingRun[]): AnalysisPlot {
  const analysis = plot.baselineAnalysis;
  if (!analysis?.result) return plot;
  const computedAt = new Date(analysis.result.computed_at).getTime();
  const stale = analysis.stale || analysis.result.traces.some((trace) => {
    const run = testingRuns.find((item) => item.id === trace.testing_run_id);
    if (!run || run.status !== 'finished') return true;
    const updatedAt = new Date(run.updated_at).getTime();
    return Number.isFinite(updatedAt) && Number.isFinite(computedAt) && updatedAt > computedAt;
  });
  return stale === analysis.stale ? plot : { ...plot, baselineAnalysis: { ...analysis, stale } };
}

function numberParam(config: AnalyticsMethodConfig, key: string, fallback: number): number {
  const value = config.params[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function stringParam(config: AnalyticsMethodConfig, key: string, fallback: string): string {
  const value = config.params[key];
  return typeof value === 'string' ? value : fallback;
}

function boolParam(config: AnalyticsMethodConfig, key: string, fallback: boolean): boolean {
  const value = config.params[key];
  return typeof value === 'boolean' ? value : fallback;
}

function finiteOrNull(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}

function ewma(values: number[], alpha: number): number[] {
  if (values.length === 0) return [];
  const boundedAlpha = Math.min(1, Math.max(0, alpha));
  const output: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    output.push(index === 0 ? values[index] : boundedAlpha * values[index] + (1 - boundedAlpha) * output[index - 1]);
  }
  return output;
}

function timeDeltaSeconds(times: number[], index: number): number {
  if (index <= 0) return 1;
  const delta = (times[index] - times[index - 1]) / 1000;
  return Number.isFinite(delta) && delta > 0 ? delta : 1;
}

function derivative(values: number[], times: number[], timeNormalized: boolean): number[] {
  return values.map((value, index) => {
    if (index === 0) return 0;
    const delta = value - values[index - 1];
    return timeNormalized ? delta / timeDeltaSeconds(times, index) : delta;
  });
}

function windowStartIndex(times: number[], index: number, config: AnalyticsMethodConfig, sampleKey = 'windowSamples', minuteKey = 'windowMinutes'): number {
  if (stringParam(config, 'windowMode', 'samples') === 'minutes') {
    const minutes = Math.max(0, numberParam(config, minuteKey, 3));
    const startMs = times[index] - minutes * 60_000;
    let start = index;
    while (start > 0 && times[start - 1] >= startMs) start -= 1;
    return start;
  }
  const samples = Math.max(1, Math.floor(numberParam(config, sampleKey, 12)));
  return Math.max(0, index - samples + 1);
}

function median(values: number[]): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((left, right) => left - right);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function rollingMap(values: number[], times: number[], config: AnalyticsMethodConfig, reducer: (slice: number[], index: number, start: number) => number, sampleKey?: string, minuteKey?: string): number[] {
  return values.map((_, index) => {
    const start = windowStartIndex(times, index, config, sampleKey, minuteKey);
    return reducer(values.slice(start, index + 1), index, start);
  });
}

function robustZ(values: number[], times: number[], config: AnalyticsMethodConfig): { z: number[]; baseline: number[]; mad: number[] } {
  const alpha = numberParam(config, 'alpha', 0.2);
  const minimumScaleRelative = Math.max(0, numberParam(config, 'minimumScaleRelative', 1e-3));
  const minimumScaleAbsolute = Math.max(0, numberParam(config, 'minimumScaleAbsolute', 1e-9));
  const smooth = ewma(values, alpha);
  const baseline = rollingMap(smooth, times, config, (slice) => median(slice));
  const mad = smooth.map((_, index) => {
    const start = windowStartIndex(times, index, config);
    const deviations = smooth.slice(start, index + 1).map((value) => Math.abs(value - baseline[index]));
    return median(deviations);
  });
  return {
    z: smooth.map((value, index) => {
      const scale = Math.max(1.4826 * mad[index], Math.abs(baseline[index]) * minimumScaleRelative, minimumScaleAbsolute);
      return (value - baseline[index]) / scale;
    }),
    baseline,
    mad,
  };
}

function positiveExceedance(values: number[], times: number[], config: AnalyticsMethodConfig): number[] {
  const z = robustZ(values, times, config).z;
  const threshold = numberParam(config, 'threshold', numberParam(config, 'zThreshold', 1));
  return z.map((value) => Math.max(0, value - threshold));
}

function rollingAreaFrom(values: number[], times: number[], config: AnalyticsMethodConfig, sampleKey = 'windowSamples', minuteKey = 'windowMinutes'): number[] {
  const exceedance = positiveExceedance(values, times, config);
  return exceedance.map((_, index) => {
    const start = windowStartIndex(times, index, config, sampleKey, minuteKey);
    return exceedance.slice(start, index + 1).reduce((sum, value) => sum + value, 0);
  });
}

function analyticsBase(values: number[], config: AnalyticsMethodConfig): number[] {
  return stringParam(config, 'source', 'smoothed') === 'raw' ? values : ewma(values, numberParam(config, 'alpha', 0.2));
}

function computeAnalyticsSeries(config: AnalyticsMethodConfig, values: number[], timestamps: string[]): Array<number | null> {
  const times = timestamps.map((timestamp) => new Date(timestamp).getTime());
  const alpha = numberParam(config, 'alpha', 0.2);
  const base = analyticsBase(values, config);
  const d = derivative(base, times, boolParam(config, 'timeNormalized', false));
  switch (config.kind) {
    case 'raw':
      return values.map(finiteOrNull);
    case 'ewma':
      return ewma(values, alpha).map(finiteOrNull);
    case 'derivative':
      return d.map(finiteOrNull);
    case 'smoothed_derivative':
      return ewma(d, numberParam(config, 'beta', 0.2)).map(finiteOrNull);
    case 'second_derivative':
      return derivative(ewma(d, numberParam(config, 'beta', 0.2)), times, boolParam(config, 'timeNormalized', false)).map(finiteOrNull);
    case 'rolling_slope':
      return base.map((value, index) => {
        const start = windowStartIndex(times, index, config);
        const delta = value - base[start];
        const denom = boolParam(config, 'timeNormalized', false) ? Math.max(1, (times[index] - times[start]) / 1000) : Math.max(1, index - start);
        return finiteOrNull(delta / denom);
      });
    case 'rolling_median':
      return rollingMap(ewma(values, alpha), times, config, (slice) => median(slice)).map(finiteOrNull);
    case 'rolling_mad': {
      const rz = robustZ(values, times, config);
      return rz.mad.map(finiteOrNull);
    }
    case 'robust_z':
      return robustZ(values, times, config).z.map(finiteOrNull);
    case 'positive_exceedance':
      return positiveExceedance(values, times, config).map(finiteOrNull);
    case 'rolling_area':
      return rollingAreaFrom(values, times, config).map(finiteOrNull);
    case 'rolling_mean':
      return rollingMap(ewma(values, alpha), times, config, (slice) => slice.reduce((sum, value) => sum + value, 0) / slice.length).map(finiteOrNull);
    case 'rolling_max':
      return rollingMap(ewma(values, alpha), times, config, (slice) => Math.max(...slice)).map(finiteOrNull);
    case 'drawdown': {
      const smooth = ewma(values, alpha);
      const maxes = rollingMap(smooth, times, config, (slice) => Math.max(...slice));
      const epsilon = numberParam(config, 'epsilon', 1e-12);
      return smooth.map((value, index) => {
        const absolute = maxes[index] - value;
        return finiteOrNull(stringParam(config, 'mode', 'relative') === 'relative' ? absolute / (maxes[index] + epsilon) : absolute);
      });
    }
    case 'positive_slope_count':
    case 'positive_slope_fraction': {
      const threshold = numberParam(config, 'slopeThreshold', 0);
      return d.map((_, index) => {
        const start = windowStartIndex(times, index, config);
        const window = d.slice(start, index + 1);
        const count = window.filter((value) => value > threshold).length;
        return finiteOrNull(config.kind === 'positive_slope_fraction' ? count / Math.max(1, window.length) : count);
      });
    }
    case 'rising_streak': {
      const threshold = numberParam(config, 'slopeThreshold', 0);
      let streak = 0;
      return d.map((value) => {
        streak = value > threshold ? streak + 1 : 0;
        return streak;
      });
    }
    case 'cusum': {
      const z = robustZ(values, times, config).z;
      const k = numberParam(config, 'k', 1);
      const zCap = numberParam(config, 'zCap', 20);
      let g = 0;
      return z.map((value) => {
        g = Math.max(0, g + Math.min(value, zCap) - k);
        return finiteOrNull(g);
      });
    }
    case 'page_hinkley': {
      const z = robustZ(values, times, config).z;
      const delta = numberParam(config, 'delta', 0.2);
      let mean = 0;
      let ph = 0;
      return z.map((value, index) => {
        mean += (value - mean) / (index + 1);
        ph = Math.max(0, ph + value - mean - delta);
        return finiteOrNull(ph);
      });
    }
    case 'evidence_score': {
      const z = robustZ(values, times, config).z;
      const smoothD = ewma(d, numberParam(config, 'beta', 0.2));
      const zThreshold = numberParam(config, 'zThreshold', 1);
      const slopeThreshold = numberParam(config, 'slopeThreshold', 0);
      const drawdownConfig = { ...config, kind: 'drawdown' as AnalyticsKind, params: { ...config.params, mode: 'relative' } };
      const drawdown = computeAnalyticsSeries(drawdownConfig, values, timestamps).map((value) => value ?? 0);
      let evidence = 0;
      return z.map((value, index) => {
        const positive = numberParam(config, 'w1', 1) * Math.max(0, value - zThreshold)
          + numberParam(config, 'w2', 1) * Math.max(0, smoothD[index] - slopeThreshold)
          + numberParam(config, 'w3', 0.2) * (smoothD[index] > slopeThreshold ? 1 : 0);
        const negative = numberParam(config, 'v1', 1) * Math.max(0, -smoothD[index])
          + numberParam(config, 'v2', 0.5) * (value < zThreshold ? 1 : 0)
          + numberParam(config, 'v3', 1) * drawdown[index];
        evidence = Math.max(0, evidence + positive - negative);
        return finiteOrNull(evidence);
      });
    }
    case 'slope_height_ratio': {
      const z = robustZ(values, times, config).z;
      const epsilon = numberParam(config, 'epsilon', 1e-12);
      return d.map((value, index) => finiteOrNull(value / (Math.abs(z[index]) + epsilon)));
    }
    case 'energy_ratio': {
      const shortArea = rollingAreaFrom(values, times, config);
      const longArea = rollingAreaFrom(values, times, config, 'longWindowSamples', 'longWindowMinutes');
      const epsilon = numberParam(config, 'epsilon', 1e-12);
      return shortArea.map((value, index) => finiteOrNull(value / (longArea[index] + epsilon)));
    }
    case 'snr_db':
    case 'snr_ratio': {
      const epsilon = numberParam(config, 'epsilon', 1e-12);
      return values.map((_, index) => {
        const start = windowStartIndex(times, index, config);
        const window = values.slice(start, index + 1).filter(Number.isFinite);
        if (window.length < 2) return null;
        const mean = window.reduce((sum, value) => sum + value, 0) / window.length;
        const variance = window.reduce((sum, value) => sum + (value - mean) ** 2, 0) / window.length;
        const ratio = Math.abs(mean) / (Math.sqrt(variance) + epsilon);
        return config.kind === 'snr_ratio' ? finiteOrNull(ratio) : finiteOrNull(20 * Math.log10(ratio));
      });
    }
    case 'rolling_std':
    case 'rolling_cv': {
      const smooth = ewma(values, alpha);
      const means = rollingMap(smooth, times, config, (slice) => slice.reduce((sum, value) => sum + value, 0) / slice.length);
      const stds = rollingMap(smooth, times, config, (slice) => {
        const mean = slice.reduce((sum, value) => sum + value, 0) / slice.length;
        return Math.sqrt(slice.reduce((sum, value) => sum + (value - mean) ** 2, 0) / slice.length);
      });
      const epsilon = numberParam(config, 'epsilon', 1e-12);
      return (config.kind === 'rolling_cv' ? stds.map((value, index) => value / (means[index] + epsilon)) : stds).map(finiteOrNull);
    }
    case 'time_since_onset': {
      const z = robustZ(values, times, config).z;
      const onsetThreshold = numberParam(config, 'onsetThreshold', 1);
      const slopeThreshold = numberParam(config, 'slopeThreshold', 0);
      const resetThreshold = numberParam(config, 'resetThreshold', 0.5);
      let onsetTime: number | null = null;
      return z.map((value, index) => {
        if (onsetTime === null && value > onsetThreshold && d[index] > slopeThreshold) onsetTime = times[index];
        if (onsetTime !== null && value < resetThreshold) onsetTime = null;
        return onsetTime === null ? 0 : finiteOrNull((times[index] - onsetTime) / 1000);
      });
    }
    case 'state_machine': {
      const z = robustZ(values, times, config).z;
      const low = numberParam(config, 'lowThreshold', 1);
      const slope = numberParam(config, 'slopeThreshold', 0);
      const hLow = numberParam(config, 'hLow', 5);
      const hHigh = numberParam(config, 'hHigh', 10);
      const off = numberParam(config, 'offThreshold', 0.5);
      const k = numberParam(config, 'k', 1);
      const zCap = numberParam(config, 'zCap', 20);
      let cusum = 0;
      let state = 0;
      return z.map((value, index) => {
        cusum = Math.max(0, cusum + Math.min(value, zCap) - k);
        if (state > 0 && value < off) {
          state = 0;
          cusum = 0;
        }
        else if (cusum >= hHigh) state = 3;
        else if (cusum >= hLow) state = 2;
        else if (value > low && d[index] > slope) state = 1;
        return state;
      });
    }
    default:
      return values.map(finiteOrNull);
  }
}

function TimeSeriesPlot({
  plot,
  results,
  selectionActive = false,
  onPointClick,
  onRangeSelected,
}: {
  plot: AnalysisPlot;
  results: CombinedResult[];
  selectionActive?: boolean;
  onPointClick?: (event: PlotlyChartClick) => void;
  onRangeSelected?: (event: PlotlyChartSelection) => void;
}) {
  const analyticsConfigs = plot.timeseriesAnalytics ?? [];
  const [manualYRanges, setManualYRanges] = useState<Record<string, TimeSeriesAxisRange>>({});
  const displayPanels = useMemo(() => {
    if (analyticsConfigs.length === 0) return [defaultAnalyticsConfig('raw')];
    if (plot.showIntermediateAnalyticsPanels === false) return [analyticsConfigs[analyticsConfigs.length - 1]];
    return [defaultAnalyticsConfig('raw'), ...analyticsConfigs];
  }, [analyticsConfigs, plot.showIntermediateAnalyticsPanels]);

  const traces = useMemo<Data[]>(() => {
    const groups = new Map<string, { name: string; color: string; metric: string; results: CombinedResult[] }>();
    if (plot.traces?.length) {
      for (const trace of plot.traces) {
        groups.set(sourceTraceKey(trace), {
          name: trace.legendLabel,
          color: trace.color,
          metric: trace.metric,
          results: [],
        });
      }
    }
    for (const result of results) {
      const key = String(result.testingRunId);
      const group = groups.get(key);
      if (group) group.results.push(result);
      else {
        groups.set(key, {
          name: result.testingRunName,
          color: TRACE_COLORS[groups.size % TRACE_COLORS.length],
          metric: 'score',
          results: [result],
        });
      }
    }
    const nextTraces: Data[] = [];
    [...groups.values()].filter((group) => group.results.length > 0).forEach((group, groupIndex) => {
      const orderedResults = group.results
        .map((result, index) => ({ result, index }))
        .sort((left, right) => new Date(left.result.timestamp).getTime() - new Date(right.result.timestamp).getTime() || left.index - right.index)
        .map((item) => item.result);
      const x = orderedResults.map((result) => result.timestamp);
      const rawValues = orderedResults.map((result) => scoreValue(result, plot.scoreSeries));
      const stageOutputs = new Map<AnalyticsKind | 'input', Array<number | null>>();
      stageOutputs.set('input', analyticsConfigs.length === 0 ? movingAverage(rawValues, plot.movingAverage).map(finiteOrNull) : rawValues.map(finiteOrNull));
      let currentValues = rawValues;
      analyticsConfigs.forEach((config) => {
        const output = computeAnalyticsSeries(config, currentValues, x);
        stageOutputs.set(config.kind, output);
        currentValues = output.map((value) => (value === null ? Number.NaN : value));
      });
      displayPanels.forEach((panel, panelIndex) => {
        const y = panel.kind === 'raw' ? stageOutputs.get('input') ?? [] : stageOutputs.get(panel.kind) ?? [];
        nextTraces.push(withLineGapPolicy({
          type: 'scatter',
          mode: selectionActive ? 'lines+markers' : 'lines',
          name: panelIndex === 0 ? group.name : `${group.name} · ${analyticsDefinition(panel.kind).label}`,
          x,
          y,
          xaxis: 'x',
          yaxis: panelIndex === 0 ? 'y' : `y${panelIndex + 1}`,
          line: { color: group.color || TRACE_COLORS[groupIndex % TRACE_COLORS.length], width: panel.kind === 'state_machine' ? 2.2 : 1.7, shape: panel.kind === 'state_machine' ? 'hv' : 'linear' },
          connectgaps: false,
          ...(selectionActive ? { marker: { size: 8, color: group.color || TRACE_COLORS[groupIndex % TRACE_COLORS.length] } } : {}),
          showlegend: panelIndex === 0,
          hovertemplate: `%{x|%Y-%m-%d %H:%M:%S}<br>${analyticsDefinition(panel.kind).label} %{y:.5g}<extra>${group.name}</extra>`,
        } as unknown as Data, {
          continuity: orderedResults.map((result) => result.continuity_segment ?? 0),
        }));
      });
    });
    return nextTraces;
  }, [analyticsConfigs, displayPanels, plot.movingAverage, plot.scoreSeries, plot.traces, results, selectionActive]);

  const traceValues = useMemo<TimeSeriesTraceValues[]>(() => traces.map((trace) => {
    const value = trace as unknown as { x?: Array<string | number | Date | null>; y?: Array<number | null>; yaxis?: string };
    return { x: value.x ?? [], y: value.y ?? [], yaxis: value.yaxis ?? 'y' };
  }), [traces]);
  const { automaticYRanges, scheduleAutomaticYRanges } = useVisibleAutomaticYRanges(traceValues);

  const fullResolutionExport = useCallback(async () => {
    const sources = plotSources(plot);
    const fullGroups = await Promise.all(sources.map(async (source, groupIndex) => {
      const configured = plot.traces?.find((trace) => Number(trace.testingRunId) === Number(source.testingRunId));
      const visible = results.find((result) => result.testingRunId === Number(source.testingRunId));
      const series = await getFullTestingRunPlotSeries(Number(source.testingRunId), {
        score_series: plot.scoreSeries,
        start_timestamp: source.start || null,
        end_timestamp: source.end || null,
      });
      return {
        name: configured?.legendLabel ?? visible?.testingRunName ?? `Testing run #${source.testingRunId}`,
        color: configured?.color ?? TRACE_COLORS[groupIndex % TRACE_COLORS.length],
        points: series.points,
      };
    }));
    const exportTraces: Data[] = [];
    fullGroups.forEach((group) => {
      const x = group.points.map((point) => point.timestamp);
      const rawValues = group.points.map((point) => point.value);
      const stageOutputs = new Map<AnalyticsKind | 'input', Array<number | null>>();
      stageOutputs.set('input', analyticsConfigs.length === 0 ? movingAverage(rawValues, plot.movingAverage).map(finiteOrNull) : rawValues.map(finiteOrNull));
      let currentValues = rawValues;
      analyticsConfigs.forEach((analytics) => {
        const output = computeAnalyticsSeries(analytics, currentValues, x);
        stageOutputs.set(analytics.kind, output);
        currentValues = output.map((value) => value === null ? Number.NaN : value);
      });
      displayPanels.forEach((panel, panelIndex) => {
        exportTraces.push({
          type: 'scatter', mode: 'lines',
          name: panelIndex === 0 ? group.name : `${group.name} · ${analyticsDefinition(panel.kind).label}`,
          x,
          y: panel.kind === 'raw' ? stageOutputs.get('input') ?? [] : stageOutputs.get(panel.kind) ?? [],
          line: { color: group.color },
        } as Data);
      });
    });
    return buildPlotExportTable(exportTraces);
  }, [analyticsConfigs, displayPanels, plot, results]);

  const handleRelayout = useCallback((event: PlotRelayoutEvent) => {
    const nextXRange = relayoutRange(event, 'xaxis');
    if (nextXRange !== undefined) scheduleAutomaticYRanges(nextXRange);

    // A box zoom can report X and Y ranges together. In that case X defines the
    // visible window and Y must remain automatic. Only a pure Y-axis gesture
    // switches a panel into manual mode.
    if (nextXRange !== undefined) return;

    const values = event as Record<string, unknown>;
    const axisKeys = new Set<string>();
    Object.keys(values).forEach((key) => {
      const match = key.match(/^(yaxis\d*)\.(?:range(?:\[[01]\])?|autorange)$/);
      if (match) axisKeys.add(match[1]);
    });
    if (axisKeys.size === 0) return;
    setManualYRanges((current) => {
      const next = { ...current };
      let changed = false;
      axisKeys.forEach((layoutAxis) => {
        const range = relayoutRange(event, layoutAxis);
        const traceAxis = layoutAxis === 'yaxis' ? 'y' : layoutAxis.replace('axis', '');
        if (range && Number.isFinite(range[0]) && Number.isFinite(range[1])) {
          next[traceAxis] = range;
          changed = true;
        }
      });
      return changed ? next : current;
    });
  }, [scheduleAutomaticYRanges]);

  const handleDoubleClick = useCallback((event: PlotlyChartDoubleClick): boolean => {
    const leftMargin = 72;
    const topMargin = 12;
    const bottomMargin = 58;
    if (event.x > leftMargin || event.y < topMargin || event.y > event.height - bottomMargin) return false;
    const paperY = (event.height - bottomMargin - event.y) / Math.max(1, event.height - topMargin - bottomMargin);
    const panelCount = Math.max(1, displayPanels.length);
    const gap = 0.035;
    const panelHeight = (1 - gap * (panelCount - 1)) / panelCount;
    const panelIndex = displayPanels.findIndex((_, index) => {
      const top = 1 - index * (panelHeight + gap);
      const bottom = top - panelHeight;
      return paperY >= bottom && paperY <= top;
    });
    if (panelIndex < 0) return false;
    const traceAxis = panelIndex === 0 ? 'y' : `y${panelIndex + 1}`;
    setManualYRanges((current) => {
      if (!current[traceAxis]) return current;
      const next = { ...current };
      delete next[traceAxis];
      return next;
    });
    return true;
  }, [displayPanels]);

  const layout = useMemo<Partial<Layout>>(
    () => {
      const panelCount = Math.max(1, displayPanels.length);
      const gap = 0.035;
      const panelHeight = (1 - gap * (panelCount - 1)) / panelCount;
      const nextLayout: Partial<Layout> & Record<string, unknown> = {
        uirevision: plot.id,
        showlegend: traces.length > panelCount,
        legend: { orientation: 'h', y: -0.18, x: 0 },
        hovermode: 'x unified',
        dragmode: selectionActive ? 'select' : 'zoom',
        selectdirection: 'h',
        xaxis: {
          title: { text: 'Time', font: { size: 12 } },
          type: 'date',
          uirevision: `${plot.id}:x`,
          rangeslider: panelCount === 1 ? { thickness: 0.08 } : undefined,
          showgrid: true,
          gridcolor: 'rgba(128,128,128,0.15)',
        },
        margin: { l: 72, r: 24, t: 12, b: 58 },
        shapes: [
          ...(plot.baselineAnalysis?.baselineRegions ?? []).map((region) => ({
            type: 'rect', xref: 'x', yref: 'paper', x0: region.start, x1: region.end, y0: 0, y1: 1,
            fillcolor: 'rgba(34, 139, 230, 0.16)', line: { color: 'rgba(34, 139, 230, 0.55)', width: 1 }, layer: 'below',
          })),
          ...(plot.baselineAnalysis?.analysisRegions ?? []).map((region, index) => ({
            type: 'rect', xref: 'x', yref: 'paper', x0: region.start, x1: region.end, y0: 0, y1: 1,
            fillcolor: index % 2 === 0 ? 'rgba(245, 159, 0, 0.16)' : 'rgba(224, 49, 49, 0.13)',
            line: { color: index % 2 === 0 ? 'rgba(245, 159, 0, 0.6)' : 'rgba(224, 49, 49, 0.55)', width: 1 }, layer: 'below',
          })),
          ...(plot.imageComparison?.reference ? [{
            type: 'line', xref: 'x', yref: 'paper', x0: plot.imageComparison.reference.timestamp, x1: plot.imageComparison.reference.timestamp, y0: 0, y1: 1,
            line: { color: '#228be6', width: 2, dash: 'dash' },
          }] : []),
          ...(plot.imageComparison?.comparisons ?? []).map((point) => ({
            type: 'line', xref: 'x', yref: 'paper', x0: point.timestamp, x1: point.timestamp, y0: 0, y1: 1,
            line: { color: '#f08c00', width: 1.5, dash: 'dot' },
          })),
        ] as NonNullable<Layout['shapes']>,
        annotations: [
          ...(plot.baselineAnalysis?.baselineRegions ?? []).map((region) => ({
            x: region.start, y: 1, xref: 'x', yref: 'paper', text: region.name, showarrow: false, xanchor: 'left', yanchor: 'bottom', font: { size: 10, color: '#228be6' },
          })),
          ...(plot.baselineAnalysis?.analysisRegions ?? []).map((region, index) => ({
            x: region.start, y: 0.96, xref: 'x', yref: 'paper', text: region.name, showarrow: false, xanchor: 'left', yanchor: 'top', font: { size: 10, color: index % 2 === 0 ? '#f08c00' : '#e03131' },
          })),
          ...(plot.imageComparison?.reference ? [{
            x: plot.imageComparison.reference.timestamp, y: 0.9, xref: 'x', yref: 'paper', text: 'Reference', showarrow: false, xanchor: 'left', yanchor: 'top', font: { size: 10, color: '#228be6' },
          }] : []),
          ...(plot.imageComparison?.comparisons ?? []).map((point, index) => ({
            x: point.timestamp, y: 0.86, xref: 'x', yref: 'paper', text: `Comparison ${index + 1}`, showarrow: false, xanchor: 'left', yanchor: 'top', font: { size: 10, color: '#f08c00' },
          })),
        ] as NonNullable<Layout['annotations']>,
      };
      displayPanels.forEach((panel, index) => {
        const top = 1 - index * (panelHeight + gap);
        const bottom = top - panelHeight;
        const traceAxis = index === 0 ? 'y' : `y${index + 1}`;
        const range = manualYRanges[traceAxis] ?? automaticYRanges[traceAxis];
        nextLayout[index === 0 ? 'yaxis' : `yaxis${index + 1}`] = {
          title: { text: analyticsDefinition(panel.kind).label, font: { size: 11 } },
          domain: [Math.max(0, bottom), Math.min(1, top)],
          showgrid: true,
          gridcolor: 'rgba(128,128,128,0.15)',
          zeroline: panel.kind !== 'raw' && panel.kind !== 'ewma',
          fixedrange: false,
          uirevision: `${plot.id}:${traceAxis}:${range?.join(':') ?? 'empty'}`,
          ...(range ? { range, autorange: false } : { autorange: true }),
        };
      });
      return nextLayout as Partial<Layout>;
    },
    [automaticYRanges, displayPanels, manualYRanges, plot.baselineAnalysis, plot.id, plot.imageComparison, selectionActive, traces.length],
  );

  if (results.length === 0) {
    return <Alert color="yellow">No results match this time range.</Alert>;
  }

  return (
    <Stack gap="xs">
      <PlotlyChart
        data={traces}
        layout={layout}
        onClick={selectionActive ? onPointClick : undefined}
        onSelected={selectionActive ? onRangeSelected : undefined}
        onRelayout={handleRelayout}
        onDoubleClick={handleDoubleClick}
        fullResolutionExport={fullResolutionExport}
        height={analyticsConfigs.length > 0 ? Math.max(520, displayPanels.length * (plot.panelHeightPx || 260)) : (plot.panelHeightPx || 420)}
      />
      <Group gap="xs">
        <Badge variant="light">{results.length} points</Badge>
        <Badge variant="light" color="cyan">Y: drag axis · double-click axis for auto</Badge>
        {plot.movingAverage > 1 && <Badge variant="light" color="blue">moving avg {plot.movingAverage}</Badge>}
        {plot.sampling > 1 && <Badge variant="light" color="gray">sample every {plot.sampling}</Badge>}
        {plot.traces?.length ? <Badge variant="light" color="teal">{plot.traces.length} traces</Badge> : null}
        {analyticsConfigs.length > 0 ? <Badge variant="light" color="violet">{analyticsConfigs.length} analytics</Badge> : null}
      </Group>
    </Stack>
  );
}

function transparentHeatmapScale(config: HeatmapVisualizationConfig) {
  const maxAlpha = config.max_clip_enabled ? 1 : config.max_opacity;
  if (config.signed_deviations) {
    return [
      [0, `rgba(0,0,255,${maxAlpha})`],
      [0.25, `rgba(0,255,255,${maxAlpha / 2})`],
      [0.5, 'rgba(255,255,255,0)'],
      [0.75, `rgba(255,255,0,${maxAlpha / 2})`],
      [1, `rgba(255,0,0,${maxAlpha})`],
    ];
  }
  return [
    [0, 'rgba(0,0,143,0)'],
    [0.25, `rgba(0,0,255,${maxAlpha * 0.25})`],
    [0.5, `rgba(0,255,255,${maxAlpha * 0.5})`],
    [0.75, `rgba(255,255,0,${maxAlpha * 0.75})`],
    [1, `rgba(255,0,0,${maxAlpha})`],
  ];
}

function HeatmapPlot({
  plot,
  results,
  heatmapCache,
  loadingHeatmaps,
  heatmapErrors,
  ensureHeatmap,
}: {
  plot: AnalysisPlot;
  results: CombinedResult[];
  heatmapCache: Record<string, HeatmapRun>;
  loadingHeatmaps: Record<string, boolean>;
  heatmapErrors: Record<string, string>;
  ensureHeatmap: (
    frame: CombinedResult,
    config: HeatmapVisualizationConfig,
    staeView: 'reconstruction' | 'prediction',
    predictionHorizon: number,
    options?: { force?: boolean },
  ) => Promise<void>;
}) {
  const current = results[0] ?? null;
  const currentKey = current ? heatmapCacheKey(current, plot.heatmapConfig, plot.staeHeatmapView, plot.predictionHorizon) : '';
  useEffect(() => {
    if (!current) return;
    ensureHeatmap(current, plot.heatmapConfig, plot.staeHeatmapView, plot.predictionHorizon);
  }, [current, ensureHeatmap, plot.heatmapConfig, plot.predictionHorizon, plot.staeHeatmapView]);

  const heatmap = currentKey ? heatmapCache[currentKey] : undefined;
  const loading = loadingHeatmaps[currentKey] === true;
  const error = heatmapErrors[currentKey];

  const relativeErrorMatrix = useMemo(() => {
    if (!heatmap?.error_matrix) return null;
    const config = heatmap.visualization_config;
    const clipFactor = config.max_clip_enabled ? config.max_clip : 1;
    const ceiling = config.fixed_ceiling_enabled
      ? config.fixed_ceiling
      : heatmap.max_error > 0
        ? heatmap.max_error * clipFactor
        : 1;
    return heatmap.error_matrix.map((row) =>
      row.map((value) =>
        config.signed_deviations
          ? Math.max(-1, Math.min(1, value / ceiling))
          : Math.max(0, Math.min(1, value / ceiling)),
      ),
    );
  }, [heatmap]);

  const errorTrace = useMemo<Data[]>(() => {
    if (!heatmap?.error_matrix || !relativeErrorMatrix) return [];
    return [
      {
        type: 'heatmap',
        z: relativeErrorMatrix,
        customdata: heatmap.error_matrix,
        colorscale: transparentHeatmapScale(heatmap.visualization_config),
        zmin: heatmap.visualization_config.signed_deviations ? -1 : 0,
        zmax: 1,
        zsmooth: false,
        showscale: false,
        hovertemplate: `x %{x}<br>y %{y}<br>${heatmap.visualization_config.signed_deviations ? 'Signed relative error' : 'Relative error'} %{z:.4f}<br>Configured pixel error %{customdata:.6g}<extra></extra>`,
      } as Data,
    ];
  }, [heatmap, relativeErrorMatrix]);

  const errorLayout = useMemo<Partial<Layout>>(() => {
    if (!heatmap) return {};
    return {
      // Keep the drawable image area identical to the adjacent image frames.
      // Pixel coordinates remain available in hover while zoom/pan stay active.
      margin: { l: 0, r: 0, t: 0, b: 0 },
      xaxis: {
        range: [-0.5, heatmap.width - 0.5],
        scaleanchor: 'y',
        constrain: 'domain',
        showgrid: false,
        zeroline: false,
        showticklabels: false,
      },
      yaxis: {
        range: [heatmap.height - 0.5, -0.5],
        showgrid: false,
        zeroline: false,
        showticklabels: false,
      },
      images: [
        {
          source: heatmap.source_image_data_url,
          xref: 'x',
          yref: 'y',
          x: -0.5,
          y: -0.5,
          sizex: heatmap.width,
          sizey: heatmap.height,
          xanchor: 'left',
          yanchor: 'top',
          sizing: 'stretch',
          layer: 'below',
          opacity: 1,
        },
      ],
    };
  }, [heatmap]);

  const frameStyle = heatmap ? { aspectRatio: `${heatmap.width} / ${heatmap.height}` } : undefined;
  const heatmapDisplaySize = Math.max(240, Math.min(1200, plot.heatmapDisplaySize ?? DEFAULT_HEATMAP_DISPLAY_SIZE));
  const heatmapPanelStyle: React.CSSProperties = { width: '100%', maxWidth: heatmapDisplaySize, marginInline: 'auto' };

  const errorPanel = heatmap ? (
    <div className="analysis-heatmap-panel" style={heatmapPanelStyle}>
      <Text size="sm" fw={500} c="dimmed" ta="center">
        Reconstruction error
      </Text>
      {heatmap.error_matrix ? (
        <>
          <div className="analysis-heatmap-image-frame analysis-heatmap-plot-frame" style={frameStyle}>
            <PlotlyChart data={errorTrace} layout={errorLayout} height="100%" />
          </div>
          <div className="analysis-relative-colorbar" aria-label="Relative reconstruction error scale from zero to one">
            <Group justify="space-between" gap="xs">
              <Text size="xs" c="dimmed">{heatmap.visualization_config.signed_deviations ? '-1' : '0'}</Text>
              <Text size="xs" c="dimmed">{heatmap.visualization_config.signed_deviations ? 'Signed relative error' : 'Relative reconstruction error'}</Text>
              <Text size="xs" c="dimmed">1</Text>
            </Group>
            <div
              className="analysis-relative-colorbar-gradient"
              style={{
                background: heatmap.visualization_config.signed_deviations ? SIGNED_GRADIENT : JET_GRADIENT,
              }}
            />
          </div>
        </>
      ) : (
        <div className="analysis-heatmap-image-frame" style={frameStyle}>
          <img src={heatmap.source_image_data_url} alt="Original with heatmap overlay" className="analysis-heatmap-image" />
          <img src={heatmap.heatmap_image_data_url} alt="Reconstruction error heatmap overlay" className="analysis-heatmap-overlay-image" />
        </div>
      )}
    </div>
  ) : null;

  if (!current) {
    return <Alert color="yellow">No result image matches this selection.</Alert>;
  }

  return (
    <Stack gap="sm">
      <Group gap="xs">
        <Badge variant="light">{resultLabel(current)}</Badge>
        <Badge variant="light" color="red">score {formatMetric(scoreValue(current))}</Badge>
        <Badge variant="light" color="gray">{current.testingRunName}</Badge>
        <Badge variant="light" color="blue">{plot.heatmapConfig.error_mode} error</Badge>
        {plot.heatmapConfig.threshold_enabled && (
          <Badge variant="light" color="yellow">threshold {formatMetric(plot.heatmapConfig.threshold)}</Badge>
        )}
        {plot.heatmapConfig.signed_deviations && (
          <Badge variant="light" color="grape">
            signed +{formatMetric(plot.heatmapConfig.positive_weight)} / -{formatMetric(plot.heatmapConfig.negative_weight)}
          </Badge>
        )}
        <Badge variant="light" color="gray">
          {plot.heatmapConfig.fixed_ceiling_enabled
            ? `ceiling ${formatMetric(plot.heatmapConfig.fixed_ceiling)}`
            : plot.heatmapConfig.max_clip_enabled
              ? `max clip ${Math.round(plot.heatmapConfig.max_clip * 100)}%`
              : `opacity ${Math.round(plot.heatmapConfig.max_opacity * 100)}%`}
        </Badge>
      </Group>
      <div className="analysis-heatmap-wrap">
        {heatmap ? (
          (plot.includeReference ?? true) ? (
            <SimpleGrid
              cols={{ base: 1, md: 3 }}
              spacing="md"
              style={{ maxWidth: heatmapDisplaySize * 3 + 32, marginInline: 'auto', alignItems: 'start' }}
            >
              <div className="analysis-heatmap-panel" style={heatmapPanelStyle}>
                <Text size="sm" fw={500} c="dimmed" ta="center">
                  Input image
                </Text>
                <div className="analysis-heatmap-image-frame" style={frameStyle}>
                  <img src={heatmap.source_image_data_url} alt="Original source" className="analysis-heatmap-image" />
                </div>
              </div>
              <div className="analysis-heatmap-panel" style={heatmapPanelStyle}>
                <Text size="sm" fw={500} c="dimmed" ta="center">
                  Reconstructed
                </Text>
                <div className="analysis-heatmap-image-frame" style={frameStyle}>
                  <img
                    src={heatmap.reconstruction_image_data_url || heatmap.source_image_data_url}
                    alt="Model reconstruction"
                    className="analysis-heatmap-image"
                  />
                </div>
              </div>
              {errorPanel}
            </SimpleGrid>
          ) : (
            errorPanel
          )
        ) : (
          <div className="analysis-heatmap-loading">
            {loading ? (
              <Stack gap="xs" align="center">
                <Loader size="sm" />
                <Text size="sm">Computing heatmap…</Text>
              </Stack>
            ) : error ? (
              <Stack gap="xs" align="center">
                <Badge color="red" variant="light">
                  Failed
                </Badge>
                <Text size="sm" ta="center">
                  {error}
                </Text>
                <Button
                  size="compact-sm"
                  variant="light"
                  onClick={() => ensureHeatmap(current, plot.heatmapConfig, plot.staeHeatmapView, plot.predictionHorizon, { force: true })}
                >
                  Retry heatmap
                </Button>
              </Stack>
            ) : (
              <Text size="sm">Heatmap queued for computation…</Text>
            )}
          </div>
        )}
      </div>
      <Text size="xs" c="dimmed">
        {(heatmap?.visualization_config.residual_source ?? plot.heatmapConfig.residual_source) === 'ssim_residual' ? 'SSIM heatmap' : 'Pixel heatmap'} · {heatmap?.visualization_config.error_mode ?? plot.heatmapConfig.error_mode} error · max pixel ({heatmap?.max_x ?? '—'}, {heatmap?.max_y ?? '—'}) · max magnitude {formatMetric(heatmap?.max_error)} · mean magnitude {formatMetric(heatmap?.mean_error)}
      </Text>
    </Stack>
  );
}

// Jet-style gradient matching the backend overlay, for the static video legend.
const JET_GRADIENT = 'linear-gradient(to right, #00008f, #0000ff, #00ffff, #ffff00, #ff0000, #800000)';
const SIGNED_GRADIENT = 'linear-gradient(to right, #0000ff, #00ffff, transparent, #ffff00, #ff0000)';

function HeatmapVideo({ plot, results }: { plot: AnalysisPlot; results: CombinedResult[] }) {
  const params = useMemo(() => {
    const source = plot.sources[0];
    if (!source || results.length === 0) return null;
    const startIso = source.start || results[0].timestamp;
    const endIso = source.end || results[results.length - 1].timestamp;
    return {
      testing_run_id: Number(source.testingRunId),
      start_timestamp: startIso,
      end_timestamp: endIso,
      stride: Math.max(1, Math.floor(source.sampling || 1)),
      testingRunName: results[0].testingRunName,
      visualizationConfig: plot.heatmapConfig,
      visualizationConfigKey: heatmapConfigKey(
        plot.heatmapConfig,
        plot.staeHeatmapView,
        plot.predictionHorizon,
      ),
      staeView: plot.staeHeatmapView,
      predictionHorizon: plot.predictionHorizon,
    };
  }, [plot.heatmapConfig, plot.predictionHorizon, plot.sources, plot.staeHeatmapView, results]);

  const [scaleMode, setScaleMode] = useState<'per_frame' | 'shared'>(plot.heatmapScaleMode ?? 'per_frame');
  const [job, setJob] = useState<HeatmapRangeRun | null>(null);
  const [starting, setStarting] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [fps, setFps] = useState(plot.heatmapFps ?? 8);
  const autoStartAttempted = useRef(false);
  const fixedCeiling = plot.heatmapConfig.fixed_ceiling_enabled;
  const effectiveScaleMode: 'per_frame' | 'shared' = fixedCeiling ? 'per_frame' : scaleMode;

  // Reset whenever the plot's range/source changes.
  useEffect(() => {
    setJob(null);
    setVideoError(null);
  }, [plot.id, params?.testing_run_id, params?.start_timestamp, params?.end_timestamp, params?.stride, params?.visualizationConfigKey, effectiveScaleMode]);

  const polling = job != null && (job.status === 'queued' || job.status === 'running');
  useEffect(() => {
    if (!polling || !job) return undefined;
    const timer = window.setInterval(async () => {
      try {
        setJob(await getHeatmapRange(job.id));
      } catch {
        /* transient; keep last state */
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [polling, job?.id]);

  const frameCount = job?.frame_count ?? 0;
  const ready = job?.status === 'finished' && frameCount > 0 && Boolean(job.video_path);

  async function startJob() {
    if (!params) return;
    setStarting(true);
    setVideoError(null);
    try {
      const created = await createHeatmapRange({
        testing_run_id: params.testing_run_id,
        start_timestamp: params.start_timestamp,
        end_timestamp: params.end_timestamp,
        stride: params.stride,
        fps,
        scale_mode: effectiveScaleMode,
        stae_view: params.staeView,
        prediction_horizon: params.predictionHorizon,
        visualization_config: params.visualizationConfig,
        force_recompute: job?.status === 'finished' && !job.video_path,
      });
      setJob(created);
    } catch (error) {
      setVideoError(errorMessage(error));
    } finally {
      setStarting(false);
    }
  }

  useEffect(() => {
    if (!plot.autoStartHeatmap || !params || autoStartAttempted.current) return;
    autoStartAttempted.current = true;
    void startJob();
  }, [params, plot.autoStartHeatmap]);

  async function abortJob() {
    if (!job) return;
    try {
      setJob(await abortHeatmapRange(job.id));
    } catch (error) {
      setVideoError(errorMessage(error));
    }
  }

  if (!params) {
    return <Alert color="yellow">Select one inference source with a time range to render a heatmap video.</Alert>;
  }

  return (
    <Stack gap="sm">
      <Group gap="xs" align="flex-end">
        <Select
          label="Color scale"
          size="xs"
          w={170}
          data={fixedCeiling
            ? [{ value: 'fixed', label: 'Fixed ceiling' }]
            : [
                { value: 'per_frame', label: 'Per-frame (auto)' },
                { value: 'shared', label: 'Shared (comparable)' },
              ]}
          value={fixedCeiling ? 'fixed' : scaleMode}
          disabled={fixedCeiling || polling}
          onChange={(value) => value && setScaleMode(value as 'per_frame' | 'shared')}
        />
        <Badge variant="light" color="gray">{params.testingRunName}</Badge>
        <Badge variant="light">stride {params.stride}</Badge>
        <Badge variant="light" color="blue">{plot.heatmapConfig.error_mode} error</Badge>
        <NumberInput
          label="fps"
          size="xs"
          w={90}
          min={1}
          max={60}
          value={fps}
          disabled={polling}
          onChange={(value) => setFps(Math.max(1, Math.min(60, Number(value) || 1)))}
        />
        {plot.heatmapConfig.signed_deviations && <Badge variant="light" color="grape">signed</Badge>}
        {fixedCeiling && <Badge variant="light" color="gray">ceiling {formatMetric(plot.heatmapConfig.fixed_ceiling)}</Badge>}
        {!ready && (
          <Button size="compact-sm" onClick={startJob} loading={starting || polling} disabled={polling}>
            {polling ? 'Rendering…' : 'Render heatmap video'}
          </Button>
        )}
        {ready && (
          <Button size="compact-sm" variant="light" onClick={startJob} loading={starting}>
            Re-render
          </Button>
        )}
      </Group>

      {videoError && <Alert color="red">{videoError}</Alert>}

      {job?.status === 'finished' && !job.video_path && (
        <Alert color="orange">
          This is a legacy frame-only result without an MP4. Render it again to create a playable video.
        </Alert>
      )}

      {polling && job && (
        <Stack gap={6}>
          <Group gap="xs">
            <Loader size="sm" />
            <Text size="sm">
              Rendering frame {job.done_count}
              {job.frame_count ? ` / ${job.frame_count}` : ''}…
            </Text>
            <Button size="compact-xs" color="red" variant="light" onClick={abortJob}>
              Abort
            </Button>
          </Group>
          {job.frame_count ? (
            <Progress value={(job.done_count / job.frame_count) * 100} size="sm" />
          ) : null}
        </Stack>
      )}

      {job && (job.status === 'failed' || job.status === 'aborted') && (
        <Alert color={job.status === 'failed' ? 'red' : 'yellow'}>
          Heatmap video {job.status}{job.error_message ? `: ${job.error_message}` : '.'}
        </Alert>
      )}

      {ready && job && (
        <Stack gap="xs">
          <div className="analysis-heatmap-image-frame analysis-heatmap-video-frame">
            <video
              src={heatmapRangeVideoUrl(job.id)}
              controls
              muted
              playsInline
              className="analysis-heatmap-image"
            />
          </div>
          <Stack gap={3}>
            <Group gap="xs" align="center">
              <Text size="xs" c="dimmed">{job.visualization_config.signed_deviations ? '-1' : '0'}</Text>
              <div
                style={{
                  flex: 1,
                  height: 10,
                  borderRadius: 3,
                  background: job.visualization_config.signed_deviations ? SIGNED_GRADIENT : JET_GRADIENT,
                }}
              />
              <Text size="xs" c="dimmed">1</Text>
            </Group>
            <Text size="xs" c="dimmed" ta="center">
              Relative reconstruction error · {job.visualization_config.fixed_ceiling_enabled
                ? `fixed ceiling ${formatMetric(job.visualization_config.fixed_ceiling)}`
                : job.scale_mode === 'shared'
                  ? 'shared scale'
                  : 'per-frame scale'} · absolute max{' '}
              {job.scale_mode === 'shared' && !job.visualization_config.fixed_ceiling_enabled
                ? formatMetric(job.global_vmax)
                : 'per-frame scale'}
            </Text>
          </Stack>
        </Stack>
      )}
    </Stack>
  );
}

function downloadBaselineCsv(plot: AnalysisPlot) {
  const analysis = plot.baselineAnalysis;
  if (!analysis?.result) return;
  const thresholds = analysis.result.thresholds;
  const header = [
    'trace', 'testing_run_id', 'region', 'region_start', 'region_end', 'samples',
    'raw_mean', 'raw_max', 'signal_mean', 'signal_max', 'signal_std', 'z_mean', 'z_median', 'z_max',
    'baseline_samples', 'baseline_mean', 'baseline_std', 'baseline_median', 'baseline_mad', 'baseline_center', 'baseline_scale',
    ...thresholds.flatMap((threshold) => [`above_${threshold}_count`, `above_${threshold}_fraction`, `above_${threshold}_longest_seconds`]),
  ];
  const quote = (value: unknown) => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const rows = analysis.result.traces.flatMap((trace) => trace.regions.map((region) => {
    const definition = analysis.analysisRegions.find((item) => item.id === region.region_id);
    const byThreshold = new Map(region.thresholds.map((item) => [item.threshold, item]));
    return [
      trace.label, trace.testing_run_id, region.region_name, definition?.start ?? '', definition?.end ?? '', region.sample_count,
      region.raw_mean, region.raw_max, region.signal_mean, region.signal_max, region.signal_std, region.z_mean, region.z_median, region.z_max,
      trace.baseline.sample_count, trace.baseline.mean, trace.baseline.std, trace.baseline.median, trace.baseline.mad, trace.baseline.center, trace.baseline.scale,
      ...thresholds.flatMap((threshold) => {
        const value = byThreshold.get(threshold);
        return [value?.sample_count ?? 0, value?.sample_fraction ?? 0, value?.longest_seconds ?? 0];
      }),
    ].map(quote).join(',');
  }));
  const blob = new Blob([[header.map(quote).join(','), ...rows].join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `baseline-analysis-${plot.id}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function legacyRegionSeries(
  trace: BaselineNormalizationResult['traces'][number],
  region: BaselineRegionStatistics,
  definition: BaselineAnalysisRegion,
): BaselineSeriesPoint[] {
  if (region.series !== undefined) return region.series;
  const start = new Date(definition.start).getTime();
  const end = new Date(definition.end).getTime();
  return (trace.series ?? []).filter((point) => {
    const timestamp = new Date(point.timestamp).getTime();
    return timestamp >= start && timestamp <= end;
  });
}

function legacyRegionEvents(
  trace: BaselineNormalizationResult['traces'][number],
  region: BaselineRegionStatistics,
  definition: BaselineAnalysisRegion,
): BaselineAnomalyEvent[] {
  if (region.events !== undefined) return region.events;
  const start = new Date(definition.start).getTime();
  const end = new Date(definition.end).getTime();
  return (trace.events ?? []).flatMap((event) => {
    const eventStart = new Date(event.start).getTime();
    const eventEnd = new Date(event.end).getTime();
    if (eventEnd < start || eventStart > end) return [];
    return [{
      ...event,
      start: new Date(Math.max(start, eventStart)).toISOString(),
      end: new Date(Math.min(end, eventEnd)).toISOString(),
    }];
  });
}

function BaselineRegionResultPlot({
  plot,
  definition,
  result,
  colorByThreshold,
}: {
  plot: AnalysisPlot;
  definition: BaselineAnalysisRegion;
  result: BaselineNormalizationResult;
  colorByThreshold: Map<number, string>;
}) {
  const plotId = plot.id;
  const [manualYRange, setManualYRange] = useState<TimeSeriesAxisRange | null>(null);
  const regionTraces = result.traces.flatMap((trace) => {
    const region = trace.regions.find((item) => item.region_id === definition.id);
    return region ? [{ trace, region }] : [];
  });
  const calculatedRegion = regionTraces[0]?.region;
  const regionStart = calculatedRegion?.start ?? definition.start;
  const regionEnd = calculatedRegion?.end ?? definition.end;
  const displayedDefinition = { ...definition, start: regionStart, end: regionEnd };
  const zData: Data[] = regionTraces.map(({ trace, region }) => {
    const series = legacyRegionSeries(trace, region, displayedDefinition);
    return withLineGapPolicy({
      type: 'scatter', mode: 'lines', name: trace.label, x: series.map((point) => point.timestamp),
      y: series.map((point) => point.z), line: { color: trace.color, width: 1.6 }, connectgaps: false,
    } as unknown as Data, {
      continuity: series.map((point) => point.continuity_segment ?? 0),
    });
  });
  const traceValues: TimeSeriesTraceValues[] = zData.map((trace) => {
    const value = trace as unknown as { x?: Array<string | number | Date | null>; y?: Array<number | null> };
    return { x: value.x ?? [], y: value.y ?? [], yaxis: 'y' };
  });
  const { automaticYRanges, scheduleAutomaticYRanges } = useVisibleAutomaticYRanges(traceValues);
  const effectiveYRange = manualYRange ?? automaticYRanges.y;
  const shapes = [
    ...result.thresholds.map((threshold) => ({
      type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: threshold, y1: threshold,
      line: { color: colorByThreshold.get(threshold), dash: 'dot', width: 1 },
    })),
    ...regionTraces.flatMap(({ trace, region }) => {
      const series = legacyRegionSeries(trace, region, displayedDefinition);
      const typicalDelta = medianPositiveTimeDelta(series.map((point) => point.timestamp)) ?? 1000;
      return legacyRegionEvents(trace, region, displayedDefinition).map((event) => ({
        type: 'rect', xref: 'x', yref: 'paper',
        x0: new Date(new Date(event.start).getTime() - typicalDelta / 2).toISOString(),
        x1: new Date(new Date(event.end).getTime() + typicalDelta / 2).toISOString(),
        y0: 0, y1: 1,
        fillcolor: colorByThreshold.get(event.threshold), opacity: 0.18,
        line: { color: colorByThreshold.get(event.threshold), width: 1 }, layer: 'below',
      }));
    }),
  ];
  const handleResultRelayout = useCallback((event: PlotRelayoutEvent) => {
    const xRange = relayoutRange(event, 'xaxis');
    if (xRange !== undefined) {
      scheduleAutomaticYRanges(xRange);
      return;
    }
    const yRange = relayoutRange(event, 'yaxis');
    if (yRange) setManualYRange(yRange);
  }, [scheduleAutomaticYRanges]);
  const handleResultDoubleClick = (event: PlotlyChartDoubleClick) => {
    if (event.x > 65 || event.y < 12 || event.y > event.height - 48) return false;
    setManualYRange(null);
    return true;
  };
  const fullResolutionExport = useCallback(async () => {
    const analytics = plot.timeseriesAnalytics ?? [];
    const stageIndex = plot.baselineAnalysis?.stageIndex ?? -1;
    const exportTraces = await Promise.all(regionTraces.map(async ({ trace }) => {
      const source = plotSources(plot).find((item) => Number(item.testingRunId) === trace.testing_run_id);
      const series = await getFullTestingRunPlotSeries(trace.testing_run_id, {
        score_series: plot.scoreSeries,
        start_timestamp: source?.start || null,
        end_timestamp: source?.end || null,
      });
      const x = series.points.map((point) => point.timestamp);
      const raw = series.points.map((point) => point.value);
      let signal: Array<number | null> = analytics.length === 0
        ? movingAverage(raw, plot.movingAverage).map(finiteOrNull)
        : raw.map(finiteOrNull);
      if (stageIndex >= 0) {
        let current = raw;
        analytics.slice(0, stageIndex + 1).forEach((config) => {
          signal = computeAnalyticsSeries(config, current, x);
          current = signal.map((value) => value === null ? Number.NaN : value);
        });
      }
      const startMs = new Date(regionStart).getTime();
      const endMs = new Date(regionEnd).getTime();
      const selected = x.map((timestamp, index) => ({ timestamp, value: signal[index] }))
        .filter((point) => {
          const time = new Date(point.timestamp).getTime();
          return time >= startMs && time <= endMs;
        });
      return {
        type: 'scatter', mode: 'lines', name: trace.label,
        x: selected.map((point) => point.timestamp),
        y: selected.map((point) => point.value === null || !Number.isFinite(point.value)
          ? null
          : (point.value - trace.baseline.center) / trace.baseline.scale),
      } as Data;
    }));
    return buildPlotExportTable(exportTraces);
  }, [plot, regionEnd, regionStart, regionTraces]);
  return (
    <Paper withBorder p="sm" radius="sm">
      <Stack gap="xs">
        <div>
          <Text fw={600}>{definition.name}</Text>
          <Text size="xs" c="dimmed">{new Date(regionStart).toLocaleString()} – {new Date(regionEnd).toLocaleString()}</Text>
        </div>
        {regionTraces.every(({ trace, region }) => legacyRegionSeries(trace, region, displayedDefinition).length === 0) ? (
          <Alert color="yellow">No valid samples exist inside this analysis region.</Alert>
        ) : (
          <PlotlyChart
            data={zData}
            layout={{
              uirevision: `${plotId}:baseline-result:${definition.id}`, hovermode: 'x unified', shapes: shapes as NonNullable<Layout['shapes']>,
              xaxis: {
                type: 'date',
                title: { text: 'Time' },
                range: [regionStart, regionEnd],
                autorange: false,
                uirevision: `${plotId}:baseline-result:${definition.id}:x`,
              },
              yaxis: { title: { text: 'Baseline z-score' }, zeroline: true, ...(effectiveYRange ? { range: effectiveYRange, autorange: false, uirevision: effectiveYRange.join(':') } : { autorange: true }) },
              margin: { l: 65, r: 20, t: 12, b: 48 },
              legend: { orientation: 'h' },
            }}
            onRelayout={handleResultRelayout}
            onDoubleClick={handleResultDoubleClick}
            fullResolutionExport={fullResolutionExport}
            height={300}
          />
        )}
      </Stack>
    </Paper>
  );
}

function BaselineAnalysisResultPanel({ plot }: { plot: AnalysisPlot }) {
  const analysis = plot.baselineAnalysis;
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  if (!analysis?.result) return null;
  const result = analysis.result;
  const thresholdColors = ['#fab005', '#fd7e14', '#e03131', '#9c36b5', '#5f3dc4', '#0c8599'];
  const colorByThreshold = new Map(result.thresholds.map((threshold, index) => [threshold, thresholdColors[index % thresholdColors.length]]));
  const decimated = result.traces.some((trace) => trace.decimated || trace.regions.some((region) => region.decimated));
  return (
    <Stack gap="sm">
      {analysis.stale && <Alert color="yellow">The saved result is stale. Recalculate to apply the current plot, regions or parameters.</Alert>}
      {decimated && <Alert color="blue">The regional plots share a limit of 8,000 points per curve. All statistics and anomaly events use the full selected resolution.</Alert>}
      <Group justify="space-between">
        <Button variant="subtle" size="compact-sm" rightSection={diagnosticsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />} onClick={() => setDiagnosticsOpen((value) => !value)}>Z-score and results</Button>
        <Button variant="default" size="compact-sm" leftSection={<Download size={14} />} onClick={() => downloadBaselineCsv(plot)}>Export CSV</Button>
      </Group>
      <Collapse in={diagnosticsOpen}>
        <Stack gap="md">
          <Group gap="xs">
            {result.thresholds.map((threshold) => <Badge key={threshold} variant="light" style={{ color: colorByThreshold.get(threshold), borderColor: colorByThreshold.get(threshold) }}>Anomaly &gt; {threshold}σ · {result.persistence_samples ?? analysis.persistenceSamples} consecutive</Badge>)}
            <Badge variant="light" color="cyan">Y: drag axis · double-click axis for auto</Badge>
          </Group>
          {analysis.analysisRegions.map((definition) => (
            <BaselineRegionResultPlot key={definition.id} plot={plot} definition={definition} result={result} colorByThreshold={colorByThreshold} />
          ))}
          <Text fw={600}>Baseline statistics</Text>
          <ScrollArea>
            <Table striped withTableBorder verticalSpacing="xs">
              <Table.Thead><Table.Tr><Table.Th>Curve</Table.Th><Table.Th>Samples</Table.Th><Table.Th>Mean</Table.Th><Table.Th>Std</Table.Th><Table.Th>Median</Table.Th><Table.Th>MAD</Table.Th><Table.Th>Center</Table.Th><Table.Th>Scale</Table.Th></Table.Tr></Table.Thead>
              <Table.Tbody>{result.traces.map((trace) => <Table.Tr key={trace.testing_run_id}><Table.Td>{trace.label}</Table.Td><Table.Td>{trace.baseline.sample_count}</Table.Td><Table.Td>{formatMetric(trace.baseline.mean)}</Table.Td><Table.Td>{formatMetric(trace.baseline.std)}</Table.Td><Table.Td>{formatMetric(trace.baseline.median)}</Table.Td><Table.Td>{formatMetric(trace.baseline.mad)}</Table.Td><Table.Td>{formatMetric(trace.baseline.center)}</Table.Td><Table.Td>{formatMetric(trace.baseline.scale)}</Table.Td></Table.Tr>)}</Table.Tbody>
            </Table>
          </ScrollArea>
          <Text fw={600}>Analysis regions</Text>
          <ScrollArea>
            <Table striped withTableBorder verticalSpacing="xs">
              <Table.Thead><Table.Tr><Table.Th>Curve</Table.Th><Table.Th>Region</Table.Th><Table.Th>N</Table.Th><Table.Th>Raw mean / max</Table.Th><Table.Th>Signal mean / max / std</Table.Th><Table.Th>Z mean / median / max</Table.Th>{result.thresholds.map((threshold) => <Table.Th key={threshold}>&gt; {threshold}σ</Table.Th>)}</Table.Tr></Table.Thead>
              <Table.Tbody>{result.traces.flatMap((trace) => trace.regions.map((region) => <Table.Tr key={`${trace.testing_run_id}:${region.region_id}`}><Table.Td>{trace.label}</Table.Td><Table.Td>{region.region_name}</Table.Td><Table.Td>{region.sample_count}</Table.Td><Table.Td>{formatMetric(region.raw_mean)} / {formatMetric(region.raw_max)}</Table.Td><Table.Td>{formatMetric(region.signal_mean)} / {formatMetric(region.signal_max)} / {formatMetric(region.signal_std)}</Table.Td><Table.Td>{formatMetric(region.z_mean)} / {formatMetric(region.z_median)} / {formatMetric(region.z_max)}</Table.Td>{region.thresholds.map((threshold) => <Table.Td key={threshold.threshold}>{threshold.sample_count} ({(threshold.sample_fraction * 100).toFixed(1)}%) · {threshold.longest_seconds.toFixed(1)}s</Table.Td>)}</Table.Tr>))}</Table.Tbody>
            </Table>
          </ScrollArea>
        </Stack>
      </Collapse>
    </Stack>
  );
}

function ImageComparisonResultPanel({ comparison }: { comparison: AnalysisImageComparisonConfig }) {
  if (!comparison.result) return null;
  const result = comparison.result;
  return (
    <Stack gap="sm">
      {comparison.stale && <Alert color="yellow">The saved image comparison is stale. Run the comparison again to apply the current selection.</Alert>}
      <Group gap="xs">
        <Badge variant="light" color="blue">{result.image_source === 'input' ? 'Input images' : 'Reconstructed images'}</Badge>
        <Badge variant="light" color="grape">Shared difference max {formatMetric(result.shared_max_difference)}</Badge>
      </Group>
      {result.comparisons.map((item) => (
        <Paper key={item.result_id} withBorder p="sm" radius="sm">
          <SimpleGrid cols={{ base: 1, md: 3 }} spacing="sm">
            <Stack gap={4} className="analysis-heatmap-panel">
              <Text size="sm" fw={600}>Reference · {new Date(result.reference_timestamp).toLocaleString()}</Text>
              <div className="analysis-heatmap-image-frame analysis-image-comparison-frame">
                <img className="analysis-heatmap-image" src={result.reference_image_data_url} alt="Reference image" />
              </div>
            </Stack>
            <Stack gap={4} className="analysis-heatmap-panel">
              <Text size="sm" fw={600}>Comparison · {new Date(item.timestamp).toLocaleString()}</Text>
              <div className="analysis-heatmap-image-frame analysis-image-comparison-frame">
                <img className="analysis-heatmap-image" src={item.image_data_url} alt="Comparison image" />
              </div>
            </Stack>
            <Stack gap={4} className="analysis-heatmap-panel">
              <Text size="sm" fw={600}>Absolute difference heatmap</Text>
              <div className="analysis-heatmap-image-frame analysis-image-comparison-frame">
                <img className="analysis-heatmap-image" src={item.heatmap_image_data_url} alt="Difference heatmap" />
              </div>
              <div className="analysis-relative-colorbar" aria-label="Shared absolute difference color scale">
                <Group justify="space-between" gap="xs"><Text size="xs" c="dimmed">0</Text><Text size="xs" c="dimmed">Shared scale</Text><Text size="xs" c="dimmed">{formatMetric(result.shared_max_difference)}</Text></Group>
                <div className="analysis-relative-colorbar-gradient" />
              </div>
              <Text size="xs" c="dimmed">Mean {formatMetric(item.mean_difference)} · Max {formatMetric(item.max_difference)}</Text>
            </Stack>
          </SimpleGrid>
        </Paper>
      ))}
    </Stack>
  );
}

const AnalysisPlotCard = memo(function AnalysisPlotCard({
  plot,
  results,
  heatmapCache,
  loadingHeatmaps,
  heatmapErrors,
  ensureHeatmap,
  onMove,
  onEdit,
  onPatch,
  onRemove,
  onHeatmapSelection,
}: {
  plot: AnalysisPlot;
  results: CombinedResult[];
  heatmapCache: Record<string, HeatmapRun>;
  loadingHeatmaps: Record<string, boolean>;
  heatmapErrors: Record<string, string>;
  ensureHeatmap: (
    frame: CombinedResult,
    config: HeatmapVisualizationConfig,
    staeView: 'reconstruction' | 'prediction',
    predictionHorizon: number,
    options?: { force?: boolean },
  ) => Promise<void>;
  onMove: (direction: -1 | 1) => void;
  onEdit: () => void;
  onPatch: (patch: Partial<AnalysisPlot>) => void;
  onRemove: () => void;
  onHeatmapSelection: (mode: HeatmapMode, start: string, end: string) => void;
}) {
  const [heatmapSelectionActive, setHeatmapSelectionActive] = useState(false);
  const [baselineSelectionActive, setBaselineSelectionActive] = useState(false);
  const [imageComparisonSelectionActive, setImageComparisonSelectionActive] = useState(false);
  const [imageComparisonSelectionKind, setImageComparisonSelectionKind] = useState<'reference' | 'comparison'>('reference');
  const [baselineRegionKind, setBaselineRegionKind] = useState<'baseline' | 'analysis'>('baseline');
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [calculatingBaseline, setCalculatingBaseline] = useState(false);
  const [calculatingImageComparison, setCalculatingImageComparison] = useState(false);
  const baselineTraces = useMemo<PlotTraceConfig[]>(() => plot.traces?.length ? plot.traces : plot.sources.map((source, index) => ({
    ...source,
    metric: 'score',
    aggregation: 'mean',
    modelLabel: `Testing run #${source.testingRunId}`,
    legendLabel: `Testing run #${source.testingRunId}`,
    color: TRACE_COLORS[index % TRACE_COLORS.length],
  })), [plot.sources, plot.traces]);

  const defaultBaselineConfig = useCallback((): BaselineAnalysisConfig => ({
    baselineRegions: [],
    analysisRegions: [],
    selectedRunIds: baselineTraces.map((trace) => trace.testingRunId),
    stageIndex: plot.timeseriesAnalytics.length - 1,
    normalization: 'classic',
    thresholds: [3, 5],
    persistenceSamples: 1,
  }), [baselineTraces, plot.timeseriesAnalytics.length]);
  const baselineConfig = plot.baselineAnalysis ?? defaultBaselineConfig();
  const comparisonSources = useMemo(() => [...new Map(plotSources(plot).map((source) => {
    const run = results.find((result) => String(result.testingRunId) === source.testingRunId);
    return [source.testingRunId, { value: source.testingRunId, label: run?.testingRunName ?? `Testing run #${source.testingRunId}` }] as const;
  })).values()], [plot, results]);
  const defaultComparisonConfig = useCallback((): AnalysisImageComparisonConfig => ({
    testingRunId: comparisonSources[0]?.value ?? '',
    imageSource: 'input',
    comparisons: [],
  }), [comparisonSources]);
  const imageComparison = plot.imageComparison ?? defaultComparisonConfig();
  const baselineStageOptions = useMemo(() => {
    const raw = [{ value: '-1', label: plot.timeseriesAnalytics.length === 0 ? `Input / moving average (${plot.movingAverage})` : 'Raw input' }];
    const stages = plot.timeseriesAnalytics.map((method, index) => ({ value: String(index), label: `${index + 1}. ${analyticsDefinition(method.kind).label}` }));
    return plot.showIntermediateAnalyticsPanels === false && stages.length > 0 ? [...raw, stages[stages.length - 1]] : [...raw, ...stages];
  }, [plot.movingAverage, plot.showIntermediateAnalyticsPanels, plot.timeseriesAnalytics]);

  const patchBaseline = useCallback((patch: Partial<BaselineAnalysisConfig>) => {
    onPatch({ baselineAnalysis: { ...baselineConfig, ...patch, stale: baselineConfig.result ? true : baselineConfig.stale } });
  }, [baselineConfig, onPatch]);

  const selectPoint = useCallback((event: PlotlyChartClick) => {
    onHeatmapSelection('single', event.timestamp, event.timestamp);
    setHeatmapSelectionActive(false);
  }, [onHeatmapSelection]);

  const selectRange = useCallback((event: PlotlyChartSelection) => {
    onHeatmapSelection('range', event.start, event.end);
    setHeatmapSelectionActive(false);
  }, [onHeatmapSelection]);

  const selectBaselineRange = useCallback((event: PlotlyChartSelection) => {
    const startMs = new Date(event.start).getTime();
    const endMs = new Date(event.end).getTime();
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return;
    const region: BaselineAnalysisRegion = {
      id: crypto.randomUUID(),
      name: baselineRegionKind === 'baseline' ? `Baseline ${baselineConfig.baselineRegions.length + 1}` : `Region ${baselineConfig.analysisRegions.length + 1}`,
      start: startMs <= endMs ? event.start : event.end,
      end: startMs <= endMs ? event.end : event.start,
    };
    patchBaseline(baselineRegionKind === 'baseline'
      ? { baselineRegions: [...baselineConfig.baselineRegions, region] }
      : { analysisRegions: [...baselineConfig.analysisRegions, region] });
  }, [baselineConfig.analysisRegions, baselineConfig.baselineRegions, baselineRegionKind, patchBaseline]);

  const selectImageComparisonPoint = useCallback((event: PlotlyChartClick) => {
    const candidates = results.filter((result) => String(result.testingRunId) === imageComparison.testingRunId);
    if (candidates.length === 0) return;
    const selectedTime = new Date(event.timestamp).getTime();
    const closest = candidates.reduce((best, result) =>
      Math.abs(new Date(result.timestamp).getTime() - selectedTime) < Math.abs(new Date(best.timestamp).getTime() - selectedTime)
        ? result
        : best,
    );
    const point = { resultId: closest.id, timestamp: closest.timestamp };
    if (imageComparisonSelectionKind === 'reference') {
      const comparisons = imageComparison.comparisons.filter((item) => item.resultId !== point.resultId);
      onPatch({ imageComparison: { ...imageComparison, reference: point, comparisons, result: imageComparison.result, stale: Boolean(imageComparison.result) } });
      setImageComparisonSelectionKind('comparison');
      return;
    }
    if (imageComparison.reference?.resultId === point.resultId) {
      notifications.show({ color: 'yellow', title: 'Reference already selected', message: 'Choose another point as a comparison image.' });
      return;
    }
    const comparisons = imageComparison.comparisons.some((item) => item.resultId === point.resultId)
      ? imageComparison.comparisons.filter((item) => item.resultId !== point.resultId)
      : [...imageComparison.comparisons, point].sort((left, right) => left.timestamp.localeCompare(right.timestamp));
    onPatch({ imageComparison: { ...imageComparison, comparisons, result: imageComparison.result, stale: Boolean(imageComparison.result) } });
  }, [imageComparison, imageComparisonSelectionKind, onPatch, results]);

  const calculateImageComparison = useCallback(async () => {
    if (!imageComparison.testingRunId || !imageComparison.reference || imageComparison.comparisons.length === 0) {
      notifications.show({ color: 'yellow', title: 'Incomplete image comparison', message: 'Select one reference point and at least one comparison point.' });
      return;
    }
    setCalculatingImageComparison(true);
    try {
      const result = await calculateAnalysisImageComparison({
        testing_run_id: Number(imageComparison.testingRunId),
        reference_result_id: imageComparison.reference.resultId,
        comparison_result_ids: imageComparison.comparisons.map((item) => item.resultId),
        image_source: imageComparison.imageSource,
      });
      onPatch({ imageComparison: { ...imageComparison, result, stale: false } });
      notifications.show({ color: 'green', title: 'Image comparison complete', message: `${result.comparisons.length} comparison${result.comparisons.length === 1 ? '' : 's'} calculated.` });
    } catch (error) {
      notifyError('Could not calculate image comparison', error);
    } finally {
      setCalculatingImageComparison(false);
    }
  }, [imageComparison, onPatch]);

  const updateRegion = useCallback((kind: 'baseline' | 'analysis', id: string, patch: Partial<BaselineAnalysisRegion>) => {
    const key = kind === 'baseline' ? 'baselineRegions' : 'analysisRegions';
    patchBaseline({ [key]: baselineConfig[key].map((region) => region.id === id ? { ...region, ...patch } : region) });
  }, [baselineConfig, patchBaseline]);

  const removeRegion = useCallback((kind: 'baseline' | 'analysis', id: string) => {
    const key = kind === 'baseline' ? 'baselineRegions' : 'analysisRegions';
    patchBaseline({ [key]: baselineConfig[key].filter((region) => region.id !== id) });
  }, [baselineConfig, patchBaseline]);

  const calculateBaseline = useCallback(async () => {
    const traces = baselineTraces.filter((trace) => baselineConfig.selectedRunIds.includes(trace.testingRunId));
    if (traces.length === 0 || baselineConfig.baselineRegions.length === 0 || baselineConfig.analysisRegions.length === 0) {
      notifications.show({ color: 'yellow', title: 'Incomplete baseline analysis', message: 'Select at least one curve, one baseline and one analysis region.' });
      return;
    }
    setCalculatingBaseline(true);
    try {
      const result = await calculateBaselineNormalization({
        traces: traces.map((trace) => ({ testing_run_id: Number(trace.testingRunId), label: trace.legendLabel, color: trace.color, start: trace.start, end: trace.end })),
        score_series: plot.scoreSeries,
        moving_average: plot.movingAverage,
        analytics_pipeline: plot.timeseriesAnalytics,
        stage_index: baselineConfig.stageIndex,
        sampling: plot.sampling,
        baseline_regions: baselineConfig.baselineRegions,
        analysis_regions: baselineConfig.analysisRegions,
        normalization: baselineConfig.normalization,
        thresholds: baselineConfig.thresholds,
        persistence_samples: baselineConfig.persistenceSamples,
        max_points: ANALYSIS_MAX_POINTS,
      });
      onPatch({ baselineAnalysis: { ...baselineConfig, result, stale: false } });
      notifications.show({ color: 'green', title: 'Baseline analysis complete', message: `${result.traces.length} curve${result.traces.length === 1 ? '' : 's'} calculated on full resolution.` });
    } catch (error) {
      notifyError('Could not calculate baseline analysis', error);
    } finally {
      setCalculatingBaseline(false);
    }
  }, [baselineConfig, baselineTraces, onPatch, plot.movingAverage, plot.sampling, plot.scoreSeries, plot.timeseriesAnalytics]);

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Group justify="space-between" align="flex-start">
          <div style={{ flex: 1, minWidth: 0 }}>
            <Group gap="xs" wrap="wrap">
              <Title order={4} fw={500} style={{ overflowWrap: 'anywhere' }}>{plot.title}</Title>
              <Badge variant="light" color={plot.plotType === 'heatmap' ? 'red' : 'blue'}>
                {plot.plotType === 'heatmap' ? 'Heatmap' : 'Time series'}
              </Badge>
            </Group>
            {plot.subtitle && <Text size="sm" c="dimmed" mt={3} style={{ overflowWrap: 'anywhere' }}>{plot.subtitle}</Text>}
            <Text size="sm" c="dimmed">
              {results.length} result rows · {plotSources(plot).length} source{plotSources(plot).length === 1 ? '' : 's'}
            </Text>
            {plot.detailSubtitle && (
              <>
                <Button variant="subtle" size="compact-xs" px={0} mt={2} rightSection={detailsOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />} onClick={() => setDetailsOpen((value) => !value)}>Show details</Button>
                <Collapse in={detailsOpen}><Stack gap={1} mt={2}>{plot.detailSubtitle.split('\n').filter(Boolean).map((line, index) => <Text key={`${index}:${line}`} size="xs" c="dimmed" style={{ overflowWrap: 'anywhere' }}>{line}</Text>)}</Stack></Collapse>
              </>
            )}
          </div>
          <Group gap={4}>
            {plot.plotType === 'timeseries' && (
              <NumberInput
                size="xs"
                w={132}
                label="Panel height"
                min={120}
                max={900}
                step={20}
                value={plot.panelHeightPx ?? (plot.timeseriesAnalytics?.length ? 260 : 420)}
                onChange={(value) => onPatch({ panelHeightPx: valueAsNumber(value, plot.panelHeightPx ?? 260) })}
              />
            )}
            {plot.plotType === 'heatmap' && plot.heatmapMode === 'single' && (
              <>
                <NumberInput
                  size="xs"
                  w={125}
                  label="Image size"
                  suffix=" px"
                  min={240}
                  max={1200}
                  step={40}
                  value={plot.heatmapDisplaySize ?? DEFAULT_HEATMAP_DISPLAY_SIZE}
                  onChange={(value) => onPatch({ heatmapDisplaySize: valueAsNumber(value, DEFAULT_HEATMAP_DISPLAY_SIZE) })}
                />
                <Switch
                  size="xs"
                  label="Reference"
                  checked={plot.includeReference ?? true}
                  onChange={(event) => onPatch({ includeReference: event.currentTarget.checked })}
                />
              </>
            )}
            {plot.plotType === 'timeseries' && (
              <Tooltip label={heatmapSelectionActive ? 'Cancel heatmap selection' : 'Create heatmap from plot selection'}>
                <ActionIcon
                  variant={heatmapSelectionActive ? 'filled' : 'subtle'}
                  color="orange"
                  aria-label={heatmapSelectionActive ? 'Cancel heatmap selection' : 'Create heatmap from plot selection'}
                  onClick={() => { setHeatmapSelectionActive((current) => !current); setBaselineSelectionActive(false); setImageComparisonSelectionActive(false); }}
                >
                  <Text span fz={17} lh={1}>🔥</Text>
                </ActionIcon>
              </Tooltip>
            )}
            {plot.plotType === 'timeseries' && (
              <Tooltip label={baselineSelectionActive ? 'Close baseline analysis selection' : 'Mark baseline and analysis regions'}>
                <ActionIcon variant={baselineSelectionActive ? 'filled' : 'subtle'} color="violet" aria-label="Baseline-normalized region analysis" onClick={() => { setBaselineSelectionActive((current) => !current); setHeatmapSelectionActive(false); setImageComparisonSelectionActive(false); }}>
                  <Activity size={17} />
                </ActionIcon>
              </Tooltip>
            )}
            {plot.plotType === 'timeseries' && (
              <Tooltip label={imageComparisonSelectionActive ? 'Close image comparison selection' : 'Compare images at selected points'}>
                <ActionIcon variant={imageComparisonSelectionActive ? 'filled' : 'subtle'} color="cyan" aria-label="Compare selected images" onClick={() => { setImageComparisonSelectionActive((current) => !current); setHeatmapSelectionActive(false); setBaselineSelectionActive(false); }}>
                  <Image size={17} />
                </ActionIcon>
              </Tooltip>
            )}
            <Tooltip label="Move up">
              <ActionIcon variant="subtle" onClick={() => onMove(-1)}>
                <ArrowUp size={16} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Move down">
              <ActionIcon variant="subtle" onClick={() => onMove(1)}>
                <ArrowDown size={16} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Edit plot">
              <ActionIcon variant="subtle" onClick={onEdit}>
                <Pencil size={16} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Remove plot">
              <ActionIcon color="red" variant="subtle" onClick={onRemove}>
                <Trash2 size={16} />
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
        {heatmapSelectionActive && (
          <Alert color="orange" icon={<Flame size={18} />}>
            Zoom or pan to the desired section first. Then click a data point for a single heatmap, or drag horizontally to create a heatmap video. Click 🔥 again to cancel.
          </Alert>
        )}
        {baselineSelectionActive && (
          <Paper withBorder p="sm" radius="sm">
            <Stack gap="sm">
              <Group justify="space-between" wrap="wrap">
                <div><Text fw={600}>Baseline-normalized region analysis</Text><Text size="xs" c="dimmed">Choose a region type, then drag horizontally in the plot. Markings appear immediately.</Text></div>
                <Select size="xs" w={170} value={baselineRegionKind} data={[{ value: 'baseline', label: 'Add baseline' }, { value: 'analysis', label: 'Add analysis region' }]} onChange={(value) => setBaselineRegionKind(value === 'analysis' ? 'analysis' : 'baseline')} />
              </Group>
              {(baselineConfig.baselineRegions.length > 0 || baselineConfig.analysisRegions.length > 0) && (
                <Stack gap="xs">
                  {([['baseline', baselineConfig.baselineRegions], ['analysis', baselineConfig.analysisRegions]] as const).flatMap(([kind, regions]) => regions.map((region) => (
                    <SimpleGrid key={region.id} cols={{ base: 1, md: 4 }} spacing="xs">
                      <TextInput size="xs" label={kind === 'baseline' ? 'Baseline name' : 'Region name'} value={region.name} onChange={(event) => updateRegion(kind, region.id, { name: event.currentTarget.value })} />
                      <DateTime24Input label="Start" value={toDateTimeLocal(region.start)} onChange={(value) => updateRegion(kind, region.id, { start: value })} />
                      <DateTime24Input label="End" value={toDateTimeLocal(region.end)} onChange={(value) => updateRegion(kind, region.id, { end: value })} />
                      <Group align="flex-end"><ActionIcon color="red" variant="subtle" aria-label={`Remove ${region.name}`} onClick={() => removeRegion(kind, region.id)}><Trash2 size={16} /></ActionIcon></Group>
                    </SimpleGrid>
                  )))}
                </Stack>
              )}
              <Text size="xs" fw={600}>Curves</Text>
              <Group gap="md">{baselineTraces.map((trace) => <Checkbox key={trace.testingRunId} label={trace.legendLabel} checked={baselineConfig.selectedRunIds.includes(trace.testingRunId)} onChange={(event) => patchBaseline({ selectedRunIds: event.currentTarget.checked ? [...baselineConfig.selectedRunIds, trace.testingRunId] : baselineConfig.selectedRunIds.filter((id) => id !== trace.testingRunId) })} />)}</Group>
              <SimpleGrid cols={{ base: 1, sm: 4 }}>
                <Select label="Signal stage" value={String(baselineConfig.stageIndex)} data={baselineStageOptions} onChange={(value) => patchBaseline({ stageIndex: Number(value ?? -1) })} />
                <Select label="Normalization" value={baselineConfig.normalization} data={[{ value: 'classic', label: 'Classic (mean / std)' }, { value: 'robust', label: 'Robust (median / MAD)' }]} onChange={(value) => patchBaseline({ normalization: value === 'robust' ? 'robust' : 'classic' })} />
                <TagsInput
                  label="Z thresholds"
                  description="Enter a positive value and press Enter"
                  value={baselineConfig.thresholds.map(String)}
                  onChange={(items) => {
                    const values = [...new Set(items.map(Number).filter((value) => Number.isFinite(value) && value > 0))].sort((left, right) => left - right);
                    if (values.length) patchBaseline({ thresholds: values });
                  }}
                  splitChars={[',', ' ']}
                />
                <NumberInput
                  label={<InfoLabel label="Consecutive samples" info="Minimum number of directly consecutive samples strictly above a Z threshold before that interval is marked as an anomaly. Data gaps reset the sequence." />}
                  min={1}
                  step={1}
                  value={baselineConfig.persistenceSamples}
                  onChange={(value) => patchBaseline({ persistenceSamples: Math.max(1, Math.floor(valueAsNumber(value, 1))) })}
                />
              </SimpleGrid>
              <Group justify="flex-end"><Button loading={calculatingBaseline} onClick={calculateBaseline} leftSection={<Activity size={16} />}>Calculate baseline analysis</Button></Group>
            </Stack>
          </Paper>
        )}
        {imageComparisonSelectionActive && (
          <Paper withBorder p="sm" radius="sm">
            <Stack gap="sm">
              <div>
                <Text fw={600}>Reference image comparison</Text>
                <Text size="xs" c="dimmed">Choose Reference, click one point, then switch to Comparison and click as many additional points as needed.</Text>
              </div>
              <SimpleGrid cols={{ base: 1, sm: 3 }}>
                <Select
                  label="Inference source"
                  data={comparisonSources}
                  value={imageComparison.testingRunId}
                  allowDeselect={false}
                  onChange={(value) => onPatch({ imageComparison: { testingRunId: value ?? '', imageSource: imageComparison.imageSource, comparisons: [], stale: Boolean(imageComparison.result), result: imageComparison.result } })}
                />
                <Select
                  label="Images to compare"
                  data={[{ value: 'input', label: 'Input images' }, { value: 'reconstruction', label: 'Reconstructed images' }]}
                  value={imageComparison.imageSource}
                  allowDeselect={false}
                  onChange={(value) => onPatch({ imageComparison: { ...imageComparison, imageSource: value === 'reconstruction' ? 'reconstruction' : 'input', stale: Boolean(imageComparison.result) } })}
                />
                <Select
                  label="Next click selects"
                  data={[{ value: 'reference', label: 'Reference point' }, { value: 'comparison', label: 'Comparison point' }]}
                  value={imageComparisonSelectionKind}
                  allowDeselect={false}
                  onChange={(value) => setImageComparisonSelectionKind(value === 'comparison' ? 'comparison' : 'reference')}
                />
              </SimpleGrid>
              <Group gap="xs">
                <Badge variant="light" color="blue">Reference: {imageComparison.reference ? new Date(imageComparison.reference.timestamp).toLocaleString() : 'not selected'}</Badge>
                {imageComparison.comparisons.map((point) => (
                  <Badge key={point.resultId} variant="light" color="orange" rightSection={<ActionIcon size="xs" variant="transparent" color="orange" onClick={() => onPatch({ imageComparison: { ...imageComparison, comparisons: imageComparison.comparisons.filter((item) => item.resultId !== point.resultId), stale: Boolean(imageComparison.result) } })}><Trash2 size={10} /></ActionIcon>}>
                    {new Date(point.timestamp).toLocaleString()}
                  </Badge>
                ))}
              </Group>
              <Group justify="flex-end">
                <Button loading={calculatingImageComparison} onClick={calculateImageComparison} leftSection={<Image size={16} />}>Calculate image comparison</Button>
              </Group>
            </Stack>
          </Paper>
        )}
        {plot.plotType === 'timeseries' ? (
          <TimeSeriesPlot
            plot={plot}
            results={results}
            selectionActive={heatmapSelectionActive || baselineSelectionActive || imageComparisonSelectionActive}
            onPointClick={heatmapSelectionActive ? selectPoint : imageComparisonSelectionActive ? selectImageComparisonPoint : undefined}
            onRangeSelected={heatmapSelectionActive ? selectRange : baselineSelectionActive ? selectBaselineRange : undefined}
          />
        ) : plot.heatmapMode === 'range' ? (
          <HeatmapVideo plot={plot} results={results} />
        ) : (
          <HeatmapPlot
            plot={plot}
            results={results}
            heatmapCache={heatmapCache}
            loadingHeatmaps={loadingHeatmaps}
            heatmapErrors={heatmapErrors}
            ensureHeatmap={ensureHeatmap}
          />
        )}
        {plot.plotType === 'timeseries' && <BaselineAnalysisResultPanel plot={plot} />}
        {plot.plotType === 'timeseries' && plot.imageComparison && <ImageComparisonResultPanel comparison={plot.imageComparison} />}
      </Stack>
    </Paper>
  );
});

function artifactTimestamp(value: string): string {
  return value.slice(0, 19).replace('T', ' ');
}

function heatmapErrorDescription(config: HeatmapVisualizationConfig): string {
  const source = config.residual_source === 'ssim_residual' ? 'SSIM residual' : 'Pixel residual';
  const error = config.residual_source === 'ssim_residual' ? '1 - SSIM' : `${config.error_mode} error`;
  const modifiers = [config.signed_deviations ? 'signed' : '', config.threshold_enabled ? `threshold ${formatValue(config.threshold)}` : '']
    .filter(Boolean)
    .join(', ');
  return `${source} · ${error}${modifiers ? ` · ${modifiers}` : ''}`;
}

function heatmapVisibilityDescription(config: HeatmapVisualizationConfig): string {
  if (config.fixed_ceiling_enabled) return `Fixed ceiling ${formatValue(config.fixed_ceiling)} · ${Math.round(config.max_opacity * 100)}% opacity`;
  if (config.max_clip_enabled) return `Max clip ${Math.round(config.max_clip * 100)}%`;
  return `Maximum opacity ${Math.round(config.max_opacity * 100)}%`;
}

function heatmapStatusColor(status: string): string {
  if (status === 'finished') return 'green';
  if (status === 'running') return 'blue';
  if (status === 'queued') return 'yellow';
  if (status === 'failed') return 'red';
  return 'gray';
}

function ExistingHeatmapArtifacts({
  singles,
  videos,
  testingRunById,
  intervalLabel,
  refreshing,
  onRefresh,
}: {
  singles: HeatmapRunSummary[];
  videos: HeatmapRangeRun[];
  testingRunById: Map<number, TestingRun>;
  intervalLabel: string;
  refreshing: boolean;
  onRefresh: () => void | Promise<void>;
}) {
  const [selectedSingle, setSelectedSingle] = useState<HeatmapRun | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<HeatmapRangeRun | null>(null);
  const [loadingSingleId, setLoadingSingleId] = useState<number | null>(null);
  const [renderingVideoId, setRenderingVideoId] = useState<number | null>(null);
  const [selectedVideoError, setSelectedVideoError] = useState<string | null>(null);
  const [artifactPage, setArtifactPage] = useState(1);

  async function openSingle(summary: HeatmapRunSummary) {
    setSelectedVideo(null);
    setLoadingSingleId(summary.id);
    try {
      setSelectedSingle(await getHeatmap(summary.id));
    } catch (error) {
      notifyError('Could not load saved heatmap', error);
    } finally {
      setLoadingSingleId(null);
    }
  }

  function openVideo(video: HeatmapRangeRun) {
    setSelectedSingle(null);
    setSelectedVideoError(null);
    setSelectedVideo(video);
  }

  async function renderLegacyVideo(video: HeatmapRangeRun) {
    setRenderingVideoId(video.id);
    try {
      const created = await createHeatmapRange({
        testing_run_id: video.testing_run_id,
        start_timestamp: video.start_timestamp,
        end_timestamp: video.end_timestamp,
        stride: video.stride,
        fps: video.fps,
        scale_mode: video.scale_mode,
        stae_view: video.stae_view,
        prediction_horizon: video.prediction_horizon,
        visualization_config: video.visualization_config,
        force_recompute: true,
      });
      notifications.show({
        color: 'blue',
        title: 'MP4 render queued',
        message: `Heatmap video run #${created.id} was added to the queue.`,
      });
      await onRefresh();
    } catch (error) {
      notifyError('Could not render heatmap MP4', error);
    } finally {
      setRenderingVideoId(null);
    }
  }

  function videoUnavailableReason(video: HeatmapRangeRun): string | null {
    if (video.status === 'finished' && !video.video_path) return 'No MP4 exists for this legacy frame-only run.';
    if (video.status === 'queued') return 'The video is queued and has not been rendered yet.';
    if (video.status === 'running') return `The video is still rendering (${video.done_count} / ${video.frame_count ?? '?'} frames).`;
    if (video.status === 'failed') return video.error_message ? `Rendering failed: ${video.error_message}` : 'Rendering failed.';
    if (video.status === 'aborted') return 'Rendering was aborted.';
    if (video.status !== 'finished') return `The artifact cannot be opened while its status is ${video.status}.`;
    return null;
  }

  const total = singles.length + videos.length;
  const pageStart = (artifactPage - 1) * DEFAULT_TABLE_PAGE_SIZE;
  const pageEnd = artifactPage * DEFAULT_TABLE_PAGE_SIZE;
  const pagedSingles = singles.slice(pageStart, pageEnd);
  const videoStart = Math.max(0, pageStart - singles.length);
  const videoEnd = Math.max(0, pageEnd - singles.length);
  const pagedVideos = videos.slice(videoStart, videoEnd);

  useEffect(() => {
    setArtifactPage((page) => Math.min(page, Math.max(1, Math.ceil(total / DEFAULT_TABLE_PAGE_SIZE))));
  }, [total]);

  return (
    <Paper withBorder p="md" radius="sm" className="analysis-artifact-browser">
      <Stack gap="md">
        <Group justify="space-between" align="flex-start" wrap="wrap">
          <div>
            <Group gap="xs">
              <Text fw={700}>Existing heatmaps</Text>
              <Badge variant="light" color="violet">{total}</Badge>
            </Group>
            <Text size="sm" c="dimmed">
              Saved single heatmaps and videos matching the selected models, inference dataset and {intervalLabel}.
            </Text>
          </div>
          <Button
            variant="default"
            size="compact-sm"
            leftSection={<RefreshCw size={15} />}
            loading={refreshing}
            onClick={onRefresh}
          >
            Refresh
          </Button>
        </Group>

        {total === 0 ? (
          <Alert color="gray">No saved single heatmaps or heatmap videos match this selection and time range yet.</Alert>
        ) : (
          <ScrollArea>
            <Table striped highlightOnHover verticalSpacing="sm" miw={1120}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Heatmap mode</Table.Th>
                  <Table.Th>Inference run</Table.Th>
                  <Table.Th>Timestamp / range</Table.Th>
                  <Table.Th>Sampling rate</Table.Th>
                  <Table.Th>Error calculation</Table.Th>
                  <Table.Th>Visibility</Table.Th>
                  <Table.Th>Status</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {pagedSingles.map((item) => (
                  <Table.Tr key={`single-${item.id}`}>
                    <Table.Td><Badge leftSection={<Image size={12} />} color="violet" variant="light">Single heatmap</Badge></Table.Td>
                    <Table.Td>{testingRunById.get(item.testing_run_id)?.name ?? `Run #${item.testing_run_id}`}</Table.Td>
                    <Table.Td>{artifactTimestamp(item.timestamp)}</Table.Td>
                    <Table.Td>1 frame</Table.Td>
                    <Table.Td>{heatmapErrorDescription(item.visualization_config)}</Table.Td>
                    <Table.Td>{heatmapVisibilityDescription(item.visualization_config)}</Table.Td>
                    <Table.Td><Badge color={heatmapStatusColor(item.status)} variant="light">{item.status}</Badge></Table.Td>
                    <Table.Td>
                      <Button size="compact-sm" variant="light" loading={loadingSingleId === item.id} onClick={() => openSingle(item)}>
                        Show
                      </Button>
                    </Table.Td>
                  </Table.Tr>
                ))}
                {pagedVideos.map((item) => {
                  const progress = item.frame_count ? Math.min(100, (item.done_count / item.frame_count) * 100) : 0;
                  const unavailableReason = videoUnavailableReason(item);
                  const canShow = item.status === 'finished' && Boolean(item.video_path);
                  return (
                    <Table.Tr key={`video-${item.id}`}>
                      <Table.Td><Badge leftSection={<Film size={12} />} color="blue" variant="light">Heatmap video</Badge></Table.Td>
                      <Table.Td>{item.testing_run_name}</Table.Td>
                      <Table.Td>{artifactTimestamp(item.start_timestamp)} – {artifactTimestamp(item.end_timestamp)}</Table.Td>
                      <Table.Td>Every {item.stride} frame{item.stride === 1 ? '' : 's'} · {item.fps} FPS</Table.Td>
                      <Table.Td>{heatmapErrorDescription(item.visualization_config)}</Table.Td>
                      <Table.Td>{heatmapVisibilityDescription(item.visualization_config)} · {item.scale_mode === 'shared' ? 'shared scale' : 'per-frame scale'}</Table.Td>
                      <Table.Td>
                        <Stack gap={4} miw={90}>
                          <Badge color={heatmapStatusColor(item.status)} variant="light">{item.status}</Badge>
                          {(item.status === 'queued' || item.status === 'running') && <Progress value={progress} size="xs" animated={item.status === 'running'} />}
                        </Stack>
                      </Table.Td>
                      <Table.Td>
                        <Stack gap={4} miw={170}>
                          {canShow ? (
                            <Button size="compact-sm" variant="light" onClick={() => openVideo(item)}>
                              Show
                            </Button>
                          ) : item.status === 'finished' && !item.video_path ? (
                            <Button size="compact-sm" color="orange" variant="light" loading={renderingVideoId === item.id} onClick={() => renderLegacyVideo(item)}>
                              Render MP4
                            </Button>
                          ) : (
                            <Button size="compact-sm" variant="light" disabled>
                              Show
                            </Button>
                          )}
                          {unavailableReason && <Text size="xs" c={item.status === 'failed' ? 'red' : 'dimmed'}>{unavailableReason}</Text>}
                        </Stack>
                      </Table.Td>
                    </Table.Tr>
                  );
                })}
              </Table.Tbody>
            </Table>
          </ScrollArea>
        )}
        <TablePagination totalItems={total} page={artifactPage} onChange={setArtifactPage} />

        {selectedSingle && (
          <Paper withBorder p="sm" radius="sm" className="analysis-saved-artifact-preview">
            <Group justify="space-between" mb="sm">
              <div>
                <Text fw={700}>Single heatmap · {artifactTimestamp(selectedSingle.timestamp)}</Text>
                <Text size="xs" c="dimmed">Maximum error {formatValue(selectedSingle.max_error)} · mean {formatValue(selectedSingle.mean_error)}</Text>
              </div>
              <Button size="compact-sm" variant="subtle" onClick={() => setSelectedSingle(null)}>Close</Button>
            </Group>
            <img className="analysis-saved-heatmap-image" src={selectedSingle.heatmap_image_data_url} alt={`Saved heatmap at ${selectedSingle.timestamp}`} />
          </Paper>
        )}

        {selectedVideo && (
          <Paper withBorder p="sm" radius="sm" className="analysis-saved-artifact-preview">
            <Group justify="space-between" mb="sm">
              <div>
                <Text fw={700}>Heatmap video · {selectedVideo.testing_run_name}</Text>
                <Text size="xs" c="dimmed">{artifactTimestamp(selectedVideo.start_timestamp)} – {artifactTimestamp(selectedVideo.end_timestamp)}</Text>
              </div>
              <Button size="compact-sm" variant="subtle" onClick={() => setSelectedVideo(null)}>Close</Button>
            </Group>
            {selectedVideoError && <Alert color="red" mb="sm">{selectedVideoError}</Alert>}
            <video
              className="inspect-video-player"
              src={heatmapRangeVideoUrl(selectedVideo.id)}
              controls
              preload="metadata"
              onError={() => setSelectedVideoError('The MP4 could not be loaded. The stored video file may be missing or invalid.')}
            />
          </Paper>
        )}
      </Stack>
    </Paper>
  );
}

function defaultDraft(): PlotDraft {
  return {
    plotType: 'timeseries',
    testingRunId: null,
    title: '',
    subtitle: '',
    scoreSeries: 'score',
    start: '',
    end: '',
    sampling: 1,
    movingAverage: 1,
    timeseriesAnalytics: [],
    analyticsDisplayMode: 'multi_panel',
    showIntermediateAnalyticsPanels: true,
    panelHeightPx: 420,
    heatmapMode: 'single',
    timestamp: null,
    includeReference: true,
    staeHeatmapView: 'reconstruction',
    predictionHorizon: 1,
    heatmapConfig: defaultHeatmapConfig(),
  };
}

function plotDraftFrom(plot: PlotDraft): PlotDraft {
  return {
    plotType: plot.plotType,
    testingRunId: plot.testingRunId,
    title: plot.title,
    subtitle: plot.subtitle,
    scoreSeries: plot.scoreSeries,
    start: plot.start,
    end: plot.end,
    sampling: plot.sampling,
    movingAverage: plot.movingAverage,
    timeseriesAnalytics: plot.timeseriesAnalytics.map((method) => ({ ...method, params: { ...method.params } })),
    analyticsDisplayMode: plot.analyticsDisplayMode,
    showIntermediateAnalyticsPanels: plot.showIntermediateAnalyticsPanels,
    panelHeightPx: plot.panelHeightPx,
    heatmapMode: plot.heatmapMode,
    timestamp: plot.timestamp,
    includeReference: plot.includeReference,
    staeHeatmapView: plot.staeHeatmapView,
    predictionHorizon: plot.predictionHorizon,
    heatmapConfig: { ...plot.heatmapConfig },
  };
}

type AnalysisBoardLayout = {
  version: 1 | 2 | 3;
  draft: PlotDraft;
  plots: AnalysisPlot[];
  selectedPipelineId: string | null;
  selectedModelIds?: string[];
  selectedInferenceDatasetId?: string | null;
  selectedMetricKeys?: string[];
  selectedAggregationKeys?: string[];
  selectedRoiKey: string | null;
  selectedSources: PlotSourceConfig[];
  addPlotOpen: boolean;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function restoreAnalytics(value: unknown): AnalyticsMethodConfig[] {
  if (!Array.isArray(value)) return [];
  const validKinds = new Set(ANALYTICS_DEFINITIONS.map((definition) => definition.kind));
  return value.filter(isRecord).map((item) => {
    const kind = String(item.kind ?? '') as AnalyticsKind;
    if (!validKinds.has(kind)) return null;
    const defaults = defaultAnalyticsConfig(kind);
    return {
      kind,
      params: {
        ...defaults.params,
        ...(isRecord(item.params) ? item.params : {}),
      },
    };
  }).filter((item): item is AnalyticsMethodConfig => item !== null);
}

function restoreDraft(value: unknown): PlotDraft {
  if (!isRecord(value)) return defaultDraft();
  const restoredAnalytics = restoreAnalytics(value.timeseriesAnalyticsPipeline ?? value.timeseriesAnalytics);
  return plotDraftFrom({
    ...defaultDraft(),
    ...(value as Partial<PlotDraft>),
    timeseriesAnalytics: restoredAnalytics,
    analyticsDisplayMode: 'multi_panel',
    showIntermediateAnalyticsPanels: typeof value.showIntermediateAnalyticsPanels === 'boolean' ? value.showIntermediateAnalyticsPanels : true,
    panelHeightPx: valueAsNumber(value.panelHeightPx as string | number, restoredAnalytics.length > 0 ? 260 : 420),
    heatmapConfig: {
      ...defaultHeatmapConfig(),
      ...(isRecord(value.heatmapConfig) ? value.heatmapConfig : {}),
    },
  });
}

function restoreSources(value: unknown): PlotSourceConfig[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((source) => ({
    testingRunId: String(source.testingRunId ?? ''),
    start: String(source.start ?? ''),
    end: String(source.end ?? ''),
    sampling: valueAsNumber(source.sampling as string | number, 1),
    timestamp: source.timestamp === null || source.timestamp === undefined ? null : String(source.timestamp),
  })).filter((source) => source.testingRunId);
}

function restoreTraces(value: unknown): PlotTraceConfig[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((trace, index) => {
    const source = restoreSources([trace])[0];
    return {
      ...source,
      metric: normalizeMetricKey(String(trace.metric ?? 'mse')),
      aggregation: normalizeAggregationKey(String(trace.aggregation ?? 'mean')),
      modelLabel: String(trace.modelLabel ?? trace.legendLabel ?? `Source ${index + 1}`),
      legendLabel: String(trace.legendLabel ?? trace.modelLabel ?? `Source ${index + 1}`),
      color: String(trace.color ?? TRACE_COLORS[index % TRACE_COLORS.length]),
    };
  }).filter((trace) => trace.testingRunId);
}

function restoreBaselineRegions(value: unknown): BaselineAnalysisRegion[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((region) => ({
    id: String(region.id ?? crypto.randomUUID()),
    name: String(region.name ?? 'Region'),
    start: String(region.start ?? ''),
    end: String(region.end ?? ''),
  })).filter((region) => region.start && region.end);
}

function restoreBaselineAnalysis(value: unknown): BaselineAnalysisConfig | undefined {
  if (!isRecord(value)) return undefined;
  return {
    baselineRegions: restoreBaselineRegions(value.baselineRegions),
    analysisRegions: restoreBaselineRegions(value.analysisRegions),
    selectedRunIds: Array.isArray(value.selectedRunIds) ? value.selectedRunIds.map(String) : [],
    stageIndex: valueAsNumber(value.stageIndex as string | number, -1),
    normalization: value.normalization === 'robust' ? 'robust' : 'classic',
    thresholds: Array.isArray(value.thresholds) ? value.thresholds.map(Number).filter((item) => Number.isFinite(item) && item > 0) : [3, 5],
    persistenceSamples: Math.max(1, Math.floor(valueAsNumber(value.persistenceSamples as string | number, 1))),
    result: isRecord(value.result) ? value.result as BaselineNormalizationResult : undefined,
    stale: typeof value.stale === 'boolean' ? value.stale : false,
  };
}

function restoreImageComparisonPoint(value: unknown): ImageComparisonPoint | undefined {
  if (!isRecord(value)) return undefined;
  const resultId = valueAsNumber(value.resultId as string | number, 0);
  const timestamp = String(value.timestamp ?? '');
  return resultId > 0 && timestamp ? { resultId, timestamp } : undefined;
}

function restoreImageComparison(value: unknown): AnalysisImageComparisonConfig | undefined {
  if (!isRecord(value)) return undefined;
  const comparisons = Array.isArray(value.comparisons)
    ? value.comparisons.map(restoreImageComparisonPoint).filter((item): item is ImageComparisonPoint => item !== undefined)
    : [];
  return {
    testingRunId: String(value.testingRunId ?? ''),
    imageSource: value.imageSource === 'reconstruction' ? 'reconstruction' : 'input',
    reference: restoreImageComparisonPoint(value.reference),
    comparisons,
    result: isRecord(value.result) ? value.result as AnalysisImageComparisonResult : undefined,
    stale: typeof value.stale === 'boolean' ? value.stale : false,
  };
}

function restorePlots(value: unknown): AnalysisPlot[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((plot) => {
    const draft = restoreDraft(plot);
    const traces = restoreTraces(plot.traces);
    const sources = restoreSources(plot.sources);
    return {
      ...draft,
      id: String(plot.id ?? crypto.randomUUID()),
      sources: sources.length > 0 ? sources : traces.map(traceToSource),
      traces,
      derivedFromPlotId: plot.derivedFromPlotId === null || plot.derivedFromPlotId === undefined ? undefined : String(plot.derivedFromPlotId),
      autoStartHeatmap: typeof plot.autoStartHeatmap === 'boolean' ? plot.autoStartHeatmap : false,
      heatmapFps: plot.heatmapFps === null || plot.heatmapFps === undefined ? undefined : valueAsNumber(plot.heatmapFps as string | number, 8),
      heatmapScaleMode: plot.heatmapScaleMode === 'shared' ? 'shared' : 'per_frame',
      heatmapDisplaySize: plot.heatmapDisplaySize === null || plot.heatmapDisplaySize === undefined
        ? DEFAULT_HEATMAP_DISPLAY_SIZE
        : valueAsNumber(plot.heatmapDisplaySize as string | number, DEFAULT_HEATMAP_DISPLAY_SIZE),
      baselineAnalysis: restoreBaselineAnalysis(plot.baselineAnalysis),
      imageComparison: restoreImageComparison(plot.imageComparison),
    };
  });
}

function restoreBoardLayout(value: Record<string, unknown>): AnalysisBoardLayout {
  return {
    version: 1,
    draft: restoreDraft(value.draft),
    plots: restorePlots(value.plots),
    selectedPipelineId: value.selectedPipelineId === null || value.selectedPipelineId === undefined ? null : String(value.selectedPipelineId),
    selectedModelIds: Array.isArray(value.selectedModelIds) ? value.selectedModelIds.map(String).filter(Boolean) : [],
    selectedInferenceDatasetId:
      value.selectedInferenceDatasetId === null || value.selectedInferenceDatasetId === undefined ? null : String(value.selectedInferenceDatasetId),
    selectedMetricKeys: Array.isArray(value.selectedMetricKeys)
      ? value.selectedMetricKeys.map((metric) => normalizeMetricKey(String(metric))).filter(Boolean)
      : [],
    selectedAggregationKeys: Array.isArray(value.selectedAggregationKeys)
      ? value.selectedAggregationKeys.map((aggregation) => normalizeAggregationKey(String(aggregation))).filter(Boolean)
      : [],
    selectedRoiKey: value.selectedRoiKey === null || value.selectedRoiKey === undefined ? null : String(value.selectedRoiKey),
    selectedSources: restoreSources(value.selectedSources),
    addPlotOpen: typeof value.addPlotOpen === 'boolean' ? value.addPlotOpen : true,
  };
}

export function AnalysisPage({ active = true }: { active?: boolean }) {
  const [testingRuns, setTestingRuns] = useState<TestingRun[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [trainingPipelines, setTrainingPipelines] = useState<TrainingPipeline[]>([]);
  const [preprocessingPipelines, setPreprocessingPipelines] = useState<PreprocessingPipeline[]>([]);
  const [methodConfigurations, setMethodConfigurations] = useState<MethodConfiguration[]>([]);
  const [savedHeatmaps, setSavedHeatmaps] = useState<HeatmapRunSummary[]>([]);
  const [savedHeatmapVideos, setSavedHeatmapVideos] = useState<HeatmapRangeRun[]>([]);
  const [refreshingHeatmapArtifacts, setRefreshingHeatmapArtifacts] = useState(false);
  const [resultsByRunId, setResultsByRunId] = useState<Record<number, TestingRunResults>>({});
  const [loadingRunId, setLoadingRunId] = useState<number | null>(null);
  const [draft, setDraft] = useState<PlotDraft>(defaultDraft());
  const [plots, setPlots] = useState<AnalysisPlot[]>([]);
  const [heatmapCache, setHeatmapCache] = useState<Record<string, HeatmapRun>>({});
  const [loadingHeatmaps, setLoadingHeatmaps] = useState<Record<string, boolean>>({});
  const [heatmapErrors, setHeatmapErrors] = useState<Record<string, string>>({});
  const heatmapRequests = useRef(new Set<string>());
  const [addPlotOpen, setAddPlotOpen] = useState(true);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);
  const [pipelineSearch, setPipelineSearch] = useState('');
  const [analysisFiltersOpen, setAnalysisFiltersOpen] = useState(true);
  const [modelTrainingDatasetFilters, setModelTrainingDatasetFilters] = useState<string[]>([]);
  const [modelPreprocessingFilters, setModelPreprocessingFilters] = useState<string[]>([]);
  const [modelMethodFilters, setModelMethodFilters] = useState<string[]>([]);
  const [modelInferenceDatasetFilters, setModelInferenceDatasetFilters] = useState<string[]>([]);
  const [modelRoiFilters, setModelRoiFilters] = useState<string[]>([]);
  const [modelMetricFilters, setModelMetricFilters] = useState<string[]>([]);
  const [modelAggregationFilters, setModelAggregationFilters] = useState<string[]>([]);
  const [analysisModelPage, setAnalysisModelPage] = useState(1);
  const [commonDatasetSearch, setCommonDatasetSearch] = useState('');
  const [commonDatasetPage, setCommonDatasetPage] = useState(1);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [selectedInferenceDatasetId, setSelectedInferenceDatasetId] = useState<string | null>(null);
  const [selectedMetricKeys, setSelectedMetricKeys] = useState<string[]>([]);
  const [selectedAggregationKeys, setSelectedAggregationKeys] = useState<string[]>([]);
  const [selectedRoiKey, setSelectedRoiKey] = useState<string | null>('none');
  const [selectedSources, setSelectedSources] = useState<PlotSourceConfig[]>([]);
  const [plotPreview, setPlotPreview] = useState<PlotPreview | null>(null);
  const [plotPreviewStale, setPlotPreviewStale] = useState(false);
  const [editingPlot, setEditingPlot] = useState<EditingPlotState>(null);
  const [preloadingPlot, setPreloadingPlot] = useState(false);
  const [detailModal, setDetailModal] = useState<DetailModalState>(null);
  const [analysisLayouts, setAnalysisLayouts] = useState<AnalysisLayout[]>([]);
  const [selectedLayoutId, setSelectedLayoutId] = useState<string | null>(null);
  const [layoutName, setLayoutName] = useState('');
  const [layoutDescription, setLayoutDescription] = useState('');
  const [layoutSaving, setLayoutSaving] = useState(false);
  const [layoutLoading, setLayoutLoading] = useState(false);
  const [layoutDeleting, setLayoutDeleting] = useState(false);
  const [timeSeriesHeatmapDraft, setTimeSeriesHeatmapDraft] = useState<TimeSeriesHeatmapDraft | null>(null);
  const [heatmapSelectionDefaults, setHeatmapSelectionDefaults] = useState<HeatmapSelectionDefaults>(() => loadHeatmapSelectionDefaults());

  async function refresh() {
    const [nextTestingRuns, nextTrainingRuns, nextDatasets, nextPipelines, nextPreprocessing, nextMethods, nextLayouts, nextHeatmaps, nextHeatmapVideos] =
      await Promise.all([
        listTestingRuns(),
        listTrainingRuns(),
        listTrainingDatasets(),
        listTrainingPipelines(),
        listPreprocessingPipelines(),
        listMethodConfigurations(),
        listAnalysisLayouts(),
        listHeatmaps(),
        listHeatmapRanges(),
      ]);
    setTestingRuns(nextTestingRuns);
    setTrainingRuns(nextTrainingRuns);
    setTrainingDatasets(nextDatasets);
    setTrainingPipelines(nextPipelines);
    setPreprocessingPipelines(nextPreprocessing);
    setMethodConfigurations(nextMethods);
    setAnalysisLayouts(nextLayouts);
    setSavedHeatmaps(nextHeatmaps);
    setSavedHeatmapVideos(nextHeatmapVideos);
  }

  const refreshHeatmapArtifacts = useCallback(async () => {
    setRefreshingHeatmapArtifacts(true);
    try {
      const [nextHeatmaps, nextVideos] = await Promise.all([listHeatmaps(), listHeatmapRanges()]);
      setSavedHeatmaps(nextHeatmaps);
      setSavedHeatmapVideos(nextVideos);
    } catch (error) {
      notifyError('Could not refresh heatmap artifacts', error);
    } finally {
      setRefreshingHeatmapArtifacts(false);
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    refresh().catch((error) => notifyError('Could not load testing runs', error));
  }, [active]);

  useEffect(() => {
    if (!active || draft.plotType !== 'heatmap' || !selectedInferenceDatasetId) return;
    const timer = window.setInterval(() => {
      if (savedHeatmapVideos.some((run) => run.status === 'queued' || run.status === 'running')) {
        refreshHeatmapArtifacts();
      }
    }, 4000);
    return () => window.clearInterval(timer);
  }, [active, draft.plotType, refreshHeatmapArtifacts, savedHeatmapVideos, selectedInferenceDatasetId]);

  const finishedRuns = useMemo(
    () => testingRuns.filter((run) => run.status === 'finished' && (run.image_count ?? 0) > 0),
    [testingRuns],
  );

  const selectedRunId = draft.testingRunId ? Number(draft.testingRunId) : null;
  const trainingRunById = useMemo(() => new Map(trainingRuns.map((run) => [run.id, run])), [trainingRuns]);
  const trainingDatasetById = useMemo(() => new Map(trainingDatasets.map((dataset) => [dataset.id, dataset])), [trainingDatasets]);
  const trainingPipelineById = useMemo(() => new Map(trainingPipelines.map((pipeline) => [pipeline.id, pipeline])), [trainingPipelines]);
  const testingRunById = useMemo(() => new Map(testingRuns.map((run) => [run.id, run])), [testingRuns]);
  const preprocessingById = useMemo(() => new Map(preprocessingPipelines.map((pipeline) => [pipeline.id, pipeline])), [preprocessingPipelines]);
  const methodById = useMemo(() => new Map(methodConfigurations.map((method) => [method.id, method])), [methodConfigurations]);
  const selectedTestingRun = selectedRunId === null ? null : testingRunById.get(selectedRunId) ?? null;
  const selectedTrainingRun = selectedTestingRun
    ? trainingRunById.get(selectedTestingRun.training_run_id) ?? null
    : null;
  const selectedRunIsStae = selectedTrainingRun?.builder_kind === 'spatiotemporal_autoencoder';
  const selectedLayout = selectedLayoutId ? analysisLayouts.find((layout) => layout.id === Number(selectedLayoutId)) ?? null : null;
  const layoutNameTrimmed = layoutName.trim();
  const layoutNameExistsForCreate = analysisLayouts.some(
    (layout) => layout.name.toLowerCase() === layoutNameTrimmed.toLowerCase(),
  );
  const layoutNameExistsForUpdate = analysisLayouts.some(
    (layout) => layout.id !== Number(selectedLayoutId) && layout.name.toLowerCase() === layoutNameTrimmed.toLowerCase(),
  );

  const allAnalysableModelRows = useMemo(() => {
    const byTrainingRun = new Map<number, { id: number; label: string; run: TrainingRun | null; testingRuns: TestingRun[] }>();
    for (const run of finishedRuns) {
      const trainingRun = trainingRunById.get(run.training_run_id) ?? null;
      const existing = byTrainingRun.get(run.training_run_id);
      if (existing) {
        existing.testingRuns.push(run);
      } else {
        byTrainingRun.set(run.training_run_id, {
          id: run.training_run_id,
          label: run.training_pipeline_name || run.training_run_name || `Training run #${run.training_run_id}`,
          run: trainingRun,
          testingRuns: [run],
        });
      }
    }
    return [...byTrainingRun.values()]
      .sort((left, right) => left.label.localeCompare(right.label));
  }, [finishedRuns, trainingRunById]);

  const analysisFacetRecords = useMemo<FacetRecord[]>(() => finishedRuns.map((run) => {
    const trainingRun = trainingRunById.get(run.training_run_id) ?? null;
    const pipeline = trainingRun ? trainingPipelineById.get(trainingRun.training_pipeline_id) ?? null : null;
    const method = pipeline ? methodById.get(pipeline.method_configuration_id) ?? null : null;
    const trainingDatasetIds = pipeline?.training_datasets.map((entry) => String(entry.training_dataset_id)) ?? [];
    const trainingDatasetNames = pipeline?.training_datasets.map((entry) => entry.name) ?? trainingRun?.dataset_names ?? [];
    return {
      id: String(run.id),
      groupId: String(run.training_run_id),
      facets: {
        trainingDataset: trainingDatasetIds,
        preprocessing: pipeline ? [String(pipeline.preprocessing_pipeline_id)] : [],
        method: pipeline ? [String(pipeline.method_configuration_id)] : [],
        inferenceDataset: [String(run.training_dataset_id)],
        roi: [roiKeyForRun(run)],
        metric: [metricKeyForRun(run)],
        aggregation: [aggregationKeyForRun(run)],
      },
      searchableValues: [
        run.name,
        run.training_run_name,
        run.training_pipeline_name,
        run.training_dataset_name,
        roiLabelForRun(run),
        run.preprocessing_pipeline_name,
        run.method_type,
        method?.name,
        metricLabel(metricKeyForRun(run)),
        aggregationLabel(aggregationKeyForRun(run)),
        ...trainingDatasetNames,
      ].filter((value): value is string => Boolean(value)),
    };
  }), [finishedRuns, methodById, trainingPipelineById, trainingRunById]);

  const analysisFacetState = useMemo<FacetFilterState>(() => ({
    query: pipelineSearch,
    selections: {
      trainingDataset: modelTrainingDatasetFilters,
      preprocessing: modelPreprocessingFilters,
      method: modelMethodFilters,
      inferenceDataset: modelInferenceDatasetFilters,
      roi: modelRoiFilters,
      metric: modelMetricFilters,
      aggregation: modelAggregationFilters,
    },
  }), [
    modelAggregationFilters,
    modelInferenceDatasetFilters,
    modelMethodFilters,
    modelMetricFilters,
    modelPreprocessingFilters,
    modelRoiFilters,
    modelTrainingDatasetFilters,
    pipelineSearch,
  ]);
  const matchingAnalysisModelIds = useMemo(
    () => matchingFacetGroupIds(analysisFacetRecords, analysisFacetState),
    [analysisFacetRecords, analysisFacetState],
  );
  const analysableModelRows = useMemo(
    () => allAnalysableModelRows.filter((row) => matchingAnalysisModelIds.has(String(row.id))),
    [allAnalysableModelRows, matchingAnalysisModelIds],
  );
  const analysisFacetCounts = useMemo(() => ({
    trainingDataset: countFacetValues(analysisFacetRecords, analysisFacetState, 'trainingDataset'),
    preprocessing: countFacetValues(analysisFacetRecords, analysisFacetState, 'preprocessing'),
    method: countFacetValues(analysisFacetRecords, analysisFacetState, 'method'),
    inferenceDataset: countFacetValues(analysisFacetRecords, analysisFacetState, 'inferenceDataset'),
    roi: countFacetValues(analysisFacetRecords, analysisFacetState, 'roi'),
    metric: countFacetValues(analysisFacetRecords, analysisFacetState, 'metric'),
    aggregation: countFacetValues(analysisFacetRecords, analysisFacetState, 'aggregation'),
  }), [analysisFacetRecords, analysisFacetState]);

  const trainingDatasetFacetOptions = useMemo(() => {
    const labels = new Map<string, string>();
    trainingPipelines.forEach((pipeline) => pipeline.training_datasets.forEach((entry) => {
      labels.set(String(entry.training_dataset_id), trainingDatasetById.get(entry.training_dataset_id)?.name ?? entry.name);
    }));
    return [...labels].map(([value, label]) => facetOption(value, label, analysisFacetCounts.trainingDataset, modelTrainingDatasetFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [analysisFacetCounts.trainingDataset, modelTrainingDatasetFilters, trainingDatasetById, trainingPipelines]);
  const preprocessingFacetOptions = useMemo(() => trainingPipelines
    .map((pipeline) => [String(pipeline.preprocessing_pipeline_id), pipeline.preprocessing_pipeline_name] as const)
    .filter(([value], index, values) => values.findIndex(([candidate]) => candidate === value) === index)
    .map(([value, label]) => facetOption(value, label, analysisFacetCounts.preprocessing, modelPreprocessingFilters))
    .sort((a, b) => a.label.localeCompare(b.label)), [analysisFacetCounts.preprocessing, modelPreprocessingFilters, trainingPipelines]);
  const methodFacetOptions = useMemo(() => trainingPipelines
    .map((pipeline) => [String(pipeline.method_configuration_id), methodById.get(pipeline.method_configuration_id)?.name ?? pipeline.method_configuration_name] as const)
    .filter(([value], index, values) => values.findIndex(([candidate]) => candidate === value) === index)
    .map(([value, label]) => facetOption(value, label, analysisFacetCounts.method, modelMethodFilters))
    .sort((a, b) => a.label.localeCompare(b.label)), [analysisFacetCounts.method, methodById, modelMethodFilters, trainingPipelines]);
  const inferenceDatasetFacetOptions = useMemo(() => {
    const labels = new Map(finishedRuns.map((run) => [String(run.training_dataset_id), trainingDatasetById.get(run.training_dataset_id)?.name ?? run.training_dataset_name]));
    return [...labels].map(([value, label]) => facetOption(value, label, analysisFacetCounts.inferenceDataset, modelInferenceDatasetFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [analysisFacetCounts.inferenceDataset, finishedRuns, modelInferenceDatasetFilters, trainingDatasetById]);
  const roiFacetOptions = useMemo(() => {
    const labels = new Map(finishedRuns.map((run) => [roiKeyForRun(run), roiLabelForRun(run)]));
    return [...labels].map(([value, label]) => facetOption(value, label, analysisFacetCounts.roi, modelRoiFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [analysisFacetCounts.roi, finishedRuns, modelRoiFilters]);
  const metricFacetOptions = useMemo(() => [...new Set(finishedRuns.map(metricKeyForRun))]
    .sort((a, b) => metricOrder(a) - metricOrder(b) || a.localeCompare(b))
    .map((value) => facetOption(value, metricLabel(value), analysisFacetCounts.metric, modelMetricFilters)), [analysisFacetCounts.metric, finishedRuns, modelMetricFilters]);
  const aggregationFacetOptions = useMemo(() => [...new Set(finishedRuns.map(aggregationKeyForRun))]
    .sort((a, b) => aggregationOrder(a) - aggregationOrder(b) || a.localeCompare(b))
    .map((value) => facetOption(value, aggregationLabel(value), analysisFacetCounts.aggregation, modelAggregationFilters)), [analysisFacetCounts.aggregation, finishedRuns, modelAggregationFilters]);
  const selectedOrFilteredModelRows = useMemo(() => {
    const included = new Set(analysableModelRows.map((row) => row.id));
    return [
      ...allAnalysableModelRows.filter((row) => selectedModelIds.includes(String(row.id)) && !included.has(row.id)),
      ...analysableModelRows,
    ];
  }, [allAnalysableModelRows, analysableModelRows, selectedModelIds]);
  const pagedAnalysableModelRows = useMemo(
    () => analysableModelRows.slice((analysisModelPage - 1) * DEFAULT_TABLE_PAGE_SIZE, analysisModelPage * DEFAULT_TABLE_PAGE_SIZE),
    [analysableModelRows, analysisModelPage],
  );

  const selectedModelIdSet = useMemo(() => new Set(selectedModelIds.map(Number)), [selectedModelIds]);
  const runsForSelectedModels = useMemo(
    () => finishedRuns.filter((run) => selectedModelIdSet.has(run.training_run_id)),
    [finishedRuns, selectedModelIdSet],
  );

  const commonDatasetOptions = useMemo(() => {
    if (selectedModelIds.length === 0) return [];
    const modelIds = selectedModelIds.map(Number);
    const sets = modelIds.map((modelId) => new Set(finishedRuns.filter((run) => run.training_run_id === modelId).map((run) => run.training_dataset_id)));
    const commonIds = [...(sets[0] ?? new Set<number>())].filter((datasetId) => sets.every((set) => set.has(datasetId)));
    return commonIds
      .map((datasetId) => {
        const dataset = trainingDatasetById.get(datasetId);
        const runs = runsForSelectedModels.filter((run) => run.training_dataset_id === datasetId);
        return {
          value: String(datasetId),
          label: dataset?.name ?? runs[0]?.training_dataset_name ?? `Inference dataset #${datasetId}`,
          dataset,
          runs,
        };
      })
      .sort((left, right) => left.label.localeCompare(right.label));
  }, [finishedRuns, runsForSelectedModels, selectedModelIds, trainingDatasetById]);
  const filteredCommonDatasetOptions = useMemo(() => {
    const query = commonDatasetSearch.trim().toLowerCase();
    if (!query) return commonDatasetOptions;
    return commonDatasetOptions.filter((option) => {
      const searchable = [option.label, option.dataset?.usage_label, ...(option.dataset?.dataset_names ?? [])];
      return searchable.some((value) => value?.toLowerCase().includes(query));
    });
  }, [commonDatasetOptions, commonDatasetSearch]);
  const pagedCommonDatasetOptions = useMemo(
    () => filteredCommonDatasetOptions.slice((commonDatasetPage - 1) * DEFAULT_TABLE_PAGE_SIZE, commonDatasetPage * DEFAULT_TABLE_PAGE_SIZE),
    [commonDatasetPage, filteredCommonDatasetOptions],
  );

  const selectedInferenceDataset = selectedInferenceDatasetId ? trainingDatasetById.get(Number(selectedInferenceDatasetId)) ?? null : null;
  const selectedInferenceBounds = sourceBounds(selectedInferenceDataset);

  const roiOptionsForSelection = useMemo(() => {
    if (!selectedInferenceDatasetId || selectedModelIds.length === 0) return [];
    const modelIds = selectedModelIds.map(Number);
    const sets = modelIds.map((modelId) => {
      const options = new Map<string, string>();
      for (const run of finishedRuns) {
        if (run.training_run_id !== modelId || run.training_dataset_id !== Number(selectedInferenceDatasetId)) continue;
        const key = run.roi_id === null ? 'none' : String(run.roi_id);
        options.set(key, run.roi_id === null ? 'No ROI' : run.roi_name ?? `ROI #${run.roi_id}`);
      }
      return options;
    });
    const commonKeys = [...(sets[0]?.keys() ?? [])].filter((key) => sets.every((set) => set.has(key)));
    return commonKeys
      .map((key) => {
        const runCount = finishedRuns.filter((run) => (
          modelIds.includes(run.training_run_id)
          && run.training_dataset_id === Number(selectedInferenceDatasetId)
          && roiKeyForRun(run) === key
        )).length;
        const label = sets[0].get(key) ?? (key === 'none' ? 'No ROI' : `ROI #${key}`);
        return { value: key, label: `${label} (${runCount} runs)` };
      })
      .sort((left, right) => (left.value === 'none' ? -1 : right.value === 'none' ? 1 : left.label.localeCompare(right.label)));
  }, [finishedRuns, selectedInferenceDatasetId, selectedModelIds]);

  const metricOptionsForSelection = useMemo(() => {
    if (!selectedInferenceDatasetId || !selectedRoiKey || selectedModelIds.length === 0) return [];
    const modelIds = selectedModelIds.map(Number);
    const sets = modelIds.map((modelId) => {
      const metrics = new Set<string>();
      for (const run of finishedRuns) {
        if (run.training_run_id !== modelId || run.training_dataset_id !== Number(selectedInferenceDatasetId)) continue;
        if ((run.roi_id === null ? 'none' : String(run.roi_id)) !== selectedRoiKey) continue;
        metrics.add(metricKeyForRun(run));
      }
      return metrics;
    });
    return [...(sets[0] ?? new Set<string>())]
      .filter((metric) => sets.every((set) => set.has(metric)))
      .sort((left, right) => metricOrder(left) - metricOrder(right) || left.localeCompare(right))
      .map((metric) => {
        const runCount = finishedRuns.filter((run) => (
          modelIds.includes(run.training_run_id)
          && run.training_dataset_id === Number(selectedInferenceDatasetId)
          && roiKeyForRun(run) === selectedRoiKey
          && metricKeyForRun(run) === metric
        )).length;
        return { value: metric, label: `${metricLabel(metric)} (${runCount} runs)` };
      });
  }, [finishedRuns, selectedInferenceDatasetId, selectedModelIds, selectedRoiKey]);

  const aggregationOptionsForSelection = useMemo(() => {
    if (!selectedInferenceDatasetId || !selectedRoiKey || selectedModelIds.length === 0) return [];
    if (selectedMetricKeys.length === 0) return [];
    const metrics = new Set(selectedMetricKeys.map(normalizeMetricKey));
    const modelIds = selectedModelIds.map(Number);
    // Nur Aggregationen anbieten, die fuer jedes Modell und jede gewaehlte Metrik als Lauf existieren.
    const sets = modelIds.flatMap((modelId) =>
      [...metrics].map((metric) => {
        const aggregations = new Set<string>();
        for (const run of finishedRuns) {
          if (run.training_run_id !== modelId || run.training_dataset_id !== Number(selectedInferenceDatasetId)) continue;
          if ((run.roi_id === null ? 'none' : String(run.roi_id)) !== selectedRoiKey) continue;
          if (metricKeyForRun(run) !== metric) continue;
          aggregations.add(aggregationKeyForRun(run));
        }
        return aggregations;
      }),
    );
    return [...(sets[0] ?? new Set<string>())]
      .filter((aggregation) => sets.every((set) => set.has(aggregation)))
      .sort((left, right) => aggregationOrder(left) - aggregationOrder(right) || left.localeCompare(right))
      .map((aggregation) => {
        const runCount = finishedRuns.filter((run) => (
          modelIds.includes(run.training_run_id)
          && run.training_dataset_id === Number(selectedInferenceDatasetId)
          && roiKeyForRun(run) === selectedRoiKey
          && metrics.has(metricKeyForRun(run))
          && aggregationKeyForRun(run) === aggregation
        )).length;
        return { value: aggregation, label: `${aggregationLabel(aggregation)} (${runCount} runs)` };
      });
  }, [finishedRuns, selectedInferenceDatasetId, selectedMetricKeys, selectedModelIds, selectedRoiKey]);

  const heatmapArtifactTestingRunIds = useMemo(() => {
    const metrics = new Set(selectedMetricKeys.map(normalizeMetricKey));
    const aggregations = new Set(selectedAggregationKeys.map(normalizeAggregationKey));
    return new Set(
      finishedRuns
        .filter((run) => selectedModelIdSet.has(run.training_run_id))
        .filter((run) => run.training_dataset_id === Number(selectedInferenceDatasetId))
        .filter((run) => (run.roi_id === null ? 'none' : String(run.roi_id)) === selectedRoiKey)
        .filter((run) => metrics.size === 0 || metrics.has(metricKeyForRun(run)))
        .filter((run) => aggregations.size === 0 || aggregations.has(aggregationKeyForRun(run)))
        .map((run) => run.id),
    );
  }, [finishedRuns, selectedAggregationKeys, selectedInferenceDatasetId, selectedMetricKeys, selectedModelIdSet, selectedRoiKey]);

  const heatmapArtifactInterval = useMemo(() => {
    const point = draft.heatmapMode === 'single' ? draft.timestamp : null;
    const rawStart = point || draft.start || selectedInferenceBounds.start;
    const rawEnd = point || draft.end || selectedInferenceBounds.end || rawStart;
    const start = rawStart ? new Date(rawStart).getTime() : Number.NEGATIVE_INFINITY;
    const end = rawEnd ? new Date(rawEnd).getTime() : Number.POSITIVE_INFINITY;
    return {
      start: Math.min(start, end),
      end: Math.max(start, end),
      label: point
        ? `timestamp ${artifactTimestamp(point)}`
        : `range ${rawStart ? artifactTimestamp(rawStart) : 'start'} – ${rawEnd ? artifactTimestamp(rawEnd) : 'end'}`,
    };
  }, [draft.end, draft.heatmapMode, draft.start, draft.timestamp, selectedInferenceBounds.end, selectedInferenceBounds.start]);

  const matchingSavedHeatmaps = useMemo(
    () => savedHeatmaps
      .filter((run) => heatmapArtifactTestingRunIds.has(run.testing_run_id))
      .filter((run) => {
        const timestamp = new Date(run.timestamp).getTime();
        return timestamp >= heatmapArtifactInterval.start && timestamp <= heatmapArtifactInterval.end;
      })
      .sort((left, right) => new Date(right.timestamp).getTime() - new Date(left.timestamp).getTime()),
    [heatmapArtifactInterval.end, heatmapArtifactInterval.start, heatmapArtifactTestingRunIds, savedHeatmaps],
  );

  const matchingSavedHeatmapVideos = useMemo(
    () => savedHeatmapVideos
      .filter((run) => heatmapArtifactTestingRunIds.has(run.testing_run_id))
      .filter((run) => {
        const start = new Date(run.start_timestamp).getTime();
        const end = new Date(run.end_timestamp).getTime();
        return start <= heatmapArtifactInterval.end && end >= heatmapArtifactInterval.start;
      })
      .sort((left, right) => new Date(right.created_at).getTime() - new Date(left.created_at).getTime()),
    [heatmapArtifactInterval.end, heatmapArtifactInterval.start, heatmapArtifactTestingRunIds, savedHeatmapVideos],
  );

  useEffect(() => {
    if (selectedInferenceDatasetId && !commonDatasetOptions.some((option) => option.value === selectedInferenceDatasetId)) {
      setSelectedInferenceDatasetId(null);
      setSelectedMetricKeys([]);
      setSelectedAggregationKeys([]);
      resetSelectionPreview();
    }
  }, [commonDatasetOptions, selectedInferenceDatasetId]);

  useEffect(() => {
    if (!roiOptionsForSelection.some((option) => option.value === selectedRoiKey)) {
      setSelectedRoiKey(null);
      setSelectedMetricKeys([]);
      setSelectedAggregationKeys([]);
      resetSelectionPreview();
    }
  }, [roiOptionsForSelection, selectedRoiKey]);

  useEffect(() => {
    const validMetrics = new Set(metricOptionsForSelection.map((option) => option.value));
    setSelectedMetricKeys((current) => {
      const filtered = current.filter((metric) => validMetrics.has(metric));
      if (filtered.length > 0) return filtered;
      if (validMetrics.has('mse')) return ['mse'];
      return metricOptionsForSelection[0]?.value ? [metricOptionsForSelection[0].value] : [];
    });
  }, [metricOptionsForSelection]);

  useEffect(() => {
    const validAggregations = new Set(aggregationOptionsForSelection.map((option) => option.value));
    setSelectedAggregationKeys((current) => {
      const filtered = current.filter((aggregation) => validAggregations.has(aggregation));
      if (filtered.length > 0) return filtered;
      if (validAggregations.has('mean')) return ['mean'];
      return aggregationOptionsForSelection[0]?.value ? [aggregationOptionsForSelection[0].value] : [];
    });
  }, [aggregationOptionsForSelection]);

  useEffect(() => setAnalysisModelPage(1), [
    pipelineSearch,
    modelTrainingDatasetFilters,
    modelPreprocessingFilters,
    modelMethodFilters,
    modelInferenceDatasetFilters,
    modelRoiFilters,
    modelMetricFilters,
    modelAggregationFilters,
  ]);
  useEffect(() => {
    setAnalysisModelPage((page) => Math.min(page, Math.max(1, Math.ceil(analysableModelRows.length / DEFAULT_TABLE_PAGE_SIZE))));
  }, [analysableModelRows.length]);
  useEffect(() => setCommonDatasetPage(1), [commonDatasetSearch, selectedModelIds]);
  useEffect(() => {
    setCommonDatasetPage((page) => Math.min(page, Math.max(1, Math.ceil(filteredCommonDatasetOptions.length / DEFAULT_TABLE_PAGE_SIZE))));
  }, [filteredCommonDatasetOptions.length]);

  const fetchResults = useCallback(
    async (runId: number) => {
      if (resultsByRunId[runId]) return resultsByRunId[runId];
      setLoadingRunId(runId);
      try {
        // Decimate large runs server-side so the page never pulls an unbounded
        // payload; first/last rows are always included for accurate bounds.
        const next = await getTestingRunResults(runId, ANALYSIS_MAX_POINTS);
        setResultsByRunId((current) => ({ ...current, [runId]: next }));
        return next;
      } finally {
        setLoadingRunId(null);
      }
    },
    [resultsByRunId],
  );

  function buildBoardLayout(): AnalysisBoardLayout {
    return {
      version: 3,
      draft,
      plots: plots.map((plot) => withAutomaticPlotMetadata(plot, testingRuns)),
      selectedPipelineId,
      selectedModelIds,
      selectedInferenceDatasetId,
      selectedMetricKeys,
      selectedAggregationKeys,
      selectedRoiKey,
      selectedSources,
      addPlotOpen,
    };
  }

  function runIdsForLayout(layout: AnalysisBoardLayout): number[] {
    const ids = new Set<number>();
    for (const source of layout.selectedSources) ids.add(Number(source.testingRunId));
    for (const plot of layout.plots) {
      for (const source of plotSources(plot)) ids.add(Number(source.testingRunId));
    }
    return [...ids].filter((id) => Number.isFinite(id));
  }

  async function loadAnalysisLayout(layoutId: number) {
    if (layoutLoading) return;
    setLayoutLoading(true);
    try {
      const saved = await getAnalysisLayout(layoutId);
      const restored = restoreBoardLayout(saved.layout);
      setSelectedLayoutId(String(saved.id));
      setLayoutName(saved.name);
      setLayoutDescription(saved.description ?? '');
      setDraft({ ...restored.draft, title: '', subtitle: '' });
      setPlots(restored.plots.map((plot) => withAutomaticPlotMetadata(plot, testingRuns)));
      setSelectedPipelineId(restored.selectedPipelineId);
      setSelectedModelIds(restored.selectedModelIds ?? []);
      setSelectedInferenceDatasetId(restored.selectedInferenceDatasetId ?? null);
      setSelectedMetricKeys(restored.selectedMetricKeys ?? []);
      setSelectedAggregationKeys(restored.selectedAggregationKeys ?? []);
      setSelectedRoiKey(restored.selectedRoiKey ?? 'none');
      setSelectedSources(restored.selectedSources);
      clearPreview();
      setAddPlotOpen(restored.addPlotOpen);
      setHeatmapCache({});
      setHeatmapErrors({});
      setLoadingHeatmaps({});
      await Promise.all(
        runIdsForLayout(restored).map((runId) =>
          fetchResults(runId).catch((error) => notifyError(`Could not load testing results for run #${runId}`, error)),
        ),
      );
      notifications.show({ color: 'green', title: 'Analysis board loaded', message: saved.name });
    } catch (error) {
      notifyError('Could not load analysis board', error);
    } finally {
      setLayoutLoading(false);
    }
  }

  async function refreshAnalysisLayouts(selectedId?: number) {
    const nextLayouts = await listAnalysisLayouts();
    setAnalysisLayouts(nextLayouts);
    if (selectedId !== undefined) setSelectedLayoutId(String(selectedId));
  }

  async function saveAnalysisLayoutAsNew() {
    if (layoutSaving) return;
    if (!layoutNameTrimmed) {
      notifications.show({ color: 'yellow', title: 'Name required', message: 'Enter a board name before saving.' });
      return;
    }
    if (layoutNameExistsForCreate) {
      notifications.show({ color: 'yellow', title: 'Name already exists', message: 'Choose a unique board name.' });
      return;
    }
    if (plots.length === 0) {
      notifications.show({ color: 'yellow', title: 'No plots to save', message: 'Add at least one plot before saving a board.' });
      return;
    }
    setLayoutSaving(true);
    try {
      const saved = await createAnalysisLayout({
        name: layoutNameTrimmed,
        description: layoutDescription.trim() || null,
        layout: buildBoardLayout() as unknown as Record<string, unknown>,
      });
      await refreshAnalysisLayouts(saved.id);
      notifications.show({ color: 'green', title: 'Analysis board saved', message: saved.name });
    } catch (error) {
      notifyError('Could not save analysis board', error);
    } finally {
      setLayoutSaving(false);
    }
  }

  async function updateSelectedAnalysisLayout() {
    if (layoutSaving || !selectedLayoutId) return;
    if (!layoutNameTrimmed) {
      notifications.show({ color: 'yellow', title: 'Name required', message: 'Enter a board name before updating.' });
      return;
    }
    if (layoutNameExistsForUpdate) {
      notifications.show({ color: 'yellow', title: 'Name already exists', message: 'Choose a unique board name.' });
      return;
    }
    if (plots.length === 0) {
      notifications.show({ color: 'yellow', title: 'No plots to save', message: 'Add at least one plot before updating a board.' });
      return;
    }
    setLayoutSaving(true);
    try {
      const saved = await updateAnalysisLayout(Number(selectedLayoutId), {
        name: layoutNameTrimmed,
        description: layoutDescription.trim() || null,
        layout: buildBoardLayout() as unknown as Record<string, unknown>,
      });
      await refreshAnalysisLayouts(saved.id);
      notifications.show({ color: 'green', title: 'Analysis board updated', message: saved.name });
    } catch (error) {
      notifyError('Could not update analysis board', error);
    } finally {
      setLayoutSaving(false);
    }
  }

  async function removeSelectedAnalysisLayout() {
    if (layoutDeleting || !selectedLayoutId) return;
    setLayoutDeleting(true);
    try {
      await deleteAnalysisLayout(Number(selectedLayoutId));
      setSelectedLayoutId(null);
      setLayoutName('');
      setLayoutDescription('');
      await refreshAnalysisLayouts();
      notifications.show({ color: 'green', title: 'Analysis board deleted', message: 'Saved board was removed.' });
    } catch (error) {
      notifyError('Could not delete analysis board', error);
    } finally {
      setLayoutDeleting(false);
    }
  }

  const ensureHeatmap = useCallback(
    async (
      frame: CombinedResult,
      config: HeatmapVisualizationConfig,
      staeView: 'reconstruction' | 'prediction',
      predictionHorizon: number,
      options?: { force?: boolean },
    ) => {
      const key = heatmapCacheKey(frame, config, staeView, predictionHorizon);
      if (heatmapRequests.current.has(key)) return;
      if (!options?.force && (heatmapCache[key] || loadingHeatmaps[key] || heatmapErrors[key])) return;
      heatmapRequests.current.add(key);
      if (options?.force) {
        setHeatmapErrors((current) => {
          const next = { ...current };
          delete next[key];
          return next;
        });
      }
      setLoadingHeatmaps((current) => ({ ...current, [key]: true }));
      try {
        const heatmap = await createHeatmap(
          frame.heatmapTimestampOnly
            ? {
                testing_run_id: frame.testingRunId,
                timestamp: frame.timestamp,
                force_recompute: options?.force ?? false,
                stae_view: staeView,
                prediction_horizon: predictionHorizon,
                visualization_config: config,
              }
            : {
                testing_run_id: frame.testingRunId,
                testing_result_id: frame.id,
                force_recompute: options?.force ?? false,
                stae_view: staeView,
                prediction_horizon: predictionHorizon,
                visualization_config: config,
              },
        );
        setHeatmapCache((current) => ({ ...current, [key]: heatmap }));
        setHeatmapErrors((current) => {
          if (!current[key]) return current;
          const next = { ...current };
          delete next[key];
          return next;
        });
      } catch (error) {
        const message = errorMessage(error);
        setHeatmapErrors((current) => ({ ...current, [key]: message }));
        notifyError('Could not compute heatmap', error);
      } finally {
        heatmapRequests.current.delete(key);
        setLoadingHeatmaps((current) => ({ ...current, [key]: false }));
      }
    },
    [heatmapCache, heatmapErrors, loadingHeatmaps],
  );

  useEffect(() => {
    if (selectedRunId === null) return;
    if (plotPreview) return;
    if (draft.plotType === 'heatmap' && draft.heatmapMode === 'single') {
      const run = testingRuns.find((item) => item.id === selectedRunId);
      const bounds = sourceBounds(run ? trainingDatasetById.get(run.training_dataset_id) : null);
      const initialize = (firstResultTimestamp?: string) =>
        setDraft((current) => {
          if (current.testingRunId !== String(selectedRunId)) return current;
          return {
            ...current,
            start: current.start || bounds.start,
            end: current.end || bounds.end,
            timestamp: current.timestamp ?? firstResultTimestamp ?? bounds.start,
          };
        });
      if (selectedRunIsStae) {
        fetchResults(selectedRunId)
          .then((data) => initialize(data.results[0]?.timestamp))
          .catch((error) => notifyError('Could not load STAE testing results', error));
      } else {
        initialize();
      }
      return;
    }
    fetchResults(selectedRunId)
      .then((data) => {
        if (data.results.length === 0) return;
        const run = testingRuns.find((item) => item.id === selectedRunId);
        const bounds = sourceBounds(run ? trainingDatasetById.get(run.training_dataset_id) : null);
        setDraft((current) => {
          if (current.testingRunId !== String(selectedRunId)) return current;
          const first = data.results[0];
          const last = data.results[data.results.length - 1];
          return {
            ...current,
            start: current.start || bounds.start || toDateTimeLocal(first.timestamp),
            end: current.end || bounds.end || toDateTimeLocal(last.timestamp),
            timestamp: current.timestamp ?? first.timestamp,
          };
        });
      })
      .catch((error) => notifyError('Could not load testing results', error));
  }, [draft.heatmapMode, draft.plotType, fetchResults, plotPreview, selectedRunId, selectedRunIsStae, testingRuns, trainingDatasetById]);

  useEffect(() => {
    if (draft.plotType === 'heatmap' && draft.heatmapMode === 'single') return;
    selectedSources.forEach((source) => {
      fetchResults(Number(source.testingRunId)).catch((error) => notifyError('Could not load testing results', error));
    });
  }, [draft.heatmapMode, draft.plotType, fetchResults, selectedSources]);

  function combinedResultsForSources(sources: PlotSourceConfig[], plotType: PlotType, heatmapMode: HeatmapMode): CombinedResult[] {
    const dedup = new Map<string, CombinedResult>();
    for (const source of sources) {
      if (plotType === 'heatmap' && heatmapMode === 'single') {
        if (!source.timestamp) continue;
        const run = testingRuns.find((item) => item.id === Number(source.testingRunId));
        const key = `${source.testingRunId}|${source.timestamp}`;
        if (!dedup.has(key)) {
          dedup.set(key, {
            id: -Number(source.testingRunId),
            position: 0,
            image_path: '',
            timestamp: source.timestamp,
            score: Number.NaN,
            full_mse: Number.NaN,
            roi_mse: null,
            tile_scores: null,
            result_metadata: null,
            width: 0,
            height: 0,
            testingRunId: Number(source.testingRunId),
            testingRunName: run?.name ?? `Testing run #${source.testingRunId}`,
            heatmapTimestampOnly: true,
          });
        }
        continue;
      }
      const data = resultsByRunId[Number(source.testingRunId)];
      if (!data) continue;
      const sourceResults =
        plotType === 'heatmap' && heatmapMode === 'single'
          ? data.results.filter((result) => result.timestamp === source.timestamp).slice(0, 1)
          : filterAndSampleResults(data.results, source.start, source.end, source.sampling);
      for (const result of sourceResults) {
        const key = `${source.testingRunId}|${result.image_path}|${result.timestamp}`;
        if (!dedup.has(key)) {
          dedup.set(key, {
            ...result,
            testingRunId: Number(source.testingRunId),
            testingRunName: data.testing_run.name,
          });
        }
      }
    }
    return [...dedup.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }

  const combinedDraftResults = useMemo(
    () => combinedResultsForSources(selectedSources, draft.plotType, draft.heatmapMode),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [draft.heatmapMode, draft.plotType, resultsByRunId, selectedSources, testingRuns],
  );

  const previewPlot = plotPreview?.plot ?? null;
  const previewResults = useMemo(
    () => (previewPlot ? combinedResultsForSources(plotSources(previewPlot), previewPlot.plotType, previewPlot.heatmapMode) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [plotPreview, resultsByRunId, testingRuns],
  );

  const plotResultsById = useMemo(() => {
    const next = new Map<string, { hasAllData: boolean; results: CombinedResult[] }>();
    for (const plot of plots) {
      const sources = plotSources(plot);
      const hasAllData =
        plot.plotType === 'heatmap' && plot.heatmapMode === 'single'
          ? true
          : sources.every((source) => resultsByRunId[Number(source.testingRunId)]);
      next.set(plot.id, {
        hasAllData,
        results: hasAllData ? combinedResultsForSources(sources, plot.plotType, plot.heatmapMode) : [],
      });
    }
    return next;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plots, resultsByRunId, testingRuns]);

  const heatmapDerivationPlot = timeSeriesHeatmapDraft
    ? plots.find((plot) => plot.id === timeSeriesHeatmapDraft.sourcePlotId) ?? null
    : null;
  const heatmapDerivationResults = timeSeriesHeatmapDraft
    ? plotResultsById.get(timeSeriesHeatmapDraft.sourcePlotId)?.results ?? []
    : [];
  const heatmapDerivationSources = useMemo(() => {
    if (!heatmapDerivationPlot) return [];
    const options = new Map<string, string>();
    for (const trace of heatmapDerivationPlot.traces ?? []) {
      options.set(trace.testingRunId, trace.legendLabel || trace.modelLabel);
    }
    for (const source of plotSources(heatmapDerivationPlot)) {
      const run = testingRuns.find((item) => item.id === Number(source.testingRunId));
      if (!options.has(source.testingRunId)) options.set(source.testingRunId, run?.name ?? `Testing run #${source.testingRunId}`);
    }
    return [...options.entries()].map(([value, label]) => ({ value, label }));
  }, [heatmapDerivationPlot, testingRuns]);

  const effectiveHeatmapDerivationResults = useMemo(() => {
    if (!timeSeriesHeatmapDraft) return [];
    const sourceResults = heatmapDerivationResults
      .filter((result) => String(result.testingRunId) === timeSeriesHeatmapDraft.testingRunId)
      .sort((left, right) => new Date(left.timestamp).getTime() - new Date(right.timestamp).getTime());
    if (sourceResults.length === 0) return [];
    const start = new Date(timeSeriesHeatmapDraft.start).getTime();
    const end = new Date(timeSeriesHeatmapDraft.end).getTime();
    if (timeSeriesHeatmapDraft.mode === 'single') {
      return [sourceResults.reduce((closest, result) =>
        Math.abs(new Date(result.timestamp).getTime() - start) < Math.abs(new Date(closest.timestamp).getTime() - start) ? result : closest,
      )];
    }
    return sourceResults.filter((result) => {
      const timestamp = new Date(result.timestamp).getTime();
      return timestamp >= Math.min(start, end) && timestamp <= Math.max(start, end);
    });
  }, [heatmapDerivationResults, timeSeriesHeatmapDraft]);

  const heatmapDerivationError = !timeSeriesHeatmapDraft
    ? null
    : !heatmapDerivationPlot
      ? 'The source time-series plot no longer exists.'
      : !timeSeriesHeatmapDraft.testingRunId
        ? 'Select one inference source.'
        : effectiveHeatmapDerivationResults.length === 0
          ? 'The selected inference source has no frames in this time selection.'
          : null;
  const heatmapDerivationVisibilityMode = timeSeriesHeatmapDraft?.heatmapConfig.fixed_ceiling_enabled
    ? 'fixed_ceiling'
    : timeSeriesHeatmapDraft?.heatmapConfig.max_clip_enabled
      ? 'max_clip'
      : 'opacity';

  function beginHeatmapDerivation(plot: AnalysisPlot, results: CombinedResult[], mode: HeatmapMode, rawStart: string, rawEnd: string) {
    const startMs = new Date(rawStart).getTime();
    const endMs = new Date(rawEnd).getTime();
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
      notifications.show({ color: 'red', title: 'Invalid time selection', message: 'The selected Plotly timestamps could not be read.' });
      return;
    }
    const lower = Math.min(startMs, endMs);
    const upper = Math.max(startMs, endMs);
    const candidates = results.filter((result) => {
      const timestamp = new Date(result.timestamp).getTime();
      return Number.isFinite(timestamp) && (mode === 'single' || (timestamp >= lower && timestamp <= upper));
    });
    if (candidates.length === 0) {
      notifications.show({ color: 'yellow', title: 'No frames selected', message: 'Select a data point or a range containing at least one visible frame.' });
      return;
    }
    const sorted = [...candidates].sort((left, right) => new Date(left.timestamp).getTime() - new Date(right.timestamp).getTime());
    const selectedPoint = mode === 'single'
      ? sorted.reduce((closest, result) =>
          Math.abs(new Date(result.timestamp).getTime() - startMs) < Math.abs(new Date(closest.timestamp).getTime() - startMs) ? result : closest,
        )
      : null;
    const firstSource = plotSources(plot)[0]?.testingRunId ?? '';
    const normalizedStart = selectedPoint?.timestamp ?? sorted[0].timestamp;
    const normalizedEnd = selectedPoint?.timestamp ?? sorted[sorted.length - 1].timestamp;
    setTimeSeriesHeatmapDraft({
      sourcePlotId: plot.id,
      mode,
      start: normalizedStart,
      end: normalizedEnd,
      testingRunId: firstSource,
      sampling: heatmapSelectionDefaults.sampling,
      fps: heatmapSelectionDefaults.fps,
      scaleMode: heatmapSelectionDefaults.scaleMode,
      heatmapConfig: { ...heatmapSelectionDefaults.heatmapConfig },
      includeReference: heatmapSelectionDefaults.includeReference,
      displaySize: heatmapSelectionDefaults.displaySize,
      advancedOpen: false,
    });
  }

  function patchDerivedHeatmapConfig(patch: Partial<HeatmapVisualizationConfig>) {
    setTimeSeriesHeatmapDraft((current) => current ? { ...current, heatmapConfig: { ...current.heatmapConfig, ...patch } } : current);
  }

  function storeCurrentHeatmapSelectionDefaults() {
    if (!timeSeriesHeatmapDraft) return;
    const nextDefaults: HeatmapSelectionDefaults = {
      sampling: Math.max(1, Math.floor(timeSeriesHeatmapDraft.sampling)),
      fps: Math.max(1, Math.min(60, Math.floor(timeSeriesHeatmapDraft.fps))),
      scaleMode: timeSeriesHeatmapDraft.scaleMode,
      includeReference: timeSeriesHeatmapDraft.includeReference,
      displaySize: Math.max(240, Math.min(1200, Math.floor(timeSeriesHeatmapDraft.displaySize))),
      heatmapConfig: { ...timeSeriesHeatmapDraft.heatmapConfig },
    };
    saveHeatmapSelectionDefaults(nextDefaults);
    setHeatmapSelectionDefaults(nextDefaults);
    notifications.show({ color: 'green', title: 'Heatmap defaults saved', message: 'These settings will be used for every new time-series heatmap selection.' });
  }

  function finishHeatmapDerivation() {
    if (!timeSeriesHeatmapDraft || !heatmapDerivationPlot || heatmapDerivationError) return;
    const selectedResults = effectiveHeatmapDerivationResults;
    const effectiveStart = selectedResults[0].timestamp;
    const effectiveEnd = selectedResults[selectedResults.length - 1].timestamp;
    const selectedTrace = heatmapDerivationPlot.traces?.find((trace) => trace.testingRunId === timeSeriesHeatmapDraft.testingRunId);
    const source: PlotSourceConfig = {
      testingRunId: timeSeriesHeatmapDraft.testingRunId,
      start: effectiveStart,
      end: effectiveEnd,
      sampling: timeSeriesHeatmapDraft.mode === 'range' ? Math.max(1, Math.floor(timeSeriesHeatmapDraft.sampling)) : 1,
      timestamp: timeSeriesHeatmapDraft.mode === 'single' ? effectiveStart : null,
    };
    const derivedPlot = withAutomaticPlotMetadata({
      ...defaultDraft(),
      id: crypto.randomUUID(),
      plotType: 'heatmap',
      heatmapMode: timeSeriesHeatmapDraft.mode,
      title: '',
      subtitle: '',
      testingRunId: timeSeriesHeatmapDraft.testingRunId,
      start: effectiveStart,
      end: effectiveEnd,
      timestamp: source.timestamp,
      sampling: source.sampling,
      includeReference: timeSeriesHeatmapDraft.includeReference,
      heatmapConfig: { ...timeSeriesHeatmapDraft.heatmapConfig },
      sources: [source],
      traces: selectedTrace ? [{ ...selectedTrace, ...source }] : undefined,
      derivedFromPlotId: heatmapDerivationPlot.id,
      autoStartHeatmap: timeSeriesHeatmapDraft.mode === 'range',
      heatmapFps: timeSeriesHeatmapDraft.fps,
      heatmapScaleMode: timeSeriesHeatmapDraft.scaleMode,
      heatmapDisplaySize: timeSeriesHeatmapDraft.displaySize,
    }, testingRuns);
    setPlots((current) => {
      const sourceIndex = current.findIndex((plot) => plot.id === heatmapDerivationPlot.id);
      const lastDerivedIndex = current.reduce(
        (last, plot, index) => plot.derivedFromPlotId === heatmapDerivationPlot.id ? Math.max(last, index) : last,
        sourceIndex,
      );
      const next = [...current];
      next.splice(Math.max(0, lastDerivedIndex + 1), 0, derivedPlot);
      return next;
    });
    setTimeSeriesHeatmapDraft(null);
    notifications.show({
      color: 'green',
      title: timeSeriesHeatmapDraft.mode === 'single' ? 'Heatmap added' : 'Heatmap video queued',
      message: 'The new block was inserted directly below its source time series.',
    });
  }

  function resolveTestingRun(modelId: number, metric: string, aggregation: string): { run: TestingRun | null; duplicateCount: number } {
    if (!selectedInferenceDatasetId || !selectedRoiKey) return { run: null, duplicateCount: 0 };
    const candidates = finishedRuns
      .filter((run) => run.training_run_id === modelId)
      .filter((run) => run.training_dataset_id === Number(selectedInferenceDatasetId))
      .filter((run) => (run.roi_id === null ? 'none' : String(run.roi_id)) === selectedRoiKey)
      .filter((run) => metricKeyForRun(run) === metric)
      .filter((run) => aggregationKeyForRun(run) === aggregation)
      .sort((left, right) => {
        const leftTime = new Date(left.ended_at ?? left.updated_at ?? left.created_at).getTime();
        const rightTime = new Date(right.ended_at ?? right.updated_at ?? right.created_at).getTime();
        return rightTime - leftTime;
      });
    return { run: candidates[0] ?? null, duplicateCount: Math.max(0, candidates.length - 1) };
  }

  function selectedModelLabel(modelId: string): string {
    return allAnalysableModelRows.find((row) => row.id === Number(modelId))?.label ?? `Training run #${modelId}`;
  }

  function markPreviewStale() {
    setPlotPreviewStale((current) => current || plotPreview !== null);
  }

  function clearPreview() {
    setPlotPreview(null);
    setPlotPreviewStale(false);
  }

  function resetSelectionPreview() {
    if (editingPlot) cancelPlotEdit();
    else clearPreview();
  }

  async function preloadPlot() {
    if (preloadingPlot) return;
    if (selectedModelIds.length === 0) {
      notifications.show({ color: 'yellow', title: 'Select models', message: 'Select one or more trained models first.' });
      return;
    }
    if (!selectedInferenceDatasetId) {
      notifications.show({ color: 'yellow', title: 'Select inference dataset', message: 'Select a shared inference dataset.' });
      return;
    }
    if (!selectedRoiKey) {
      notifications.show({ color: 'yellow', title: 'Select ROI', message: 'Select No ROI or a shared ROI.' });
      return;
    }
    if (selectedMetricKeys.length === 0) {
      notifications.show({ color: 'yellow', title: 'Select metrics', message: 'Select one or more metrics for the plot.' });
      return;
    }
    if (selectedAggregationKeys.length === 0) {
      notifications.show({ color: 'yellow', title: 'Select aggregation', message: 'Select one or more score aggregations for the plot.' });
      return;
    }
    if (draft.plotType === 'heatmap' && (selectedModelIds.length > 1 || selectedMetricKeys.length > 1 || selectedAggregationKeys.length > 1)) {
      notifications.show({ color: 'yellow', title: 'Heatmap needs one source', message: 'Heatmaps currently support one model, one metric and one aggregation. Use time series for multi-source comparisons.' });
      return;
    }
    const start = draft.start || selectedInferenceBounds.start;
    const end = draft.end || selectedInferenceBounds.end;
    if (!start || !end) {
      notifications.show({ color: 'yellow', title: 'Missing time bounds', message: 'The selected inference dataset has no start/end bounds.' });
      return;
    }
    const sampling = Math.max(1, Math.floor(draft.sampling));
    const duplicateNotes: string[] = [];
    const traces: PlotTraceConfig[] = [];
    setPreloadingPlot(true);
    try {
      for (const modelId of selectedModelIds) {
        for (const metric of selectedMetricKeys) {
          for (const aggregation of selectedAggregationKeys) {
            const { run, duplicateCount } = resolveTestingRun(Number(modelId), metric, aggregation);
            if (!run) {
              throw new Error(
                `No finished ${metricLabel(metric)} / ${aggregationLabel(aggregation)} inference run found for ${selectedModelLabel(modelId)}.`,
              );
            }
            if (duplicateCount > 0) {
              duplicateNotes.push(`${selectedModelLabel(modelId)} / ${metricLabel(metric)} / ${aggregationLabel(aggregation)}: newest run used, ${duplicateCount} older duplicate${duplicateCount === 1 ? '' : 's'} ignored.`);
            }
            const previousTrace = plotPreview?.traces.find(
              (trace) => trace.testingRunId === String(run.id) && trace.metric === metric && (trace.aggregation ?? 'mean') === aggregation,
            );
            traces.push({
              testingRunId: String(run.id),
              metric,
              aggregation,
              modelLabel: selectedModelLabel(modelId),
              legendLabel:
                previousTrace?.legendLabel
                ?? traceLabelForRun(run, metric, selectedMetricKeys.length > 1, aggregation, selectedAggregationKeys.length > 1),
              color: previousTrace?.color ?? TRACE_COLORS[traces.length % TRACE_COLORS.length],
              start,
              end,
              sampling,
              timestamp: draft.timestamp ?? start,
            });
          }
        }
      }
      if (draft.plotType !== 'heatmap' || draft.heatmapMode !== 'single') {
        await Promise.all(traces.map((trace) => fetchResults(Number(trace.testingRunId))));
      }
      const previewPlot = withAutomaticPlotMetadata({
        ...draft,
        id: editingPlot?.plot.id ?? 'preview',
        title: '',
        subtitle: '',
        sources: traces.map(traceToSource),
        traces,
        testingRunId: traces[0]?.testingRunId ?? draft.testingRunId,
        start,
        end,
        sampling,
        timestamp: draft.timestamp ?? traces[0]?.timestamp ?? start,
        scoreSeries: 'score',
        baselineAnalysis: editingPlot?.plot.baselineAnalysis,
        imageComparison: editingPlot?.plot.imageComparison,
      }, testingRuns);
      setDraft((current) => ({
        ...current,
        title: previewPlot.title,
        subtitle: previewPlot.subtitle,
        testingRunId: previewPlot.testingRunId,
        start,
        end,
        sampling,
        timestamp: previewPlot.timestamp,
        scoreSeries: previewPlot.scoreSeries,
      }));
      setSelectedSources(traces.map(traceToSource));
      setPlotPreview({
        title: previewPlot.title,
        subtitle: previewPlot.subtitle,
        traces,
        duplicateNotes,
        plot: previewPlot,
      });
      setPlotPreviewStale(false);
    } catch (error) {
      notifyError('Could not preload plot', error);
    } finally {
      setPreloadingPlot(false);
    }
  }

  function updatePreviewTrace(index: number, patch: Partial<PlotTraceConfig>) {
    setPlotPreview((current) => {
      if (!current) return current;
      const traces = current.traces.map((trace, traceIndex) => (traceIndex === index ? { ...trace, ...patch } : trace));
      const plot = withAutomaticPlotMetadata({
        ...current.plot,
        sources: traces.map(traceToSource),
        traces,
      }, testingRuns);
      return {
        ...current,
        title: plot.title,
        subtitle: plot.subtitle,
        traces,
        plot,
      };
    });
  }

  function finishPlot() {
    if (!plotPreview) {
      notifications.show({ color: 'yellow', title: 'Preload required', message: 'Preload the plot before adding it to the board.' });
      return;
    }
    if (plotPreviewStale) {
      notifications.show({ color: 'yellow', title: 'Preview is stale', message: 'Update the preview before finishing this plot.' });
      return;
    }
    const availableResults = combinedResultsForSources(plotPreview.traces.map(traceToSource), draft.plotType, draft.heatmapMode);
    if (draft.plotType !== 'heatmap' || draft.heatmapMode !== 'single') {
      if (availableResults.length === 0) {
        notifications.show({ color: 'yellow', title: 'No matching results', message: 'Adjust time range, sampling or metric selection.' });
        return;
      }
    }
    const nextPlot = withAutomaticPlotMetadata({
      ...plotPreview.plot,
      id: editingPlot?.plot.id ?? crypto.randomUUID(),
      timestamp: plotPreview.plot.timestamp ?? availableResults[0]?.timestamp ?? plotPreview.traces[0]?.timestamp ?? null,
      baselineAnalysis: editingPlot?.plot.baselineAnalysis
        ? { ...editingPlot.plot.baselineAnalysis, stale: true }
        : undefined,
      imageComparison: editingPlot?.plot.imageComparison
        ? { ...editingPlot.plot.imageComparison, stale: true }
        : undefined,
    }, testingRuns);
    setPlots((current) => {
      if (!editingPlot) return [...current, nextPlot];
      const next = [...current];
      next.splice(Math.min(editingPlot.index, next.length), 0, nextPlot);
      return next;
    });
    setEditingPlot(null);
    setDraft(plotDraftFrom(nextPlot));
    clearPreview();
  }

  function cancelPlotEdit() {
    if (editingPlot) {
      setPlots((current) => {
        const next = [...current];
        next.splice(Math.min(editingPlot.index, next.length), 0, editingPlot.plot);
        return next;
      });
    }
    setEditingPlot(null);
    clearPreview();
  }

  function editPlot(plot: AnalysisPlot, index: number) {
    const traces = plot.traces ?? plot.sources.map((source, sourceIndex) => {
      const run = testingRuns.find((item) => item.id === Number(source.testingRunId));
      const metric = run ? metricKeyForRun(run) : 'mse';
      return {
        ...source,
        metric,
        aggregation: run ? aggregationKeyForRun(run) : 'mean',
        modelLabel: run?.training_pipeline_name ?? `Source ${sourceIndex + 1}`,
        legendLabel: run?.training_pipeline_name ?? `Source ${sourceIndex + 1}`,
        color: TRACE_COLORS[sourceIndex % TRACE_COLORS.length],
      };
    });
    const canonicalPlot = withAutomaticPlotMetadata({ ...plot, sources: traces.map(traceToSource), traces }, testingRuns);
    const firstRun = testingRuns.find((run) => run.id === Number(traces[0]?.testingRunId));
    setEditingPlot({ plot: canonicalPlot, index });
    setPlots((current) => current.filter((item) => item.id !== plot.id));
    setDraft(plotDraftFrom(canonicalPlot));
    setSelectedSources(traces.map(traceToSource));
    setSelectedModelIds([...new Set(traces.map((trace) => testingRuns.find((run) => run.id === Number(trace.testingRunId))?.training_run_id).filter((id): id is number => typeof id === 'number').map(String))]);
    setSelectedInferenceDatasetId(firstRun ? String(firstRun.training_dataset_id) : null);
    setSelectedRoiKey(firstRun?.roi_id === null || firstRun?.roi_id === undefined ? 'none' : String(firstRun.roi_id));
    setSelectedMetricKeys([...new Set(traces.map((trace) => trace.metric))]);
    setSelectedAggregationKeys([...new Set(traces.map((trace) => trace.aggregation ?? 'mean'))]);
    setPlotPreview({
      title: canonicalPlot.title,
      subtitle: canonicalPlot.subtitle,
      traces,
      duplicateNotes: [],
      plot: canonicalPlot,
    });
    setPlotPreviewStale(false);
    setAddPlotOpen(true);
  }

  function movePlot(plotId: string, direction: -1 | 1) {
    setPlots((current) => {
      const index = current.findIndex((plot) => plot.id === plotId);
      const nextIndex = index + direction;
      if (index < 0 || nextIndex < 0 || nextIndex >= current.length) return current;
      const next = [...current];
      [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
      return next;
    });
  }

  function detailObjectsForRun(run: TestingRun) {
    const trainingRun = trainingRunById.get(run.training_run_id) ?? null;
    const trainingPipeline = trainingRun ? trainingPipelineById.get(trainingRun.training_pipeline_id) ?? null : null;
    const trainset = trainingDatasetById.get(run.training_dataset_id) ?? null;
    const preprocessing = trainingPipeline ? preprocessingById.get(trainingPipeline.preprocessing_pipeline_id) ?? null : null;
    const method = trainingPipeline ? methodById.get(trainingPipeline.method_configuration_id) ?? null : null;
    return { trainingRun, trainingPipeline, trainset, preprocessing, method };
  }

  function updateHeatmapConfig(patch: Partial<HeatmapVisualizationConfig>) {
    setDraft((current) => ({
      ...current,
      heatmapConfig: { ...current.heatmapConfig, ...patch },
    }));
    markPreviewStale();
  }

  function updateAnalyticsConfig(index: number, patch: Partial<AnalyticsMethodConfig>) {
    setDraft((current) => ({
      ...current,
      timeseriesAnalytics: current.timeseriesAnalytics.map((config, configIndex) =>
        configIndex === index
          ? { ...config, ...patch, params: { ...config.params, ...(patch.params ?? {}) } }
          : config,
      ),
    }));
    markPreviewStale();
  }

  function renderAnalyticsParamInput(config: AnalyticsMethodConfig, index: number, key: string, value: number | string | boolean) {
    if (key === 'windowSamples' && config.params.windowMode === 'minutes') return null;
    if (key === 'windowMinutes' && config.params.windowMode !== 'minutes') return null;
    if (key === 'baselineWindowSamples' && config.params.windowMode === 'minutes') return null;
    if (key === 'baselineWindowMinutes' && config.params.windowMode !== 'minutes') return null;
    if (key === 'longWindowSamples' && config.params.windowMode === 'minutes') return null;
    if (key === 'longWindowMinutes' && config.params.windowMode !== 'minutes') return null;
    if (key === 'windowMode') {
      return (
        <Select
          key={key}
          label={<InfoLabel label={analyticsParamLabel(key)} info={analyticsParamInfo(key)} />}
          data={[
            { value: 'samples', label: 'Samples' },
            { value: 'minutes', label: 'Minutes' },
          ]}
          value={String(value)}
          onChange={(nextValue) => updateAnalyticsConfig(index, { params: { [key]: nextValue ?? 'samples' } })}
        />
      );
    }
    if (key === 'source') {
      return (
        <Select
          key={key}
          label={<InfoLabel label={analyticsParamLabel(key)} info={analyticsParamInfo(key)} />}
          data={[
            { value: 'raw', label: 'Raw score' },
            { value: 'smoothed', label: 'EWMA smoothed' },
          ]}
          value={String(value)}
          onChange={(nextValue) => updateAnalyticsConfig(index, { params: { [key]: nextValue ?? 'smoothed' } })}
        />
      );
    }
    if (key === 'mode') {
      return (
        <Select
          key={key}
          label={<InfoLabel label={analyticsParamLabel(key)} info={analyticsParamInfo(key)} />}
          data={[
            { value: 'relative', label: 'Relative' },
            { value: 'absolute', label: 'Absolute' },
          ]}
          value={String(value)}
          onChange={(nextValue) => updateAnalyticsConfig(index, { params: { [key]: nextValue ?? 'relative' } })}
        />
      );
    }
    if (typeof value === 'boolean') {
      return (
        <Switch
          key={key}
          label={<InfoLabel label={analyticsParamLabel(key)} info={analyticsParamInfo(key)} />}
          checked={value}
          onChange={(event) => updateAnalyticsConfig(index, { params: { [key]: event.currentTarget.checked } })}
        />
      );
    }
    if (typeof value === 'number') {
      const isInteger = key.toLowerCase().includes('samples') || key.toLowerCase().includes('window') || key === 'h' || key === 'hLow' || key === 'hHigh';
      return (
        <NumberInput
          key={key}
          label={<InfoLabel label={analyticsParamLabel(key)} info={analyticsParamInfo(key)} />}
          value={value}
          min={key === 'epsilon' ? 0 : undefined}
          step={key === 'epsilon' ? 1e-12 : isInteger ? 1 : 0.1}
          decimalScale={key === 'epsilon' ? 14 : undefined}
          onChange={(nextValue) => updateAnalyticsConfig(index, { params: { [key]: valueAsNumber(nextValue, value) } })}
        />
      );
    }
    return null;
  }

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={2}>Model Analysis</Title>
          <Text c="dimmed" size="sm">
            Compose interactive plots from saved inference runs and compare them on one board.
          </Text>
        </div>
        <Button
          variant="default"
          leftSection={<RotateCcw size={18} />}
          disabled={plots.length === 0}
          onClick={() => setPlots([])}
        >
          Reset board
        </Button>
      </Group>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Group justify="space-between" align="flex-start" wrap="wrap">
            <div>
              <Text fw={700}>Saved boards</Text>
              <Text c="dimmed" size="sm">
                Save the current plot board and restore it later with the same selected runs, ranges and plot settings.
              </Text>
            </div>
            <Button
              variant="default"
              leftSection={<Plus size={16} />}
              onClick={() => {
                setSelectedLayoutId(null);
                setLayoutName('');
                setLayoutDescription('');
              }}
            >
              New board name
            </Button>
          </Group>
          <SimpleGrid cols={{ base: 1, md: 3 }} spacing="md">
            <Select
              label="Saved board"
              placeholder="Select saved board"
              data={analysisLayouts.map((layout) => ({ value: String(layout.id), label: layout.name }))}
              value={selectedLayoutId}
              onChange={(value) => {
                setSelectedLayoutId(value);
                const layout = value ? analysisLayouts.find((item) => item.id === Number(value)) : null;
                setLayoutName(layout?.name ?? '');
                setLayoutDescription(layout?.description ?? '');
              }}
              searchable
              clearable
            />
            <TextInput
              label="Board name"
              placeholder="e.g. AE baseline comparison"
              value={layoutName}
              onChange={(event) => setLayoutName(event.currentTarget.value)}
              error={
                layoutNameTrimmed && !selectedLayoutId && layoutNameExistsForCreate
                  ? 'Name already exists'
                  : selectedLayoutId && layoutNameExistsForUpdate
                    ? 'Name already exists'
                    : null
              }
            />
            <TextInput
              label="Description"
              placeholder="Optional note"
              value={layoutDescription}
              onChange={(event) => setLayoutDescription(event.currentTarget.value)}
            />
          </SimpleGrid>
          <Group justify="space-between" align="center" wrap="wrap">
            <Text size="sm" c="dimmed">
              {selectedLayout
                ? `Selected: ${selectedLayout.name}`
                : `${analysisLayouts.length} saved board${analysisLayouts.length === 1 ? '' : 's'}`}
            </Text>
            <Group gap="sm">
              <Button
                variant="default"
                leftSection={<Upload size={16} />}
                disabled={!selectedLayoutId}
                loading={layoutLoading}
                onClick={() => selectedLayoutId && loadAnalysisLayout(Number(selectedLayoutId))}
              >
                Load
              </Button>
              <Button
                leftSection={<Save size={16} />}
                disabled={!layoutNameTrimmed || layoutNameExistsForCreate || plots.length === 0}
                loading={layoutSaving}
                onClick={saveAnalysisLayoutAsNew}
              >
                Save as new
              </Button>
              <Button
                variant="light"
                disabled={!selectedLayoutId || !layoutNameTrimmed || layoutNameExistsForUpdate || plots.length === 0}
                loading={layoutSaving}
                onClick={updateSelectedAnalysisLayout}
              >
                Update
              </Button>
              <Button
                variant="subtle"
                color="red"
                leftSection={<Trash2 size={16} />}
                disabled={!selectedLayoutId}
                loading={layoutDeleting}
                onClick={removeSelectedAnalysisLayout}
              >
                Delete
              </Button>
            </Group>
          </Group>
        </Stack>
      </Paper>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Group justify="space-between" align="center">
            <Button
              variant="subtle"
              leftSection={addPlotOpen ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
              onClick={() => setAddPlotOpen((open) => !open)}
            >
              Add plot
            </Button>
            {selectedRunId !== null && loadingRunId === selectedRunId && <Loader size="sm" />}
          </Group>
          <Collapse in={addPlotOpen}>
            <Stack gap="md">
              <StepCard index={1} title="Trained models" color="blue">
                <Stack gap="sm">
                  <Group justify="space-between" wrap="wrap">
                    <Group gap="xs">
                      <Button
                        variant="subtle"
                        size="compact-sm"
                        leftSection={<SlidersHorizontal size={16} />}
                        rightSection={analysisFiltersOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                        onClick={() => setAnalysisFiltersOpen((open) => !open)}
                      >
                        Filters
                      </Button>
                      <Badge variant="light">{analysableModelRows.length} matching model{analysableModelRows.length === 1 ? '' : 's'}</Badge>
                    </Group>
                    {(pipelineSearch.trim() || modelTrainingDatasetFilters.length || modelPreprocessingFilters.length || modelMethodFilters.length || modelInferenceDatasetFilters.length || modelRoiFilters.length || modelMetricFilters.length || modelAggregationFilters.length) ? (
                      <Button variant="subtle" color="gray" size="compact-sm" onClick={() => {
                        setPipelineSearch('');
                        setModelTrainingDatasetFilters([]);
                        setModelPreprocessingFilters([]);
                        setModelMethodFilters([]);
                        setModelInferenceDatasetFilters([]);
                        setModelRoiFilters([]);
                        setModelMetricFilters([]);
                        setModelAggregationFilters([]);
                      }}>Reset filters</Button>
                    ) : null}
                  </Group>
                  <Collapse in={analysisFiltersOpen}>
                    <Stack gap="sm">
                      <TextInput
                        placeholder="Search models, pipelines and inference runs"
                        leftSection={<Search size={16} />}
                        value={pipelineSearch}
                        onChange={(event) => setPipelineSearch(event.currentTarget.value)}
                      />
                      <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
                        <MultiSelect label="Training datasets" data={trainingDatasetFacetOptions} value={modelTrainingDatasetFilters} onChange={setModelTrainingDatasetFilters} searchable clearable />
                        <MultiSelect label="Preprocessing" data={preprocessingFacetOptions} value={modelPreprocessingFilters} onChange={setModelPreprocessingFilters} searchable clearable />
                        <MultiSelect label="Methods" data={methodFacetOptions} value={modelMethodFilters} onChange={setModelMethodFilters} searchable clearable />
                        <MultiSelect label="Inference datasets" data={inferenceDatasetFacetOptions} value={modelInferenceDatasetFilters} onChange={setModelInferenceDatasetFilters} searchable clearable />
                        <MultiSelect label="ROI" data={roiFacetOptions} value={modelRoiFilters} onChange={setModelRoiFilters} searchable clearable />
                        <MultiSelect label="Metrics" data={metricFacetOptions} value={modelMetricFilters} onChange={setModelMetricFilters} searchable clearable />
                        <MultiSelect label="Score aggregation" data={aggregationFacetOptions} value={modelAggregationFilters} onChange={setModelAggregationFilters} searchable clearable />
                      </SimpleGrid>
                      <Text size="xs" c="dimmed">Counts show unique models that would remain after the search and all other filter categories. Values inside one category use OR.</Text>
                    </Stack>
                  </Collapse>
                  <MultiSelect
                    label="Selected models"
                    placeholder="Choose one or more trained models"
                    data={selectedOrFilteredModelRows.map((row) => ({ value: String(row.id), label: row.label }))}
                    value={selectedModelIds}
                    searchable
                    clearable
                    onChange={(values) => {
                      setSelectedModelIds(values);
                      setSelectedSources([]);
                      resetSelectionPreview();
                    }}
                  />
                  <ScrollArea h={220}>
                    <Table striped highlightOnHover verticalSpacing="sm" miw={900}>
                      <Table.Thead>
                        <Table.Tr>
                          <Table.Th>Model</Table.Th>
                          <Table.Th>Inference datasets</Table.Th>
                          <Table.Th>Metrics</Table.Th>
                          <Table.Th>Finished runs</Table.Th>
                          <Table.Th />
                        </Table.Tr>
                      </Table.Thead>
                      <Table.Tbody>
                        {pagedAnalysableModelRows.map((row) => {
                          const selected = selectedModelIds.includes(String(row.id));
                          const metrics = [...new Set(row.testingRuns.map(metricKeyForRun))].sort((left, right) => metricOrder(left) - metricOrder(right));
                          const datasets = [...new Set(row.testingRuns.map((run) => run.training_dataset_name))];
                          return (
                            <Table.Tr key={row.id} className={selected ? 'analysis-selected-row' : undefined}>
                              <Table.Td>{row.label}</Table.Td>
                              <Table.Td>{datasets.length}</Table.Td>
                              <Table.Td>
                                <Group gap={4}>
                                  {metrics.map((metric) => (
                                    <Badge key={metric} size="xs" variant="light" color="blue">
                                      {metricLabel(metric)}
                                    </Badge>
                                  ))}
                                </Group>
                              </Table.Td>
                              <Table.Td>{row.testingRuns.length}</Table.Td>
                              <Table.Td>
                                <Group justify="flex-end">
                                  <Button
                                    size="compact-sm"
                                    variant={selected ? 'filled' : 'light'}
                                    color={selected ? 'green' : 'blue'}
                                    leftSection={<Check size={14} />}
                                    onClick={() => {
                                      setSelectedModelIds((current) =>
                                        current.includes(String(row.id))
                                          ? current.filter((id) => id !== String(row.id))
                                          : [...current, String(row.id)],
                                      );
                                      setSelectedSources([]);
                                      resetSelectionPreview();
                                    }}
                                  >
                                    {selected ? 'Selected' : 'Use'}
                                  </Button>
                                </Group>
                              </Table.Td>
                            </Table.Tr>
                          );
                        })}
                        {analysableModelRows.length === 0 && (
                          <Table.Tr>
                            <Table.Td colSpan={5}>
                              <Text c="dimmed" ta="center" py="md">
                                No trained models with finished inference runs yet.
                              </Text>
                            </Table.Td>
                          </Table.Tr>
                        )}
                      </Table.Tbody>
                    </Table>
                  </ScrollArea>
                  <TablePagination totalItems={analysableModelRows.length} page={analysisModelPage} onChange={setAnalysisModelPage} />
                </Stack>
              </StepCard>

              {selectedModelIds.length > 0 && (
                <StepCard index={2} title="Inference datasets" color="teal">
                  <Stack gap="sm">
                    <Group justify="space-between" wrap="wrap">
                      <TextInput
                        placeholder="Search common inference datasets"
                        leftSection={<Search size={16} />}
                        value={commonDatasetSearch}
                        onChange={(event) => setCommonDatasetSearch(event.currentTarget.value)}
                        style={{ flex: 1 }}
                      />
                      <Badge variant="light" color="teal">{filteredCommonDatasetOptions.length} matching dataset{filteredCommonDatasetOptions.length === 1 ? '' : 's'}</Badge>
                      {commonDatasetSearch && <Button variant="subtle" color="gray" size="compact-sm" onClick={() => setCommonDatasetSearch('')}>Reset</Button>}
                    </Group>
                    <Paper withBorder radius="sm" className="analysis-run-picker">
                      <ScrollArea>
                        <Table striped highlightOnHover verticalSpacing="sm" miw={900}>
                          <Table.Thead>
                            <Table.Tr>
                              <Table.Th>Name</Table.Th>
                              <Table.Th>Label</Table.Th>
                              <Table.Th>Datasets</Table.Th>
                              <Table.Th>Image size</Table.Th>
                              <Table.Th>Stride</Table.Th>
                              <Table.Th>Images</Table.Th>
                              <Table.Th>Shared runs</Table.Th>
                              <Table.Th />
                              <Table.Th />
                            </Table.Tr>
                          </Table.Thead>
                          <Table.Tbody>
                            {pagedCommonDatasetOptions.map((option) => {
                              const dataset = option.dataset;
                              const selected = selectedInferenceDatasetId === option.value;
                              return (
                                <Table.Tr key={option.value} className={selected ? 'analysis-selected-row' : undefined}>
                                  <Table.Td>{option.label}</Table.Td>
                                  <Table.Td>
                                    <Badge size="xs" variant="light" color={usageColor(dataset?.usage_label)}>
                                      {usageLabel(dataset?.usage_label)}
                                    </Badge>
                                  </Table.Td>
                                  <Table.Td>
                                    <Group gap={4}>
                                      {(dataset?.dataset_names ?? []).map((name) => (
                                        <Badge key={name} size="xs" variant="light" color="gray">
                                          {name}
                                        </Badge>
                                      ))}
                                    </Group>
                                  </Table.Td>
                                  <Table.Td>
                                    <Group gap={4}>
                                      {(dataset ? datasetResolutions(dataset) : []).map((res) => (
                                        <Badge key={res} size="xs" variant="light" color="teal">
                                          {res}
                                        </Badge>
                                      ))}
                                    </Group>
                                  </Table.Td>
                                  <Table.Td>{datasetStrides(dataset ?? null)}</Table.Td>
                                  <Table.Td>{dataset?.total_selected_images ?? option.runs[0]?.image_count ?? '—'}</Table.Td>
                                  <Table.Td>{option.runs.length}</Table.Td>
                                  <Table.Td>
                                    <DetailButton title="Inference dataset details" body={renderTrainsetDetails(dataset ?? null)} onOpen={setDetailModal} />
                                  </Table.Td>
                                  <Table.Td>
                                    <Button
                                      size="compact-sm"
                                      variant={selected ? 'filled' : 'light'}
                                      color={selected ? 'green' : 'blue'}
                                      onClick={() => {
                                        setSelectedInferenceDatasetId(option.value);
                                        setSelectedSources([]);
                                        resetSelectionPreview();
                                        const bounds = sourceBounds(dataset);
                                        setDraft((current) => ({ ...current, start: bounds.start, end: bounds.end, timestamp: bounds.start || current.timestamp }));
                                      }}
                                    >
                                      {selected ? 'Selected' : 'Use'}
                                    </Button>
                                  </Table.Td>
                                </Table.Tr>
                              );
                            })}
                            {filteredCommonDatasetOptions.length === 0 && (
                              <Table.Tr>
                                <Table.Td colSpan={9}>
                                  <Text c="dimmed" ta="center" py="md">
                                    {commonDatasetOptions.length === 0 ? 'No inference dataset is available for all selected models.' : 'No common inference dataset matches the search.'}
                                  </Text>
                                </Table.Td>
                              </Table.Tr>
                            )}
                          </Table.Tbody>
                        </Table>
                      </ScrollArea>
                    </Paper>
                    <TablePagination totalItems={filteredCommonDatasetOptions.length} page={commonDatasetPage} onChange={setCommonDatasetPage} />
                  </Stack>
                </StepCard>
              )}

              {selectedInferenceDatasetId && (
                <StepCard index={3} title="ROI & plot type" color="violet">
                  <SimpleGrid cols={{ base: 1, md: 2 }}>
                    <Select
                      label="ROI"
                      placeholder="Select ROI variant"
                      data={roiOptionsForSelection}
                      value={selectedRoiKey}
                      searchable
                      onChange={(value) => {
                        setSelectedRoiKey(value);
                        setSelectedSources([]);
                        resetSelectionPreview();
                      }}
                    />
                    <Select
                      label="Plot type"
                      data={[
                        { value: 'timeseries', label: 'Time series' },
                        { value: 'heatmap', label: 'Heatmap' },
                      ]}
                      value={draft.plotType}
                      onChange={(value) => {
                        setDraft((current) => ({
                          ...current,
                          plotType: (value ?? 'timeseries') as PlotType,
                        }));
                        markPreviewStale();
                      }}
                    />
                  </SimpleGrid>
                  {selectedRoiKey === 'none' && <Text size="sm" c="dimmed" mt="xs">No ROI is selected. Scores use the full image result from the matching inference runs.</Text>}
                  {roiOptionsForSelection.length === 0 && <Alert color="yellow" mt="sm">No ROI variant is available for every selected model on this inference dataset.</Alert>}
                </StepCard>
              )}

              {selectedInferenceDatasetId && selectedRoiKey && (
                <StepCard index={4} title="Plot configuration" color="gray">
                  <Stack gap="md">
                    <MultiSelect
                      label="Metrics"
                      placeholder="Select one or more metrics"
                      data={metricOptionsForSelection}
                      value={selectedMetricKeys}
                      searchable
                      clearable
                      onChange={(values) => {
                        setSelectedMetricKeys(values.map(normalizeMetricKey));
                        setSelectedSources([]);
                        markPreviewStale();
                      }}
                    />
                    <MultiSelect
                      label={
                        <Group gap={4}>
                          <Text span size="sm" fw={500}>Score aggregation</Text>
                          <Tooltip
                            label="How each inference run reduced its per-pixel error map to one score per frame. Only aggregations that already exist as finished runs for every selected model and metric are listed."
                            multiline
                            w={320}
                            withArrow
                          >
                            <Info size={14} />
                          </Tooltip>
                        </Group>
                      }
                      placeholder={aggregationOptionsForSelection.length === 0 ? 'No inference runs available' : 'Select one or more aggregations'}
                      data={aggregationOptionsForSelection}
                      value={selectedAggregationKeys}
                      disabled={aggregationOptionsForSelection.length === 0}
                      searchable
                      clearable
                      onChange={(values) => {
                        setSelectedAggregationKeys(values.map(normalizeAggregationKey));
                        setSelectedSources([]);
                        markPreviewStale();
                      }}
                    />
                    <SimpleGrid cols={{ base: 1, md: 3 }}>
                      {draft.plotType === 'heatmap' && draft.heatmapMode === 'single' ? (
                        <DateTime24Input
                          label="Timestamp"
                          min={selectedInferenceBounds.start}
                          max={selectedInferenceBounds.end}
                          value={toDateTimeLocal(draft.timestamp ?? selectedInferenceBounds.start)}
                          description={selectedInferenceBounds.start && selectedInferenceBounds.end ? `${selectedInferenceBounds.start.replace('T', ' ')} to ${selectedInferenceBounds.end.replace('T', ' ')}` : undefined}
                          onChange={(value) => {
                            setDraft((current) => ({ ...current, timestamp: value }));
                            markPreviewStale();
                          }}
                        />
                      ) : (
                        <>
                          <DateTime24Input
                            label="Start"
                            min={selectedInferenceBounds.start}
                            max={selectedInferenceBounds.end}
                            value={draft.start}
                            onChange={(value) => {
                              setDraft((current) => ({ ...current, start: value }));
                              markPreviewStale();
                            }}
                          />
                          <DateTime24Input
                            label="End"
                            min={selectedInferenceBounds.start}
                            max={selectedInferenceBounds.end}
                            value={draft.end}
                            onChange={(value) => {
                              setDraft((current) => ({ ...current, end: value }));
                              markPreviewStale();
                            }}
                          />
                        </>
                      )}
                      <NumberInput
                        label="Sampling rate"
                        min={1}
                        value={draft.sampling}
                        onChange={(value) => {
                          setDraft((current) => ({ ...current, sampling: valueAsNumber(value, 1) }));
                          markPreviewStale();
                        }}
                      />
                    </SimpleGrid>
                    {draft.plotType === 'heatmap' && (
                      <ExistingHeatmapArtifacts
                        singles={matchingSavedHeatmaps}
                        videos={matchingSavedHeatmapVideos}
                        testingRunById={testingRunById}
                        intervalLabel={heatmapArtifactInterval.label}
                        refreshing={refreshingHeatmapArtifacts}
                        onRefresh={refreshHeatmapArtifacts}
                      />
                    )}
                    {draft.plotType === 'timeseries' ? (
                      <Stack gap="md">
                        <SimpleGrid cols={{ base: 1, md: 3 }}>
                          <Select
                            label="Score line"
                            data={scoreSeriesOptions(combinedDraftResults)}
                            value={draft.scoreSeries}
                            onChange={(value) => {
                              setDraft((current) => ({ ...current, scoreSeries: value ?? 'score' }));
                              markPreviewStale();
                            }}
                          />
                          <NumberInput
                            label="Moving average window"
                            min={1}
                            value={draft.movingAverage}
                            onChange={(value) => {
                              setDraft((current) => ({ ...current, movingAverage: valueAsNumber(value, 1) }));
                              markPreviewStale();
                            }}
                          />
                          <TextInput label="X-axis" value="Time" disabled />
                        </SimpleGrid>
                        <Paper withBorder p="sm" radius="sm">
                          <Stack gap="sm">
                            <Group justify="space-between" align="flex-start">
                              <div>
                                <Text fw={700}>Timeseries analytics pipeline</Text>
                                <Text size="sm" c="dimmed">
                                  Stages run left-to-right on the previous output. Example: MSE {'->'} First derivative {'->'} CUSUM.
                                </Text>
                              </div>
                              <Switch
                                label={<InfoLabel label="Show intermediate panels" info="On shows the raw score and every analytics stage as separate aligned panels. Off shows only the final pipeline output." />}
                                checked={draft.showIntermediateAnalyticsPanels}
                                onChange={(event) => {
                                  const checked = event.currentTarget.checked;
                                  setDraft((current) => ({ ...current, showIntermediateAnalyticsPanels: checked }));
                                  markPreviewStale();
                                }}
                              />
                            </Group>
                            <NumberInput
                              label={<InfoLabel label="Panel height" info="Vertical height per time-series panel. Increase this when multiple analytics stages make the graph too flat." />}
                              min={120}
                              max={900}
                              step={20}
                              value={draft.panelHeightPx}
                              onChange={(value) => {
                                const nextHeight = valueAsNumber(value, draft.panelHeightPx);
                                setDraft((current) => ({ ...current, panelHeightPx: nextHeight }));
                                setPlotPreview((current) =>
                                  current
                                    ? { ...current, plot: { ...current.plot, panelHeightPx: nextHeight } }
                                    : current,
                                );
                              }}
                            />
                            <div>
                              <Text size="sm" c="dimmed">
                                Add one or more causal stages. Empty means raw score only.
                              </Text>
                            </div>
                            <MultiSelect
                              label="Analytics stages"
                              placeholder="None"
                              data={ANALYTICS_DEFINITIONS.map((definition) => ({ value: definition.kind, label: definition.label }))}
                              value={draft.timeseriesAnalytics.map((config) => config.kind)}
                              searchable
                              clearable
                              onChange={(values) => {
                                setDraft((current) => {
                                  const existing = new Map(current.timeseriesAnalytics.map((config) => [config.kind, config]));
                                  return {
                                    ...current,
                                    timeseriesAnalytics: values.map((value) => existing.get(value as AnalyticsKind) ?? defaultAnalyticsConfig(value as AnalyticsKind)),
                                    analyticsDisplayMode: 'multi_panel',
                                    panelHeightPx: values.length > 0 && current.timeseriesAnalytics.length === 0 ? 260 : current.panelHeightPx,
                                  };
                                });
                                markPreviewStale();
                              }}
                            />
                            {draft.timeseriesAnalytics.length > 0 && (
                              <Stack gap="sm">
                                {draft.timeseriesAnalytics.map((config, index) => (
                                  <Paper key={config.kind} withBorder p="sm" radius="sm">
                                    <Stack gap="xs">
                                      <Group justify="space-between" align="flex-start">
                                        <div>
                                          <Group gap="xs">
                                            <Badge variant="light" color="violet">Stage {index + 1}</Badge>
                                            <InfoLabel label={analyticsDefinition(config.kind).label} info={analyticsDefinition(config.kind).description} />
                                          </Group>
                                          <Text size="xs" c="dimmed">{analyticsDefinition(config.kind).description}</Text>
                                        </div>
                                        <Group gap={4}>
                                          <Tooltip label="Move stage up">
                                            <ActionIcon
                                              variant="subtle"
                                              disabled={index === 0}
                                              onClick={() => {
                                                setDraft((current) => {
                                                  const next = [...current.timeseriesAnalytics];
                                                  [next[index - 1], next[index]] = [next[index], next[index - 1]];
                                                  return { ...current, timeseriesAnalytics: next };
                                                });
                                                markPreviewStale();
                                              }}
                                            >
                                              <ArrowUp size={16} />
                                            </ActionIcon>
                                          </Tooltip>
                                          <Tooltip label="Move stage down">
                                            <ActionIcon
                                              variant="subtle"
                                              disabled={index === draft.timeseriesAnalytics.length - 1}
                                              onClick={() => {
                                                setDraft((current) => {
                                                  const next = [...current.timeseriesAnalytics];
                                                  [next[index], next[index + 1]] = [next[index + 1], next[index]];
                                                  return { ...current, timeseriesAnalytics: next };
                                                });
                                                markPreviewStale();
                                              }}
                                            >
                                              <ArrowDown size={16} />
                                            </ActionIcon>
                                          </Tooltip>
                                          <Tooltip label="Delete stage">
                                            <ActionIcon
                                              variant="subtle"
                                              color="red"
                                              onClick={() => {
                                                setDraft((current) => ({
                                                  ...current,
                                                  timeseriesAnalytics: current.timeseriesAnalytics.filter((_, configIndex) => configIndex !== index),
                                                }));
                                                markPreviewStale();
                                              }}
                                            >
                                              <Trash2 size={16} />
                                            </ActionIcon>
                                          </Tooltip>
                                        </Group>
                                      </Group>
                                      <SimpleGrid cols={{ base: 1, md: 3 }}>
                                        {Object.entries(config.params).map(([key, value]) => renderAnalyticsParamInput(config, index, key, value))}
                                      </SimpleGrid>
                                    </Stack>
                                  </Paper>
                                ))}
                              </Stack>
                            )}
                          </Stack>
                        </Paper>
                      </Stack>
                    ) : (
                <Stack gap="md">
                  <SimpleGrid cols={{ base: 1, md: 3 }}>
                    <Select
                      label={<InfoLabel label="Heatmap mode" info="Single timestamp computes one interactive heatmap. Date range renders a sampled heatmap video." />}
                      data={[
                        { value: 'single', label: 'Single timestamp' },
                        { value: 'range', label: 'Date range video' },
                      ]}
                      value={draft.heatmapMode}
                      onChange={(value) => {
                        setDraft((current) => ({
                          ...current,
                          heatmapMode: (value ?? 'single') as HeatmapMode,
                          timestamp: value === 'range' ? null : current.timestamp,
                        }));
                        markPreviewStale();
                      }}
                    />
                    <TextInput
                      label={<InfoLabel label="Frames" info="Number of deduplicated source timestamps before the selected sampling rate is applied." />}
                      value={`${combinedDraftResults.length} deduplicated frames`}
                      disabled
                    />
                    <Switch
                      label={<InfoLabel label="Include reference" info="Shows original and reconstructed images next to the transparent error overlay." />}
                      checked={draft.includeReference}
                      onChange={(event) => {
                        const checked = event.currentTarget.checked;
                        setDraft((current) => ({ ...current, includeReference: checked }));
                        markPreviewStale();
                      }}
                    />
                  </SimpleGrid>
                  {selectedRunIsStae && (
                    <SimpleGrid cols={{ base: 1, md: 3 }}>
                      <Select
                        label={<InfoLabel label="STAE view" info="Reconstruction compares the last input frame with its reconstruction. Prediction compares a future target frame with the predicted future frame." />}
                        data={[
                          { value: 'reconstruction', label: 'Reconstruction' },
                          { value: 'prediction', label: 'Future prediction' },
                        ]}
                        value={draft.staeHeatmapView}
                        onChange={(value) => {
                          setDraft((current) => ({
                            ...current,
                            staeHeatmapView: (value ?? 'reconstruction') as 'reconstruction' | 'prediction',
                          }));
                          markPreviewStale();
                        }}
                      />
                      <NumberInput
                        label={<InfoLabel label="Prediction horizon" info="Future frame index for STAE prediction heatmaps. future+1 means the first predicted future frame." />}
                        min={1}
                        value={draft.predictionHorizon}
                        disabled={draft.staeHeatmapView !== 'prediction'}
                        onChange={(value) => {
                          setDraft((current) => ({ ...current, predictionHorizon: valueAsNumber(value, 1) }));
                          markPreviewStale();
                        }}
                      />
                    </SimpleGrid>
                  )}

                  <Text fw={600} size="sm">Error calculation</Text>
                  <SimpleGrid cols={{ base: 1, md: 3 }}>
                    <Select
                      label={<InfoLabel label="Residual source" info="Pixel residual compares pixel values directly. SSIM residual uses local 1 - SSIM structure distance with standard K constants." />}
                      data={[
                        { value: 'pixel_residual', label: 'Pixel residual' },
                        { value: 'ssim_residual', label: 'SSIM residual' },
                      ]}
                      value={draft.heatmapConfig.residual_source}
                      onChange={(value) => updateHeatmapConfig({ residual_source: (value ?? 'pixel_residual') as 'pixel_residual' | 'ssim_residual' })}
                    />
                    <Select
                      label={<InfoLabel label="Pixel error" info="Absolute uses |input - reconstruction|. Squared uses (input - reconstruction)² and emphasizes large deviations more strongly." />}
                      data={[
                        { value: 'squared', label: 'Squared error' },
                        { value: 'absolute', label: 'Absolute error' },
                      ]}
                      value={draft.heatmapConfig.error_mode}
                      disabled={draft.heatmapConfig.residual_source === 'ssim_residual'}
                      onChange={(value) => updateHeatmapConfig({ error_mode: (value ?? 'squared') as 'squared' | 'absolute' })}
                    />
                    <Switch
                      label={<InfoLabel label="Signed deviations" info="Off treats only deviation magnitude. On preserves direction: input brighter than reconstruction is positive; input darker is negative." />}
                      checked={draft.heatmapConfig.signed_deviations}
                      disabled={draft.heatmapConfig.residual_source === 'ssim_residual'}
                      onChange={(event) => updateHeatmapConfig({ signed_deviations: event.currentTarget.checked })}
                    />
                    <Switch
                      label={<InfoLabel label="Threshold" info="Suppresses pixels whose absolute input/reconstruction difference is below the specified value. The value uses preprocessed pixel units." />}
                      checked={draft.heatmapConfig.threshold_enabled}
                      onChange={(event) => updateHeatmapConfig({ threshold_enabled: event.currentTarget.checked })}
                    />
                  </SimpleGrid>
                  {draft.heatmapConfig.residual_source === 'ssim_residual' && (
                    <SimpleGrid cols={{ base: 1, md: 3 }}>
                      <NumberInput
                        label={<InfoLabel label="SSIM window" info="Odd local window size used for 1 - SSIM residual maps. 11 is the standard default." />}
                        min={3}
                        step={2}
                        value={draft.heatmapConfig.ssim_window_size}
                        onChange={(value) => updateHeatmapConfig({ ssim_window_size: valueAsNumber(value, 11) })}
                      />
                      <NumberInput
                        label={<InfoLabel label="SSIM K1" info="Standard SSIM K constant. MLTrace computes C1=(K1*data_range)^2." />}
                        min={0}
                        step={0.001}
                        decimalScale={4}
                        value={draft.heatmapConfig.ssim_k1}
                        onChange={(value) => updateHeatmapConfig({ ssim_k1: valueAsNumber(value, 0.01) })}
                      />
                      <NumberInput
                        label={<InfoLabel label="SSIM K2" info="Standard SSIM K constant. MLTrace computes C2=(K2*data_range)^2." />}
                        min={0}
                        step={0.001}
                        decimalScale={4}
                        value={draft.heatmapConfig.ssim_k2}
                        onChange={(value) => updateHeatmapConfig({ ssim_k2: valueAsNumber(value, 0.03) })}
                      />
                      <NumberInput
                        label={<InfoLabel label="SSIM data range" info="Expected image value range. Use 1.0 for normalized float model inputs." />}
                        min={0.000001}
                        step={0.1}
                        decimalScale={4}
                        value={draft.heatmapConfig.ssim_data_range}
                        onChange={(value) => updateHeatmapConfig({ ssim_data_range: valueAsNumber(value, 1) })}
                      />
                    </SimpleGrid>
                  )}

                  {(draft.heatmapConfig.signed_deviations || draft.heatmapConfig.threshold_enabled) && (
                    <SimpleGrid cols={{ base: 1, md: 3 }}>
                      {draft.heatmapConfig.signed_deviations && (
                        <>
                          <NumberInput
                            label={<InfoLabel label="Positive weight" info="Multiplier for pixels where input is brighter than reconstruction. Zero suppresses positive deviations." />}
                            min={0}
                            step={0.1}
                            decimalScale={3}
                            value={draft.heatmapConfig.positive_weight}
                            onChange={(value) => updateHeatmapConfig({ positive_weight: valueAsNumber(value, 1) })}
                          />
                          <NumberInput
                            label={<InfoLabel label="Negative weight" info="Multiplier for pixels where input is darker than reconstruction. Zero suppresses negative deviations." />}
                            min={0}
                            step={0.1}
                            decimalScale={3}
                            value={draft.heatmapConfig.negative_weight}
                            onChange={(value) => updateHeatmapConfig({ negative_weight: valueAsNumber(value, 1) })}
                          />
                        </>
                      )}
                      {draft.heatmapConfig.threshold_enabled && (
                        <NumberInput
                          label={<InfoLabel label="Threshold value" info="Absolute difference below this value becomes zero before absolute/squared error and sign weighting are applied." />}
                          min={0}
                          step={0.01}
                          decimalScale={6}
                          value={draft.heatmapConfig.threshold}
                          onChange={(value) => updateHeatmapConfig({ threshold: valueAsNumber(value, 0) })}
                        />
                      )}
                    </SimpleGrid>
                  )}

                  <Text fw={600} size="sm">Visibility</Text>
                  <SimpleGrid cols={{ base: 1, md: 3 }}>
                    <Switch
                      label={<InfoLabel label="Fixed ceiling" info="Uses this absolute error value as relative 1.0. Tiny residual errors therefore remain faint instead of being stretched to full visibility. The value uses the selected error metric's units." />}
                      checked={draft.heatmapConfig.fixed_ceiling_enabled}
                      onChange={(event) => {
                        const checked = event.currentTarget.checked;
                        updateHeatmapConfig({
                          fixed_ceiling_enabled: checked,
                          max_clip_enabled: checked ? false : draft.heatmapConfig.max_clip_enabled,
                        });
                      }}
                    />
                    <Switch
                      label={<InfoLabel label="Max clip" info="When enabled, errors at max clip × the strongest current error already reach full opacity. At 0.33, the strongest third saturates like the legacy visualization." />}
                      checked={draft.heatmapConfig.max_clip_enabled}
                      onChange={(event) => {
                        const checked = event.currentTarget.checked;
                        updateHeatmapConfig({
                          max_clip_enabled: checked,
                          fixed_ceiling_enabled: checked ? false : draft.heatmapConfig.fixed_ceiling_enabled,
                        });
                      }}
                    />
                    {!draft.heatmapConfig.max_clip_enabled && (
                      <NumberInput
                        label={<InfoLabel label="Maximum opacity (%)" info="Maximum overlay coverage when max clip is disabled. This remains configurable with fixed ceiling; the previous MLTrace behavior used 55%." />}
                        min={0}
                        max={100}
                        step={5}
                        value={Math.round(draft.heatmapConfig.max_opacity * 100)}
                        onChange={(value) => updateHeatmapConfig({ max_opacity: valueAsNumber(value, 55) / 100 })}
                      />
                    )}
                  </SimpleGrid>
                  {(draft.heatmapConfig.fixed_ceiling_enabled || draft.heatmapConfig.max_clip_enabled) && (
                    <SimpleGrid cols={{ base: 1, md: 3 }}>
                      {draft.heatmapConfig.fixed_ceiling_enabled && (
                        <NumberInput
                          label={<InfoLabel label="Ceiling value" info="Error magnitude mapped to relative 1.0. For absolute error this is a direct pixel difference; for squared error it is a squared pixel difference." />}
                          min={Number.EPSILON}
                          step={0.01}
                          decimalScale={8}
                          value={draft.heatmapConfig.fixed_ceiling}
                          onChange={(value) => updateHeatmapConfig({ fixed_ceiling: valueAsNumber(value, 1) })}
                        />
                      )}
                      {draft.heatmapConfig.max_clip_enabled && (
                        <NumberInput
                          label={<InfoLabel label="Max clip (%)" info="Relative fraction of the strongest remaining error at which the overlay becomes fully opaque." />}
                          min={1}
                          max={100}
                          step={1}
                          value={Math.round(draft.heatmapConfig.max_clip * 100)}
                          onChange={(value) => updateHeatmapConfig({ max_clip: valueAsNumber(value, 33) / 100 })}
                        />
                      )}
                    </SimpleGrid>
                  )}
                </Stack>
              )}
                    {finishedRuns.length === 0 && <Alert color="blue">No finished testing runs with images are available yet.</Alert>}
                    {plotPreview && (
                      <Paper withBorder p="sm" radius="sm">
                        <Stack gap="sm">
                          <div>
                            <Group gap="xs">
                              <Text fw={700}>{editingPlot ? 'Editing plot preview' : 'Preloaded plot preview'}</Text>
                              {plotPreviewStale && <Badge color="yellow" variant="light">Preview stale</Badge>}
                            </Group>
                            <Text size="sm" fw={600} style={{ overflowWrap: 'anywhere' }}>{plotPreview.title}</Text>
                            <Text size="xs" c="dimmed" mt={2} style={{ overflowWrap: 'anywhere' }}>{plotPreview.subtitle}</Text>
                            {plotPreview.plot.detailSubtitle && (
                              <details style={{ marginTop: 4 }}>
                                <summary style={{ cursor: 'pointer', fontSize: 12 }}>Show details</summary>
                                <Stack gap={1} mt={2}>{plotPreview.plot.detailSubtitle.split('\n').filter(Boolean).map((line, index) => <Text key={`${index}:${line}`} size="xs" c="dimmed" style={{ overflowWrap: 'anywhere' }}>{line}</Text>)}</Stack>
                              </details>
                            )}
                          </div>
                          {plotPreviewStale && (
                            <Alert color="yellow">
                              Parameter changes are not applied to this preview yet. Click Update preview to recompute only this plot.
                            </Alert>
                          )}
                          {plotPreview.duplicateNotes.length > 0 && (
                            <Alert color="yellow">
                              {plotPreview.duplicateNotes.map((note) => (
                                <Text key={note} size="sm">{note}</Text>
                              ))}
                            </Alert>
                          )}
                          {previewPlot && (
                            <Paper withBorder p="xs" radius="sm">
                              {previewPlot.plotType === 'timeseries' ? (
                                <TimeSeriesPlot plot={previewPlot} results={previewResults} />
                              ) : previewPlot.heatmapMode === 'range' ? (
                                <HeatmapVideo plot={previewPlot} results={previewResults} />
                              ) : (
                                <HeatmapPlot
                                  plot={previewPlot}
                                  results={previewResults}
                                  heatmapCache={heatmapCache}
                                  loadingHeatmaps={loadingHeatmaps}
                                  heatmapErrors={heatmapErrors}
                                  ensureHeatmap={ensureHeatmap}
                                />
                              )}
                            </Paper>
                          )}
                          {plotPreview.traces.map((trace, index) => (
                            <SimpleGrid key={`${trace.testingRunId}:${trace.metric}`} cols={{ base: 1, md: 3 }} spacing="sm">
                              <TextInput
                                label="Legend"
                                value={trace.legendLabel}
                                onChange={(event) => updatePreviewTrace(index, { legendLabel: event.currentTarget.value })}
                              />
                              <ColorInput
                                label="Color"
                                value={trace.color}
                                onChange={(value) => updatePreviewTrace(index, { color: value })}
                              />
                              <TextInput label="Metric" value={metricLabel(trace.metric)} disabled />
                              <TextInput label="Aggregation" value={aggregationLabel(trace.aggregation ?? 'mean')} disabled />
                            </SimpleGrid>
                          ))}
                        </Stack>
                      </Paper>
                    )}
                    <Group justify="space-between" align="center">
                      <Text size="sm" c="dimmed">
                        {plotPreview
                          ? `${plotPreview.traces.length} trace${plotPreview.traces.length === 1 ? '' : 's'} ready`
                          : 'Preload a plot to review legend labels and colors before adding it to the board.'}
                      </Text>
                      <Group gap="sm">
                        {editingPlot && (
                          <Button variant="default" onClick={cancelPlotEdit}>
                            Cancel edit
                          </Button>
                        )}
                        <Button
                          variant="light"
                          leftSection={<Upload size={18} />}
                          loading={preloadingPlot}
                          onClick={preloadPlot}
                          disabled={selectedModelIds.length === 0 || !selectedInferenceDatasetId || selectedMetricKeys.length === 0}
                        >
                          {plotPreview ? 'Update preview' : 'Preload plot'}
                        </Button>
                        <Button leftSection={<Plus size={18} />} onClick={finishPlot} disabled={!plotPreview || plotPreviewStale}>
                          {editingPlot ? 'Finish edit' : 'Finish plot'}
                        </Button>
                      </Group>
                    </Group>
                  </Stack>
                </StepCard>
              )}
            </Stack>
          </Collapse>
        </Stack>
      </Paper>

      {plots.length === 0 ? (
        <Alert color="blue">Add time series or heatmap plots to start comparing testing runs.</Alert>
      ) : (
        <Stack gap="md">
          {plots.map((plot) => {
            const plotData = plotResultsById.get(plot.id) ?? { hasAllData: false, results: [] };
            const displayPlot = withBaselineFreshness(withAutomaticPlotMetadata(plot, testingRuns), testingRuns);
            return plotData.hasAllData ? (
              <AnalysisPlotCard
                key={plot.id}
                plot={displayPlot}
                results={plotData.results}
                heatmapCache={heatmapCache}
                loadingHeatmaps={loadingHeatmaps}
                heatmapErrors={heatmapErrors}
                ensureHeatmap={ensureHeatmap}
                onMove={(direction) => movePlot(plot.id, direction)}
                onEdit={() => editPlot(displayPlot, plots.findIndex((item) => item.id === plot.id))}
                onPatch={(patch) =>
                  setPlots((current) =>
                    current.map((item) => (item.id === plot.id ? withAutomaticPlotMetadata({ ...item, ...patch }, testingRuns) : item)),
                  )
                }
                onRemove={() => setPlots((current) => current.filter((item) => item.id !== plot.id))}
                onHeatmapSelection={(mode, start, end) => beginHeatmapDerivation(displayPlot, plotData.results, mode, start, end)}
              />
            ) : (
              <Paper key={plot.id} withBorder p="md" radius="sm">
                <Group gap="sm">
                  <Loader size="sm" />
                  <Text>Loading plot data…</Text>
                </Group>
              </Paper>
            );
          })}
        </Stack>
      )}
      <Modal
        opened={timeSeriesHeatmapDraft !== null}
        onClose={() => setTimeSeriesHeatmapDraft(null)}
        title={timeSeriesHeatmapDraft?.mode === 'single' ? 'Create heatmap from data point' : 'Create heatmap video from time range'}
        size="xl"
      >
        {timeSeriesHeatmapDraft && (
          <Stack gap="md">
            <Alert color={heatmapDerivationError ? 'yellow' : 'blue'}>
              {heatmapDerivationError ?? (
                timeSeriesHeatmapDraft.mode === 'single'
                  ? `Selected timestamp: ${artifactTimestamp(effectiveHeatmapDerivationResults[0].timestamp)}`
                  : `Selected range: ${artifactTimestamp(effectiveHeatmapDerivationResults[0].timestamp)} – ${artifactTimestamp(effectiveHeatmapDerivationResults[effectiveHeatmapDerivationResults.length - 1].timestamp)} · ${effectiveHeatmapDerivationResults.length} visible frames`
              )}
            </Alert>

            <SimpleGrid cols={{ base: 1, md: timeSeriesHeatmapDraft.mode === 'range' ? 3 : 1 }}>
              <Select
                label="Inference source"
                data={heatmapDerivationSources}
                value={timeSeriesHeatmapDraft.testingRunId}
                onChange={(value) => setTimeSeriesHeatmapDraft((current) => current ? { ...current, testingRunId: value ?? '' } : current)}
                allowDeselect={false}
              />
              {timeSeriesHeatmapDraft.mode === 'range' && (
                <>
                  <NumberInput
                    label="Sampling rate"
                    description="Render every Nth source frame"
                    min={1}
                    value={timeSeriesHeatmapDraft.sampling}
                    onChange={(value) => setTimeSeriesHeatmapDraft((current) => current ? { ...current, sampling: valueAsNumber(value, 1) } : current)}
                  />
                  <NumberInput
                    label="Video FPS"
                    min={1}
                    max={60}
                    value={timeSeriesHeatmapDraft.fps}
                    onChange={(value) => setTimeSeriesHeatmapDraft((current) => current ? { ...current, fps: valueAsNumber(value, 8) } : current)}
                  />
                </>
              )}
            </SimpleGrid>

            <SimpleGrid cols={{ base: 1, md: 2 }}>
              <Switch
                label="Show reference images"
                description="Display input, reconstruction and transparent heatmap side by side"
                checked={timeSeriesHeatmapDraft.includeReference}
                onChange={(event) => {
                  const checked = event.currentTarget.checked;
                  setTimeSeriesHeatmapDraft((current) => current ? { ...current, includeReference: checked } : current);
                }}
              />
              <NumberInput
                label="Heatmap image size"
                description="Maximum width and height per image panel"
                suffix=" px"
                min={240}
                max={1200}
                step={40}
                value={timeSeriesHeatmapDraft.displaySize}
                onChange={(value) => setTimeSeriesHeatmapDraft((current) => current ? { ...current, displaySize: valueAsNumber(value, DEFAULT_HEATMAP_DISPLAY_SIZE) } : current)}
              />
            </SimpleGrid>

            <SimpleGrid cols={{ base: 1, md: 3 }}>
              <Select
                label="Residual source"
                data={[
                  { value: 'pixel_residual', label: 'Pixel residual' },
                  { value: 'ssim_residual', label: 'SSIM residual' },
                ]}
                value={timeSeriesHeatmapDraft.heatmapConfig.residual_source}
                onChange={(value) => patchDerivedHeatmapConfig({ residual_source: (value ?? 'pixel_residual') as 'pixel_residual' | 'ssim_residual' })}
              />
              <Select
                label="Error calculation"
                data={[
                  { value: 'squared', label: 'Squared error' },
                  { value: 'absolute', label: 'Absolute error' },
                ]}
                value={timeSeriesHeatmapDraft.heatmapConfig.error_mode}
                disabled={timeSeriesHeatmapDraft.heatmapConfig.residual_source === 'ssim_residual'}
                onChange={(value) => patchDerivedHeatmapConfig({ error_mode: (value ?? 'squared') as 'squared' | 'absolute' })}
              />
              {timeSeriesHeatmapDraft.mode === 'range' && (
                <Select
                  label="Video color scale"
                  data={[
                    { value: 'per_frame', label: 'Per-frame (auto)' },
                    { value: 'shared', label: 'Shared (comparable)' },
                  ]}
                  value={timeSeriesHeatmapDraft.scaleMode}
                  disabled={timeSeriesHeatmapDraft.heatmapConfig.fixed_ceiling_enabled}
                  onChange={(value) => setTimeSeriesHeatmapDraft((current) => current ? { ...current, scaleMode: (value ?? 'per_frame') as 'per_frame' | 'shared' } : current)}
                />
              )}
            </SimpleGrid>

            <SimpleGrid cols={{ base: 1, md: 2 }}>
              <Select
                label="Visibility mode"
                data={[
                  { value: 'opacity', label: 'Maximum opacity' },
                  { value: 'max_clip', label: 'Relative max clip' },
                  { value: 'fixed_ceiling', label: 'Fixed error ceiling' },
                ]}
                value={heatmapDerivationVisibilityMode}
                onChange={(value) => patchDerivedHeatmapConfig({
                  fixed_ceiling_enabled: value === 'fixed_ceiling',
                  max_clip_enabled: value === 'max_clip',
                })}
              />
              {heatmapDerivationVisibilityMode === 'fixed_ceiling' ? (
                <NumberInput
                  label="Ceiling value"
                  min={Number.EPSILON}
                  decimalScale={8}
                  value={timeSeriesHeatmapDraft.heatmapConfig.fixed_ceiling}
                  onChange={(value) => patchDerivedHeatmapConfig({ fixed_ceiling: valueAsNumber(value, 1) })}
                />
              ) : heatmapDerivationVisibilityMode === 'max_clip' ? (
                <NumberInput
                  label="Max clip (%)"
                  min={1}
                  max={100}
                  value={Math.round(timeSeriesHeatmapDraft.heatmapConfig.max_clip * 100)}
                  onChange={(value) => patchDerivedHeatmapConfig({ max_clip: valueAsNumber(value, 33) / 100 })}
                />
              ) : (
                <NumberInput
                  label="Maximum opacity (%)"
                  min={0}
                  max={100}
                  value={Math.round(timeSeriesHeatmapDraft.heatmapConfig.max_opacity * 100)}
                  onChange={(value) => patchDerivedHeatmapConfig({ max_opacity: valueAsNumber(value, 55) / 100 })}
                />
              )}
            </SimpleGrid>

            <Button
              variant="subtle"
              leftSection={<SlidersHorizontal size={16} />}
              rightSection={timeSeriesHeatmapDraft.advancedOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              onClick={() => setTimeSeriesHeatmapDraft((current) => current ? { ...current, advancedOpen: !current.advancedOpen } : current)}
            >
              Advanced heatmap parameters
            </Button>
            <Collapse in={timeSeriesHeatmapDraft.advancedOpen}>
              <Stack gap="md">
                <SimpleGrid cols={{ base: 1, md: 3 }}>
                  <Switch
                    label="Signed deviations"
                    checked={timeSeriesHeatmapDraft.heatmapConfig.signed_deviations}
                    disabled={timeSeriesHeatmapDraft.heatmapConfig.residual_source === 'ssim_residual'}
                    onChange={(event) => patchDerivedHeatmapConfig({ signed_deviations: event.currentTarget.checked })}
                  />
                  <Switch
                    label="Threshold"
                    checked={timeSeriesHeatmapDraft.heatmapConfig.threshold_enabled}
                    onChange={(event) => patchDerivedHeatmapConfig({ threshold_enabled: event.currentTarget.checked })}
                  />
                  {timeSeriesHeatmapDraft.heatmapConfig.threshold_enabled && (
                    <NumberInput
                      label="Threshold value"
                      min={0}
                      decimalScale={6}
                      value={timeSeriesHeatmapDraft.heatmapConfig.threshold}
                      onChange={(value) => patchDerivedHeatmapConfig({ threshold: valueAsNumber(value, 0) })}
                    />
                  )}
                </SimpleGrid>
                {timeSeriesHeatmapDraft.heatmapConfig.signed_deviations && (
                  <SimpleGrid cols={{ base: 1, md: 2 }}>
                    <NumberInput label="Positive weight" min={0} value={timeSeriesHeatmapDraft.heatmapConfig.positive_weight} onChange={(value) => patchDerivedHeatmapConfig({ positive_weight: valueAsNumber(value, 1) })} />
                    <NumberInput label="Negative weight" min={0} value={timeSeriesHeatmapDraft.heatmapConfig.negative_weight} onChange={(value) => patchDerivedHeatmapConfig({ negative_weight: valueAsNumber(value, 1) })} />
                  </SimpleGrid>
                )}
                {timeSeriesHeatmapDraft.heatmapConfig.residual_source === 'ssim_residual' && (
                  <SimpleGrid cols={{ base: 1, md: 4 }}>
                    <NumberInput label="SSIM window" min={3} step={2} value={timeSeriesHeatmapDraft.heatmapConfig.ssim_window_size} onChange={(value) => patchDerivedHeatmapConfig({ ssim_window_size: valueAsNumber(value, 11) })} />
                    <NumberInput label="SSIM K1" min={0} decimalScale={4} value={timeSeriesHeatmapDraft.heatmapConfig.ssim_k1} onChange={(value) => patchDerivedHeatmapConfig({ ssim_k1: valueAsNumber(value, 0.01) })} />
                    <NumberInput label="SSIM K2" min={0} decimalScale={4} value={timeSeriesHeatmapDraft.heatmapConfig.ssim_k2} onChange={(value) => patchDerivedHeatmapConfig({ ssim_k2: valueAsNumber(value, 0.03) })} />
                    <NumberInput label="SSIM data range" min={Number.EPSILON} value={timeSeriesHeatmapDraft.heatmapConfig.ssim_data_range} onChange={(value) => patchDerivedHeatmapConfig({ ssim_data_range: valueAsNumber(value, 1) })} />
                    <NumberInput label="SSIM alpha" min={0} value={timeSeriesHeatmapDraft.heatmapConfig.ssim_alpha} onChange={(value) => patchDerivedHeatmapConfig({ ssim_alpha: valueAsNumber(value, 1) })} />
                    <NumberInput label="SSIM beta" min={0} value={timeSeriesHeatmapDraft.heatmapConfig.ssim_beta} onChange={(value) => patchDerivedHeatmapConfig({ ssim_beta: valueAsNumber(value, 1) })} />
                    <NumberInput label="SSIM gamma" min={0} value={timeSeriesHeatmapDraft.heatmapConfig.ssim_gamma} onChange={(value) => patchDerivedHeatmapConfig({ ssim_gamma: valueAsNumber(value, 1) })} />
                  </SimpleGrid>
                )}
              </Stack>
            </Collapse>

            <Group justify="flex-end">
              <Button variant="light" leftSection={<Save size={16} />} onClick={storeCurrentHeatmapSelectionDefaults}>
                Set as default
              </Button>
              <Button variant="default" onClick={() => setTimeSeriesHeatmapDraft(null)}>Cancel</Button>
              <Button
                color="red"
                leftSection={<Flame size={16} />}
                disabled={Boolean(heatmapDerivationError)}
                onClick={finishHeatmapDerivation}
              >
                {timeSeriesHeatmapDraft.mode === 'single' ? 'Create heatmap' : 'Create and render video'}
              </Button>
            </Group>
          </Stack>
        )}
      </Modal>
      <DetailModal detail={detailModal} onClose={() => setDetailModal(null)} />
    </Stack>
  );
}
