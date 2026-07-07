import {
  ActionIcon,
  Alert,
  Badge,
  Button,
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
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { ArrowDown, ArrowUp, Check, ChevronDown, ChevronRight, Info, Pause, Pencil, Play, Plus, RotateCcw, Save, Search, Trash2, Upload } from 'lucide-react';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import type React from 'react';

import {
  abortHeatmapRange,
  createAnalysisLayout,
  createHeatmap,
  createHeatmapRange,
  deleteAnalysisLayout,
  getAnalysisLayout,
  getHeatmapRange,
  getTestingRunResults,
  heatmapRangeFrameUrl,
  listAnalysisLayouts,
  listMethodConfigurations,
  listPreprocessingPipelines,
  listTestingRuns,
  listTrainingDatasets,
  listTrainingPipelines,
  listTrainingRuns,
  updateAnalysisLayout,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart } from '../components/PlotlyChart';
import { StepCard } from '../components/StepCard';
import type { Data, Layout } from '../lib/plotly';
import { formatValue } from '../methods/utils';
import { datasetResolutions, formatResolution, orderedGraphNodes, stepDetail } from '../training/graph';
import type {
  AnalysisLayout,
  HeatmapRangeRun,
  HeatmapRun,
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
};

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

function metricKeyForRun(run: TestingRun): string {
  const configMetric = run.inference_config?.error_metric ?? run.inference_config?.residual_metric;
  if (typeof configMetric === 'string' && configMetric.trim()) return normalizeMetricKey(configMetric);
  return 'mse';
}

function normalizeMetricKey(metric: string): string {
  const normalized = metric.trim().toLowerCase();
  if (normalized === 'ssim' || normalized === 'ssim_distance') return 'ssim_distance';
  if (normalized === 'l1' || normalized === 'mae') return 'mae';
  if (normalized === 'l2' || normalized === 'mse') return 'mse';
  return normalized || 'mse';
}

function metricLabel(metric: string): string {
  const normalized = normalizeMetricKey(metric);
  if (normalized === 'mse') return 'MSE';
  if (normalized === 'mae') return 'MAE';
  if (normalized === 'ssim_distance') return 'SSIM';
  return normalized.replaceAll('_', ' ').toUpperCase();
}

function metricOrder(metric: string): number {
  const normalized = normalizeMetricKey(metric);
  if (normalized === 'mse') return 0;
  if (normalized === 'mae') return 1;
  if (normalized === 'ssim_distance') return 2;
  return 10;
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
  return plot.traces?.length ? plot.traces.map(traceToSource) : plot.sources;
}

function sourceTraceKey(source: PlotSourceConfig): string {
  return `${source.testingRunId}`;
}

function traceLabelForRun(run: TestingRun, metric: string, multipleMetrics: boolean): string {
  const modelLabel = run.training_pipeline_name || run.training_run_name || `Training run #${run.training_run_id}`;
  return multipleMetrics ? `${modelLabel} · ${metricLabel(metric)}` : modelLabel;
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
  { kind: 'robust_z', label: 'Robust z-score', description: 'Score relative to rolling median and MAD.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, epsilon: 1e-12 } },
  { kind: 'positive_exceedance', label: 'Positive exceedance', description: 'Positive part above a z-score threshold.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, threshold: 1, epsilon: 1e-12 } },
  { kind: 'rolling_area', label: 'Rolling area', description: 'Accumulated positive exceedance in a causal window.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, baselineWindowSamples: 60, baselineWindowMinutes: 60, alpha: 0.2, threshold: 1, epsilon: 1e-12 } },
  { kind: 'rolling_mean', label: 'Rolling mean', description: 'Causal local average of the smoothed score.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, alpha: 0.2 } },
  { kind: 'rolling_max', label: 'Rolling maximum', description: 'Causal local maximum.', defaultParams: { windowMode: 'samples', windowSamples: 20, windowMinutes: 5, alpha: 0.2 } },
  { kind: 'drawdown', label: 'Drawdown', description: 'Drop from causal rolling maximum.', defaultParams: { windowMode: 'samples', windowSamples: 20, windowMinutes: 5, alpha: 0.2, mode: 'relative', epsilon: 1e-12 } },
  { kind: 'positive_slope_count', label: 'Positive slope count', description: 'Number of positive slopes in the causal window.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, alpha: 0.2, slopeThreshold: 0 } },
  { kind: 'positive_slope_fraction', label: 'Positive slope fraction', description: 'Fraction of slopes above threshold in the causal window.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, alpha: 0.2, slopeThreshold: 0 } },
  { kind: 'rising_streak', label: 'Rising streak', description: 'Current consecutive count of positive slopes.', defaultParams: { alpha: 0.2, slopeThreshold: 0 } },
  { kind: 'cusum', label: 'CUSUM', description: 'Positive evidence accumulator on robust z-score.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, k: 1, h: 8, epsilon: 1e-12 } },
  { kind: 'page_hinkley', label: 'Page-Hinkley', description: 'Online mean-shift accumulator.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, delta: 0.2, lambda: 8, epsilon: 1e-12 } },
  { kind: 'evidence_score', label: 'Evidence score', description: 'Online positive/negative evidence score.', defaultParams: { windowMode: 'samples', windowSamples: 20, windowMinutes: 5, alpha: 0.2, zThreshold: 1, slopeThreshold: 0, w1: 1, w2: 1, w3: 0.2, v1: 1, v2: 0.5, v3: 1, epsilon: 1e-12 } },
  { kind: 'slope_height_ratio', label: 'Slope / height ratio', description: 'Current slope relative to robust z-score height.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, epsilon: 1e-12 } },
  { kind: 'energy_ratio', label: 'Short / long energy ratio', description: 'Short rolling area divided by long rolling area.', defaultParams: { windowMode: 'samples', windowSamples: 12, windowMinutes: 3, longWindowSamples: 40, longWindowMinutes: 10, baselineWindowSamples: 60, baselineWindowMinutes: 60, alpha: 0.2, threshold: 1, epsilon: 1e-12 } },
  { kind: 'rolling_std', label: 'Rolling std', description: 'Causal local standard deviation.', defaultParams: { windowMode: 'samples', windowSamples: 20, windowMinutes: 5, alpha: 0.2 } },
  { kind: 'rolling_cv', label: 'Rolling coefficient of variation', description: 'Rolling std divided by rolling mean.', defaultParams: { windowMode: 'samples', windowSamples: 20, windowMinutes: 5, alpha: 0.2, epsilon: 1e-12 } },
  { kind: 'time_since_onset', label: 'Time since onset', description: 'Elapsed time since z and slope crossed onset thresholds.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, onsetThreshold: 1, slopeThreshold: 0, resetThreshold: 0.5, epsilon: 1e-12 } },
  { kind: 'state_machine', label: 'State machine', description: 'Visual state band from z-score, slope and CUSUM thresholds.', defaultParams: { windowMode: 'samples', windowSamples: 60, windowMinutes: 60, alpha: 0.2, lowThreshold: 1, slopeThreshold: 0, hLow: 5, hHigh: 10, offThreshold: 0.5, epsilon: 1e-12 } },
];

function analyticsDefinition(kind: AnalyticsKind): AnalyticsDefinition {
  return ANALYTICS_DEFINITIONS.find((definition) => definition.kind === kind) ?? ANALYTICS_DEFINITIONS[0];
}

function defaultAnalyticsConfig(kind: AnalyticsKind): AnalyticsMethodConfig {
  const definition = analyticsDefinition(kind);
  return { kind, params: { ...definition.defaultParams } };
}

function analyticsSummary(configs: AnalyticsMethodConfig[]): string {
  if (configs.length === 0) return 'Analytics: none';
  return `Analytics: ${configs.map((config) => {
    const params = config.params;
    const details: string[] = [];
    if (typeof params.alpha === 'number') details.push(`alpha=${params.alpha}`);
    if (typeof params.beta === 'number') details.push(`beta=${params.beta}`);
    if (typeof params.windowSamples === 'number' && params.windowMode === 'samples') details.push(`W=${params.windowSamples} samples`);
    if (typeof params.windowMinutes === 'number' && params.windowMode === 'minutes') details.push(`W=${params.windowMinutes} min`);
    if (typeof params.k === 'number') details.push(`k=${params.k}`);
    if (typeof params.h === 'number') details.push(`h=${params.h}`);
    return `${analyticsDefinition(config.kind).label}${details.length ? ` (${details.join(', ')})` : ''}`;
  }).join(', ')}`;
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
  };
  return infos[key] ?? 'Parameter used by this causal time-series analytics stage.';
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
  const epsilon = numberParam(config, 'epsilon', 1e-12);
  const smooth = ewma(values, alpha);
  const baseline = rollingMap(smooth, times, config, (slice) => median(slice));
  const mad = smooth.map((_, index) => {
    const start = windowStartIndex(times, index, config);
    const deviations = smooth.slice(start, index + 1).map((value) => Math.abs(value - baseline[index]));
    return median(deviations);
  });
  return {
    z: smooth.map((value, index) => (value - baseline[index]) / (1.4826 * mad[index] + epsilon)),
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
      let g = 0;
      return z.map((value) => {
        g = Math.max(0, g + value - k);
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
      const cusumConfig = { ...config, kind: 'cusum' as AnalyticsKind, params: { ...config.params, h: numberParam(config, 'hHigh', 10) } };
      const cusum = computeAnalyticsSeries(cusumConfig, values, timestamps).map((value) => value ?? 0);
      const low = numberParam(config, 'lowThreshold', 1);
      const slope = numberParam(config, 'slopeThreshold', 0);
      const hLow = numberParam(config, 'hLow', 5);
      const hHigh = numberParam(config, 'hHigh', 10);
      const off = numberParam(config, 'offThreshold', 0.5);
      let state = 0;
      return z.map((value, index) => {
        if (cusum[index] >= hHigh) state = 3;
        else if (cusum[index] >= hLow) state = 2;
        else if (value > low && d[index] > slope) state = 1;
        else if (state > 0 && value < off) state = 0;
        return state;
      });
    }
    default:
      return values.map(finiteOrNull);
  }
}

function TimeSeriesPlot({ plot, results }: { plot: AnalysisPlot; results: CombinedResult[] }) {
  const analyticsConfigs = plot.timeseriesAnalytics ?? [];
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
      const x = group.results.map((result) => result.timestamp);
      const rawValues = group.results.map((result) => scoreValue(result, plot.scoreSeries));
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
        nextTraces.push({
          type: 'scatter',
          mode: 'lines',
          name: panelIndex === 0 ? group.name : `${group.name} · ${analyticsDefinition(panel.kind).label}`,
          x,
          y,
          xaxis: 'x',
          yaxis: panelIndex === 0 ? 'y' : `y${panelIndex + 1}`,
          line: { color: group.color || TRACE_COLORS[groupIndex % TRACE_COLORS.length], width: panel.kind === 'state_machine' ? 2.2 : 1.7, shape: panel.kind === 'state_machine' ? 'hv' : 'linear' },
          showlegend: panelIndex === 0,
          hovertemplate: `%{x|%Y-%m-%d %H:%M:%S}<br>${analyticsDefinition(panel.kind).label} %{y:.5g}<extra>${group.name}</extra>`,
        } as unknown as Data);
      });
    });
    return nextTraces;
  }, [analyticsConfigs, displayPanels, plot.movingAverage, plot.scoreSeries, plot.traces, results]);

  const layout = useMemo<Partial<Layout>>(
    () => {
      const panelCount = Math.max(1, displayPanels.length);
      const gap = 0.035;
      const panelHeight = (1 - gap * (panelCount - 1)) / panelCount;
      const nextLayout: Partial<Layout> & Record<string, unknown> = {
        showlegend: traces.length > panelCount,
        legend: { orientation: 'h', y: -0.18, x: 0 },
        hovermode: 'x unified',
        xaxis: {
          title: { text: 'Time', font: { size: 12 } },
          type: 'date',
          rangeslider: panelCount === 1 ? { thickness: 0.08 } : undefined,
          showgrid: true,
          gridcolor: 'rgba(128,128,128,0.15)',
        },
        margin: { l: 72, r: 24, t: 12, b: 58 },
      };
      displayPanels.forEach((panel, index) => {
        const top = 1 - index * (panelHeight + gap);
        const bottom = top - panelHeight;
        nextLayout[index === 0 ? 'yaxis' : `yaxis${index + 1}`] = {
          title: { text: analyticsDefinition(panel.kind).label, font: { size: 11 } },
          domain: [Math.max(0, bottom), Math.min(1, top)],
          showgrid: true,
          gridcolor: 'rgba(128,128,128,0.15)',
          zeroline: panel.kind !== 'raw' && panel.kind !== 'ewma',
        };
      });
      return nextLayout as Partial<Layout>;
    },
    [displayPanels, traces.length],
  );

  if (results.length === 0) {
    return <Alert color="yellow">No results match this time range.</Alert>;
  }

  return (
    <Stack gap="xs">
      <PlotlyChart
        data={traces}
        layout={layout}
        height={analyticsConfigs.length > 0 ? Math.max(520, displayPanels.length * (plot.panelHeightPx || 260)) : (plot.panelHeightPx || 420)}
      />
      <Group gap="xs">
        <Badge variant="light">{results.length} points</Badge>
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

  const errorPanel = heatmap ? (
    <div className="analysis-heatmap-panel">
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
            <SimpleGrid cols={{ base: 1, md: 3 }} spacing="md">
              <div className="analysis-heatmap-panel">
                <Text size="sm" fw={500} c="dimmed" ta="center">
                  Original
                </Text>
                <div className="analysis-heatmap-image-frame" style={frameStyle}>
                  <img src={heatmap.source_image_data_url} alt="Original source" className="analysis-heatmap-image" />
                </div>
              </div>
              <div className="analysis-heatmap-panel">
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
      visualizationConfigKey: heatmapConfigKey(plot.heatmapConfig),
    };
  }, [plot.heatmapConfig, plot.sources, results]);

  const [scaleMode, setScaleMode] = useState<'per_frame' | 'shared'>('per_frame');
  const [job, setJob] = useState<HeatmapRangeRun | null>(null);
  const [starting, setStarting] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(8);
  const fixedCeiling = plot.heatmapConfig.fixed_ceiling_enabled;
  const effectiveScaleMode: 'per_frame' | 'shared' = fixedCeiling ? 'per_frame' : scaleMode;

  // Reset whenever the plot's range/source changes.
  useEffect(() => {
    setJob(null);
    setVideoError(null);
    setFrameIndex(0);
    setPlaying(false);
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
  const ready = job?.status === 'finished' && frameCount > 0;

  useEffect(() => {
    if (!ready || !playing || frameCount <= 1) return undefined;
    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % frameCount);
    }, Math.max(33, 1000 / Math.max(1, fps)));
    return () => window.clearInterval(timer);
  }, [ready, playing, frameCount, fps]);

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
        scale_mode: effectiveScaleMode,
        visualization_config: params.visualizationConfig,
      });
      setJob(created);
      setFrameIndex(0);
    } catch (error) {
      setVideoError(errorMessage(error));
    } finally {
      setStarting(false);
    }
  }

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
          <Group justify="space-between" align="center">
            <Group gap="xs">
              <Button
                size="compact-sm"
                variant="light"
                leftSection={playing ? <Pause size={14} /> : <Play size={14} />}
                onClick={() => setPlaying((current) => !current)}
              >
                {playing ? 'Pause' : 'Play'}
              </Button>
              <Text size="xs" c="dimmed">Frame {frameIndex + 1}/{frameCount}</Text>
            </Group>
            <NumberInput
              label="fps"
              size="xs"
              w={90}
              min={1}
              max={60}
              value={fps}
              onChange={(value) => setFps(Math.max(1, Math.min(60, Number(value) || 1)))}
            />
          </Group>
          <div className="analysis-heatmap-image-frame analysis-heatmap-video-frame">
            <img
              src={heatmapRangeFrameUrl(job.id, frameIndex)}
              alt={`Heatmap frame ${frameIndex + 1}`}
              className="analysis-heatmap-image"
            />
          </div>
          <input
            type="range"
            min={0}
            max={frameCount - 1}
            value={frameIndex}
            className="analysis-frame-slider"
            onChange={(event) => setFrameIndex(Number(event.currentTarget.value))}
          />
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
                : formatMetric(job.frame_max_errors?.[frameIndex])}
            </Text>
          </Stack>
        </Stack>
      )}
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
}) {
  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Group justify="space-between" align="flex-start">
          <div>
            <Group gap="xs">
              <Title order={4} fw={500}>{plot.title}</Title>
              <Badge variant="light" color={plot.plotType === 'heatmap' ? 'red' : 'blue'}>
                {plot.plotType === 'heatmap' ? 'Heatmap' : 'Time series'}
              </Badge>
            </Group>
            {plot.subtitle && (
              <Text size="sm" c="dimmed" mt={2}>
                {plot.subtitle}
              </Text>
            )}
            <Text size="sm" c="dimmed">
              {results.length} result rows · {plotSources(plot).length} source{plotSources(plot).length === 1 ? '' : 's'}
            </Text>
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
        {plot.plotType === 'timeseries' ? (
          <TimeSeriesPlot plot={plot} results={results} />
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
      </Stack>
    </Paper>
  );
});

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

type AnalysisBoardLayout = {
  version: 1 | 2;
  draft: PlotDraft;
  plots: AnalysisPlot[];
  selectedPipelineId: string | null;
  selectedModelIds?: string[];
  selectedInferenceDatasetId?: string | null;
  selectedMetricKeys?: string[];
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
  return {
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
  };
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
      modelLabel: String(trace.modelLabel ?? trace.legendLabel ?? `Source ${index + 1}`),
      legendLabel: String(trace.legendLabel ?? trace.modelLabel ?? `Source ${index + 1}`),
      color: String(trace.color ?? TRACE_COLORS[index % TRACE_COLORS.length]),
    };
  }).filter((trace) => trace.testingRunId);
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
  const [resultsByRunId, setResultsByRunId] = useState<Record<number, TestingRunResults>>({});
  const [loadingRunId, setLoadingRunId] = useState<number | null>(null);
  const [draft, setDraft] = useState<PlotDraft>(defaultDraft());
  const [plots, setPlots] = useState<AnalysisPlot[]>([]);
  const [heatmapCache, setHeatmapCache] = useState<Record<string, HeatmapRun>>({});
  const [loadingHeatmaps, setLoadingHeatmaps] = useState<Record<string, boolean>>({});
  const [heatmapErrors, setHeatmapErrors] = useState<Record<string, string>>({});
  const [addPlotOpen, setAddPlotOpen] = useState(true);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);
  const [pipelineSearch, setPipelineSearch] = useState('');
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [selectedInferenceDatasetId, setSelectedInferenceDatasetId] = useState<string | null>(null);
  const [selectedMetricKeys, setSelectedMetricKeys] = useState<string[]>([]);
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

  async function refresh() {
    const [nextTestingRuns, nextTrainingRuns, nextDatasets, nextPipelines, nextPreprocessing, nextMethods, nextLayouts] =
      await Promise.all([
        listTestingRuns(),
        listTrainingRuns(),
        listTrainingDatasets(),
        listTrainingPipelines(),
        listPreprocessingPipelines(),
        listMethodConfigurations(),
        listAnalysisLayouts(),
      ]);
    setTestingRuns(nextTestingRuns);
    setTrainingRuns(nextTrainingRuns);
    setTrainingDatasets(nextDatasets);
    setTrainingPipelines(nextPipelines);
    setPreprocessingPipelines(nextPreprocessing);
    setMethodConfigurations(nextMethods);
    setAnalysisLayouts(nextLayouts);
  }

  useEffect(() => {
    if (!active) return;
    refresh().catch((error) => notifyError('Could not load testing runs', error));
  }, [active]);

  const finishedRuns = useMemo(
    () => testingRuns.filter((run) => run.status === 'finished' && (run.image_count ?? 0) > 0),
    [testingRuns],
  );

  const selectedRunId = draft.testingRunId ? Number(draft.testingRunId) : null;
  const trainingRunById = useMemo(() => new Map(trainingRuns.map((run) => [run.id, run])), [trainingRuns]);
  const trainingDatasetById = useMemo(() => new Map(trainingDatasets.map((dataset) => [dataset.id, dataset])), [trainingDatasets]);
  const trainingPipelineById = useMemo(() => new Map(trainingPipelines.map((pipeline) => [pipeline.id, pipeline])), [trainingPipelines]);
  const preprocessingById = useMemo(() => new Map(preprocessingPipelines.map((pipeline) => [pipeline.id, pipeline])), [preprocessingPipelines]);
  const methodById = useMemo(() => new Map(methodConfigurations.map((method) => [method.id, method])), [methodConfigurations]);
  const selectedLayout = selectedLayoutId ? analysisLayouts.find((layout) => layout.id === Number(selectedLayoutId)) ?? null : null;
  const layoutNameTrimmed = layoutName.trim();
  const layoutNameExistsForCreate = analysisLayouts.some(
    (layout) => layout.name.toLowerCase() === layoutNameTrimmed.toLowerCase(),
  );
  const layoutNameExistsForUpdate = analysisLayouts.some(
    (layout) => layout.id !== Number(selectedLayoutId) && layout.name.toLowerCase() === layoutNameTrimmed.toLowerCase(),
  );

  const analysableModelRows = useMemo(() => {
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
    const query = pipelineSearch.trim().toLowerCase();
    return [...byTrainingRun.values()]
      .filter((row) => {
        if (!query) return true;
        return row.label.toLowerCase().includes(query) || row.testingRuns.some((run) => run.training_dataset_name.toLowerCase().includes(query));
      })
      .sort((left, right) => left.label.localeCompare(right.label));
  }, [finishedRuns, pipelineSearch, trainingRunById]);

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

  const selectedInferenceDataset = selectedInferenceDatasetId ? trainingDatasetById.get(Number(selectedInferenceDatasetId)) ?? null : null;
  const selectedInferenceBounds = sourceBounds(selectedInferenceDataset);

  const roiOptionsForSelection = useMemo(() => {
    if (!selectedInferenceDatasetId || selectedModelIds.length === 0) return [{ value: 'none', label: 'No ROI' }];
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
    const sorted = commonKeys
      .map((key) => ({ value: key, label: sets[0].get(key) ?? (key === 'none' ? 'No ROI' : `ROI #${key}`) }))
      .sort((left, right) => (left.value === 'none' ? -1 : right.value === 'none' ? 1 : left.label.localeCompare(right.label)));
    if (!sorted.some((option) => option.value === 'none')) sorted.unshift({ value: 'none', label: 'No ROI' });
    return sorted.length > 0 ? sorted : [{ value: 'none', label: 'No ROI' }];
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
      .map((metric) => ({ value: metric, label: metricLabel(metric) }));
  }, [finishedRuns, selectedInferenceDatasetId, selectedModelIds, selectedRoiKey]);

  useEffect(() => {
    if (selectedInferenceDatasetId && !commonDatasetOptions.some((option) => option.value === selectedInferenceDatasetId)) {
      setSelectedInferenceDatasetId(null);
      setSelectedMetricKeys([]);
      resetSelectionPreview();
    }
  }, [commonDatasetOptions, selectedInferenceDatasetId]);

  useEffect(() => {
    if (!roiOptionsForSelection.some((option) => option.value === selectedRoiKey)) {
      setSelectedRoiKey('none');
      setSelectedMetricKeys([]);
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
      version: 2,
      draft,
      plots,
      selectedPipelineId,
      selectedModelIds,
      selectedInferenceDatasetId,
      selectedMetricKeys,
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
      setDraft(restored.draft);
      setPlots(restored.plots);
      setSelectedPipelineId(restored.selectedPipelineId);
      setSelectedModelIds(restored.selectedModelIds ?? []);
      setSelectedInferenceDatasetId(restored.selectedInferenceDatasetId ?? null);
      setSelectedMetricKeys(restored.selectedMetricKeys ?? []);
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
      if (!options?.force && (heatmapCache[key] || loadingHeatmaps[key] || heatmapErrors[key])) return;
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
      setDraft((current) => {
        if (current.testingRunId !== String(selectedRunId)) return current;
        return {
          ...current,
          title: current.title || `Heatmap · ${run?.name ?? `Testing run #${selectedRunId}`}`,
          start: current.start || bounds.start,
          end: current.end || bounds.end,
          timestamp: current.timestamp ?? bounds.start,
        };
      });
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
            title: current.title || `${current.plotType === 'heatmap' ? 'Heatmap' : 'Time series'} · ${data.testing_run.name}`,
            start: current.start || bounds.start || toDateTimeLocal(first.timestamp),
            end: current.end || bounds.end || toDateTimeLocal(last.timestamp),
            timestamp: current.timestamp ?? first.timestamp,
          };
        });
      })
      .catch((error) => notifyError('Could not load testing results', error));
  }, [draft.heatmapMode, draft.plotType, fetchResults, plotPreview, selectedRunId, testingRuns, trainingDatasetById]);

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

  function resolveTestingRun(modelId: number, metric: string): { run: TestingRun | null; duplicateCount: number } {
    if (!selectedInferenceDatasetId || !selectedRoiKey) return { run: null, duplicateCount: 0 };
    const candidates = finishedRuns
      .filter((run) => run.training_run_id === modelId)
      .filter((run) => run.training_dataset_id === Number(selectedInferenceDatasetId))
      .filter((run) => (run.roi_id === null ? 'none' : String(run.roi_id)) === selectedRoiKey)
      .filter((run) => metricKeyForRun(run) === metric)
      .sort((left, right) => {
        const leftTime = new Date(left.ended_at ?? left.updated_at ?? left.created_at).getTime();
        const rightTime = new Date(right.ended_at ?? right.updated_at ?? right.created_at).getTime();
        return rightTime - leftTime;
      });
    return { run: candidates[0] ?? null, duplicateCount: Math.max(0, candidates.length - 1) };
  }

  function selectedModelLabel(modelId: string): string {
    return analysableModelRows.find((row) => row.id === Number(modelId))?.label ?? `Training run #${modelId}`;
  }

  function autoPlotTitle(): string {
    const modelPart = selectedModelIds.length === 1 ? selectedModelLabel(selectedModelIds[0]) : `${selectedModelIds.length} models`;
    const datasetPart = selectedInferenceDataset?.name ?? 'Inference dataset';
    const roiPart = selectedRoiKey && selectedRoiKey !== 'none'
      ? ` · ${roiOptionsForSelection.find((option) => option.value === selectedRoiKey)?.label ?? `ROI #${selectedRoiKey}`}`
      : '';
    return `${modelPart} · ${datasetPart}${roiPart}`;
  }

  function buildPlotSubtitle(metrics: string[], start: string, end: string, sampling: number): string {
    const parts = [
      `Metrics: ${metrics.map(metricLabel).join(', ')}`,
      `Range: ${start.replace('T', ' ')} to ${end.replace('T', ' ')}`,
      `Sampling: every ${sampling}`,
    ];
    if (draft.movingAverage > 1) parts.push(`Moving average: ${draft.movingAverage}`);
    if (draft.plotType === 'timeseries') parts.push(analyticsSummary(draft.timeseriesAnalytics));
    if (selectedRoiKey && selectedRoiKey !== 'none') {
      parts.push(`ROI: ${roiOptionsForSelection.find((option) => option.value === selectedRoiKey)?.label ?? selectedRoiKey}`);
    } else {
      parts.push('ROI: No ROI');
    }
    return parts.join(' · ');
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
    if (draft.plotType === 'heatmap' && (selectedModelIds.length > 1 || selectedMetricKeys.length > 1)) {
      notifications.show({ color: 'yellow', title: 'Heatmap needs one source', message: 'Heatmaps currently support one model and one metric. Use time series for multi-model comparisons.' });
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
          const { run, duplicateCount } = resolveTestingRun(Number(modelId), metric);
          if (!run) {
            throw new Error(`No finished ${metricLabel(metric)} inference run found for ${selectedModelLabel(modelId)}.`);
          }
          if (duplicateCount > 0) {
            duplicateNotes.push(`${selectedModelLabel(modelId)} / ${metricLabel(metric)}: newest run used, ${duplicateCount} older duplicate${duplicateCount === 1 ? '' : 's'} ignored.`);
          }
          const previousTrace = plotPreview?.traces.find((trace) => trace.testingRunId === String(run.id) && trace.metric === metric);
          traces.push({
            testingRunId: String(run.id),
            metric,
            modelLabel: selectedModelLabel(modelId),
            legendLabel: previousTrace?.legendLabel ?? traceLabelForRun(run, metric, selectedMetricKeys.length > 1),
            color: previousTrace?.color ?? TRACE_COLORS[traces.length % TRACE_COLORS.length],
            start,
            end,
            sampling,
            timestamp: draft.timestamp ?? start,
          });
        }
      }
      if (draft.plotType !== 'heatmap' || draft.heatmapMode !== 'single') {
        await Promise.all(traces.map((trace) => fetchResults(Number(trace.testingRunId))));
      }
      setDraft((current) => ({
        ...current,
        title: current.title,
        subtitle: buildPlotSubtitle(selectedMetricKeys, start, end, sampling),
        testingRunId: traces[0]?.testingRunId ?? current.testingRunId,
        start,
        end,
        sampling,
        timestamp: current.timestamp ?? traces[0]?.timestamp ?? start,
        scoreSeries: 'score',
      }));
      const title = draft.title.trim() || autoPlotTitle();
      const subtitle = buildPlotSubtitle(selectedMetricKeys, start, end, sampling);
      const previewPlot: AnalysisPlot = {
        ...draft,
        id: editingPlot?.plot.id ?? 'preview',
        title,
        subtitle,
        sources: traces.map(traceToSource),
        traces,
        testingRunId: traces[0]?.testingRunId ?? draft.testingRunId,
        timestamp: draft.timestamp ?? traces[0]?.timestamp ?? start,
      };
      setSelectedSources(traces.map(traceToSource));
      setPlotPreview({
        title,
        subtitle,
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
      return {
        ...current,
        traces,
        plot: {
          ...current.plot,
          sources: traces.map(traceToSource),
          traces,
        },
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
    const nextPlot: AnalysisPlot = {
      ...plotPreview.plot,
      id: editingPlot?.plot.id ?? crypto.randomUUID(),
      timestamp: plotPreview.plot.timestamp ?? availableResults[0]?.timestamp ?? plotPreview.traces[0]?.timestamp ?? null,
    };
    setPlots((current) => {
      if (!editingPlot) return [...current, nextPlot];
      const next = [...current];
      next.splice(Math.min(editingPlot.index, next.length), 0, nextPlot);
      return next;
    });
    setEditingPlot(null);
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
        modelLabel: run?.training_pipeline_name ?? `Source ${sourceIndex + 1}`,
        legendLabel: run?.training_pipeline_name ?? `Source ${sourceIndex + 1}`,
        color: TRACE_COLORS[sourceIndex % TRACE_COLORS.length],
      };
    });
    const firstRun = testingRuns.find((run) => run.id === Number(traces[0]?.testingRunId));
    setEditingPlot({ plot, index });
    setPlots((current) => current.filter((item) => item.id !== plot.id));
    setDraft({ ...plot, traces: undefined } as PlotDraft);
    setSelectedSources(traces.map(traceToSource));
    setSelectedModelIds([...new Set(traces.map((trace) => testingRuns.find((run) => run.id === Number(trace.testingRunId))?.training_run_id).filter((id): id is number => typeof id === 'number').map(String))]);
    setSelectedInferenceDatasetId(firstRun ? String(firstRun.training_dataset_id) : null);
    setSelectedRoiKey(firstRun?.roi_id === null || firstRun?.roi_id === undefined ? 'none' : String(firstRun.roi_id));
    setSelectedMetricKeys([...new Set(traces.map((trace) => trace.metric))]);
    setPlotPreview({
      title: plot.title,
      subtitle: plot.subtitle,
      traces,
      duplicateNotes: [],
      plot: {
        ...plot,
        sources: traces.map(traceToSource),
        traces,
      },
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
          <Title order={2}>Analysis</Title>
          <Text c="dimmed" size="sm">
            Compose interactive plots from saved testing runs and compare them on one board.
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
                  <TextInput
                    placeholder="Search by model, pipeline or inference dataset"
                    leftSection={<Search size={16} />}
                    value={pipelineSearch}
                    onChange={(event) => setPipelineSearch(event.currentTarget.value)}
                  />
                  <MultiSelect
                    label="Selected models"
                    placeholder="Choose one or more trained models"
                    data={analysableModelRows.map((row) => ({ value: String(row.id), label: row.label }))}
                    value={selectedModelIds}
                    searchable
                    clearable
                    onChange={(values) => {
                      setSelectedModelIds(values);
                      setSelectedInferenceDatasetId(null);
                      setSelectedMetricKeys([]);
                      setSelectedRoiKey('none');
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
                        {analysableModelRows.map((row) => {
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
                                      setSelectedInferenceDatasetId(null);
                                      setSelectedMetricKeys([]);
                                      setSelectedRoiKey('none');
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
                </Stack>
              </StepCard>

              {selectedModelIds.length > 0 && (
                <StepCard index={2} title="Inference datasets" color="teal">
                  <Stack gap="sm">
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
                            {commonDatasetOptions.map((option) => {
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
                                        setSelectedMetricKeys([]);
                                        setSelectedRoiKey('none');
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
                            {commonDatasetOptions.length === 0 && (
                              <Table.Tr>
                                <Table.Td colSpan={9}>
                                  <Text c="dimmed" ta="center" py="md">
                                    No inference dataset is available for all selected models.
                                  </Text>
                                </Table.Td>
                              </Table.Tr>
                            )}
                          </Table.Tbody>
                        </Table>
                      </ScrollArea>
                    </Paper>
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
                        setSelectedRoiKey(value ?? 'none');
                        setSelectedMetricKeys([]);
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
                </StepCard>
              )}

              {selectedInferenceDatasetId && selectedRoiKey && (
                <StepCard index={4} title="Plot configuration" color="gray">
                  <Stack gap="md">
                    <TextInput
                      label="Plot title"
                      placeholder={autoPlotTitle()}
                      value={draft.title}
                      onChange={(event) => {
                        const value = event.currentTarget.value;
                        setDraft((current) => ({ ...current, title: value }));
                        markPreviewStale();
                      }}
                    />
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
                  {combinedDraftResults.some((result) => result.result_metadata?.sample_kind === 'clip') && (
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
                            <Text size="sm" c="dimmed">{plotPreview.title}</Text>
                            <Text size="xs" c="dimmed">{plotPreview.subtitle}</Text>
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
            return plotData.hasAllData ? (
              <AnalysisPlotCard
                key={plot.id}
                plot={plot}
                results={plotData.results}
                heatmapCache={heatmapCache}
                loadingHeatmaps={loadingHeatmaps}
                heatmapErrors={heatmapErrors}
                ensureHeatmap={ensureHeatmap}
                onMove={(direction) => movePlot(plot.id, direction)}
                onEdit={() => editPlot(plot, plots.findIndex((item) => item.id === plot.id))}
                onPatch={(patch) =>
                  setPlots((current) =>
                    current.map((item) => (item.id === plot.id ? { ...item, ...patch } : item)),
                  )
                }
                onRemove={() => setPlots((current) => current.filter((item) => item.id !== plot.id))}
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
      <DetailModal detail={detailModal} onClose={() => setDetailModal(null)} />
    </Stack>
  );
}
