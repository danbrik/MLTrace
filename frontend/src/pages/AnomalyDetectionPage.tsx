import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Collapse,
  Group,
  Loader,
  MultiSelect,
  NumberInput,
  Paper,
  Progress,
  ScrollArea,
  SegmentedControl,
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
import { Activity, ChevronDown, ChevronRight, Download, Info, Play, RefreshCw, Search, SlidersHorizontal, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  createAnomalyDetectionRun,
  deleteAnomalyDetectionRun,
  getAnomalyDetectionDiagnostics,
  getAnomalyDetectionProgress,
  getAnomalyDetectionRun,
  getTestingRunResults,
  listAnomalyDetectionRuns,
  listTestingRuns,
  previewAnomalyDetectionThreshold,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart, type PlotlyChartSelection } from '../components/PlotlyChart';
import { DEFAULT_TABLE_PAGE_SIZE, TablePagination } from '../components/TablePagination';
import type { Data, Layout } from '../lib/plotly';
import {
  countFacetValues,
  facetOption,
  matchingFacetRecords,
  type FacetFilterState,
  type FacetRecord,
} from '../testing/facetFilters';
import {
  aggregationKeyForRun,
  aggregationLabel,
  metricKeyForRun,
  metricLabel,
  roiKeyForRun,
  roiLabelForRun,
} from '../testing/inferenceRunMetadata';
import type {
  AnomalyDetectionConfig,
  AnomalyDetectionAlgorithm,
  AnomalyDetectionProgress,
  AnomalyDetectionRun,
  AnomalyDetectionRunSummary,
  AnomalyDetectionScoreSeries,
  AnomalyDetectionSeriesPoint,
  AnomalyDetectionThresholdPreview,
  TestingRun,
  TestingRunResult,
} from '../types';

const MAX_PREVIEW_POINTS = 8000;
const MAX_DIAGNOSTIC_POINTS = 50000;

const FAST_CONFIG: AnomalyDetectionConfig = {
  algorithm: 'robust_cusum',
  smoothing_half_life_minutes: 5,
  baseline_window_minutes: 120,
  warmup_minutes: 30,
  minimum_warmup_points: 30,
  warning_z: 3,
  high_z: 5,
  minimum_scale_relative: 1e-3,
  minimum_scale_absolute: 1e-9,
  cusum_drift: 1,
  cusum_threshold: 10,
  cusum_z_cap: 20,
  confirmation_mode: 'minutes',
  confirmation_minutes: 5,
  confirmation_samples: 1,
  recovery_z: 1,
  recovery_minutes: 15,
  fallback_recovery_minutes: 60,
  preroll_minutes: 120,
  gap_multiplier: 5,
  minimum_gap_minutes: 15,
  event_smoothing_enabled: true,
  event_smoothing_method: 'median',
  event_smoothing_window_seconds: 5,
  threshold_mode: 'quantile',
  manual_threshold: null,
  threshold_quantile: 0.9999,
  persistence_k: 10,
  persistence_n: 15,
  threshold_off_factor: 0.8,
  normal_close_seconds: 30,
  merge_gap_seconds: 60,
  event_minimum_gap_seconds: 15,
  sigma_threshold: 3,
};

const EVENT_CONFIG: AnomalyDetectionConfig = {
  ...FAST_CONFIG,
  algorithm: 'event_threshold',
};

const SIGMA_CONFIG: AnomalyDetectionConfig = {
  ...FAST_CONFIG,
  algorithm: 'rolling_sigma',
  baseline_window_minutes: 30,
  warmup_minutes: 5,
  minimum_warmup_points: 30,
  sigma_threshold: 3,
  preroll_minutes: 30,
  confirmation_mode: 'samples',
  confirmation_samples: 1,
};

const BALANCED_CONFIG: AnomalyDetectionConfig = {
  ...FAST_CONFIG,
  smoothing_half_life_minutes: 10,
  baseline_window_minutes: 360,
  warmup_minutes: 60,
  minimum_warmup_points: 60,
  warning_z: 3.5,
  high_z: 5.5,
  cusum_threshold: 12,
  confirmation_minutes: 10,
  recovery_minutes: 30,
  fallback_recovery_minutes: 120,
};

const ROBUST_CONFIG: AnomalyDetectionConfig = {
  ...FAST_CONFIG,
  smoothing_half_life_minutes: 20,
  baseline_window_minutes: 720,
  warmup_minutes: 120,
  minimum_warmup_points: 120,
  warning_z: 4,
  high_z: 6,
  cusum_drift: 1.5,
  cusum_threshold: 15,
  confirmation_minutes: 20,
  recovery_z: 0.75,
  recovery_minutes: 45,
  fallback_recovery_minutes: 240,
};

const PRESETS: Record<string, AnomalyDetectionConfig> = {
  fast: FAST_CONFIG,
  balanced: BALANCED_CONFIG,
  robust: ROBUST_CONFIG,
};

const ALGORITHMS: Array<{
  value: AnomalyDetectionAlgorithm;
  label: string;
  description: string;
}> = [
  {
    value: 'robust_cusum',
    label: 'Robust Z-score + CUSUM (recommended)',
    description: 'Starts a warning from the robust Z-score and confirms sustained changes using either a high Z-score or accumulated CUSUM evidence.',
  },
  {
    value: 'robust_zscore',
    label: 'Robust Z-score',
    description: 'Uses the rolling median and MAD only. It confirms an event when the high Z-score remains present for the confirmation time.',
  },
  {
    value: 'event_threshold',
    label: 'Event Threshold (K-out-of-N)',
    description: 'Smooths the reconstruction error, applies a manual or validation quantile threshold, and confirms an event when K of the last N samples exceed it.',
  },
  {
    value: 'rolling_sigma',
    label: 'Rolling baseline + 3σ',
    description: 'Uses the unchanged raw reconstruction error and confirms an anomaly after the configured consecutive time or sample count above the rolling standard-deviation threshold.',
  },
];

type NumericConfigKey = {
  [Key in keyof AnomalyDetectionConfig]: AnomalyDetectionConfig[Key] extends number ? Key : never
}[keyof AnomalyDetectionConfig];

const ROBUST_ALGORITHMS: AnomalyDetectionAlgorithm[] = ['robust_zscore', 'robust_cusum'];
const BASELINE_ALGORITHMS: AnomalyDetectionAlgorithm[] = [...ROBUST_ALGORITHMS, 'rolling_sigma'];

const PARAMETER_DEFINITIONS: Array<{
  key: NumericConfigKey;
  label: string;
  description: string;
  min?: number;
  decimalScale?: number;
  algorithms?: AnomalyDetectionAlgorithm[];
}> = [
  { key: 'smoothing_half_life_minutes', label: 'EWMA half-life (min)', description: 'Controls time-weighted smoothing. A shorter half-life reacts faster but follows noise more closely.', min: 0.1, algorithms: ROBUST_ALGORITHMS },
  { key: 'baseline_window_minutes', label: 'Baseline window (min)', description: 'Length of normal history used for the rolling baseline. Rolling Sigma uses its mean and standard deviation; robust methods use median and MAD.', min: 1, algorithms: BASELINE_ALGORITHMS },
  { key: 'warmup_minutes', label: 'Warm-up (min)', description: 'Minimum normal-history duration required before anomalies can be detected.', min: 0, algorithms: BASELINE_ALGORITHMS },
  { key: 'minimum_warmup_points', label: 'Minimum warm-up points', description: 'Minimum number of valid normal measurements required before the baseline is considered reliable.', min: 3, algorithms: BASELINE_ALGORITHMS },
  { key: 'warning_z', label: 'Warning Z-score', description: 'Starts a yellow early-warning interval when the robust Z-score reaches this value.', min: 0.1, decimalScale: 2, algorithms: ROBUST_ALGORITHMS },
  { key: 'high_z', label: 'Confirmation Z-score', description: 'Confirms a red anomaly when this robust Z-score persists for the confirmation time.', min: 0.1, decimalScale: 2, algorithms: ROBUST_ALGORITHMS },
  { key: 'minimum_scale_relative', label: 'Relative scale floor', description: 'Minimum robust scale as a fraction of the rolling median magnitude. Increase this if a near-zero MAD creates implausibly large Z-scores.', min: 0, decimalScale: 6, algorithms: ROBUST_ALGORITHMS },
  { key: 'minimum_scale_absolute', label: 'Absolute scale floor', description: 'Absolute lower bound for the robust scale in score units. Use this when the normal score level is near zero.', min: 0, decimalScale: 12, algorithms: ROBUST_ALGORITHMS },
  { key: 'cusum_drift', label: 'CUSUM drift', description: 'Evidence subtracted during every minute. Larger values make CUSUM less sensitive to small shifts.', min: 0, decimalScale: 2, algorithms: ['robust_cusum'] },
  { key: 'cusum_threshold', label: 'CUSUM threshold', description: 'Accumulated positive evidence required to confirm an anomaly when the high Z-score is not reached.', min: 0.1, decimalScale: 2, algorithms: ['robust_cusum'] },
  { key: 'cusum_z_cap', label: 'CUSUM Z-score cap', description: 'Maximum Z-score contribution used by CUSUM per update. The displayed Robust Z-score remains uncapped.', min: 0.1, decimalScale: 2, algorithms: ['robust_cusum'] },
  { key: 'recovery_z', label: 'Recovery Z-score', description: 'The signal must fall below this Z-score before recovery timing begins.', decimalScale: 2, algorithms: ROBUST_ALGORITHMS },
  { key: 'recovery_minutes', label: 'Recovery hold (min)', description: 'Continuous time below the recovery Z-score required to close an event.', min: 0, algorithms: ROBUST_ALGORITHMS },
  { key: 'fallback_recovery_minutes', label: 'Fallback recovery (min)', description: 'Closes an event after this continuous time below Warning Z-score. Set to 0 to disable.', min: 0, algorithms: ROBUST_ALGORITHMS },
  { key: 'preroll_minutes', label: 'Pre-roll (min)', description: 'Hidden history loaded before a selected range so its baseline is already initialized.', min: 0 },
  { key: 'gap_multiplier', label: 'Gap multiplier', description: 'A time gap larger than this multiple of the normal sample interval resets detector state.', min: 1.1, decimalScale: 1 },
  { key: 'minimum_gap_minutes', label: 'Minimum gap (min)', description: 'Absolute minimum duration that is treated as a data gap and resets the baseline.', min: 0.1, algorithms: BASELINE_ALGORITHMS },
  { key: 'sigma_threshold', label: 'Standard-deviation threshold (σ)', description: 'A raw score is marked anomalous when it is strictly greater than rolling mean plus this many standard deviations. The default is 3σ.', min: 0.1, decimalScale: 2, algorithms: ['rolling_sigma'] },
  { key: 'event_smoothing_window_seconds', label: 'Smoothing window (s)', description: 'Trailing causal time window (t − window, t] used by median or moving-average smoothing.', min: 0.1, algorithms: ['event_threshold'] },
  { key: 'persistence_k', label: 'Required candidates K', description: 'Number of above-threshold samples required within the last N samples.', min: 1, algorithms: ['event_threshold'] },
  { key: 'persistence_n', label: 'Persistence window N', description: 'Number of most recent samples considered by the K-out-of-N rule.', min: 1, algorithms: ['event_threshold'] },
  { key: 'threshold_off_factor', label: 'Threshold-off factor', description: 'Multiplies Threshold On to create the lower recovery threshold and prevent event chattering.', min: 0.01, decimalScale: 3, algorithms: ['event_threshold'] },
  { key: 'normal_close_seconds', label: 'Normal close duration (s)', description: 'Continuous time below Threshold Off required before an active event closes.', min: 0, algorithms: ['event_threshold'] },
  { key: 'merge_gap_seconds', label: 'Merge gap (s)', description: 'Events separated by less than this time are combined unless a data gap lies between them.', min: 0, algorithms: ['event_threshold'] },
  { key: 'event_minimum_gap_seconds', label: 'Minimum data gap (s)', description: 'Smallest timestamp gap that resets smoothing, K-out-of-N evidence, and active event state.', min: 0.1, algorithms: ['event_threshold'] },
];

function InfoLabel({ label, description }: { label: string; description: string }) {
  return (
    <Group gap={4} wrap="nowrap">
      <Text span size="sm" fw={500}>{label}</Text>
      <Tooltip label={description} multiline w={320} withArrow>
        <ActionIcon component="span" size="xs" variant="subtle" color="gray" aria-label={`${label} information`}>
          <Info size={13} />
        </ActionIcon>
      </Tooltip>
    </Group>
  );
}

function algorithmDefinition(value: AnomalyDetectionAlgorithm) {
  return ALGORITHMS.find((algorithm) => algorithm.value === value) ?? ALGORITHMS[0];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown error';
}

function localInput(timestamp: string): string {
  return timestamp ? timestamp.slice(0, 19) : '';
}

function formatTimestamp(timestamp: string | null): string {
  if (!timestamp) return '—';
  const value = new Date(timestamp);
  return Number.isNaN(value.getTime()) ? timestamp : value.toLocaleString('de-DE', { hour12: false });
}

function formatDuration(start: string, end: string): string {
  const minutes = Math.max(0, (new Date(end).getTime() - new Date(start).getTime()) / 60_000);
  if (minutes >= 60) return `${(minutes / 60).toFixed(1)} h`;
  return `${Math.round(minutes)} min`;
}

function scoreValue(result: TestingRunResult, series: AnomalyDetectionScoreSeries): number | null {
  if (series === 'score') return result.score;
  if (series === 'full_mse') return result.full_mse;
  return result.roi_mse;
}

function numberValue(value: string | number, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function sameJson(left: unknown, right: unknown): boolean {
  const normalize = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(normalize);
    if (value && typeof value === 'object') {
      return Object.fromEntries(
        Object.entries(value as Record<string, unknown>)
          .sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey))
          .map(([key, item]) => [key, normalize(item)]),
      );
    }
    return value ?? null;
  };
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right));
}

function compatibleThresholdRun(
  target: TestingRun,
  candidate: TestingRun,
  scoreSeries: AnomalyDetectionScoreSeries,
): boolean {
  if (candidate.id === target.id || candidate.training_run_id !== target.training_run_id) return false;
  if (!sameJson(candidate.inference_config ?? {}, target.inference_config ?? {})) return false;
  if (scoreSeries === 'roi_mse' && candidate.roi_mse_mean === null) return false;
  if ((scoreSeries === 'score' || scoreSeries === 'roi_mse') && !sameJson(candidate.roi_geometry, target.roi_geometry)) return false;
  return true;
}

function inferenceFacetRecord(run: TestingRun): FacetRecord {
  return {
    id: String(run.id),
    facets: {
      model: [String(run.training_run_id)],
      dataset: [String(run.training_dataset_id)],
      roi: [roiKeyForRun(run)],
      metric: [metricKeyForRun(run)],
      aggregation: [aggregationKeyForRun(run)],
      preprocessing: run.preprocessing_pipeline_name ? [run.preprocessing_pipeline_name] : [],
      method: run.method_type ? [run.method_type] : [],
    },
    searchableValues: [
      run.name,
      run.training_run_name,
      run.training_pipeline_name,
      run.training_dataset_name,
      roiLabelForRun(run),
      run.preprocessing_pipeline_name,
      run.method_type,
      metricLabel(metricKeyForRun(run)),
      aggregationLabel(aggregationKeyForRun(run)),
    ].filter(Boolean),
  };
}

function progressToken(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function beginProgressPolling(
  token: string,
  onProgress: (progress: AnomalyDetectionProgress) => void,
): () => void {
  let stopped = false;
  let polling = false;
  const poll = async () => {
    if (stopped || polling) return;
    polling = true;
    try {
      const progress = await getAnomalyDetectionProgress(token);
      if (!stopped) onProgress(progress);
    } catch {
      // The worker may not have registered the token when the first poll wins
      // the race against the POST/GET request. The operation response remains
      // the authoritative error channel.
    } finally {
      polling = false;
    }
  };
  void poll();
  const timer = window.setInterval(() => { void poll(); }, 500);
  return () => {
    stopped = true;
    window.clearInterval(timer);
  };
}

function DetectionProgressPanel({
  progress,
  elapsedSeconds,
  fallbackMessage,
}: {
  progress: AnomalyDetectionProgress | null;
  elapsedSeconds: number;
  fallbackMessage: string;
}) {
  const percent = progress?.percent ?? 0;
  const showCount = Boolean(progress?.total && progress.phase !== 'complete');
  const countUnit = progress?.phase === 'saving' ? 'events' : 'points';
  return (
    <Paper withBorder p="sm" radius="sm">
      <Stack gap={6}>
        <Group justify="space-between">
          <Text size="sm" fw={600}>{progress?.message ?? fallbackMessage}</Text>
          <Text size="sm" c="dimmed">{Math.round(percent)}%</Text>
        </Group>
        <Progress value={percent} animated={progress?.status !== 'error'} size="md" />
        <Text size="xs" c="dimmed">
          Backend progress
          {showCount && progress ? ` · ${progress.completed.toLocaleString()} / ${progress.total.toLocaleString()} ${countUnit}` : ''}
          {` · elapsed ${elapsedSeconds}s · client timeout 15 minutes`}
        </Text>
      </Stack>
    </Paper>
  );
}

export function AnomalyDetectionPage({ active }: { active: boolean }) {
  const [testingRuns, setTestingRuns] = useState<TestingRun[]>([]);
  const [savedRuns, setSavedRuns] = useState<AnomalyDetectionRunSummary[]>([]);
  const [sourceFiltersOpen, setSourceFiltersOpen] = useState(true);
  const [sourceSearch, setSourceSearch] = useState('');
  const [sourceModelFilters, setSourceModelFilters] = useState<string[]>([]);
  const [sourceDatasetFilters, setSourceDatasetFilters] = useState<string[]>([]);
  const [sourceRoiFilters, setSourceRoiFilters] = useState<string[]>([]);
  const [sourceMetricFilters, setSourceMetricFilters] = useState<string[]>([]);
  const [sourceAggregationFilters, setSourceAggregationFilters] = useState<string[]>([]);
  const [sourcePreprocessingFilters, setSourcePreprocessingFilters] = useState<string[]>([]);
  const [sourceMethodFilters, setSourceMethodFilters] = useState<string[]>([]);
  const [sourcePage, setSourcePage] = useState(1);
  const [testingRunId, setTestingRunId] = useState<string | null>(null);
  const [previewResults, setPreviewResults] = useState<TestingRunResult[]>([]);
  const [previewDecimated, setPreviewDecimated] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [rangeMode, setRangeMode] = useState<'full' | 'selection'>('full');
  const [fullStart, setFullStart] = useState('');
  const [fullEnd, setFullEnd] = useState('');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [scoreSeries, setScoreSeries] = useState<AnomalyDetectionScoreSeries>('score');
  const [runName, setRunName] = useState('');
  const [preset, setPreset] = useState('fast');
  const [config, setConfig] = useState<AnomalyDetectionConfig>({ ...FAST_CONFIG });
  const [thresholdTestingRunId, setThresholdTestingRunId] = useState<string | null>(null);
  const [thresholdRunSearch, setThresholdRunSearch] = useState('');
  const [thresholdDatasetFilters, setThresholdDatasetFilters] = useState<string[]>([]);
  const [thresholdRunPage, setThresholdRunPage] = useState(1);
  const [thresholdResults, setThresholdResults] = useState<TestingRunResult[]>([]);
  const [thresholdDecimated, setThresholdDecimated] = useState(false);
  const [thresholdLoading, setThresholdLoading] = useState(false);
  const [thresholdRangeMode, setThresholdRangeMode] = useState<'full' | 'selection'>('full');
  const [thresholdFullStart, setThresholdFullStart] = useState('');
  const [thresholdFullEnd, setThresholdFullEnd] = useState('');
  const [thresholdStart, setThresholdStart] = useState('');
  const [thresholdEnd, setThresholdEnd] = useState('');
  const [thresholdPreview, setThresholdPreview] = useState<AnomalyDetectionThresholdPreview | null>(null);
  const [thresholdCalculating, setThresholdCalculating] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [running, setRunning] = useState(false);
  const [operationProgress, setOperationProgress] = useState<AnomalyDetectionProgress | null>(null);
  const [runElapsedSeconds, setRunElapsedSeconds] = useState(0);
  const [activeRun, setActiveRun] = useState<AnomalyDetectionRun | null>(null);
  const [diagnosticAnchor, setDiagnosticAnchor] = useState('');
  const [exactDiagnosticRows, setExactDiagnosticRows] = useState<AnomalyDetectionSeriesPoint[] | null>(null);
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false);
  const [loadingSavedRunId, setLoadingSavedRunId] = useState<number | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(true);
  const [savedPage, setSavedPage] = useState(1);
  const [savedSearch, setSavedSearch] = useState('');
  const [savedAlgorithmFilters, setSavedAlgorithmFilters] = useState<string[]>([]);
  const [savedModelFilters, setSavedModelFilters] = useState<string[]>([]);
  const [savedDatasetFilters, setSavedDatasetFilters] = useState<string[]>([]);
  const [savedRoiFilters, setSavedRoiFilters] = useState<string[]>([]);
  const [savedScoreFilters, setSavedScoreFilters] = useState<string[]>([]);

  const refresh = useCallback(async () => {
    const [inferences, detections] = await Promise.all([listTestingRuns(), listAnomalyDetectionRuns()]);
    setTestingRuns(inferences.filter((run) => run.status === 'finished'));
    setSavedRuns(detections);
  }, []);

  useEffect(() => {
    if (!active) return;
    refresh().catch((error) => notifications.show({ color: 'red', title: 'Could not load anomaly detection', message: errorMessage(error) }));
  }, [active, refresh]);

  useEffect(() => {
    if (!running && loadingSavedRunId === null) return;
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      const elapsed = (Date.now() - startedAt) / 1000;
      setRunElapsedSeconds(Math.floor(elapsed));
    }, 500);
    return () => window.clearInterval(timer);
  }, [running, loadingSavedRunId]);

  const selectedRun = testingRuns.find((run) => run.id === Number(testingRunId)) ?? null;
  const inferenceFacetRecords = useMemo(() => testingRuns.map(inferenceFacetRecord), [testingRuns]);
  const sourceFacetState = useMemo<FacetFilterState>(() => ({
    query: sourceSearch,
    selections: {
      model: sourceModelFilters,
      dataset: sourceDatasetFilters,
      roi: sourceRoiFilters,
      metric: sourceMetricFilters,
      aggregation: sourceAggregationFilters,
      preprocessing: sourcePreprocessingFilters,
      method: sourceMethodFilters,
    },
  }), [sourceAggregationFilters, sourceDatasetFilters, sourceMethodFilters, sourceMetricFilters, sourceModelFilters, sourcePreprocessingFilters, sourceRoiFilters, sourceSearch]);
  const filteredRunRecords = useMemo(
    () => matchingFacetRecords(inferenceFacetRecords, sourceFacetState),
    [inferenceFacetRecords, sourceFacetState],
  );
  const testingRunById = useMemo(() => new Map(testingRuns.map((run) => [run.id, run])), [testingRuns]);
  const filteredRuns = useMemo(
    () => filteredRunRecords.map((record) => testingRunById.get(Number(record.id))).filter((run): run is TestingRun => Boolean(run)),
    [filteredRunRecords, testingRunById],
  );
  const pagedFilteredRuns = useMemo(
    () => filteredRuns.slice((sourcePage - 1) * DEFAULT_TABLE_PAGE_SIZE, sourcePage * DEFAULT_TABLE_PAGE_SIZE),
    [filteredRuns, sourcePage],
  );
  const sourceFacetCounts = useMemo(() => ({
    model: countFacetValues(inferenceFacetRecords, sourceFacetState, 'model'),
    dataset: countFacetValues(inferenceFacetRecords, sourceFacetState, 'dataset'),
    roi: countFacetValues(inferenceFacetRecords, sourceFacetState, 'roi'),
    metric: countFacetValues(inferenceFacetRecords, sourceFacetState, 'metric'),
    aggregation: countFacetValues(inferenceFacetRecords, sourceFacetState, 'aggregation'),
    preprocessing: countFacetValues(inferenceFacetRecords, sourceFacetState, 'preprocessing'),
    method: countFacetValues(inferenceFacetRecords, sourceFacetState, 'method'),
  }), [inferenceFacetRecords, sourceFacetState]);
  const sourceModelOptions = useMemo(() => {
    const labels = new Map(testingRuns.map((run) => [String(run.training_run_id), run.training_pipeline_name || run.training_run_name || `Training run #${run.training_run_id}`]));
    return [...labels].map(([value, label]) => facetOption(value, label, sourceFacetCounts.model, sourceModelFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [sourceFacetCounts.model, sourceModelFilters, testingRuns]);
  const sourceDatasetOptions = useMemo(() => {
    const labels = new Map(testingRuns.map((run) => [String(run.training_dataset_id), run.training_dataset_name || `Inference dataset #${run.training_dataset_id}`]));
    return [...labels].map(([value, label]) => facetOption(value, label, sourceFacetCounts.dataset, sourceDatasetFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [sourceDatasetFilters, sourceFacetCounts.dataset, testingRuns]);
  const sourceRoiOptions = useMemo(() => {
    const labels = new Map(testingRuns.map((run) => [roiKeyForRun(run), roiLabelForRun(run)]));
    return [...labels].map(([value, label]) => facetOption(value, label, sourceFacetCounts.roi, sourceRoiFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [sourceFacetCounts.roi, sourceRoiFilters, testingRuns]);
  const sourceMetricOptions = useMemo(() => [...new Set(testingRuns.map(metricKeyForRun))].map((value) => facetOption(value, metricLabel(value), sourceFacetCounts.metric, sourceMetricFilters)), [sourceFacetCounts.metric, sourceMetricFilters, testingRuns]);
  const sourceAggregationOptions = useMemo(() => [...new Set(testingRuns.map(aggregationKeyForRun))].map((value) => facetOption(value, aggregationLabel(value), sourceFacetCounts.aggregation, sourceAggregationFilters)), [sourceAggregationFilters, sourceFacetCounts.aggregation, testingRuns]);
  const sourcePreprocessingOptions = useMemo(() => [...new Set(testingRuns.map((run) => run.preprocessing_pipeline_name).filter(Boolean))].map((value) => facetOption(value, value, sourceFacetCounts.preprocessing, sourcePreprocessingFilters)).sort((a, b) => a.label.localeCompare(b.label)), [sourceFacetCounts.preprocessing, sourcePreprocessingFilters, testingRuns]);
  const sourceMethodOptions = useMemo(() => [...new Set(testingRuns.map((run) => run.method_type).filter(Boolean))].map((value) => facetOption(value, value, sourceFacetCounts.method, sourceMethodFilters)).sort((a, b) => a.label.localeCompare(b.label)), [sourceFacetCounts.method, sourceMethodFilters, testingRuns]);
  const compatibleThresholdRuns = useMemo(
    () => selectedRun
      ? testingRuns.filter((run) => compatibleThresholdRun(selectedRun, run, scoreSeries))
      : [],
    [scoreSeries, selectedRun, testingRuns],
  );
  const thresholdTestingRun = compatibleThresholdRuns.find(
    (run) => run.id === Number(thresholdTestingRunId),
  ) ?? null;
  const thresholdFacetRecords = useMemo(() => compatibleThresholdRuns.map(inferenceFacetRecord), [compatibleThresholdRuns]);
  const thresholdFacetState = useMemo<FacetFilterState>(() => ({
    query: thresholdRunSearch,
    selections: { dataset: thresholdDatasetFilters },
  }), [thresholdDatasetFilters, thresholdRunSearch]);
  const filteredThresholdRuns = useMemo(() => {
    const ids = new Set(matchingFacetRecords(thresholdFacetRecords, thresholdFacetState).map((record) => Number(record.id)));
    return compatibleThresholdRuns.filter((run) => ids.has(run.id));
  }, [compatibleThresholdRuns, thresholdFacetRecords, thresholdFacetState]);
  const pagedThresholdRuns = useMemo(
    () => filteredThresholdRuns.slice((thresholdRunPage - 1) * DEFAULT_TABLE_PAGE_SIZE, thresholdRunPage * DEFAULT_TABLE_PAGE_SIZE),
    [filteredThresholdRuns, thresholdRunPage],
  );
  const thresholdDatasetCounts = useMemo(() => countFacetValues(thresholdFacetRecords, thresholdFacetState, 'dataset'), [thresholdFacetRecords, thresholdFacetState]);
  const thresholdDatasetOptions = useMemo(() => {
    const labels = new Map(compatibleThresholdRuns.map((run) => [String(run.training_dataset_id), run.training_dataset_name || `Inference dataset #${run.training_dataset_id}`]));
    return [...labels].map(([value, label]) => facetOption(value, label, thresholdDatasetCounts, thresholdDatasetFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [compatibleThresholdRuns, thresholdDatasetCounts, thresholdDatasetFilters]);

  const savedFacetRecords = useMemo<FacetRecord[]>(() => savedRuns.map((run) => {
    const source = testingRunById.get(run.testing_run_id) ?? null;
    return {
      id: String(run.id),
      facets: {
        algorithm: [run.config.algorithm],
        model: source ? [String(source.training_run_id)] : [],
        dataset: source ? [String(source.training_dataset_id)] : [],
        roi: source ? [roiKeyForRun(source)] : [],
        score: [run.score_series],
      },
      searchableValues: [run.name, run.testing_run_name, source?.training_pipeline_name, source?.training_dataset_name, source ? roiLabelForRun(source) : null, algorithmDefinition(run.config.algorithm).label, run.score_series].filter((value): value is string => Boolean(value)),
    };
  }), [savedRuns, testingRunById]);
  const savedFacetState = useMemo<FacetFilterState>(() => ({
    query: savedSearch,
    selections: {
      algorithm: savedAlgorithmFilters,
      model: savedModelFilters,
      dataset: savedDatasetFilters,
      roi: savedRoiFilters,
      score: savedScoreFilters,
    },
  }), [savedAlgorithmFilters, savedDatasetFilters, savedModelFilters, savedRoiFilters, savedScoreFilters, savedSearch]);
  const filteredSavedRuns = useMemo(() => {
    const ids = new Set(matchingFacetRecords(savedFacetRecords, savedFacetState).map((record) => Number(record.id)));
    return savedRuns.filter((run) => ids.has(run.id));
  }, [savedFacetRecords, savedFacetState, savedRuns]);
  const pagedSavedRuns = useMemo(
    () => filteredSavedRuns.slice((savedPage - 1) * DEFAULT_TABLE_PAGE_SIZE, savedPage * DEFAULT_TABLE_PAGE_SIZE),
    [filteredSavedRuns, savedPage],
  );
  const savedFacetCounts = useMemo(() => ({
    algorithm: countFacetValues(savedFacetRecords, savedFacetState, 'algorithm'),
    model: countFacetValues(savedFacetRecords, savedFacetState, 'model'),
    dataset: countFacetValues(savedFacetRecords, savedFacetState, 'dataset'),
    roi: countFacetValues(savedFacetRecords, savedFacetState, 'roi'),
    score: countFacetValues(savedFacetRecords, savedFacetState, 'score'),
  }), [savedFacetRecords, savedFacetState]);
  const savedAlgorithmOptions = useMemo(() => [...new Set(savedRuns.map((run) => run.config.algorithm))].map((value) => facetOption(value, algorithmDefinition(value).label, savedFacetCounts.algorithm, savedAlgorithmFilters)), [savedAlgorithmFilters, savedFacetCounts.algorithm, savedRuns]);
  const savedModelOptions = useMemo(() => {
    const labels = new Map(testingRuns.map((run) => [String(run.training_run_id), run.training_pipeline_name || run.training_run_name || `Training run #${run.training_run_id}`]));
    return [...labels].map(([value, label]) => facetOption(value, label, savedFacetCounts.model, savedModelFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [savedFacetCounts.model, savedModelFilters, testingRuns]);
  const savedDatasetOptions = useMemo(() => {
    const labels = new Map(testingRuns.map((run) => [String(run.training_dataset_id), run.training_dataset_name || `Inference dataset #${run.training_dataset_id}`]));
    return [...labels].map(([value, label]) => facetOption(value, label, savedFacetCounts.dataset, savedDatasetFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [savedDatasetFilters, savedFacetCounts.dataset, testingRuns]);
  const savedRoiOptions = useMemo(() => {
    const labels = new Map(testingRuns.map((run) => [roiKeyForRun(run), roiLabelForRun(run)]));
    return [...labels].map(([value, label]) => facetOption(value, label, savedFacetCounts.roi, savedRoiFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [savedFacetCounts.roi, savedRoiFilters, testingRuns]);
  const savedScoreOptions = useMemo(() => [...new Set(savedRuns.map((run) => run.score_series))].map((value) => facetOption(value, value.replaceAll('_', ' ').toUpperCase(), savedFacetCounts.score, savedScoreFilters)), [savedFacetCounts.score, savedRuns, savedScoreFilters]);

  useEffect(() => setSourcePage(1), [sourceSearch, sourceModelFilters, sourceDatasetFilters, sourceRoiFilters, sourceMetricFilters, sourceAggregationFilters, sourcePreprocessingFilters, sourceMethodFilters]);
  useEffect(() => setSourcePage((page) => Math.min(page, Math.max(1, Math.ceil(filteredRuns.length / DEFAULT_TABLE_PAGE_SIZE)))), [filteredRuns.length]);
  useEffect(() => setThresholdRunPage(1), [thresholdRunSearch, thresholdDatasetFilters, selectedRun?.id, scoreSeries]);
  useEffect(() => setThresholdRunPage((page) => Math.min(page, Math.max(1, Math.ceil(filteredThresholdRuns.length / DEFAULT_TABLE_PAGE_SIZE)))), [filteredThresholdRuns.length]);
  useEffect(() => setSavedPage(1), [savedSearch, savedAlgorithmFilters, savedModelFilters, savedDatasetFilters, savedRoiFilters, savedScoreFilters]);
  useEffect(() => setSavedPage((page) => Math.min(page, Math.max(1, Math.ceil(filteredSavedRuns.length / DEFAULT_TABLE_PAGE_SIZE)))), [filteredSavedRuns.length]);

  useEffect(() => {
    if (!testingRunId || testingRuns.some((run) => run.id === Number(testingRunId))) return;
    setTestingRunId(null);
    setPreviewResults([]);
  }, [testingRunId, testingRuns]);

  useEffect(() => {
    if (!testingRunId) return;
    let cancelled = false;
    setPreviewLoading(true);
    getTestingRunResults(Number(testingRunId), MAX_PREVIEW_POINTS)
      .then((response) => {
        if (cancelled) return;
        setPreviewResults(response.results);
        setPreviewDecimated(response.decimated);
        const first = response.results[0]?.timestamp ?? '';
        const last = response.results.at(-1)?.timestamp ?? first;
        setFullStart(localInput(first));
        setFullEnd(localInput(last));
        setStart(localInput(first));
        setEnd(localInput(last));
        setRangeMode('full');
        setRunName(`${response.testing_run.name} anomaly detection`);
        setScoreSeries(response.testing_run.roi_mse_mean === null ? 'score' : scoreSeries);
      })
      .catch((error) => notifications.show({ color: 'red', title: 'Could not load inference results', message: errorMessage(error) }))
      .finally(() => { if (!cancelled) setPreviewLoading(false); });
    return () => { cancelled = true; };
    // scoreSeries intentionally remains the user's current preference.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [testingRunId]);

  useEffect(() => {
    if (!thresholdTestingRunId) return;
    if (compatibleThresholdRuns.some((run) => run.id === Number(thresholdTestingRunId))) return;
    setThresholdTestingRunId(null);
    setThresholdResults([]);
    setThresholdPreview(null);
  }, [compatibleThresholdRuns, thresholdTestingRunId]);

  useEffect(() => {
    if (!thresholdTestingRunId) {
      setThresholdResults([]);
      setThresholdDecimated(false);
      return;
    }
    let cancelled = false;
    setThresholdLoading(true);
    getTestingRunResults(Number(thresholdTestingRunId), MAX_PREVIEW_POINTS)
      .then((response) => {
        if (cancelled) return;
        setThresholdResults(response.results);
        setThresholdDecimated(response.decimated);
        const first = response.results[0]?.timestamp ?? '';
        const last = response.results.at(-1)?.timestamp ?? first;
        setThresholdFullStart(localInput(first));
        setThresholdFullEnd(localInput(last));
        setThresholdStart(localInput(first));
        setThresholdEnd(localInput(last));
        setThresholdRangeMode('full');
      })
      .catch((error) => notifications.show({ color: 'red', title: 'Could not load validation inference', message: errorMessage(error) }))
      .finally(() => { if (!cancelled) setThresholdLoading(false); });
    return () => { cancelled = true; };
  }, [thresholdTestingRunId]);

  useEffect(() => {
    setThresholdPreview(null);
  }, [
    config.event_smoothing_enabled,
    config.event_smoothing_method,
    config.event_smoothing_window_seconds,
    config.event_minimum_gap_seconds,
    config.gap_multiplier,
    config.threshold_quantile,
    config.threshold_mode,
    scoreSeries,
    thresholdEnd,
    thresholdStart,
    thresholdTestingRunId,
  ]);

  const previewData = useMemo<Data[]>(() => [{
    type: 'scatter',
    mode: rangeMode === 'selection' ? 'lines+markers' : 'lines',
    name: scoreSeries.replace('_', ' ').toUpperCase(),
    x: previewResults.map((result) => result.timestamp),
    y: previewResults.map((result) => scoreValue(result, scoreSeries)),
    line: { color: '#228be6', width: 1.6 },
    marker: { size: 5 },
    connectgaps: false,
  }], [previewResults, rangeMode, scoreSeries]);

  const previewLayout = useMemo<Partial<Layout>>(() => ({
    title: { text: rangeMode === 'selection' ? 'Drag across the plot to select a time range' : 'Inference reconstruction error', font: { size: 14 } },
    dragmode: rangeMode === 'selection' ? 'select' : 'zoom',
    hovermode: 'x unified',
    xaxis: { type: 'date', title: { text: 'Time' } },
    yaxis: { title: { text: scoreSeries.replace('_', ' ').toUpperCase() } },
    showlegend: false,
  }), [rangeMode, scoreSeries]);

  const thresholdPreviewData = useMemo<Data[]>(() => [{
    type: 'scatter',
    mode: thresholdRangeMode === 'selection' ? 'lines+markers' : 'lines',
    name: 'Validation score',
    x: thresholdResults.map((result) => result.timestamp),
    y: thresholdResults.map((result) => scoreValue(result, scoreSeries)),
    line: { color: '#12b886', width: 1.5 },
    marker: { size: 5 },
    connectgaps: false,
  }], [scoreSeries, thresholdRangeMode, thresholdResults]);

  const thresholdPreviewLayout = useMemo<Partial<Layout>>(() => ({
    title: { text: thresholdRangeMode === 'selection' ? 'Select normal validation range' : 'Normal validation scores', font: { size: 14 } },
    dragmode: thresholdRangeMode === 'selection' ? 'select' : 'zoom',
    hovermode: 'x unified',
    xaxis: { type: 'date', title: { text: 'Time' } },
    yaxis: { title: { text: scoreSeries.replace('_', ' ').toUpperCase() } },
    showlegend: false,
  }), [scoreSeries, thresholdRangeMode]);

  const selectRange = useCallback((selection: PlotlyChartSelection) => {
    const left = new Date(selection.start);
    const right = new Date(selection.end);
    if (Number.isNaN(left.getTime()) || Number.isNaN(right.getTime())) return;
    const first = left <= right ? selection.start : selection.end;
    const last = left <= right ? selection.end : selection.start;
    setStart(localInput(first.replace(' ', 'T')));
    setEnd(localInput(last.replace(' ', 'T')));
  }, []);

  const selectThresholdRange = useCallback((selection: PlotlyChartSelection) => {
    const left = new Date(selection.start);
    const right = new Date(selection.end);
    if (Number.isNaN(left.getTime()) || Number.isNaN(right.getTime())) return;
    const first = left <= right ? selection.start : selection.end;
    const last = left <= right ? selection.end : selection.start;
    setThresholdStart(localInput(first.replace(' ', 'T')));
    setThresholdEnd(localInput(last.replace(' ', 'T')));
  }, []);

  function applyPreset(value: string | null) {
    const next = value ?? (config.algorithm === 'event_threshold' ? 'prototype' : config.algorithm === 'rolling_sigma' ? 'simple' : 'fast');
    setPreset(next);
    if (next === 'prototype') {
      setConfig({ ...EVENT_CONFIG });
      return;
    }
    if (next === 'simple') {
      setConfig({ ...SIGMA_CONFIG });
      return;
    }
    if (PRESETS[next]) {
      setConfig((current) => ({ ...PRESETS[next], algorithm: current.algorithm }));
    }
  }

  function patchConfig(key: NumericConfigKey, value: string | number) {
    setPreset('custom');
    setConfig((current) => ({ ...current, [key]: numberValue(value, current[key]) }));
  }

  function selectAlgorithm(value: string | null) {
    const algorithm = (value ?? 'robust_cusum') as AnomalyDetectionAlgorithm;
    if (algorithm === 'event_threshold') {
      setConfig({ ...EVENT_CONFIG });
      setPreset('prototype');
    } else if (algorithm === 'rolling_sigma') {
      setConfig({ ...SIGMA_CONFIG });
      setPreset('simple');
    } else {
      setConfig({ ...FAST_CONFIG, algorithm });
      setPreset('fast');
    }
  }

  async function calculateThreshold() {
    if (!thresholdTestingRun || !thresholdStart || !thresholdEnd) {
      notifications.show({ color: 'yellow', title: 'Incomplete validation source', message: 'Select a compatible validation inference and time range.' });
      return;
    }
    setThresholdCalculating(true);
    try {
      const calculated = await previewAnomalyDetectionThreshold({
        testing_run_id: thresholdTestingRun.id,
        score_series: scoreSeries,
        start_timestamp: thresholdRangeMode === 'full' ? thresholdFullStart : thresholdStart,
        end_timestamp: thresholdRangeMode === 'full' ? thresholdFullEnd : thresholdEnd,
        smoothing_enabled: config.event_smoothing_enabled,
        smoothing_method: config.event_smoothing_method,
        smoothing_window_seconds: config.event_smoothing_window_seconds,
        gap_multiplier: config.gap_multiplier,
        minimum_gap_seconds: config.event_minimum_gap_seconds,
        quantile: config.threshold_quantile,
      });
      setThresholdPreview(calculated);
    } catch (error) {
      notifications.show({ color: 'red', title: 'Threshold calculation failed', message: errorMessage(error) });
    } finally {
      setThresholdCalculating(false);
    }
  }

  async function executeDetection() {
    if (!selectedRun || !start || !end || !runName.trim()) {
      notifications.show({ color: 'yellow', title: 'Incomplete configuration', message: 'Select an inference, time range and run name.' });
      return;
    }
    if (
      config.algorithm === 'event_threshold'
      && config.threshold_mode === 'quantile'
      && (!thresholdTestingRun || !thresholdStart || !thresholdEnd)
    ) {
      notifications.show({ color: 'yellow', title: 'Incomplete validation source', message: 'Select a compatible normal-validation inference and range.' });
      return;
    }
    if (config.algorithm === 'event_threshold' && config.threshold_mode === 'manual' && config.manual_threshold === null) {
      notifications.show({ color: 'yellow', title: 'Missing threshold', message: 'Enter a manual Threshold On value.' });
      return;
    }
    const token = progressToken();
    setOperationProgress(null);
    setRunElapsedSeconds(0);
    setRunning(true);
    const stopPolling = beginProgressPolling(token, setOperationProgress);
    try {
      const created = await createAnomalyDetectionRun({
        name: runName.trim(),
        testing_run_id: selectedRun.id,
        score_series: scoreSeries,
        start_timestamp: rangeMode === 'full' ? fullStart : start,
        end_timestamp: rangeMode === 'full' ? fullEnd : end,
        config,
        ...(config.algorithm === 'event_threshold' && config.threshold_mode === 'quantile' && thresholdTestingRun ? {
          threshold_testing_run_id: thresholdTestingRun.id,
          threshold_start_timestamp: thresholdRangeMode === 'full' ? thresholdFullStart : thresholdStart,
          threshold_end_timestamp: thresholdRangeMode === 'full' ? thresholdFullEnd : thresholdEnd,
        } : {}),
        progress_token: token,
      });
      const fullResolution = created.point_count <= MAX_DIAGNOSTIC_POINTS
        ? await getAnomalyDetectionRun(created.id, MAX_DIAGNOSTIC_POINTS)
        : created;
      setActiveRun(fullResolution);
      await refresh();
      notifications.show({ color: 'green', title: 'Anomaly detection complete', message: `${created.anomaly_count} confirmed anomalies detected.` });
    } catch (error) {
      notifications.show({ color: 'red', title: 'Anomaly detection failed', message: errorMessage(error) });
    } finally {
      stopPolling();
      setRunning(false);
    }
  }

  async function openSaved(runId: number) {
    if (loadingSavedRunId !== null) return;
    const token = progressToken();
    setOperationProgress(null);
    setRunElapsedSeconds(0);
    setLoadingSavedRunId(runId);
    const stopPolling = beginProgressPolling(token, setOperationProgress);
    try {
      const loaded = await getAnomalyDetectionRun(runId, MAX_DIAGNOSTIC_POINTS, token);
      setActiveRun(loaded);
      setDetailsOpen(true);
    } catch (error) {
      notifications.show({ color: 'red', title: 'Could not open detection run', message: errorMessage(error) });
    } finally {
      stopPolling();
      setLoadingSavedRunId(null);
    }
  }

  async function removeSaved(run: AnomalyDetectionRunSummary) {
    if (loadingSavedRunId !== null) return;
    if (!window.confirm(`Delete anomaly detection run “${run.name}”?`)) return;
    try {
      await deleteAnomalyDetectionRun(run.id);
      if (activeRun?.id === run.id) setActiveRun(null);
      await refresh();
    } catch (error) {
      notifications.show({ color: 'red', title: 'Delete failed', message: errorMessage(error) });
    }
  }

  const resultShapes = useMemo(() => {
    if (!activeRun) return [];
    const shapes: NonNullable<Partial<Layout>['shapes']> = [];
    const warmup = activeRun.series.filter((point) => point.state === 'warmup');
    if (warmup.length) {
      shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: warmup[0].timestamp, x1: warmup.at(-1)?.timestamp, y0: 0, y1: 1, fillcolor: 'rgba(134,142,150,0.14)', line: { width: 0 }, layer: 'below' });
    }
    activeRun.events.forEach((event) => {
      if (activeRun.config.algorithm !== 'event_threshold') {
        const yellowEnd = event.confirmed_at ?? event.end_timestamp;
        shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: event.warning_start, x1: yellowEnd, y0: 0, y1: 1, fillcolor: 'rgba(250,176,5,0.20)', line: { width: 0 }, layer: 'below' });
      }
      if (event.confirmed_at) {
        shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: event.confirmed_at, x1: event.end_timestamp, y0: 0, y1: 1, fillcolor: 'rgba(250,82,82,0.22)', line: { width: 0 }, layer: 'below' });
      }
    });
    return shapes;
  }, [activeRun]);

  const resultData = useMemo<Data[]>(() => {
    if (!activeRun) return [];
    const timestamps = activeRun.series.map((point) => point.timestamp);
    if (activeRun.config.algorithm === 'event_threshold') {
      const smoothingLabel = !activeRun.config.event_smoothing_enabled
        ? 'Raw score (smoothing disabled)'
        : activeRun.config.event_smoothing_method === 'median' ? 'Rolling median' : 'Moving average';
      return [
        { type: 'scatter', mode: 'lines', name: 'Raw error', x: timestamps, y: activeRun.series.map((point) => point.score), line: { color: '#868e96', width: 1 } },
        { type: 'scatter', mode: 'lines', name: smoothingLabel, x: timestamps, y: activeRun.series.map((point) => point.smoothed), line: { color: '#228be6', width: 2 } },
        { type: 'scatter', mode: 'lines', name: 'Threshold On', x: timestamps, y: activeRun.series.map((point) => point.threshold_on), line: { color: '#fa5252', width: 1.5, dash: 'dash' } },
        { type: 'scatter', mode: 'lines', name: 'Threshold Off', x: timestamps, y: activeRun.series.map((point) => point.threshold_off), line: { color: '#f59f00', width: 1.5, dash: 'dot' } },
      ];
    }
    if (activeRun.config.algorithm === 'rolling_sigma') {
      return [
        { type: 'scatter', mode: 'lines', name: 'Raw error', x: timestamps, y: activeRun.series.map((point) => point.score), line: { color: '#228be6', width: 1.5 } },
        { type: 'scatter', mode: 'lines', name: 'Rolling mean', x: timestamps, y: activeRun.series.map((point) => point.baseline), line: { color: '#868e96', width: 1.5 } },
        { type: 'scatter', mode: 'lines', name: `Mean + ${activeRun.config.sigma_threshold}σ`, x: timestamps, y: activeRun.series.map((point) => point.high_threshold), line: { color: '#fa5252', width: 1.5, dash: 'dash' } },
      ];
    }
    return [
      { type: 'scatter', mode: 'lines', name: 'Raw error', x: timestamps, y: activeRun.series.map((point) => point.score), line: { color: '#868e96', width: 1 } },
      { type: 'scatter', mode: 'lines', name: 'EWMA', x: timestamps, y: activeRun.series.map((point) => point.smoothed), line: { color: '#228be6', width: 2 } },
      { type: 'scatter', mode: 'lines', name: 'Warning threshold', x: timestamps, y: activeRun.series.map((point) => point.warning_threshold), line: { color: '#fab005', width: 1.5, dash: 'dash' } },
      { type: 'scatter', mode: 'lines', name: 'High threshold', x: timestamps, y: activeRun.series.map((point) => point.high_threshold), line: { color: '#fa5252', width: 1.5, dash: 'dot' } },
    ];
  }, [activeRun]);

  const diagnosticData = useMemo<Data[]>(() => {
    if (!activeRun) return [];
    if (activeRun.config.algorithm === 'event_threshold') {
      const candidatePoints = activeRun.series.filter((point) => point.candidate);
      return [
        { type: 'scatter', mode: 'lines', name: 'Candidates in last N samples', x: activeRun.series.map((point) => point.timestamp), y: activeRun.series.map((point) => point.persistence_count), line: { color: '#7950f2', width: 1.5 } },
        { type: 'scatter', mode: 'lines', name: `Required K = ${activeRun.config.persistence_k}`, x: activeRun.series.map((point) => point.timestamp), y: activeRun.series.map(() => activeRun.config.persistence_k), line: { color: '#fa5252', width: 1.2, dash: 'dash' } },
        { type: 'scatter', mode: 'markers', name: 'Above-threshold sample', x: candidatePoints.map((point) => point.timestamp), y: candidatePoints.map((point) => point.persistence_count), marker: { color: '#f59f00', size: 5 } },
      ];
    }
    const data: Data[] = [
      { type: 'scatter', mode: 'lines', name: activeRun.config.algorithm === 'rolling_sigma' ? 'Standard deviations above mean' : 'Robust z-score', x: activeRun.series.map((point) => point.timestamp), y: activeRun.series.map((point) => point.robust_z), line: { color: '#7950f2', width: 1.5 } },
    ];
    if (activeRun.config.algorithm === 'rolling_sigma') {
      data.push({ type: 'scatter', mode: 'lines', name: `Anomaly at ${activeRun.config.sigma_threshold}σ`, x: activeRun.series.map((point) => point.timestamp), y: activeRun.series.map(() => activeRun.config.sigma_threshold), line: { color: '#fa5252', width: 1.2, dash: 'dash' } });
    }
    if (activeRun.config.algorithm === 'robust_cusum') {
      data.push({ type: 'scatter', mode: 'lines', name: 'CUSUM', x: activeRun.series.map((point) => point.timestamp), y: activeRun.series.map((point) => point.cusum), yaxis: 'y2', line: { color: '#0ca678', width: 1.5 } });
    }
    return data;
  }, [activeRun]);

  useEffect(() => {
    if (!activeRun) {
      setDiagnosticAnchor('');
      return;
    }
    setDiagnosticAnchor(activeRun.events.find((event) => event.confirmed_at)?.peak_timestamp ?? activeRun.start_timestamp);
    setExactDiagnosticRows(null);
  }, [activeRun?.id]);

  const diagnosticRows = useMemo(() => {
    if (!activeRun || !ROBUST_ALGORITHMS.includes(activeRun.config.algorithm)) return [];
    const anchorTime = Date.parse(diagnosticAnchor);
    const startIndex = Number.isFinite(anchorTime)
      ? activeRun.series.findIndex((point) => Date.parse(point.timestamp) >= anchorTime)
      : 0;
    return activeRun.series.slice(Math.max(0, startIndex), Math.max(0, startIndex) + 200);
  }, [activeRun, diagnosticAnchor]);

  const displayedDiagnosticRows = exactDiagnosticRows ?? diagnosticRows;

  async function loadExactDiagnostics() {
    if (!activeRun || !diagnosticAnchor) return;
    setDiagnosticsLoading(true);
    try {
      const rows = await getAnomalyDetectionDiagnostics(activeRun.id, diagnosticAnchor, 200);
      setExactDiagnosticRows(rows);
    } catch (error) {
      notifications.show({ color: 'red', title: 'Could not load diagnostics', message: errorMessage(error) });
    } finally {
      setDiagnosticsLoading(false);
    }
  }

  function downloadDiagnosticCsv() {
    if (!activeRun || displayedDiagnosticRows.length === 0) return;
    const header = ['timestamp', 'error', 'median', 'mad', 'scale', 'robust_z', 'ewma', 'cusum_increment', 'cusum', 'state'];
    const rows = displayedDiagnosticRows.map((point) => [
      point.timestamp,
      point.score,
      point.baseline ?? '',
      point.mad ?? '',
      point.scale ?? '',
      point.robust_z ?? '',
      point.smoothed,
      point.cusum_increment ?? '',
      point.cusum,
      point.state,
    ]);
    const csv = [header, ...rows].map((row) => row.join(',')).join('\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `anomaly-run-${activeRun.id}-diagnostics.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={2}>Anomaly Detection</Title>
          <Text c="dimmed">Detect sustained crucible disturbances from reconstruction-error time series.</Text>
        </div>
        <Button variant="subtle" leftSection={<RefreshCw size={16} />} onClick={() => refresh()}>Refresh</Button>
      </Group>

      <Paper withBorder p="md">
        <Stack gap="md">
          <Group justify="space-between" wrap="wrap">
            <Text fw={700}>1. Select inference source</Text>
            <Badge variant="light">{filteredRuns.length} matching inference{filteredRuns.length === 1 ? '' : 's'}</Badge>
          </Group>
          <Group justify="space-between" wrap="wrap">
            <Button variant="subtle" size="compact-sm" leftSection={<SlidersHorizontal size={16} />} rightSection={sourceFiltersOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />} onClick={() => setSourceFiltersOpen((open) => !open)}>Filters</Button>
            {(sourceSearch.trim() || sourceModelFilters.length || sourceDatasetFilters.length || sourceRoiFilters.length || sourceMetricFilters.length || sourceAggregationFilters.length || sourcePreprocessingFilters.length || sourceMethodFilters.length) ? (
              <Button variant="subtle" color="gray" size="compact-sm" onClick={() => {
                setSourceSearch('');
                setSourceModelFilters([]);
                setSourceDatasetFilters([]);
                setSourceRoiFilters([]);
                setSourceMetricFilters([]);
                setSourceAggregationFilters([]);
                setSourcePreprocessingFilters([]);
                setSourceMethodFilters([]);
              }}>Reset filters</Button>
            ) : null}
          </Group>
          <Collapse in={sourceFiltersOpen}>
            <Stack gap="sm">
              <TextInput placeholder="Search inference, model, pipeline or dataset" leftSection={<Search size={16} />} value={sourceSearch} onChange={(event) => setSourceSearch(event.currentTarget.value)} />
              <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
                <MultiSelect label="Models" searchable clearable data={sourceModelOptions} value={sourceModelFilters} onChange={setSourceModelFilters} />
                <MultiSelect label="Inference datasets" searchable clearable data={sourceDatasetOptions} value={sourceDatasetFilters} onChange={setSourceDatasetFilters} />
                <MultiSelect label="ROI" searchable clearable data={sourceRoiOptions} value={sourceRoiFilters} onChange={setSourceRoiFilters} />
                <MultiSelect label="Metrics" searchable clearable data={sourceMetricOptions} value={sourceMetricFilters} onChange={setSourceMetricFilters} />
                <MultiSelect label="Score aggregation" searchable clearable data={sourceAggregationOptions} value={sourceAggregationFilters} onChange={setSourceAggregationFilters} />
                <MultiSelect label="Preprocessing" searchable clearable data={sourcePreprocessingOptions} value={sourcePreprocessingFilters} onChange={setSourcePreprocessingFilters} />
                <MultiSelect label="Methods" searchable clearable data={sourceMethodOptions} value={sourceMethodFilters} onChange={setSourceMethodFilters} />
              </SimpleGrid>
              <Text size="xs" c="dimmed">Counts show finished inferences available after the search and all other categories. Multiple values inside one category use OR.</Text>
            </Stack>
          </Collapse>
          <ScrollArea.Autosize mah={360} type="auto">
            <Table striped highlightOnHover miw={1040}>
              <Table.Thead><Table.Tr><Table.Th>Inference</Table.Th><Table.Th>Model</Table.Th><Table.Th>Dataset</Table.Th><Table.Th>ROI</Table.Th><Table.Th>Score configuration</Table.Th><Table.Th>Frames</Table.Th><Table.Th>Finished</Table.Th><Table.Th /></Table.Tr></Table.Thead>
              <Table.Tbody>
                {pagedFilteredRuns.map((run) => {
                  const selected = testingRunId === String(run.id);
                  return <Table.Tr key={run.id} bg={selected ? 'var(--mantine-color-green-light)' : undefined}>
                    <Table.Td><Text fw={selected ? 700 : 500}>{run.name}</Text></Table.Td>
                    <Table.Td>{run.training_pipeline_name || run.training_run_name || `Training run #${run.training_run_id}`}</Table.Td>
                    <Table.Td>{run.training_dataset_name || `Inference dataset #${run.training_dataset_id}`}</Table.Td>
                    <Table.Td>{roiLabelForRun(run)}</Table.Td>
                    <Table.Td>{metricLabel(metricKeyForRun(run))} · {aggregationLabel(aggregationKeyForRun(run))}</Table.Td>
                    <Table.Td>{(run.image_count ?? 0).toLocaleString()}</Table.Td>
                    <Table.Td>{formatTimestamp(run.ended_at)}</Table.Td>
                    <Table.Td><Button size="compact-sm" variant={selected ? 'filled' : 'light'} color={selected ? 'green' : 'blue'} disabled={running || thresholdCalculating || previewLoading} onClick={() => setTestingRunId(String(run.id))}>{selected ? 'Selected' : 'Use'}</Button></Table.Td>
                  </Table.Tr>;
                })}
                {filteredRuns.length === 0 && <Table.Tr><Table.Td colSpan={8}><Stack align="center" gap="xs" py="md"><Text size="sm" c="dimmed">No finished inference matches the combined filters.</Text><Button variant="light" size="compact-sm" onClick={() => { setSourceSearch(''); setSourceModelFilters([]); setSourceDatasetFilters([]); setSourceRoiFilters([]); setSourceMetricFilters([]); setSourceAggregationFilters([]); setSourcePreprocessingFilters([]); setSourceMethodFilters([]); }}>Reset filters</Button></Stack></Table.Td></Table.Tr>}
              </Table.Tbody>
            </Table>
          </ScrollArea.Autosize>
          <TablePagination totalItems={filteredRuns.length} page={sourcePage} onChange={setSourcePage} />
          {selectedRun && (
            <Alert color="green" title="Selected inference">
              <Group gap="xs">
              <Badge variant="light">{selectedRun.name}</Badge>
              <Badge variant="light">{selectedRun.training_pipeline_name || selectedRun.training_run_name}</Badge>
              <Badge variant="light" color="teal">{selectedRun.training_dataset_name}</Badge>
              <Badge variant="light" color="violet">{roiLabelForRun(selectedRun)}</Badge>
              <Badge variant="light" color="gray">{selectedRun.image_count ?? 0} frames</Badge>
              {!filteredRuns.some((run) => run.id === selectedRun.id) && <Badge variant="light" color="yellow">Hidden by current filters</Badge>}
              </Group>
            </Alert>
          )}
        </Stack>
      </Paper>

      {testingRunId && (
        <Paper withBorder p="md">
          <Stack gap="md">
            <Group justify="space-between"><Text fw={700}>2. Select score and time range</Text>{previewLoading && <Loader size="sm" />}</Group>
            <SimpleGrid cols={{ base: 1, md: 2 }}>
              <Select
                label="Reconstruction-error series"
                data={[
                  { value: 'score', label: 'Configured score' },
                  { value: 'full_mse', label: 'Full-image MSE' },
                  { value: 'roi_mse', label: 'ROI MSE', disabled: selectedRun?.roi_mse_mean === null },
                ]}
                value={scoreSeries}
                disabled={running || thresholdCalculating}
                onChange={(value) => setScoreSeries((value ?? 'score') as AnomalyDetectionScoreSeries)}
              />
              <div>
                <Text size="sm" fw={500} mb={5}>Range mode</Text>
                <SegmentedControl fullWidth data={[{ value: 'full', label: 'Entire period' }, { value: 'selection', label: 'Select in plot' }]} value={rangeMode} onChange={(value) => { setRangeMode(value as 'full' | 'selection'); if (value === 'full') { setStart(fullStart); setEnd(fullEnd); } }} />
              </div>
            </SimpleGrid>
            <PlotlyChart data={previewData} layout={previewLayout} height={360} onSelected={rangeMode === 'selection' ? selectRange : undefined} />
            <SimpleGrid cols={{ base: 1, md: 2 }}>
              <DateTime24Input label="Start" value={rangeMode === 'full' ? fullStart : start} min={fullStart} max={fullEnd} disabled={rangeMode === 'full'} onChange={setStart} />
              <DateTime24Input label="End" value={rangeMode === 'full' ? fullEnd : end} min={fullStart} max={fullEnd} disabled={rangeMode === 'full'} onChange={setEnd} />
            </SimpleGrid>
            {previewDecimated && <Alert color="blue">The preview is decimated. Detection always uses every stored inference point.</Alert>}
          </Stack>
        </Paper>
      )}

      {testingRunId && (
        <Paper withBorder p="md">
          <Stack gap="md">
            <Text fw={700}>3. Configure and run detector</Text>
            <SimpleGrid cols={{ base: 1, md: 3 }}>
              <TextInput label="Run name" value={runName} disabled={running} onChange={(event) => setRunName(event.currentTarget.value)} />
              <Select
                label={<InfoLabel label="Detection algorithm" description={`${algorithmDefinition(config.algorithm).description} Exactly one algorithm is executed per run.`} />}
                value={config.algorithm}
                data={ALGORITHMS.map((algorithm) => ({ value: algorithm.value, label: algorithm.label }))}
                disabled={running || thresholdCalculating}
                onChange={selectAlgorithm}
              />
              <Select
                label={<InfoLabel label="Sensitivity preset" description="Applies a complete parameter set to the selected algorithm. Editing an advanced value changes the preset to Custom." />}
                value={preset}
                data={config.algorithm === 'event_threshold'
                  ? [{ value: 'prototype', label: 'Prototype defaults (recommended)' }, { value: 'custom', label: 'Custom', disabled: preset !== 'custom' }]
                  : config.algorithm === 'rolling_sigma'
                    ? [{ value: 'simple', label: '30 min baseline · 3σ (recommended)' }, { value: 'custom', label: 'Custom', disabled: preset !== 'custom' }]
                  : [{ value: 'fast', label: 'Fast response (recommended)' }, { value: 'balanced', label: 'Balanced' }, { value: 'robust', label: 'Very robust' }, { value: 'custom', label: 'Custom', disabled: preset !== 'custom' }]}
                disabled={running || thresholdCalculating}
                onChange={applyPreset}
              />
            </SimpleGrid>
            <Alert color="blue" variant="light">
              <Text fw={600}>{algorithmDefinition(config.algorithm).label}</Text>
              <Text size="sm">{algorithmDefinition(config.algorithm).description}</Text>
            </Alert>
            {config.algorithm === 'event_threshold' && (
              <Stack gap="md">
                <SimpleGrid cols={{ base: 1, md: 2 }}>
                  <Select
                    label={<InfoLabel label="Threshold mode" description="Manual uses a fixed score. Quantile calculates the score only from a separate compatible normal-validation inference." />}
                    value={config.threshold_mode}
                    data={[{ value: 'quantile', label: 'Quantile from normal validation' }, { value: 'manual', label: 'Manual threshold' }]}
                    disabled={running || thresholdCalculating}
                    onChange={(value) => {
                      setPreset('custom');
                      setConfig((current) => ({ ...current, threshold_mode: (value ?? 'quantile') as 'manual' | 'quantile' }));
                    }}
                  />
                  {config.threshold_mode === 'manual' && (
                    <NumberInput
                      label={<InfoLabel label="Threshold On" description="An individual sample is an anomaly candidate only when its smoothed score is strictly greater than this value." />}
                      value={config.manual_threshold ?? ''}
                      min={0}
                      decimalScale={8}
                      disabled={running || thresholdCalculating}
                      onChange={(value) => {
                        setPreset('custom');
                        setConfig((current) => ({ ...current, manual_threshold: typeof value === 'number' ? value : null }));
                      }}
                    />
                  )}
                </SimpleGrid>

                {config.threshold_mode === 'quantile' && (
                  <Paper withBorder p="md" radius="sm">
                    <Stack gap="md">
                      <Group justify="space-between">
                        <div>
                          <Text fw={700}>Normal validation threshold</Text>
                          <Text size="sm" c="dimmed">Only compatible runs using the same trained model, scoring configuration and ROI are available.</Text>
                        </div>
                        {thresholdLoading && <Loader size="sm" />}
                      </Group>
                      <Text size="sm" fw={500}><InfoLabel label="Validation inference" description="This inference must contain normal operation only. Its scores never come from the analyzed target range." /></Text>
                      <SimpleGrid cols={{ base: 1, md: 2 }}>
                        <TextInput placeholder="Search compatible inference or dataset" leftSection={<Search size={16} />} value={thresholdRunSearch} onChange={(event) => setThresholdRunSearch(event.currentTarget.value)} />
                        <MultiSelect label="Inference datasets" searchable clearable data={thresholdDatasetOptions} value={thresholdDatasetFilters} onChange={setThresholdDatasetFilters} />
                      </SimpleGrid>
                      <Group justify="space-between" wrap="wrap">
                        <Badge variant="light" color="teal">{filteredThresholdRuns.length} compatible match{filteredThresholdRuns.length === 1 ? '' : 'es'}</Badge>
                        {(thresholdRunSearch || thresholdDatasetFilters.length > 0) && <Button variant="subtle" color="gray" size="compact-sm" onClick={() => { setThresholdRunSearch(''); setThresholdDatasetFilters([]); }}>Reset filters</Button>}
                      </Group>
                      <ScrollArea.Autosize mah={280} type="auto">
                        <Table striped highlightOnHover miw={760}>
                          <Table.Thead><Table.Tr><Table.Th>Inference</Table.Th><Table.Th>Dataset</Table.Th><Table.Th>ROI</Table.Th><Table.Th>Frames</Table.Th><Table.Th>Finished</Table.Th><Table.Th /></Table.Tr></Table.Thead>
                          <Table.Tbody>
                            {pagedThresholdRuns.map((run) => {
                              const selected = thresholdTestingRunId === String(run.id);
                              return <Table.Tr key={run.id} bg={selected ? 'var(--mantine-color-teal-light)' : undefined}>
                                <Table.Td><Text fw={selected ? 700 : 500}>{run.name}</Text></Table.Td>
                                <Table.Td>{run.training_dataset_name}</Table.Td>
                                <Table.Td>{roiLabelForRun(run)}</Table.Td>
                                <Table.Td>{(run.image_count ?? 0).toLocaleString()}</Table.Td>
                                <Table.Td>{formatTimestamp(run.ended_at)}</Table.Td>
                                <Table.Td><Button size="compact-sm" variant={selected ? 'filled' : 'light'} color="teal" disabled={running || thresholdCalculating || thresholdLoading} onClick={() => setThresholdTestingRunId(String(run.id))}>{selected ? 'Selected' : 'Use'}</Button></Table.Td>
                              </Table.Tr>;
                            })}
                            {filteredThresholdRuns.length === 0 && <Table.Tr><Table.Td colSpan={6}><Text c="dimmed" size="sm" ta="center" py="md">{compatibleThresholdRuns.length ? 'No compatible validation inference matches the filters.' : 'No compatible validation inference is available.'}</Text></Table.Td></Table.Tr>}
                          </Table.Tbody>
                        </Table>
                      </ScrollArea.Autosize>
                      <TablePagination totalItems={filteredThresholdRuns.length} page={thresholdRunPage} onChange={setThresholdRunPage} />
                      {thresholdTestingRun && !filteredThresholdRuns.some((run) => run.id === thresholdTestingRun.id) && <Alert color="yellow">Selected validation inference: {thresholdTestingRun.name}. It is hidden by the current filters.</Alert>}
                      {thresholdTestingRun && (
                        <>
                          <SegmentedControl
                            fullWidth
                            data={[{ value: 'full', label: 'Entire validation period' }, { value: 'selection', label: 'Select in plot' }]}
                            value={thresholdRangeMode}
                            disabled={thresholdCalculating}
                            onChange={(value) => {
                              setThresholdRangeMode(value as 'full' | 'selection');
                              if (value === 'full') {
                                setThresholdStart(thresholdFullStart);
                                setThresholdEnd(thresholdFullEnd);
                              }
                            }}
                          />
                          <PlotlyChart
                            data={thresholdPreviewData}
                            layout={thresholdPreviewLayout}
                            height={280}
                            onSelected={thresholdRangeMode === 'selection' ? selectThresholdRange : undefined}
                          />
                          <SimpleGrid cols={{ base: 1, md: 2 }}>
                            <DateTime24Input label="Validation start" value={thresholdRangeMode === 'full' ? thresholdFullStart : thresholdStart} min={thresholdFullStart} max={thresholdFullEnd} disabled={thresholdRangeMode === 'full' || thresholdCalculating} onChange={setThresholdStart} />
                            <DateTime24Input label="Validation end" value={thresholdRangeMode === 'full' ? thresholdFullEnd : thresholdEnd} min={thresholdFullStart} max={thresholdFullEnd} disabled={thresholdRangeMode === 'full' || thresholdCalculating} onChange={setThresholdEnd} />
                          </SimpleGrid>
                          {thresholdDecimated && <Alert color="blue">Validation preview is decimated. The threshold calculation uses every stored point in the selected range.</Alert>}
                        </>
                      )}
                      <SimpleGrid cols={{ base: 1, md: 2 }}>
                        <div>
                          <NumberInput
                            label={<InfoLabel label="Validation quantile (%)" description="The calculated Threshold On is this quantile of the smoothed normal-validation scores." />}
                            value={config.threshold_quantile * 100}
                            min={0.000001}
                            max={99.999999}
                            decimalScale={6}
                            disabled={running || thresholdCalculating}
                            onChange={(value) => {
                              const percent = numberValue(value, config.threshold_quantile * 100);
                              setPreset('custom');
                              setConfig((current) => ({ ...current, threshold_quantile: percent / 100 }));
                            }}
                          />
                          <Group gap={4} mt={6}>
                            {[99.9, 99.99, 99.999].map((quantile) => (
                              <Button key={quantile} size="compact-xs" variant="light" disabled={thresholdCalculating || running} onClick={() => {
                                setPreset('custom');
                                setConfig((current) => ({ ...current, threshold_quantile: quantile / 100 }));
                              }}>{quantile}%</Button>
                            ))}
                          </Group>
                        </div>
                        <Button
                          mt={25}
                          variant="light"
                          color="teal"
                          loading={thresholdCalculating}
                          disabled={!thresholdTestingRun || !thresholdStart || !thresholdEnd || running}
                          onClick={calculateThreshold}
                        >
                          Calculate threshold
                        </Button>
                      </SimpleGrid>
                      {thresholdPreview && (
                        <Alert color="teal" title="Calculated full-resolution threshold">
                          <Text size="sm">{thresholdPreview.testing_run_name} · {(thresholdPreview.quantile * 100).toFixed(5)}% quantile · {thresholdPreview.point_count.toLocaleString()} points</Text>
                          <Text fw={700} mt={4}>Threshold On: {thresholdPreview.threshold.toPrecision(8)}</Text>
                          <Text size="sm">Threshold Off: {(thresholdPreview.threshold * config.threshold_off_factor).toPrecision(8)}</Text>
                        </Alert>
                      )}
                    </Stack>
                  </Paper>
                )}
              </Stack>
            )}
            <Button variant="subtle" justify="space-between" rightSection={advancedOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />} onClick={() => setAdvancedOpen((value) => !value)}>Advanced detector parameters</Button>
            <Collapse in={advancedOpen}>
              <Stack gap="md">
                {config.algorithm === 'event_threshold' && (
                  <SimpleGrid cols={{ base: 1, sm: 2 }}>
                    <Switch
                      label={<InfoLabel label="Enable smoothing" description="Keeps the raw score unchanged and applies the selected causal smoother only to candidate detection." />}
                      checked={config.event_smoothing_enabled}
                      disabled={running || thresholdCalculating}
                      onChange={(event) => {
                        setPreset('custom');
                        setConfig((current) => ({ ...current, event_smoothing_enabled: event.currentTarget.checked }));
                      }}
                    />
                    <Select
                      label={<InfoLabel label="Smoothing method" description="Median rejects isolated peaks; Moving Average reacts more smoothly but is influenced by outliers." />}
                      value={config.event_smoothing_method}
                      data={[{ value: 'median', label: 'Median' }, { value: 'moving_average', label: 'Moving Average' }]}
                      disabled={running || thresholdCalculating || !config.event_smoothing_enabled}
                      onChange={(value) => {
                        setPreset('custom');
                        setConfig((current) => ({ ...current, event_smoothing_method: (value ?? 'median') as 'median' | 'moving_average' }));
                      }}
                    />
                  </SimpleGrid>
                )}
                {BASELINE_ALGORITHMS.includes(config.algorithm) && (
                  <SimpleGrid cols={{ base: 1, sm: 2 }}>
                    <Select
                      label={<InfoLabel label="Confirmation duration type" description="Choose whether the score must remain above the confirmation threshold for a continuous number of elapsed minutes or consecutive samples. The counter resets as soon as the score falls below the threshold." />}
                      value={config.confirmation_mode}
                      data={[
                        { value: 'minutes', label: 'Minutes' },
                        { value: 'samples', label: 'Samples' },
                      ]}
                      disabled={running || thresholdCalculating}
                      onChange={(value) => {
                        setPreset('custom');
                        setConfig((current) => ({ ...current, confirmation_mode: (value ?? 'minutes') as 'minutes' | 'samples' }));
                      }}
                    />
                    {config.confirmation_mode === 'minutes' ? (
                      <NumberInput
                        label={<InfoLabel label="Required time above threshold (min)" description="Continuous elapsed time that the signal must meet the algorithm's confirmation condition before the interval is marked as an anomaly. Any drop below it resets the timer." />}
                        min={0}
                        decimalScale={2}
                        value={config.confirmation_minutes}
                        disabled={running || thresholdCalculating}
                        onChange={(value) => patchConfig('confirmation_minutes', value)}
                      />
                    ) : (
                      <NumberInput
                        label={<InfoLabel label="Required consecutive samples" description="Number of consecutive samples that must meet the algorithm's confirmation condition before the interval is marked as an anomaly. The default is 1 sample." />}
                        min={1}
                        value={config.confirmation_samples}
                        disabled={running || thresholdCalculating}
                        onChange={(value) => patchConfig('confirmation_samples', value)}
                      />
                    )}
                  </SimpleGrid>
                )}
                <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
                {PARAMETER_DEFINITIONS
                  .filter((parameter) => !parameter.algorithms || parameter.algorithms.includes(config.algorithm))
                  .map((parameter) => (
                    <NumberInput
                      key={parameter.key}
                      label={<InfoLabel label={parameter.label} description={parameter.description} />}
                      min={parameter.key === 'persistence_n' ? config.persistence_k : parameter.min}
                      decimalScale={parameter.decimalScale}
                      max={parameter.key === 'persistence_k' ? config.persistence_n : parameter.key === 'threshold_off_factor' ? 1 : undefined}
                      value={config[parameter.key]}
                      disabled={running || thresholdCalculating}
                      onChange={(value) => patchConfig(parameter.key, value)}
                    />
                  ))}
                </SimpleGrid>
              </Stack>
            </Collapse>
            {running && (
              <DetectionProgressPanel
                progress={operationProgress}
                elapsedSeconds={runElapsedSeconds}
                fallbackMessage={`Starting ${algorithmDefinition(config.algorithm).label}…`}
              />
            )}
            <Button leftSection={running ? <Loader size="xs" color="white" /> : <Play size={16} />} onClick={executeDetection} disabled={running || thresholdCalculating || loadingSavedRunId !== null || !previewResults.length}>Run anomaly detection</Button>
          </Stack>
        </Paper>
      )}

      <Paper withBorder p="md">
        <Stack gap="sm">
          <Group justify="space-between"><Text fw={700}>Saved detection runs</Text><Badge variant="light">{filteredSavedRuns.length} matching</Badge></Group>
          <TextInput placeholder="Search detection name, source, model or dataset" leftSection={<Search size={16} />} value={savedSearch} onChange={(event) => setSavedSearch(event.currentTarget.value)} />
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 5 }}>
            <MultiSelect label="Algorithms" searchable clearable data={savedAlgorithmOptions} value={savedAlgorithmFilters} onChange={setSavedAlgorithmFilters} />
            <MultiSelect label="Models" searchable clearable data={savedModelOptions} value={savedModelFilters} onChange={setSavedModelFilters} />
            <MultiSelect label="Inference datasets" searchable clearable data={savedDatasetOptions} value={savedDatasetFilters} onChange={setSavedDatasetFilters} />
            <MultiSelect label="ROI" searchable clearable data={savedRoiOptions} value={savedRoiFilters} onChange={setSavedRoiFilters} />
            <MultiSelect label="Score series" searchable clearable data={savedScoreOptions} value={savedScoreFilters} onChange={setSavedScoreFilters} />
          </SimpleGrid>
          {(savedSearch || savedAlgorithmFilters.length || savedModelFilters.length || savedDatasetFilters.length || savedRoiFilters.length || savedScoreFilters.length) ? <Group justify="flex-end"><Button variant="subtle" color="gray" size="compact-sm" onClick={() => { setSavedSearch(''); setSavedAlgorithmFilters([]); setSavedModelFilters([]); setSavedDatasetFilters([]); setSavedRoiFilters([]); setSavedScoreFilters([]); }}>Reset filters</Button></Group> : null}
          {savedRuns.length === 0 ? <Text size="sm" c="dimmed">No saved anomaly detection runs yet.</Text> : filteredSavedRuns.length === 0 ? (
            <Stack align="center" gap="xs" py="md"><Text size="sm" c="dimmed">No saved detection run matches the combined filters.</Text><Button variant="light" size="compact-sm" onClick={() => { setSavedSearch(''); setSavedAlgorithmFilters([]); setSavedModelFilters([]); setSavedDatasetFilters([]); setSavedRoiFilters([]); setSavedScoreFilters([]); }}>Reset filters</Button></Stack>
          ) : (
            <ScrollArea.Autosize mah={360} type="auto">
              <Table highlightOnHover>
                <Table.Thead><Table.Tr><Table.Th>Name</Table.Th><Table.Th>Algorithm</Table.Th><Table.Th>Inference</Table.Th><Table.Th>Range</Table.Th><Table.Th>Warnings</Table.Th><Table.Th>Anomalies</Table.Th><Table.Th /></Table.Tr></Table.Thead>
                <Table.Tbody>{pagedSavedRuns.map((run) => <Table.Tr key={run.id} bg={activeRun?.id === run.id ? 'var(--mantine-color-violet-light)' : undefined}>
                  <Table.Td><Text fw={activeRun?.id === run.id ? 700 : 500}>{run.name}</Text></Table.Td>
                  <Table.Td><Badge variant="light" color="violet">{algorithmDefinition(run.config.algorithm).label}</Badge></Table.Td>
                  <Table.Td>{run.testing_run_name}</Table.Td><Table.Td>{formatDuration(run.start_timestamp, run.end_timestamp)}</Table.Td><Table.Td>{run.warning_count}</Table.Td><Table.Td><Badge color={run.anomaly_count ? 'red' : 'gray'}>{run.anomaly_count}</Badge></Table.Td>
                  <Table.Td>
                    <Group gap={4} wrap="nowrap">
                      <Button
                        size="compact-sm"
                        variant={activeRun?.id === run.id ? 'light' : 'subtle'}
                        color="violet"
                        leftSection={loadingSavedRunId === run.id ? <Loader size="xs" /> : <Activity size={15} />}
                        disabled={running || loadingSavedRunId !== null || activeRun?.id === run.id}
                        onClick={() => openSaved(run.id)}
                      >
                        {activeRun?.id === run.id ? 'Opened' : 'Open'}
                      </Button>
                      <Tooltip label="Delete run">
                        <ActionIcon color="red" variant="subtle" disabled={running || loadingSavedRunId !== null} onClick={() => removeSaved(run)}><Trash2 size={16} /></ActionIcon>
                      </Tooltip>
                    </Group>
                  </Table.Td>
                </Table.Tr>)}</Table.Tbody>
              </Table>
            </ScrollArea.Autosize>
          )}
          <TablePagination totalItems={filteredSavedRuns.length} page={savedPage} onChange={setSavedPage} />
          {activeRun && !filteredSavedRuns.some((run) => run.id === activeRun.id) && <Alert color="yellow">The opened run remains visible below but is hidden by the current Saved Runs filters.</Alert>}
          {loadingSavedRunId !== null && (
            <DetectionProgressPanel
              progress={operationProgress}
              elapsedSeconds={runElapsedSeconds}
              fallbackMessage="Starting saved detection run…"
            />
          )}
        </Stack>
      </Paper>

      {activeRun && (
        <Paper withBorder p="md">
          <Stack gap="md">
            <Group justify="space-between" align="flex-start">
              <div><Title order={3}>{activeRun.name}</Title><Text size="sm" c="dimmed">{activeRun.testing_run_name} · {formatTimestamp(activeRun.start_timestamp)} – {formatTimestamp(activeRun.end_timestamp)}</Text></div>
              <Group gap="xs">
                <Badge color="violet">{algorithmDefinition(activeRun.config.algorithm).label}</Badge>
                {activeRun.config.algorithm === 'event_threshold' || activeRun.config.algorithm === 'rolling_sigma'
                  ? <Badge color="red">{activeRun.anomaly_count} events</Badge>
                  : <><Badge color="yellow">{activeRun.warning_count} warnings</Badge><Badge color="red">{activeRun.anomaly_count} confirmed</Badge></>}
                <Badge color="gray">{activeRun.point_count} points</Badge>
              </Group>
            </Group>
            {activeRun.config.algorithm === 'event_threshold' && (
              <Alert color="teal" title={`Threshold On: ${activeRun.resolved_threshold?.toPrecision(8) ?? '—'}`}>
                {activeRun.config.threshold_mode === 'quantile'
                  ? `${activeRun.threshold_testing_run_name ?? 'Validation inference'} · ${(activeRun.config.threshold_quantile * 100).toFixed(5)}% quantile · ${formatTimestamp(activeRun.threshold_start_timestamp)} – ${formatTimestamp(activeRun.threshold_end_timestamp)}`
                  : 'Manual threshold'}
                <Text size="sm">Threshold Off: {activeRun.resolved_threshold == null ? '—' : (activeRun.resolved_threshold * activeRun.config.threshold_off_factor).toPrecision(8)}</Text>
              </Alert>
            )}
            {activeRun.decimated && <Alert color="blue">Plot reduced from {activeRun.total.toLocaleString()} points; event detection used the full series.</Alert>}
            <PlotlyChart data={resultData} layout={{ hovermode: 'x unified', shapes: resultShapes, xaxis: { type: 'date', title: { text: 'Time' } }, yaxis: { title: { text: activeRun.score_series.replace('_', ' ').toUpperCase() } }, legend: { orientation: 'h' } }} height={480} />
            <Button variant="subtle" justify="space-between" rightSection={detailsOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />} onClick={() => setDetailsOpen((value) => !value)}>Diagnostics</Button>
            <Collapse in={detailsOpen}>
              <Stack gap="md">
                <PlotlyChart data={diagnosticData} layout={{ hovermode: 'x unified', xaxis: { type: 'date' }, yaxis: { title: { text: activeRun.config.algorithm === 'event_threshold' ? 'Candidate count' : activeRun.config.algorithm === 'rolling_sigma' ? 'Standard deviations' : 'Robust z-score' } }, ...(activeRun.config.algorithm === 'robust_cusum' ? { yaxis2: { title: { text: 'CUSUM' }, overlaying: 'y', side: 'right' } } : {}), legend: { orientation: 'h' } }} height={300} />
                {ROBUST_ALGORITHMS.includes(activeRun.config.algorithm) && (
                  <Stack gap="sm">
                    <Group justify="space-between" align="flex-end" wrap="wrap">
                      <DateTime24Input
                        label="Diagnostic anchor"
                        value={diagnosticAnchor}
                        min={activeRun.start_timestamp}
                        max={activeRun.end_timestamp}
                        description="Shows the first 200 points at or after this physical event-end timestamp."
                        onChange={(value) => {
                          setDiagnosticAnchor(value);
                          setExactDiagnosticRows(null);
                        }}
                      />
                      <Group gap="xs">
                        <Button variant="light" loading={diagnosticsLoading} onClick={loadExactDiagnostics}>Load exact 200 points</Button>
                        <Button variant="light" leftSection={<Download size={16} />} disabled={displayedDiagnosticRows.length === 0} onClick={downloadDiagnosticCsv}>
                          Export CSV
                        </Button>
                      </Group>
                    </Group>
                    {activeRun.decimated && exactDiagnosticRows === null && (
                      <Alert color="yellow">The plot is decimated. Select the physical event-end timestamp and load the exact 200 full-resolution points before exporting.</Alert>
                    )}
                    <ScrollArea h={360}>
                      <Table striped highlightOnHover withColumnBorders>
                        <Table.Thead><Table.Tr><Table.Th>Timestamp</Table.Th><Table.Th>Error</Table.Th><Table.Th>Median</Table.Th><Table.Th>MAD</Table.Th><Table.Th>Scale</Table.Th><Table.Th>Robust Z</Table.Th><Table.Th>EWMA</Table.Th><Table.Th>CUSUM Δ</Table.Th><Table.Th>CUSUM</Table.Th><Table.Th>State</Table.Th></Table.Tr></Table.Thead>
                        <Table.Tbody>{displayedDiagnosticRows.map((point) => (
                          <Table.Tr key={point.timestamp}>
                            <Table.Td>{formatTimestamp(point.timestamp)}</Table.Td>
                            <Table.Td>{point.score.toPrecision(8)}</Table.Td>
                            <Table.Td>{point.baseline?.toPrecision(8) ?? '—'}</Table.Td>
                            <Table.Td>{point.mad?.toPrecision(8) ?? '—'}</Table.Td>
                            <Table.Td>{point.scale?.toPrecision(8) ?? '—'}</Table.Td>
                            <Table.Td>{point.robust_z?.toPrecision(8) ?? '—'}</Table.Td>
                            <Table.Td>{point.smoothed.toPrecision(8)}</Table.Td>
                            <Table.Td>{point.cusum_increment?.toPrecision(8) ?? '—'}</Table.Td>
                            <Table.Td>{point.cusum.toPrecision(8)}</Table.Td>
                            <Table.Td><Badge variant="light">{point.state}</Badge></Table.Td>
                          </Table.Tr>
                        ))}</Table.Tbody>
                      </Table>
                    </ScrollArea>
                  </Stack>
                )}
              </Stack>
            </Collapse>
            <Text fw={700}>{activeRun.config.algorithm === 'event_threshold' ? 'Detected events' : 'Detected intervals'}</Text>
            {activeRun.events.length === 0 ? <Alert color="green">No warnings or confirmed anomalies detected.</Alert> : (
              <ScrollArea>
                {activeRun.config.algorithm === 'event_threshold' ? (
                  <Table striped highlightOnHover>
                    <Table.Thead><Table.Tr><Table.Th>Start</Table.Th><Table.Th>End</Table.Th><Table.Th>Duration</Table.Th><Table.Th>Raw peak</Table.Th><Table.Th>Max smoothed</Table.Th><Table.Th>Mean smoothed</Table.Th><Table.Th>Threshold</Table.Th><Table.Th>End reason</Table.Th></Table.Tr></Table.Thead>
                    <Table.Tbody>{activeRun.events.map((event) => <Table.Tr key={event.id}>
                      <Table.Td>{formatTimestamp(event.warning_start)}</Table.Td><Table.Td>{formatTimestamp(event.end_timestamp)}</Table.Td>
                      <Table.Td>{event.duration_seconds == null ? formatDuration(event.warning_start, event.end_timestamp) : `${event.duration_seconds.toFixed(1)} s`}</Table.Td>
                      <Table.Td>{event.max_score.toPrecision(5)}<Text size="xs" c="dimmed">{formatTimestamp(event.peak_timestamp)}</Text></Table.Td>
                      <Table.Td>{event.max_smoothed_score?.toPrecision(5) ?? '—'}</Table.Td><Table.Td>{event.mean_smoothed_score?.toPrecision(5) ?? '—'}</Table.Td>
                      <Table.Td>{event.threshold?.toPrecision(5) ?? '—'}</Table.Td><Table.Td>{event.end_reason.replace('_', ' ')}</Table.Td>
                    </Table.Tr>)}</Table.Tbody>
                  </Table>
                ) : (
                  <Table striped highlightOnHover>
                  <Table.Thead><Table.Tr><Table.Th>Status</Table.Th><Table.Th>Warning start</Table.Th><Table.Th>Confirmed</Table.Th><Table.Th>End</Table.Th><Table.Th>Duration</Table.Th><Table.Th>Peak</Table.Th><Table.Th>Max z</Table.Th><Table.Th>End reason</Table.Th></Table.Tr></Table.Thead>
                  <Table.Tbody>{activeRun.events.map((event) => <Table.Tr key={event.id}>
                    <Table.Td><Badge color={event.confirmed_at ? 'red' : 'yellow'}>{event.confirmed_at ? 'Confirmed' : 'Warning'}</Badge></Table.Td>
                    <Table.Td>{formatTimestamp(event.warning_start)}</Table.Td><Table.Td>{formatTimestamp(event.confirmed_at)}</Table.Td><Table.Td>{formatTimestamp(event.end_timestamp)}</Table.Td>
                    <Table.Td>{formatDuration(event.warning_start, event.end_timestamp)}</Table.Td><Table.Td>{event.max_score.toPrecision(5)}<Text size="xs" c="dimmed">{formatTimestamp(event.peak_timestamp)}</Text></Table.Td>
                    <Table.Td>{event.max_robust_z?.toFixed(2) ?? '—'}</Table.Td><Table.Td>{event.end_reason.replace('_', ' ')}</Table.Td>
                  </Table.Tr>)}</Table.Tbody>
                  </Table>
                )}
              </ScrollArea>
            )}
          </Stack>
        </Paper>
      )}
    </Stack>
  );
}
