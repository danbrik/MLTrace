import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Collapse,
  Group,
  Loader,
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
import { Activity, ChevronDown, ChevronRight, Info, Play, RefreshCw, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  createAnomalyDetectionRun,
  deleteAnomalyDetectionRun,
  getAnomalyDetectionProgress,
  getAnomalyDetectionRun,
  getTestingRunResults,
  listAnomalyDetectionRuns,
  listTestingRuns,
  previewAnomalyDetectionThreshold,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart, type PlotlyChartSelection } from '../components/PlotlyChart';
import type { Data, Layout } from '../lib/plotly';
import type {
  AnomalyDetectionConfig,
  AnomalyDetectionAlgorithm,
  AnomalyDetectionProgress,
  AnomalyDetectionRun,
  AnomalyDetectionRunSummary,
  AnomalyDetectionScoreSeries,
  AnomalyDetectionThresholdPreview,
  TestingRun,
  TestingRunResult,
} from '../types';

const MAX_PREVIEW_POINTS = 8000;

const FAST_CONFIG: AnomalyDetectionConfig = {
  algorithm: 'robust_cusum',
  smoothing_half_life_minutes: 5,
  baseline_window_minutes: 120,
  warmup_minutes: 30,
  minimum_warmup_points: 30,
  warning_z: 3,
  high_z: 5,
  cusum_drift: 1,
  cusum_threshold: 10,
  confirmation_minutes: 5,
  recovery_z: 1,
  recovery_minutes: 15,
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
    description: 'Uses the unchanged raw reconstruction error. A point is anomalous immediately when it exceeds the mean of the preceding normal baseline by the configured number of standard deviations.',
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
  { key: 'cusum_drift', label: 'CUSUM drift', description: 'Evidence subtracted during every minute. Larger values make CUSUM less sensitive to small shifts.', min: 0, decimalScale: 2, algorithms: ['robust_cusum'] },
  { key: 'cusum_threshold', label: 'CUSUM threshold', description: 'Accumulated positive evidence required to confirm an anomaly when the high Z-score is not reached.', min: 0.1, decimalScale: 2, algorithms: ['robust_cusum'] },
  { key: 'confirmation_minutes', label: 'Confirmation hold (min)', description: 'Minimum time between the start of a warning and confirmation of an anomaly.', min: 0, algorithms: ROBUST_ALGORITHMS },
  { key: 'recovery_z', label: 'Recovery Z-score', description: 'The signal must fall below this Z-score before recovery timing begins.', decimalScale: 2, algorithms: ROBUST_ALGORITHMS },
  { key: 'recovery_minutes', label: 'Recovery hold (min)', description: 'Continuous time below the recovery Z-score required to close an event.', min: 0, algorithms: ROBUST_ALGORITHMS },
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

function uniqueOptions(runs: TestingRun[], value: (run: TestingRun) => number | string, label: (run: TestingRun) => string) {
  const options = new Map<string, string>();
  runs.forEach((run) => options.set(String(value(run)), label(run)));
  return [...options.entries()].map(([optionValue, optionLabel]) => ({ value: optionValue, label: optionLabel }));
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
  const [modelFilter, setModelFilter] = useState<string | null>(null);
  const [datasetFilter, setDatasetFilter] = useState<string | null>(null);
  const [roiFilter, setRoiFilter] = useState<string | null>(null);
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
  const [loadingSavedRunId, setLoadingSavedRunId] = useState<number | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(true);

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

  const modelRuns = useMemo(
    () => testingRuns.filter((run) => !modelFilter || run.training_run_id === Number(modelFilter)),
    [testingRuns, modelFilter],
  );
  const datasetRuns = useMemo(
    () => modelRuns.filter((run) => !datasetFilter || run.training_dataset_id === Number(datasetFilter)),
    [modelRuns, datasetFilter],
  );
  const filteredRuns = useMemo(
    () => datasetRuns.filter((run) => !roiFilter || (run.roi_id === null ? 'none' : String(run.roi_id)) === roiFilter),
    [datasetRuns, roiFilter],
  );
  const selectedRun = testingRuns.find((run) => run.id === Number(testingRunId)) ?? null;
  const compatibleThresholdRuns = useMemo(
    () => selectedRun
      ? testingRuns.filter((run) => compatibleThresholdRun(selectedRun, run, scoreSeries))
      : [],
    [scoreSeries, selectedRun, testingRuns],
  );
  const thresholdTestingRun = compatibleThresholdRuns.find(
    (run) => run.id === Number(thresholdTestingRunId),
  ) ?? null;

  useEffect(() => {
    if (!testingRunId || filteredRuns.some((run) => run.id === Number(testingRunId))) return;
    setTestingRunId(null);
    setPreviewResults([]);
  }, [filteredRuns, testingRunId]);

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
      setActiveRun(created);
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
      const loaded = await getAnomalyDetectionRun(runId, MAX_PREVIEW_POINTS, token);
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
      if (!['event_threshold', 'rolling_sigma'].includes(activeRun.config.algorithm)) {
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
          <Text fw={700}>1. Select inference source</Text>
          <SimpleGrid cols={{ base: 1, md: 4 }}>
            <Select label="Model" searchable clearable data={uniqueOptions(testingRuns, (run) => run.training_run_id, (run) => run.training_run_name)} value={modelFilter} onChange={(value) => { setModelFilter(value); setDatasetFilter(null); setRoiFilter(null); }} />
            <Select label="Inference dataset" searchable clearable data={uniqueOptions(modelRuns, (run) => run.training_dataset_id, (run) => run.training_dataset_name)} value={datasetFilter} onChange={(value) => { setDatasetFilter(value); setRoiFilter(null); }} />
            <Select label="ROI" clearable data={uniqueOptions(datasetRuns, (run) => run.roi_id ?? 'none', (run) => run.roi_name ?? 'No ROI')} value={roiFilter} onChange={setRoiFilter} />
            <Select label="Finished inference" searchable data={filteredRuns.map((run) => ({ value: String(run.id), label: run.name }))} value={testingRunId} onChange={setTestingRunId} placeholder="Select inference" disabled={running || thresholdCalculating} />
          </SimpleGrid>
          {selectedRun && (
            <Group gap="xs">
              <Badge variant="light">{selectedRun.training_run_name}</Badge>
              <Badge variant="light" color="teal">{selectedRun.training_dataset_name}</Badge>
              <Badge variant="light" color="gray">{selectedRun.image_count ?? 0} frames</Badge>
            </Group>
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
                      <Select
                        label={<InfoLabel label="Validation inference" description="This inference must contain normal operation only. Its scores never come from the analyzed target range." />}
                        searchable
                        data={compatibleThresholdRuns.map((run) => ({ value: String(run.id), label: `${run.name} · ${run.training_dataset_name}` }))}
                        value={thresholdTestingRunId}
                        placeholder={compatibleThresholdRuns.length ? 'Select normal inference' : 'No compatible inference available'}
                        disabled={running || thresholdCalculating}
                        onChange={setThresholdTestingRunId}
                      />
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
          <Group justify="space-between"><Text fw={700}>Saved detection runs</Text><Badge variant="light">{savedRuns.length}</Badge></Group>
          {savedRuns.length === 0 ? <Text size="sm" c="dimmed">No saved anomaly detection runs yet.</Text> : (
            <ScrollArea.Autosize mah={360} type="auto">
              <Table highlightOnHover>
                <Table.Thead><Table.Tr><Table.Th>Name</Table.Th><Table.Th>Algorithm</Table.Th><Table.Th>Inference</Table.Th><Table.Th>Range</Table.Th><Table.Th>Warnings</Table.Th><Table.Th>Anomalies</Table.Th><Table.Th /></Table.Tr></Table.Thead>
                <Table.Tbody>{savedRuns.map((run) => <Table.Tr key={run.id} bg={activeRun?.id === run.id ? 'var(--mantine-color-violet-light)' : undefined}>
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
              <PlotlyChart data={diagnosticData} layout={{ hovermode: 'x unified', xaxis: { type: 'date' }, yaxis: { title: { text: activeRun.config.algorithm === 'event_threshold' ? 'Candidate count' : activeRun.config.algorithm === 'rolling_sigma' ? 'Standard deviations' : 'Robust z-score' } }, ...(activeRun.config.algorithm === 'robust_cusum' ? { yaxis2: { title: { text: 'CUSUM' }, overlaying: 'y', side: 'right' } } : {}), legend: { orientation: 'h' } }} height={300} />
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
