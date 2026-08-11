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
  getAnomalyDetectionRun,
  getTestingRunResults,
  listAnomalyDetectionRuns,
  listTestingRuns,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart, type PlotlyChartSelection } from '../components/PlotlyChart';
import type { Data, Layout } from '../lib/plotly';
import type {
  AnomalyDetectionConfig,
  AnomalyDetectionAlgorithm,
  AnomalyDetectionRun,
  AnomalyDetectionRunSummary,
  AnomalyDetectionScoreSeries,
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
];

type NumericConfigKey = Exclude<keyof AnomalyDetectionConfig, 'algorithm'>;

const PARAMETER_DEFINITIONS: Array<{
  key: NumericConfigKey;
  label: string;
  description: string;
  min?: number;
  decimalScale?: number;
  algorithms?: AnomalyDetectionAlgorithm[];
}> = [
  { key: 'smoothing_half_life_minutes', label: 'EWMA half-life (min)', description: 'Controls time-weighted smoothing. A shorter half-life reacts faster but follows noise more closely.', min: 0.1 },
  { key: 'baseline_window_minutes', label: 'Baseline window (min)', description: 'Length of normal history used for the rolling median and median absolute deviation (MAD).', min: 1 },
  { key: 'warmup_minutes', label: 'Warm-up (min)', description: 'Minimum normal-history duration required before warnings can start.', min: 0 },
  { key: 'minimum_warmup_points', label: 'Minimum warm-up points', description: 'Minimum number of valid measurements required before the baseline is considered reliable.', min: 3 },
  { key: 'warning_z', label: 'Warning Z-score', description: 'Starts a yellow early-warning interval when the robust Z-score reaches this value.', min: 0.1, decimalScale: 2 },
  { key: 'high_z', label: 'Confirmation Z-score', description: 'Confirms a red anomaly when this robust Z-score persists for the confirmation time.', min: 0.1, decimalScale: 2 },
  { key: 'cusum_drift', label: 'CUSUM drift', description: 'Evidence subtracted during every minute. Larger values make CUSUM less sensitive to small shifts.', min: 0, decimalScale: 2, algorithms: ['robust_cusum'] },
  { key: 'cusum_threshold', label: 'CUSUM threshold', description: 'Accumulated positive evidence required to confirm an anomaly when the high Z-score is not reached.', min: 0.1, decimalScale: 2, algorithms: ['robust_cusum'] },
  { key: 'confirmation_minutes', label: 'Confirmation hold (min)', description: 'Minimum time between the start of a warning and confirmation of an anomaly.', min: 0 },
  { key: 'recovery_z', label: 'Recovery Z-score', description: 'The signal must fall below this Z-score before recovery timing begins.', decimalScale: 2 },
  { key: 'recovery_minutes', label: 'Recovery hold (min)', description: 'Continuous time below the recovery Z-score required to close an event.', min: 0 },
  { key: 'preroll_minutes', label: 'Pre-roll (min)', description: 'Hidden history loaded before a selected range so its baseline is already initialized.', min: 0 },
  { key: 'gap_multiplier', label: 'Gap multiplier', description: 'A time gap larger than this multiple of the normal sample interval resets detector state.', min: 1.1, decimalScale: 1 },
  { key: 'minimum_gap_minutes', label: 'Minimum gap (min)', description: 'Absolute minimum duration that is treated as a data gap.', min: 0.1 },
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
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [running, setRunning] = useState(false);
  const [runProgress, setRunProgress] = useState(0);
  const [runElapsedSeconds, setRunElapsedSeconds] = useState(0);
  const [activeRun, setActiveRun] = useState<AnomalyDetectionRun | null>(null);
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
    if (!running) return;
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      const elapsed = (Date.now() - startedAt) / 1000;
      setRunElapsedSeconds(Math.floor(elapsed));
      setRunProgress(Math.min(95, 5 + 90 * (1 - Math.exp(-elapsed / 45))));
    }, 500);
    return () => window.clearInterval(timer);
  }, [running]);

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

  const selectRange = useCallback((selection: PlotlyChartSelection) => {
    const left = new Date(selection.start);
    const right = new Date(selection.end);
    if (Number.isNaN(left.getTime()) || Number.isNaN(right.getTime())) return;
    const first = left <= right ? selection.start : selection.end;
    const last = left <= right ? selection.end : selection.start;
    setStart(localInput(first.replace(' ', 'T')));
    setEnd(localInput(last.replace(' ', 'T')));
  }, []);

  function applyPreset(value: string | null) {
    const next = value ?? 'fast';
    setPreset(next);
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
    setConfig((current) => ({ ...current, algorithm }));
  }

  async function executeDetection() {
    if (!selectedRun || !start || !end || !runName.trim()) {
      notifications.show({ color: 'yellow', title: 'Incomplete configuration', message: 'Select an inference, time range and run name.' });
      return;
    }
    setRunProgress(3);
    setRunElapsedSeconds(0);
    setRunning(true);
    try {
      const created = await createAnomalyDetectionRun({
        name: runName.trim(),
        testing_run_id: selectedRun.id,
        score_series: scoreSeries,
        start_timestamp: rangeMode === 'full' ? fullStart : start,
        end_timestamp: rangeMode === 'full' ? fullEnd : end,
        config,
      });
      setRunProgress(100);
      setActiveRun(created);
      await refresh();
      notifications.show({ color: 'green', title: 'Anomaly detection complete', message: `${created.anomaly_count} confirmed anomalies detected.` });
    } catch (error) {
      notifications.show({ color: 'red', title: 'Anomaly detection failed', message: errorMessage(error) });
    } finally {
      setRunning(false);
    }
  }

  async function openSaved(runId: number) {
    try {
      setActiveRun(await getAnomalyDetectionRun(runId));
    } catch (error) {
      notifications.show({ color: 'red', title: 'Could not open detection run', message: errorMessage(error) });
    }
  }

  async function removeSaved(run: AnomalyDetectionRunSummary) {
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
      const yellowEnd = event.confirmed_at ?? event.end_timestamp;
      shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: event.warning_start, x1: yellowEnd, y0: 0, y1: 1, fillcolor: 'rgba(250,176,5,0.20)', line: { width: 0 }, layer: 'below' });
      if (event.confirmed_at) {
        shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: event.confirmed_at, x1: event.end_timestamp, y0: 0, y1: 1, fillcolor: 'rgba(250,82,82,0.22)', line: { width: 0 }, layer: 'below' });
      }
    });
    return shapes;
  }, [activeRun]);

  const resultData = useMemo<Data[]>(() => activeRun ? [
    { type: 'scatter', mode: 'lines', name: 'Raw error', x: activeRun.series.map((point) => point.timestamp), y: activeRun.series.map((point) => point.score), line: { color: '#868e96', width: 1 } },
    { type: 'scatter', mode: 'lines', name: 'EWMA', x: activeRun.series.map((point) => point.timestamp), y: activeRun.series.map((point) => point.smoothed), line: { color: '#228be6', width: 2 } },
    { type: 'scatter', mode: 'lines', name: 'Warning threshold', x: activeRun.series.map((point) => point.timestamp), y: activeRun.series.map((point) => point.warning_threshold), line: { color: '#fab005', width: 1.5, dash: 'dash' } },
    { type: 'scatter', mode: 'lines', name: 'High threshold', x: activeRun.series.map((point) => point.timestamp), y: activeRun.series.map((point) => point.high_threshold), line: { color: '#fa5252', width: 1.5, dash: 'dot' } },
  ] : [], [activeRun]);

  const diagnosticData = useMemo<Data[]>(() => {
    if (!activeRun) return [];
    const data: Data[] = [
      { type: 'scatter', mode: 'lines', name: 'Robust z-score', x: activeRun.series.map((point) => point.timestamp), y: activeRun.series.map((point) => point.robust_z), line: { color: '#7950f2', width: 1.5 } },
    ];
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
            <Select label="Finished inference" searchable data={filteredRuns.map((run) => ({ value: String(run.id), label: run.name }))} value={testingRunId} onChange={setTestingRunId} placeholder="Select inference" />
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
                disabled={running}
                onChange={selectAlgorithm}
              />
              <Select
                label={<InfoLabel label="Sensitivity preset" description="Applies a complete parameter set to the selected algorithm. Editing an advanced value changes the preset to Custom." />}
                value={preset}
                data={[{ value: 'fast', label: 'Fast response (recommended)' }, { value: 'balanced', label: 'Balanced' }, { value: 'robust', label: 'Very robust' }, { value: 'custom', label: 'Custom', disabled: preset !== 'custom' }]}
                disabled={running}
                onChange={applyPreset}
              />
            </SimpleGrid>
            <Alert color="blue" variant="light">
              <Text fw={600}>{algorithmDefinition(config.algorithm).label}</Text>
              <Text size="sm">{algorithmDefinition(config.algorithm).description}</Text>
            </Alert>
            <Button variant="subtle" justify="space-between" rightSection={advancedOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />} onClick={() => setAdvancedOpen((value) => !value)}>Advanced detector parameters</Button>
            <Collapse in={advancedOpen}>
              <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
                {PARAMETER_DEFINITIONS
                  .filter((parameter) => !parameter.algorithms || parameter.algorithms.includes(config.algorithm))
                  .map((parameter) => (
                    <NumberInput
                      key={parameter.key}
                      label={<InfoLabel label={parameter.label} description={parameter.description} />}
                      min={parameter.min}
                      decimalScale={parameter.decimalScale}
                      value={config[parameter.key]}
                      disabled={running}
                      onChange={(value) => patchConfig(parameter.key, value)}
                    />
                  ))}
              </SimpleGrid>
            </Collapse>
            {running && (
              <Paper withBorder p="sm" radius="sm">
                <Stack gap={6}>
                  <Group justify="space-between">
                    <Text size="sm" fw={600}>{runProgress < 25 ? 'Loading full-resolution score series…' : runProgress < 85 ? `Running ${algorithmDefinition(config.algorithm).label}…` : 'Finalizing events and plot series…'}</Text>
                    <Text size="sm" c="dimmed">{Math.round(runProgress)}%</Text>
                  </Group>
                  <Progress value={runProgress} animated size="md" />
                  <Text size="xs" c="dimmed">Estimated progress · elapsed {runElapsedSeconds}s · client timeout 15 minutes</Text>
                </Stack>
              </Paper>
            )}
            <Button leftSection={running ? <Loader size="xs" color="white" /> : <Play size={16} />} onClick={executeDetection} disabled={running || !previewResults.length}>Run anomaly detection</Button>
          </Stack>
        </Paper>
      )}

      {activeRun && (
        <Paper withBorder p="md">
          <Stack gap="md">
            <Group justify="space-between" align="flex-start">
              <div><Title order={3}>{activeRun.name}</Title><Text size="sm" c="dimmed">{activeRun.testing_run_name} · {formatTimestamp(activeRun.start_timestamp)} – {formatTimestamp(activeRun.end_timestamp)}</Text></div>
              <Group gap="xs"><Badge color="violet">{algorithmDefinition(activeRun.config.algorithm).label}</Badge><Badge color="yellow">{activeRun.warning_count} warnings</Badge><Badge color="red">{activeRun.anomaly_count} confirmed</Badge><Badge color="gray">{activeRun.point_count} points</Badge></Group>
            </Group>
            {activeRun.decimated && <Alert color="blue">Plot reduced from {activeRun.total.toLocaleString()} points; event detection used the full series.</Alert>}
            <PlotlyChart data={resultData} layout={{ hovermode: 'x unified', shapes: resultShapes, xaxis: { type: 'date', title: { text: 'Time' } }, yaxis: { title: { text: activeRun.score_series.replace('_', ' ').toUpperCase() } }, legend: { orientation: 'h' } }} height={480} />
            <Button variant="subtle" justify="space-between" rightSection={detailsOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />} onClick={() => setDetailsOpen((value) => !value)}>Diagnostics</Button>
            <Collapse in={detailsOpen}>
              <PlotlyChart data={diagnosticData} layout={{ hovermode: 'x unified', xaxis: { type: 'date' }, yaxis: { title: { text: 'Robust z-score' } }, ...(activeRun.config.algorithm === 'robust_cusum' ? { yaxis2: { title: { text: 'CUSUM' }, overlaying: 'y', side: 'right' } } : {}), legend: { orientation: 'h' } }} height={300} />
            </Collapse>
            <Text fw={700}>Detected intervals</Text>
            {activeRun.events.length === 0 ? <Alert color="green">No warnings or confirmed anomalies detected.</Alert> : (
              <ScrollArea>
                <Table striped highlightOnHover>
                  <Table.Thead><Table.Tr><Table.Th>Status</Table.Th><Table.Th>Warning start</Table.Th><Table.Th>Confirmed</Table.Th><Table.Th>End</Table.Th><Table.Th>Duration</Table.Th><Table.Th>Peak</Table.Th><Table.Th>Max z</Table.Th><Table.Th>End reason</Table.Th></Table.Tr></Table.Thead>
                  <Table.Tbody>{activeRun.events.map((event) => <Table.Tr key={event.id}>
                    <Table.Td><Badge color={event.confirmed_at ? 'red' : 'yellow'}>{event.confirmed_at ? 'Confirmed' : 'Warning'}</Badge></Table.Td>
                    <Table.Td>{formatTimestamp(event.warning_start)}</Table.Td><Table.Td>{formatTimestamp(event.confirmed_at)}</Table.Td><Table.Td>{formatTimestamp(event.end_timestamp)}</Table.Td>
                    <Table.Td>{formatDuration(event.warning_start, event.end_timestamp)}</Table.Td><Table.Td>{event.max_score.toPrecision(5)}<Text size="xs" c="dimmed">{formatTimestamp(event.peak_timestamp)}</Text></Table.Td>
                    <Table.Td>{event.max_robust_z.toFixed(2)}</Table.Td><Table.Td>{event.end_reason.replace('_', ' ')}</Table.Td>
                  </Table.Tr>)}</Table.Tbody>
                </Table>
              </ScrollArea>
            )}
          </Stack>
        </Paper>
      )}

      <Paper withBorder p="md">
        <Stack gap="sm">
          <Group justify="space-between"><Text fw={700}>Saved detection runs</Text><Badge variant="light">{savedRuns.length}</Badge></Group>
          {savedRuns.length === 0 ? <Text size="sm" c="dimmed">No saved anomaly detection runs yet.</Text> : (
            <ScrollArea>
              <Table highlightOnHover>
                <Table.Thead><Table.Tr><Table.Th>Name</Table.Th><Table.Th>Algorithm</Table.Th><Table.Th>Inference</Table.Th><Table.Th>Range</Table.Th><Table.Th>Warnings</Table.Th><Table.Th>Anomalies</Table.Th><Table.Th /></Table.Tr></Table.Thead>
                <Table.Tbody>{savedRuns.map((run) => <Table.Tr key={run.id}>
                  <Table.Td><Button variant="subtle" px={0} leftSection={<Activity size={15} />} onClick={() => openSaved(run.id)}>{run.name}</Button></Table.Td>
                  <Table.Td><Badge variant="light" color="violet">{algorithmDefinition(run.config.algorithm).label}</Badge></Table.Td>
                  <Table.Td>{run.testing_run_name}</Table.Td><Table.Td>{formatDuration(run.start_timestamp, run.end_timestamp)}</Table.Td><Table.Td>{run.warning_count}</Table.Td><Table.Td><Badge color={run.anomaly_count ? 'red' : 'gray'}>{run.anomaly_count}</Badge></Table.Td>
                  <Table.Td><Tooltip label="Delete run"><ActionIcon color="red" variant="subtle" onClick={() => removeSaved(run)}><Trash2 size={16} /></ActionIcon></Tooltip></Table.Td>
                </Table.Tr>)}</Table.Tbody>
              </Table>
            </ScrollArea>
          )}
        </Stack>
      </Paper>
    </Stack>
  );
}
