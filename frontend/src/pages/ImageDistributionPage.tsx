import { ActionIcon, Alert, Badge, Button, Group, Loader, Paper, Progress, ScrollArea, Select, SimpleGrid, Stack, Table, Text, TextInput, Title } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Calculator, Download, Play, Plus, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import {
  calculateImageDistribution,
  calculateImageDistributionIntervals,
  getImageDistributionRun,
  imageDistributionCsvUrl,
  listPreprocessingPipelines,
  listImageDistributionRuns,
  listTrainingDatasets,
} from '../api';
import { PlotlyChart } from '../components/PlotlyChart';
import { DateTime24Input } from '../components/DateTime24Input';
import { withLineGapPolicy } from '../lib/plotGaps';
import type { Data, Layout } from '../lib/plotly';
import type {
  ImageDistributionHourlyPoint,
  ImageDistributionIntervalDraft,
  ImageDistributionIntervalMetric,
  ImageDistributionIntervalSummary,
  ImageDistributionMetricSummary,
  ImageDistributionResult,
  ImageDistributionRun,
  PreprocessingPipeline,
  TrainingDataset,
} from '../types';

type Props = { active: boolean };
type MetricKey = 'mean_intensity' | 'spatial_std_intensity' | 'q95_intensity';

const PERIOD_COLORS: Record<string, string> = {
  train: 'rgba(34, 139, 230, 0.10)',
  validation: 'rgba(64, 192, 87, 0.12)',
  test: 'rgba(250, 176, 5, 0.12)',
  mixed: 'rgba(132, 94, 247, 0.10)',
};
const PERIOD_BADGE_COLORS: Record<string, string> = {
  train: 'blue',
  validation: 'green',
  test: 'yellow',
  mixed: 'violet',
};
const STEP_LABELS: Record<string, string> = {
  queued: 'Wartet im Scheduler',
  queued_for_resume: 'Wartet auf automatische Fortsetzung',
  loading_configuration: 'Konfiguration wird geladen',
  waiting_for_folder_index: 'Wartet auf einen laufenden Ordnerindex',
  loading_folder_index: 'Gespeicherter Ordnerindex wird geladen',
  indexing_folders: 'NAS-Ordner werden streamingbasiert indexiert',
  selecting_images: 'Train/Test-Regeln und Stride werden angewendet',
  calibrating_workers: 'Optimale CPU-Parallelität wird kalibriert',
  resolving_images: 'Ausgewählte Bilder werden ermittelt',
  checking_cache: 'Gespeicherte CSV wird geprüft',
  loading_cache: 'Gespeicherte CSV wird geladen',
  preparing_pipeline: 'Preprocessing wird vorbereitet',
  processing_images: 'Bilder werden verarbeitet',
  writing_csv: 'Ergebnisse werden als CSV gespeichert',
  aggregating_hourly: 'Stündliche Kennzahlen werden aggregiert',
  finished: 'Analyse abgeschlossen',
  failed: 'Analyse fehlgeschlagen',
  aborted: 'Analyse abgebrochen',
};

function formatDurationSeconds(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return 'wird ermittelt';
  const seconds = Math.max(0, Math.round(value));
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days > 0) return `${days} T ${hours} Std`;
  if (hours > 0) return `${hours} Std ${minutes} Min`;
  if (minutes > 0) return `${minutes} Min`;
  return `${seconds} Sek`;
}

function formatBytes(value: number | null): string {
  if (value == null) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let scaled = value;
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${scaled.toLocaleString('de-DE', { maximumFractionDigits: unit >= 3 ? 2 : 1 })} ${units[unit]}`;
}

function formatMetric(value: number): string {
  return value.toLocaleString('de-DE', { maximumSignificantDigits: 6 });
}

function formatIqr(metric: ImageDistributionIntervalMetric): string {
  return `${formatMetric(metric.q25)} – ${formatMetric(metric.q75)} (Δ ${formatMetric(metric.iqr)})`;
}

function intervalInputValue(value: string): string {
  return value.slice(0, 19);
}

function newIntervalId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `interval-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function defaultIntervalBounds(result: ImageDistributionResult): { start: string; end: string } {
  if (result.hourly.length === 0) return { start: '', end: '' };
  const start = intervalInputValue(result.hourly[0].hour);
  const lastHour = parseDatasetTime(result.hourly[result.hourly.length - 1].hour);
  return {
    start,
    end: new Date(lastHour + 3_599_999).toISOString().slice(0, 19),
  };
}

function parseDatasetTime(value: string): number {
  return Date.parse(/[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`);
}

function continuity(points: ImageDistributionHourlyPoint[]): number[] {
  let segment = 0;
  return points.map((point, index) => {
    if (index > 0 && parseDatasetTime(point.hour) - parseDatasetTime(points[index - 1].hour) !== 3_600_000) segment += 1;
    return segment;
  });
}

function periodShapes(result: ImageDistributionResult): NonNullable<Partial<Layout>['shapes']> {
  return result.periods.map((period) => ({
    type: 'rect',
    xref: 'x',
    yref: 'paper',
    x0: period.start,
    x1: period.end,
    y0: 0,
    y1: 1,
    fillcolor: PERIOD_COLORS[period.usage_label] ?? 'rgba(134, 142, 150, 0.10)',
    line: { width: 0 },
    layer: 'below',
  }));
}

function metricTraces(points: ImageDistributionHourlyPoint[], key: MetricKey): Data[] {
  const x = points.map((point) => point.hour);
  const segments = continuity(points);
  const metric = (point: ImageDistributionHourlyPoint): ImageDistributionMetricSummary => point[key];
  const hover = points.map((point) => `${point.image_count} Bild${point.image_count === 1 ? '' : 'er'}`);
  return [
    withLineGapPolicy({
      type: 'scatter', mode: 'lines', x, y: points.map((point) => metric(point).q25),
      line: { width: 0 }, hoverinfo: 'skip', showlegend: false,
    } as Data, { continuity: segments }),
    withLineGapPolicy({
      type: 'scatter', mode: 'lines', x, y: points.map((point) => metric(point).q75),
      line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(34, 139, 230, 0.20)',
      name: 'Q25–Q75', hoverinfo: 'skip',
    } as Data, { continuity: segments }),
    withLineGapPolicy({
      type: 'scatter', mode: 'lines', x, y: points.map((point) => metric(point).median),
      line: { color: '#228be6', width: 2 }, name: 'Stündlicher Median',
      text: hover, hovertemplate: '%{x}<br>Median: %{y:.5g}<br>%{text}<extra></extra>',
    } as Data, { continuity: segments }),
  ];
}

function MetricPlot({ result, metricKey, title, yTitle }: {
  result: ImageDistributionResult;
  metricKey: MetricKey;
  title: string;
  yTitle: string;
}) {
  return (
    <Paper withBorder p="md">
      <Title order={4} mb="xs">{title}</Title>
      <PlotlyChart
        data={metricTraces(result.hourly, metricKey)}
        height={360}
        rescaleYOnVisibleX
        layout={{
          xaxis: { title: { text: 'Zeit' }, type: 'date', rangeslider: { visible: true } },
          yaxis: { title: { text: yTitle } },
          hovermode: 'x unified',
          legend: { orientation: 'h' },
          shapes: periodShapes(result),
          margin: { l: 72, r: 24, t: 12, b: 62 },
        }}
      />
    </Paper>
  );
}

export function ImageDistributionPage({ active }: Props) {
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [trainingDatasetId, setTrainingDatasetId] = useState<string | null>(null);
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [result, setResult] = useState<ImageDistributionResult | null>(null);
  const [activeRun, setActiveRun] = useState<ImageDistributionRun | null>(null);
  const [loadingOptions, setLoadingOptions] = useState(false);
  const [calculating, setCalculating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [intervals, setIntervals] = useState<ImageDistributionIntervalDraft[]>([]);
  const [intervalResults, setIntervalResults] = useState<ImageDistributionIntervalSummary[]>([]);
  const [intervalCalculating, setIntervalCalculating] = useState(false);
  const [intervalError, setIntervalError] = useState<string | null>(null);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    setLoadingOptions(true);
    Promise.all([listTrainingDatasets(), listPreprocessingPipelines(), listImageDistributionRuns()])
      .then(([nextTrainingDatasets, nextPipelines, runs]) => {
        if (cancelled) return;
        setTrainingDatasets(nextTrainingDatasets);
        setPipelines(nextPipelines);
        const ongoing = runs.find((run) => run.status === 'running' || run.status === 'queued');
        if (ongoing) {
          setActiveRun(ongoing);
          setTrainingDatasetId(String(ongoing.training_dataset_id));
          setPipelineId(String(ongoing.preprocessing_pipeline_id));
        }
      })
      .catch((reason) => { if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason)); })
      .finally(() => { if (!cancelled) setLoadingOptions(false); });
    return () => { cancelled = true; };
  }, [active]);

  useEffect(() => {
    setIntervals([]);
    setIntervalResults([]);
    setIntervalError(null);
  }, [result?.cache_key]);

  useEffect(() => {
    if (!active || !activeRun || !['queued', 'running'].includes(activeRun.status)) return undefined;
    let cancelled = false;
    let timer: number | null = null;
    let retryDelay = 1000;
    const poll = async () => {
      try {
        const next = await getImageDistributionRun(activeRun.id);
        if (cancelled) return;
        retryDelay = 1000;
        setActiveRun(next);
        setError(null);
        if (next.status === 'finished' && next.result) {
          setResult(next.result);
          notifications.show({
            color: next.failed_images ? 'yellow' : 'green',
            title: next.cache_hit ? 'Gespeicherte Analyse geladen' : 'Analyse abgeschlossen',
            message: `${next.successful_images.toLocaleString('de-DE')} von ${(next.total_images ?? 0).toLocaleString('de-DE')} Bildern ausgewertet.`,
          });
        } else if (next.status === 'failed' || next.status === 'aborted') {
          setError(next.error_message ?? (next.status === 'aborted' ? 'Analyse wurde abgebrochen.' : 'Analyse fehlgeschlagen.'));
        }
      } catch (reason) {
        if (!cancelled) {
          setError(`Fortschritt vorübergehend nicht erreichbar; der Scheduler-Job läuft weiter. ${reason instanceof Error ? reason.message : String(reason)}`);
          retryDelay = Math.min(15_000, retryDelay * 2);
        }
      } finally {
        if (!cancelled) timer = window.setTimeout(poll, retryDelay);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [active, activeRun?.id, activeRun?.status]);

  const datasetOptions = useMemo(() => trainingDatasets.map((dataset) => ({
    value: String(dataset.id),
    label: `${dataset.name} · ${dataset.usage_label}`,
    disabled: dataset.invalid_rule_count > 0,
  })), [trainingDatasets]);
  const pipelineOptions = useMemo(() => pipelines.map((pipeline) => ({ value: String(pipeline.id), label: pipeline.name })), [pipelines]);
  const hasActiveRun = activeRun != null && (activeRun.status === 'queued' || activeRun.status === 'running');
  const phaseProgress = activeRun?.phase_total && activeRun.phase_total > 0
    ? activeRun.phase_processed / activeRun.phase_total * 100
    : null;
  const imageProgress = activeRun?.total_images && activeRun.total_images > 0
    ? activeRun.processed_images / activeRun.total_images * 100
    : null;
  const displayedProgress = activeRun?.current_step === 'processing_images' || activeRun?.current_step === 'calibrating_workers'
    ? imageProgress
    : phaseProgress ?? imageProgress;
  const heartbeatStale = activeRun?.status === 'running' && activeRun.heartbeat_at != null
    && Date.now() - new Date(activeRun.heartbeat_at).getTime() > 120_000;

  async function run() {
    if (!trainingDatasetId || !pipelineId) return;
    setCalculating(true);
    setError(null);
    setResult(null);
    setActiveRun(null);
    try {
      const next = await calculateImageDistribution(Number(trainingDatasetId), Number(pipelineId));
      setActiveRun(next);
      notifications.show({
        color: 'blue',
        title: 'Analyse eingeplant',
        message: 'Der Vorgang ist jetzt im Scheduler sichtbar.',
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setCalculating(false);
    }
  }

  function addInterval() {
    if (!result) return;
    const bounds = defaultIntervalBounds(result);
    setIntervals((current) => [
      ...current,
      {
        id: newIntervalId(),
        name: `Zeitraum ${current.length + 1}`,
        ...bounds,
      },
    ]);
    setIntervalResults([]);
    setIntervalError(null);
  }

  function updateInterval(id: string, update: Partial<ImageDistributionIntervalDraft>) {
    setIntervals((current) => current.map((interval) => interval.id === id ? { ...interval, ...update } : interval));
    setIntervalResults([]);
    setIntervalError(null);
  }

  function removeInterval(id: string) {
    setIntervals((current) => current.filter((interval) => interval.id !== id));
    setIntervalResults((current) => current.filter((interval) => interval.id !== id));
    setIntervalError(null);
  }

  async function calculateIntervals() {
    if (!activeRun || activeRun.status !== 'finished' || intervals.length === 0) return;
    const invalid = intervals.find((interval) => (
      !interval.name.trim() || !interval.start || !interval.end || interval.end < interval.start
    ));
    if (invalid) {
      setIntervalError(`Bitte Name, Start und Ende für „${invalid.name || 'unbenannter Zeitraum'}“ prüfen.`);
      return;
    }
    setIntervalCalculating(true);
    setIntervalError(null);
    try {
      const response = await calculateImageDistributionIntervals(
        activeRun.id,
        intervals.map((interval) => ({ ...interval, name: interval.name.trim() })),
      );
      setIntervalResults(response.intervals);
    } catch (reason) {
      setIntervalError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setIntervalCalculating(false);
    }
  }

  return (
    <Stack gap="md">
      <div>
        <Title order={2}>Zeitliche Bildverteilung</Title>
        <Text c="dimmed">Beobachte Änderungen der gesamten Intensitätsverteilung und ihres oberen Bereichs über den vollständigen Datensatz.</Text>
      </div>

      <Paper withBorder p="md">
        <Stack gap="md">
          <SimpleGrid cols={{ base: 1, md: 2 }}>
            <Select label="Train/Test Dataset" placeholder="Train/Test Dataset auswählen" data={datasetOptions} value={trainingDatasetId} onChange={setTrainingDatasetId} searchable disabled={loadingOptions || calculating} />
            <Select label="Preprocessing" placeholder="Pipeline auswählen" data={pipelineOptions} value={pipelineId} onChange={setPipelineId} searchable disabled={loadingOptions || calculating} />
          </SimpleGrid>
          <Group>
            <Button leftSection={calculating ? <Loader size={16} color="white" /> : <Play size={16} />} onClick={run} disabled={!trainingDatasetId || !pipelineId || calculating || hasActiveRun}>
              {calculating ? 'Analyse wird eingeplant …' : hasActiveRun ? 'Analyse läuft im Scheduler' : 'Analyse starten'}
            </Button>
            <Text size="sm" c="dimmed">Gleiche, unveränderte Konfigurationen werden direkt aus der CSV geladen.</Text>
          </Group>
        </Stack>
      </Paper>

      {error && <Alert color="red" title="Analyse nicht möglich">{error}</Alert>}

      {activeRun && (
        <Paper withBorder p="md">
          <Stack gap="xs">
            <Group justify="space-between">
              <div>
                <Text fw={600}>{STEP_LABELS[activeRun.current_step] ?? activeRun.current_step}</Text>
                <Text size="sm" c="dimmed">
                  Lauf #{activeRun.id} · Status {activeRun.status}
                  {activeRun.total_images != null ? ` · ${activeRun.processed_images.toLocaleString('de-DE')} / ${activeRun.total_images.toLocaleString('de-DE')} Bilder` : ''}
                </Text>
              </div>
              <Badge color={activeRun.status === 'finished' ? 'green' : activeRun.status === 'failed' ? 'red' : activeRun.status === 'aborted' ? 'orange' : 'blue'}>
                {activeRun.status}
              </Badge>
            </Group>
            <Progress
              value={Math.min(100, displayedProgress ?? 0)}
              animated={activeRun.status === 'running'}
              color={activeRun.status === 'failed' ? 'red' : activeRun.status === 'aborted' ? 'orange' : 'blue'}
            />
            {activeRun.total_images != null && (
              <Text size="xs" c="dimmed">
                {activeRun.successful_images.toLocaleString('de-DE')} erfolgreich · {activeRun.failed_images.toLocaleString('de-DE')} fehlgeschlagen
              </Text>
            )}
            {heartbeatStale && (
              <Alert color="yellow" title="Keine aktuelle Fortschrittsmeldung">
                Seit mehr als zwei Minuten wurde kein Heartbeat gespeichert. Der Job wird nicht automatisch beendet und kann auf das NAS oder einen Ordnerindex warten.
              </Alert>
            )}
            <SimpleGrid cols={{ base: 2, md: 4 }}>
              <div><Text size="xs" c="dimmed">Durchsatz</Text><Text size="sm" fw={600}>{activeRun.throughput_images_per_second != null ? `${activeRun.throughput_images_per_second.toFixed(1)} Bilder/s` : 'wird kalibriert'}</Text></div>
              <div><Text size="xs" c="dimmed">NAS-Durchsatz</Text><Text size="sm" fw={600}>{activeRun.throughput_mb_per_second != null ? `${activeRun.throughput_mb_per_second.toFixed(1)} MB/s` : '—'}</Text></div>
              <div><Text size="xs" c="dimmed">Restlaufzeit</Text><Text size="sm" fw={600}>{formatDurationSeconds(activeRun.eta_seconds)}</Text></div>
              <div><Text size="xs" c="dimmed">CPU-Worker</Text><Text size="sm" fw={600}>{activeRun.effective_worker_count ?? '—'}</Text></div>
              <div><Text size="xs" c="dimmed">Gelesen</Text><Text size="sm" fw={600}>{formatBytes(activeRun.processed_bytes)}</Text></div>
              <div><Text size="xs" c="dimmed">Gesamtvolumen</Text><Text size="sm" fw={600}>{formatBytes(activeRun.total_bytes)}</Text></div>
              <div><Text size="xs" c="dimmed">Vergangen</Text><Text size="sm" fw={600}>{activeRun.started_at ? formatDurationSeconds((Date.now() - new Date(activeRun.started_at).getTime()) / 1000) : '—'}</Text></div>
              <div><Text size="xs" c="dimmed">Fortsetzungen</Text><Text size="sm" fw={600}>{activeRun.resume_count}</Text></div>
            </SimpleGrid>
            {(activeRun.stride_projections?.length ?? 0) > 0 && (
              <div>
                <Text size="sm" fw={600} mb={4}>Projektion für größeren Train/Test-Stride</Text>
                <SimpleGrid cols={{ base: 1, md: 3 }}>
                  {activeRun.stride_projections?.map((projection) => (
                    <Paper key={projection.factor} withBorder p="xs">
                      <Text size="sm" fw={600}>Stride × {projection.factor}</Text>
                      <Text size="xs" c="dimmed">
                        ≈ {projection.estimated_images.toLocaleString('de-DE')} Bilder · {formatDurationSeconds(projection.estimated_seconds)}
                      </Text>
                      <Text size="xs" c={projection.estimated_median_images_per_hour >= 300 ? 'green' : 'orange'}>
                        Median ≈ {projection.estimated_median_images_per_hour.toFixed(0)} Bilder/Stunde
                      </Text>
                    </Paper>
                  ))}
                </SimpleGrid>
                <Text size="xs" c="dimmed" mt={4}>Nur Empfehlung; geändert wird der Stride ausschließlich im Train/Test Dataset.</Text>
              </div>
            )}
          </Stack>
        </Paper>
      )}

      {result && (
        <>
          <Paper withBorder p="md">
            <Group justify="space-between" align="flex-start">
              <div>
                <Group gap="xs">
                  <Badge color={result.cache_hit ? 'blue' : 'green'}>{result.cache_hit ? 'Aus CSV geladen' : 'Neu berechnet'}</Badge>
                  <Text fw={600}>{result.training_dataset_name} ({result.usage_label}) · {result.preprocessing_pipeline_name}</Text>
                </Group>
                <Text size="sm" c="dimmed" mt={6}>
                  {result.successful_images.toLocaleString('de-DE')} Bilder, {result.hourly.length.toLocaleString('de-DE')} Stundenfenster. Pixelwerte werden in der von der Pipeline ausgegebenen Skala ausgewertet.
                </Text>
                {result.failed_images > 0 && <Text size="sm" c="orange" mt={4}>{result.failed_images.toLocaleString('de-DE')} Bilder konnten nicht verarbeitet werden; Details stehen in der CSV.</Text>}
              </div>
              <Button component="a" href={imageDistributionCsvUrl(result.cache_key)} leftSection={<Download size={16} />} variant="light">CSV herunterladen</Button>
            </Group>
            {result.periods.length > 0 && (
              <Group gap="xs" mt="md">
                <Text size="sm" fw={600}>Markierte Zeiträume:</Text>
                {Array.from(new Set(result.periods.map((period) => period.usage_label))).map((label) => (
                  <Badge key={label} variant="light" color={PERIOD_BADGE_COLORS[label] ?? 'gray'}>
                    {label} ({result.periods.filter((period) => period.usage_label === label).length})
                  </Badge>
                ))}
                <Text size="xs" c="dimmed">Die transparenten Flächen zeigen die gespeicherten Train-/Validierungs-/Testbereiche.</Text>
              </Group>
            )}
          </Paper>

          {activeRun?.status === 'finished' && (
            <Paper withBorder p="md">
              <Stack gap="md">
                <Group justify="space-between" align="flex-start">
                  <div>
                    <Title order={4}>Kennzahlen für frei gewählte Zeiträume</Title>
                    <Text size="sm" c="dimmed">
                      Die Quantile werden exakt aus den Einzelbildwerten der CSV berechnet, nicht aus den bereits aggregierten Stundenwerten.
                    </Text>
                  </div>
                  <Button variant="light" leftSection={<Plus size={16} />} onClick={addInterval}>
                    Zeitraum hinzufügen
                  </Button>
                </Group>

                {intervals.length === 0 ? (
                  <Alert color="blue">Füge einen oder mehrere Zeiträume hinzu, um sie miteinander zu vergleichen.</Alert>
                ) : (
                  <Stack gap="sm">
                    {intervals.map((interval) => {
                      const bounds = defaultIntervalBounds(result);
                      const rangeInvalid = Boolean(interval.start && interval.end && interval.end < interval.start);
                      return (
                        <Paper key={interval.id} withBorder p="sm">
                          <SimpleGrid cols={{ base: 1, lg: 4 }}>
                            <TextInput
                              label="Bezeichnung"
                              value={interval.name}
                              error={!interval.name.trim() ? 'Bezeichnung erforderlich' : undefined}
                              onChange={(event) => updateInterval(interval.id, { name: event.currentTarget.value })}
                            />
                            <DateTime24Input
                              label="Start"
                              value={interval.start}
                              min={bounds.start}
                              max={bounds.end}
                              error={rangeInvalid ? 'Start liegt nach Ende' : undefined}
                              onChange={(value) => updateInterval(interval.id, { start: value })}
                            />
                            <DateTime24Input
                              label="Ende"
                              value={interval.end}
                              min={bounds.start}
                              max={bounds.end}
                              error={rangeInvalid ? 'Ende liegt vor Start' : undefined}
                              onChange={(value) => updateInterval(interval.id, { end: value })}
                            />
                            <Group align="flex-end" justify="flex-end">
                              <ActionIcon
                                color="red"
                                variant="light"
                                size="lg"
                                aria-label={`Zeitraum ${interval.name} entfernen`}
                                onClick={() => removeInterval(interval.id)}
                              >
                                <Trash2 size={18} />
                              </ActionIcon>
                            </Group>
                          </SimpleGrid>
                        </Paper>
                      );
                    })}
                    <Group justify="flex-end">
                      <Button
                        leftSection={intervalCalculating ? <Loader size={16} color="white" /> : <Calculator size={16} />}
                        onClick={calculateIntervals}
                        disabled={intervalCalculating}
                      >
                        Zeitraumkennzahlen berechnen
                      </Button>
                    </Group>
                  </Stack>
                )}

                {intervalError && <Alert color="red">{intervalError}</Alert>}

                {intervalResults.length > 0 && (
                  <ScrollArea>
                    <Table striped highlightOnHover miw={1250} verticalSpacing="sm">
                      <Table.Thead>
                        <Table.Tr>
                          <Table.Th>Zeitraum</Table.Th>
                          <Table.Th>Bilder</Table.Th>
                          <Table.Th>Mean Median</Table.Th>
                          <Table.Th>Mean IQR (Q25–Q75)</Table.Th>
                          <Table.Th>Q95 Median</Table.Th>
                          <Table.Th>Q95 IQR (Q25–Q75)</Table.Th>
                          <Table.Th>Std Median</Table.Th>
                          <Table.Th>Std IQR (Q25–Q75)</Table.Th>
                        </Table.Tr>
                      </Table.Thead>
                      <Table.Tbody>
                        {intervalResults.map((interval) => (
                          <Table.Tr key={interval.id}>
                            <Table.Td>
                              <Text fw={600}>{interval.name}</Text>
                              <Text size="xs" c="dimmed">{intervalInputValue(interval.start)} – {intervalInputValue(interval.end)}</Text>
                            </Table.Td>
                            <Table.Td>{interval.image_count.toLocaleString('de-DE')}</Table.Td>
                            <Table.Td>{interval.mean_intensity ? formatMetric(interval.mean_intensity.median) : '—'}</Table.Td>
                            <Table.Td>{interval.mean_intensity ? formatIqr(interval.mean_intensity) : '—'}</Table.Td>
                            <Table.Td>{interval.q95_intensity ? formatMetric(interval.q95_intensity.median) : '—'}</Table.Td>
                            <Table.Td>{interval.q95_intensity ? formatIqr(interval.q95_intensity) : '—'}</Table.Td>
                            <Table.Td>{interval.spatial_std_intensity ? formatMetric(interval.spatial_std_intensity.median) : '—'}</Table.Td>
                            <Table.Td>{interval.spatial_std_intensity ? formatIqr(interval.spatial_std_intensity) : '—'}</Table.Td>
                          </Table.Tr>
                        ))}
                      </Table.Tbody>
                    </Table>
                  </ScrollArea>
                )}
              </Stack>
            </Paper>
          )}

          {result.hourly.length === 0 ? (
            <Alert color="yellow">Keine erfolgreich verarbeiteten Bilder für die Zeitaggregation vorhanden.</Alert>
          ) : (
            <Stack gap="md">
              <MetricPlot result={result} metricKey="mean_intensity" title="Mittlere Bildintensität über die Zeit" yTitle="Mittlere Pixelintensität" />
              <MetricPlot result={result} metricKey="spatial_std_intensity" title="Räumliche Pixel-Standardabweichung über die Zeit" yTitle="Räumliche Standardabweichung" />
              <MetricPlot result={result} metricKey="q95_intensity" title="95%-Quantil der Pixelintensität über die Zeit" yTitle="Q95 der Pixelintensität" />
            </Stack>
          )}
        </>
      )}
    </Stack>
  );
}
