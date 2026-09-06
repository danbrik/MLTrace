import {
  ActionIcon, Alert, Badge, Button, Group, Loader, MultiSelect, NumberInput, Paper,
  Progress, ScrollArea, Select, SimpleGrid, Stack, Switch, Table, Text, TextInput, Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Download, Plus, Play, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import {
  createResolutionSensitivityRun, getResolutionSensitivityRun, listEvaluationLabelSets,
  listPreprocessingPipelines, listResolutionSensitivityRuns, listTrainingDatasets,
  resolutionSensitivityExportUrl,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart } from '../components/PlotlyChart';
import type { Data } from '../lib/plotly';
import type {
  EvaluationLabelSet, PreprocessingPipeline, ResolutionSensitivityInterval,
  ResolutionSensitivityRun, TrainingDataset,
} from '../types';
import {
  hasRequiredPipelineMapping, mapSelectedPipelines, REQUIRED_RESOLUTIONS, targetIntervalsFromLabelSet,
} from '../resolutionSensitivity/helpers';

type Props = { active: boolean };
type Feature = 'mean_intensity' | 'q95_intensity' | 'spatial_std_intensity';
const RESOLUTIONS: number[] = [...REQUIRED_RESOLUTIONS];
const FEATURE_LABELS: Record<Feature, string> = {
  mean_intensity: 'Mean', q95_intensity: 'Q95', spatial_std_intensity: 'Std',
};
const STEP_LABELS: Record<string, string> = {
  queued: 'Wartet im Scheduler', loading_configuration: 'Konfiguration wird geladen',
  resolving_images: 'Verfügbare Bilder werden ermittelt', determining_data_range: 'SSIM-Wertespanne wird bestimmt',
  processing_images: 'Preprocessing-Pipelines werden ausgewertet', finished: 'Analyse abgeschlossen',
  failed: 'Analyse fehlgeschlagen', aborted: 'Analyse abgebrochen',
};

function uid(prefix: string): string {
  return `${prefix}-${globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`}`;
}

function fmt(value: number | null | undefined, digits = 4): string {
  return value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('de-DE', { maximumFractionDigits: digits });
}

export function ResolutionSensitivityPage({ active }: Props) {
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [labelSets, setLabelSets] = useState<EvaluationLabelSet[]>([]);
  const [runs, setRuns] = useState<ResolutionSensitivityRun[]>([]);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [pipelineIds, setPipelineIds] = useState<string[]>([]);
  const [labelSetId, setLabelSetId] = useState<string | null>(null);
  const [intervals, setIntervals] = useState<ResolutionSensitivityInterval[]>([]);
  const [samples, setSamples] = useState<number | string>(100);
  const [overrideEnabled, setOverrideEnabled] = useState(false);
  const [dataRange, setDataRange] = useState<number | string>(255);
  const [run, setRun] = useState<ResolutionSensitivityRun | null>(null);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [feature, setFeature] = useState<Feature>('mean_intensity');

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    setLoading(true);
    Promise.all([listTrainingDatasets(), listPreprocessingPipelines(), listEvaluationLabelSets(), listResolutionSensitivityRuns()])
      .then(([nextDatasets, nextPipelines, nextLabels, nextRuns]) => {
        if (cancelled) return;
        setDatasets(nextDatasets); setPipelines(nextPipelines); setLabelSets(nextLabels); setRuns(nextRuns);
        const preferred = nextRuns.find((item) => ['queued', 'running'].includes(item.status)) ?? nextRuns[0];
        if (preferred) setRun(preferred);
      })
      .catch((reason) => { if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [active]);

  useEffect(() => {
    if (!active || !run || !['queued', 'running'].includes(run.status)) return undefined;
    let cancelled = false;
    const timer = window.setInterval(() => {
      getResolutionSensitivityRun(run.id).then((next) => {
        if (cancelled) return;
        setRun(next);
        setRuns((current) => current.map((item) => item.id === next.id ? next : item));
        if (next.status === 'finished') notifications.show({ color: 'green', title: 'Sensitivitätsanalyse abgeschlossen', message: `Lauf #${next.id} ist fertig.` });
      }).catch(() => undefined);
    }, 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active, run?.id, run?.status]);

  const eligiblePipelines = useMemo(() => pipelines.filter((pipeline) => (
    pipeline.output_width === pipeline.output_height && RESOLUTIONS.includes(pipeline.output_width ?? 0)
  )), [pipelines]);
  const selectedMapping = useMemo(() => mapSelectedPipelines(pipelineIds, eligiblePipelines), [pipelineIds, eligiblePipelines]);
  const validMapping = hasRequiredPipelineMapping(selectedMapping);
  const availableLabelSets = labelSets.filter((item) => String(item.training_dataset_id) === datasetId);
  const activeProgress = run?.total_images ? Math.min(100, run.processed_images / run.total_images * 100) : 0;

  function addInterval(type: 'normal' | 'event') {
    setIntervals((current) => [...current, { id: uid(type), name: type === 'normal' ? `Normal ${current.filter((i) => i.type === type).length + 1}` : `Ereignis ${current.filter((i) => i.type === type).length + 1}`, type, start: '', end: '' }]);
  }

  function updateInterval(id: string, update: Partial<ResolutionSensitivityInterval>) {
    setIntervals((current) => current.map((item) => item.id === id ? { ...item, ...update } : item));
  }

  function chooseLabelSet(value: string | null) {
    setLabelSetId(value);
    const selected = labelSets.find((item) => String(item.id) === value);
    if (!selected) return;
    const imported = targetIntervalsFromLabelSet(selected);
    setIntervals((current) => [...current.filter((item) => item.type === 'normal'), ...imported]);
  }

  async function start() {
    setError(null);
    const sampleCount = Number(samples);
    if (!datasetId || !validMapping || !Number.isInteger(sampleCount) || sampleCount < 1) return;
    setStarting(true);
    try {
      const next = await createResolutionSensitivityRun({
        training_dataset_id: Number(datasetId), pipeline_ids: pipelineIds.map(Number), intervals,
        samples_per_interval: sampleCount, label_set_id: labelSetId ? Number(labelSetId) : null,
        ssim_data_range: overrideEnabled ? Number(dataRange) : null,
      });
      setRun(next); setRuns((current) => [next, ...current]);
      notifications.show({ color: 'blue', title: 'Analyse eingeplant', message: 'Der CPU-Lauf ist jetzt im Scheduler sichtbar.' });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally { setStarting(false); }
  }

  const result = run?.result;
  const separationTraces: Data[] = result ? RESOLUTIONS.map((resolution) => ({
    type: 'bar', name: `${resolution}×${resolution}`,
    x: result.separations.filter((item) => item.feature === feature && item.resolution === resolution).map((item) => item.event_name),
    y: result.separations.filter((item) => item.feature === feature && item.resolution === resolution).map((item) => item.separation),
  } as Data)) : [];

  return <Stack gap="md">
    <div><Title order={2}>Auflösungs-Sensitivität</Title><Text c="dimmed">Vergleiche den Informationsgehalt vollständiger Preprocessing-Pipelines bei 840, 512, 256 und 128 Pixeln.</Text></div>
    <Paper withBorder p="md"><Stack gap="md">
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        <Select label="Train/Test Dataset" data={datasets.map((item) => ({ value: String(item.id), label: item.name, disabled: item.invalid_rule_count > 0 }))} value={datasetId} onChange={(value) => { setDatasetId(value); setLabelSetId(null); }} searchable />
        <MultiSelect label="Preprocessing-Pipelines" description="Je genau eine Ausgabe mit 840, 512, 256 und 128 Pixeln" data={eligiblePipelines.map((item) => ({ value: String(item.id), label: `${item.name} · ${item.output_width}×${item.output_height}` }))} value={pipelineIds} onChange={setPipelineIds} maxValues={4} searchable />
        <Select label="Evaluation-Labelset (optional)" description="Zielereignisse werden als editierbare Vorlage übernommen" data={availableLabelSets.map((item) => ({ value: String(item.id), label: `${item.name} · v${item.version}` }))} value={labelSetId} onChange={chooseLabelSet} clearable disabled={!datasetId} />
        <NumberInput label="Sollzeitpunkte pro Zeitraum" min={1} max={10000} value={samples} onChange={setSamples} />
      </SimpleGrid>
      <Group gap="xs">{RESOLUTIONS.map((resolution) => <Badge key={resolution} color={selectedMapping.get(resolution) ? 'green' : 'gray'}>{resolution}: {selectedMapping.get(resolution)?.name ?? 'fehlt'}</Badge>)}</Group>
      <Group align="flex-end"><Switch label="SSIM data_range manuell setzen" checked={overrideEnabled} onChange={(event) => setOverrideEnabled(event.currentTarget.checked)} />{overrideEnabled && <NumberInput label="data_range" min={Number.EPSILON} value={dataRange} onChange={setDataRange} />}</Group>
      <Group justify="space-between"><Group><Button variant="light" leftSection={<Plus size={16} />} onClick={() => addInterval('normal')}>Normalzeitraum</Button><Button variant="light" color="orange" leftSection={<Plus size={16} />} onClick={() => addInterval('event')}>Ereigniszeitraum</Button></Group><Button leftSection={starting ? <Loader size={16} color="white" /> : <Play size={16} />} disabled={!datasetId || !validMapping || intervals.length < 2 || starting} onClick={start}>Analyse starten</Button></Group>
      {intervals.map((interval) => <Paper key={interval.id} withBorder p="sm"><SimpleGrid cols={{ base: 1, lg: 4 }}>
        <TextInput label={interval.type === 'normal' ? 'Normalzeitraum' : 'Ereignis'} value={interval.name} onChange={(event) => updateInterval(interval.id, { name: event.currentTarget.value })} />
        <DateTime24Input label="Start" value={interval.start} onChange={(value) => updateInterval(interval.id, { start: value })} />
        <DateTime24Input label="Ende" value={interval.end} onChange={(value) => updateInterval(interval.id, { end: value })} />
        <Group align="flex-end" justify="flex-end"><Badge color={interval.type === 'normal' ? 'blue' : 'orange'}>{interval.type === 'normal' ? 'Normal' : 'Event'}</Badge><ActionIcon color="red" variant="light" onClick={() => setIntervals((current) => current.filter((item) => item.id !== interval.id))}><Trash2 size={17} /></ActionIcon></Group>
      </SimpleGrid></Paper>)}
    </Stack></Paper>
    {error && <Alert color="red" title="Analyse nicht möglich">{error}</Alert>}

    {runs.length > 0 && <Paper withBorder p="md"><Select label="Gespeicherter Lauf" data={runs.map((item) => ({ value: String(item.id), label: `#${item.id} · ${item.training_dataset_name} · ${item.status}` }))} value={run ? String(run.id) : null} onChange={(value) => { const selected = runs.find((item) => String(item.id) === value); if (selected) setRun(selected); }} /></Paper>}
    {run && <Paper withBorder p="md"><Stack gap="xs"><Group justify="space-between"><div><Text fw={600}>{STEP_LABELS[run.current_step] ?? run.current_step}</Text><Text size="sm" c="dimmed">Lauf #{run.id} · {run.training_dataset_name}</Text></div><Badge color={run.status === 'finished' ? 'green' : run.status === 'failed' ? 'red' : 'blue'}>{run.status}</Badge></Group>{['queued', 'running'].includes(run.status) && <Progress value={activeProgress} animated />}{run.error_message && <Alert color="red">{run.error_message}</Alert>}</Stack></Paper>}

    {result && <>
      <Paper withBorder p="md"><Group justify="space-between"><div><Title order={4}>Ergebnisübersicht</Title><Text size="sm" c="dimmed">{result.sample_count} Stichproben · {result.unique_image_count} eindeutige Bilder · {result.duplicate_sample_count} Ersatz-Duplikate · SSIM data_range {fmt(result.data_range)}</Text></div><Group><Button component="a" href={resolutionSensitivityExportUrl(run.id, 'details')} variant="light" leftSection={<Download size={16} />}>Details CSV</Button><Button component="a" href={resolutionSensitivityExportUrl(run.id, 'summary')} variant="light" leftSection={<Download size={16} />}>Summary CSV</Button></Group></Group></Paper>
      <ScrollArea><Table striped miw={1250}><Table.Thead><Table.Tr><Table.Th>Auflösung</Table.Th><Table.Th>Pixel / Anteil</Table.Th><Table.Th>SSIM Median / IQR</Table.Th><Table.Th>MAE Median / IQR</Table.Th>{(Object.keys(FEATURE_LABELS) as Feature[]).map((key) => <Table.Th key={key}>{FEATURE_LABELS[key]} Sep. Median / Minimum |Sep|</Table.Th>)}<Table.Th>Schwächstes Ereignis</Table.Th></Table.Tr></Table.Thead><Table.Tbody>{result.overview.map((item) => <Table.Tr key={item.resolution}><Table.Td fw={600}>{item.resolution}×{item.resolution}</Table.Td><Table.Td>{item.pixel_count.toLocaleString('de-DE')} / {fmt(item.pixel_share * 100, 1)} %</Table.Td><Table.Td>{fmt(item.ssim.median)} / {fmt(item.ssim.iqr)}</Table.Td><Table.Td>{fmt(item.mae.median)} / {fmt(item.mae.iqr)}</Table.Td>{(Object.keys(FEATURE_LABELS) as Feature[]).map((key) => <Table.Td key={key}>{fmt(item.features[key].median_separation)} / {fmt(item.features[key].minimum_absolute_separation)}</Table.Td>)}<Table.Td>{item.weakest_event ?? '—'}{item.weakest_feature ? ` · ${FEATURE_LABELS[item.weakest_feature as Feature]}` : ''}</Table.Td></Table.Tr>)}</Table.Tbody></Table></ScrollArea>
      <SimpleGrid cols={{ base: 1, lg: 2 }}><Paper withBorder p="md"><Title order={4}>SSIM und MAE</Title><PlotlyChart height={350} data={[{ type: 'bar', name: 'SSIM', x: result.overview.map((item) => String(item.resolution)), y: result.overview.map((item) => item.ssim.median), yaxis: 'y' } as Data, { type: 'scatter', mode: 'lines+markers', name: 'MAE', x: result.overview.map((item) => String(item.resolution)), y: result.overview.map((item) => item.mae.median), yaxis: 'y2' } as Data]} layout={{ xaxis: { title: { text: 'Auflösung' } }, yaxis: { title: { text: 'SSIM' } }, yaxis2: { title: { text: 'MAE' }, overlaying: 'y', side: 'right' }, margin: { l: 60, r: 60, t: 20, b: 50 } }} /></Paper><Paper withBorder p="md"><Group justify="space-between"><Title order={4}>Event-Separation</Title><Select w={150} data={(Object.keys(FEATURE_LABELS) as Feature[]).map((key) => ({ value: key, label: FEATURE_LABELS[key] }))} value={feature} onChange={(value) => setFeature((value ?? 'mean_intensity') as Feature)} /></Group><PlotlyChart height={350} data={separationTraces} layout={{ barmode: 'group', yaxis: { title: { text: 'Robuster z-Wert' } }, margin: { l: 60, r: 20, t: 20, b: 70 } }} /></Paper></SimpleGrid>
      <ScrollArea><Table striped miw={900}><Table.Thead><Table.Tr><Table.Th>Ereignis</Table.Th><Table.Th>Merkmal</Table.Th>{RESOLUTIONS.map((resolution) => <Table.Th key={resolution}>{resolution} Sep / R</Table.Th>)}</Table.Tr></Table.Thead><Table.Tbody>{Array.from(new Set(result.separations.map((item) => `${item.event_id}|${item.feature}`))).map((key) => { const [eventId, metric] = key.split('|'); const rows = result.separations.filter((item) => item.event_id === eventId && item.feature === metric); return <Table.Tr key={key}><Table.Td>{rows[0]?.event_name}</Table.Td><Table.Td>{FEATURE_LABELS[metric as Feature]}</Table.Td>{RESOLUTIONS.map((resolution) => { const item = rows.find((entry) => entry.resolution === resolution); return <Table.Td key={resolution}>{fmt(item?.separation)} / {fmt(item?.retention)}</Table.Td>; })}</Table.Tr>; })}</Table.Tbody></Table></ScrollArea>
    </>}
    {loading && <Group justify="center"><Loader /></Group>}
  </Stack>;
}
