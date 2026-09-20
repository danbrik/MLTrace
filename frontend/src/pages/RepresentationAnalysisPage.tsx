import {
  ActionIcon, Alert, Badge, Button, Group, Image, Loader, NumberInput, Paper, Progress,
  ScrollArea, Select, SimpleGrid, Stack, Switch, Table, Text, TextInput, Title,
} from '@mantine/core';
import { Plus, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import {
  abortRepresentationRun, createRepresentationRun, deleteRepresentationRun, getRepresentationLog,
  getRepresentationResults, getRepresentationRun, listPreprocessingPipelines, listRepresentationRuns,
  listTrainingDatasets, previewRepresentation, representationArtifactUrl,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart } from '../components/PlotlyChart';
import { LABELS, PHASES, nextEventId, projectionTraces, validateConfig } from '../representation/helpers';
import type {
  PreprocessingPipeline, RepresentationConfig, RepresentationInterval, RepresentationLabel,
  RepresentationPreview, RepresentationResults, RepresentationRun, TrainingDataset,
} from '../types';

const initial: RepresentationConfig = { training_dataset_id: 0, preprocessing_pipeline_id: 0, intervals: [], cluster_count: 3, pca_variance: 0.95, seed: 42 };
const activeRun = (run: RepresentationRun | null) => Boolean(run && ['queued', 'running'].includes(run.status));
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);

export function RepresentationAnalysisPage({ active }: { active: boolean }) {
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [config, setConfig] = useState<RepresentationConfig>(initial);
  const [preview, setPreview] = useState<{ signature: string; value: RepresentationPreview } | null>(null);
  const [runs, setRuns] = useState<RepresentationRun[]>([]);
  const [run, setRun] = useState<RepresentationRun | null>(null);
  const [results, setResults] = useState<RepresentationResults | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [log, setLog] = useState<string | null>(null);
  const eventCounter = useRef(1);
  const signature = JSON.stringify(config);
  const currentPreview = preview?.signature === signature ? preview.value : null;
  const validation = validateConfig(config);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    setLoading(true);
    Promise.all([listTrainingDatasets(), listPreprocessingPipelines(), listRepresentationRuns()])
      .then(([nextDatasets, nextPipelines, nextRuns]) => {
        if (cancelled) return;
        setDatasets(nextDatasets); setPipelines(nextPipelines); setRuns(nextRuns);
        setRun(current => nextRuns.find(item => item.id === current?.id) ?? nextRuns.find(item => activeRun(item)) ?? nextRuns[0] ?? null);
      }).catch(reason => { if (!cancelled) setError(errorText(reason)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [active]);

  useEffect(() => {
    if (!active || !run || !activeRun(run)) return;
    let cancelled = false;
    let pending = false;
    const timer = window.setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const next = await getRepresentationRun(run.id);
        if (cancelled) return;
        setPollError(null); setRun(next);
        setRuns(current => current.map(item => item.id === next.id ? next : item));
      } catch (reason) { if (!cancelled) setPollError(errorText(reason)); }
      finally { pending = false; }
    }, 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active, run?.id, run?.status]);

  useEffect(() => {
    setResults(null); setLog(null);
    if (!active || run?.status !== 'finished') return;
    let cancelled = false;
    getRepresentationResults(run.id).then(next => { if (!cancelled) setResults(next); })
      .catch(reason => { if (!cancelled) setError(errorText(reason)); });
    return () => { cancelled = true; };
  }, [active, run?.id, run?.status]);

  const plots = useMemo(() => results ? (['pca', 'umap'] as const).flatMap(projection => [false, true].map(events => ({
    id: `${projection}-${events}`, projection, title: `${projection.toUpperCase()} · ${events ? 'Anomalie-Ereignisse' : 'Ground Truth'}`,
    data: projectionTraces(results.points, projection, events),
  }))) : [], [results]);

  function addInterval() {
    setConfig(current => ({ ...current, intervals: [...current.intervals, {
      id: crypto.randomUUID(), name: `Bereich ${current.intervals.length + 1}`, label: 'normal',
      start: '', end: '', sampling_rate: 30, random: false, event_id: null,
    }] }));
  }
  function updateInterval(id: string, values: Partial<RepresentationInterval>) {
    setConfig(current => ({ ...current, intervals: current.intervals.map(item => item.id === id ? { ...item, ...values } : item) }));
  }
  function changeLabel(item: RepresentationInterval, label: RepresentationLabel) {
    const event = label === 'anomaly' ? item.event_id ?? `A${eventCounter.current++}` : null;
    updateInterval(item.id, { label, event_id: event });
  }
  async function action(work: () => Promise<void>) {
    setBusy(true); setError(null);
    try { await work(); } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  }
  function useTemplate() {
    if (!run) return;
    const copy = structuredClone(run.config);
    setConfig(copy); setPreview(null);
    eventCounter.current = Number(nextEventId(copy.intervals).slice(1));
  }

  return <Stack gap="lg">
    <div><Title order={2}>DINOv3-Repräsentationsanalyse</Title><Text c="dimmed">Explorative Struktur von Bildrepräsentationen und Vergleich mit Ground Truth.</Text></div>
    {error && <Alert color="red" withCloseButton onClose={() => setError(null)}>{error}</Alert>}
    {loading && <Loader size="sm" />}
    <Paper withBorder p="lg"><Stack>
      <Title order={4}>1 · Datensatz und Preprocessing</Title>
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        <Select label="Train/Test Dataset" searchable value={config.training_dataset_id ? String(config.training_dataset_id) : null}
          data={datasets.map(item => ({ value: String(item.id), label: item.name, disabled: item.invalid_rule_count > 0 }))}
          disabled={busy} onChange={value => { setConfig(current => ({ ...current, training_dataset_id: Number(value), intervals: [] })); setPreview(null); eventCounter.current = 1; }} />
        <Select label="Preprocessing-Pipeline" description="Crop oder Warp der Pipeline bestimmt die gemeinsame ROI." searchable
          value={config.preprocessing_pipeline_id ? String(config.preprocessing_pipeline_id) : null}
          data={pipelines.map(item => ({ value: String(item.id), label: item.name }))} disabled={busy}
          onChange={value => setConfig(current => ({ ...current, preprocessing_pipeline_id: Number(value) }))} />
      </SimpleGrid>
      <Text size="sm" c="dimmed">Der gespeicherte Datensatz-Stride wird zuerst angewendet. Nur hinzugefügte Bereiche werden analysiert.</Text>
    </Stack></Paper>
    <Paper withBorder p="lg"><Stack>
      <Group justify="space-between"><Title order={4}>2 · Bereiche auswählen</Title><Button leftSection={<Plus size={16} />} onClick={addInterval} disabled={!config.training_dataset_id || busy}>Bereich hinzufügen</Button></Group>
      <Text size="sm">Normal, Anomalie und Puffer nehmen gleichberechtigt an der Analyse teil. Beginn einschließlich, Ende ausschließlich.</Text>
      {config.intervals.map(item => <Paper key={item.id} withBorder p="md"><Stack gap="sm">
        <Group align="end">
          <TextInput label="Bereich" value={item.name} disabled={busy} onChange={event => updateInterval(item.id, { name: event.currentTarget.value })} />
          <Select label="Ground Truth" data={Object.entries(LABELS).map(([value, label]) => ({ value, label }))} value={item.label} allowDeselect={false}
            disabled={busy} onChange={value => changeLabel(item, value as RepresentationLabel)} />
          {item.event_id && <Badge>{item.event_id}</Badge>}
          <NumberInput label="Samplingrate (Bilder)" value={item.sampling_rate} min={1} allowDecimal={false} disabled={busy}
            onChange={value => updateInterval(item.id, { sampling_rate: Number(value) })} />
          <Switch label="Zufall" checked={item.random} disabled={busy} onChange={event => updateInterval(item.id, { random: event.currentTarget.checked })} />
          <ActionIcon color="red" variant="subtle" aria-label={`Bereich ${item.name} entfernen`} disabled={busy}
            onClick={() => setConfig(current => ({ ...current, intervals: current.intervals.filter(row => row.id !== item.id) }))}><Trash2 size={18} /></ActionIcon>
        </Group>
        <SimpleGrid cols={{ base: 1, md: 2 }}><DateTime24Input label="Beginn" value={item.start} disabled={busy} onChange={value => updateInterval(item.id, { start: value })} />
          <DateTime24Input label="Ende (ausschließlich)" value={item.end} disabled={busy} onChange={value => updateInterval(item.id, { end: value })} /></SimpleGrid>
      </Stack></Paper>)}
      {!config.intervals.length && <Text c="dimmed">Füge Bereiche für Normal, Anomalie und optional Puffer hinzu.</Text>}
      <Text size="sm" c="dimmed">Rate 30: Bild 30, 60, … oder mit Zufall ein Bild je vollständigem 30er-Block. Restblöcke entfallen. Die Rate zählt Bilder, keine Sekunden.</Text>
    </Stack></Paper>
    <Paper withBorder p="lg"><Stack>
      <Title order={4}>3 · Analyse und Vorschau</Title>
      <SimpleGrid cols={{ base: 1, md: 3 }}>
        <NumberInput label="Clusterzahl k" min={2} allowDecimal={false} value={config.cluster_count} disabled={busy} onChange={value => setConfig(current => ({ ...current, cluster_count: Number(value) }))} />
        <NumberInput label="Erklärte PCA-Varianz (%)" min={0.1} max={99.99} decimalScale={2} value={Number((config.pca_variance * 100).toFixed(2))} disabled={busy} onChange={value => setConfig(current => ({ ...current, pca_variance: Number(value) / 100 }))} />
        <NumberInput label="Zufallsseed" min={0} max={4294967295} allowDecimal={false} value={config.seed} disabled={busy} onChange={value => setConfig(current => ({ ...current, seed: Number(value) }))} />
      </SimpleGrid>
      <Text size="sm">DINOv3 ViT-S/16 · eingefrorener Encoder · CLS-Features · 224 × 224 Pixel</Text>
      {validation && <Text size="sm" c="dimmed">{validation}</Text>}
      <Group><Button variant="light" disabled={!!validation || busy} loading={busy} onClick={() => void action(async () => {
        const value = await previewRepresentation(config); setPreview({ signature, value });
      })}>Auswahl und Bildvorschau prüfen</Button>
        <Button disabled={busy || !!validation || !currentPreview || currentPreview.errors.length > 0} onClick={() => void action(async () => {
          const next = await createRepresentationRun(config); setRun(next); setRuns(current => [next, ...current]);
        })}>Analyse starten</Button></Group>
      {currentPreview && <>
        <Group>{Object.entries(currentPreview.label_counts).map(([label, count]) => <Badge key={label} variant="light">{LABELS[label as RepresentationLabel]}: {count}</Badge>)}<Text size="sm">Gesamt: {currentPreview.total}</Text></Group>
        <Table><Table.Thead><Table.Tr><Table.Th>Bereich</Table.Th><Table.Th>Verfügbar</Table.Th><Table.Th>Ausgewählt</Table.Th><Table.Th>Rest</Table.Th></Table.Tr></Table.Thead>
          <Table.Tbody>{currentPreview.intervals.map(item => <Table.Tr key={item.id}><Table.Td>{item.name} · {LABELS[item.label]} {item.event_id}</Table.Td><Table.Td>{item.available}</Table.Td><Table.Td>{item.selected}</Table.Td><Table.Td>{item.remainder}</Table.Td></Table.Tr>)}</Table.Tbody></Table>
        {currentPreview.errors.map(message => <Alert color="orange" key={message}>{message}</Alert>)}
        {currentPreview.image && <><Text size="sm">Beispielbild: {currentPreview.timestamp}</Text><SimpleGrid cols={2}>
          <div><Text size="sm">Pipeline-Ausgabe / ROI</Text><Image src={currentPreview.image} h={240} fit="contain" alt="Preprocessing-Ausgabe" /></div>
          <div><Text size="sm">Modelleingabe vor Normalisierung</Text><Image src={currentPreview.model_image ?? undefined} h={240} fit="contain" alt="DINOv3-Eingabe" /></div>
        </SimpleGrid></>}
      </>}
    </Stack></Paper>
    <Paper withBorder p="lg"><Stack>
      <Title order={4}>Gespeicherte Analysen</Title>
      <Select label="Lauf öffnen" searchable value={run ? String(run.id) : null} allowDeselect={false}
        data={runs.map(item => ({ value: String(item.id), label: `#${item.id} · ${item.training_dataset_name} · ${PHASES[item.status] ?? item.status}` }))}
        onChange={value => { setRun(runs.find(item => String(item.id) === value) ?? null); setPollError(null); }} />
      {run && <>
        <Group><Badge>{PHASES[run.status] ?? run.status}</Badge><Text>{run.training_dataset_name} · {run.pipeline_snapshot.name}</Text>
          <Button variant="subtle" disabled={busy} onClick={useTemplate}>Als Vorlage übernehmen</Button>
          <Button variant="subtle" disabled={busy} onClick={() => void action(async () => setLog((await getRepresentationLog(run.id)).log))}>Log anzeigen</Button>
          {activeRun(run) ? <Button color="orange" disabled={busy || run.cancel_requested} onClick={() => void action(async () => setRun(await abortRepresentationRun(run.id)))}>Abbrechen</Button>
            : <Button color="red" variant="subtle" disabled={busy} onClick={() => void action(async () => {
              await deleteRepresentationRun(run.id); setRuns(current => current.filter(item => item.id !== run.id)); setRun(null);
            })}>Lauf löschen</Button>}
        </Group>
        <Text size="sm">k={run.config.cluster_count} · PCA {(run.config.pca_variance * 100).toFixed(2)} % · Seed {run.config.seed} · {run.device ?? 'Gerät wird zugewiesen'}</Text>
        {activeRun(run) && <Group><Loader size="sm" /><Text>{run.cancel_requested ? 'Abbruch angefordert' : PHASES[run.current_step] ?? run.current_step}</Text></Group>}
        {activeRun(run) && run.total_images != null && run.total_images > 0 && <><Progress value={100 * run.processed_images / run.total_images} /><Text size="sm">{run.processed_images} / {run.total_images} Bilder</Text></>}
        {pollError && <Alert color="orange">Status konnte nicht aktualisiert werden: {pollError}</Alert>}
        {run.error_message && <Alert color={run.status === 'aborted' ? 'orange' : 'red'}>{run.error_message}</Alert>}
        {log !== null && <ScrollArea h={200}><Text component="pre" size="xs">{log || 'Noch keine Logeinträge.'}</Text></ScrollArea>}
      </>}
    </Stack></Paper>
    {run && results && <>
      <Paper withBorder p="lg"><Stack><Title order={4}>Cluster und Ground Truth</Title>
        <Group><Badge size="lg">ARI {results.metrics.ari.toFixed(4)}</Badge><Badge size="lg">NMI {results.metrics.nmi.toFixed(4)}</Badge></Group>
        <Text>{results.metrics.sample_count} Bilder · {results.metrics.feature_dimension} Features · {results.metrics.pca_components} PCA-Komponenten für {(results.metrics.pca_retained_variance * 100).toFixed(2)} % erklärte Varianz</Text>
        <Table><Table.Thead><Table.Tr><Table.Th>Cluster</Table.Th>{Object.values(LABELS).map(label => <Table.Th key={label}>{label}</Table.Th>)}</Table.Tr></Table.Thead>
          <Table.Tbody>{results.metrics.contingency.map(row => <Table.Tr key={row.cluster}><Table.Td>Cluster {row.cluster}</Table.Td><Table.Td>{row.normal}</Table.Td><Table.Td>{row.anomaly}</Table.Td><Table.Td>{row.buffer}</Table.Td></Table.Tr>)}</Table.Tbody></Table>
        <Text size="sm" c="dimmed">Die Cluster wurden ohne Ground Truth berechnet. Ihre Nummern sind keine Klassennamen. Die Diagramme zeigen explorative Projektionen.</Text>
        <Group>{[['samples.csv', 'Bilddaten CSV'], ['features.npz', 'Features NPZ'], ['analysis.json', 'Konfiguration und Metriken JSON']].map(([name, label]) => <Button key={name} component="a" href={representationArtifactUrl(run.id, name)} variant="light">{label}</Button>)}</Group>
      </Stack></Paper>
      <SimpleGrid cols={{ base: 1, xl: 2 }}>{plots.map(plot => <Paper withBorder p="md" key={plot.id}><Title order={5}>{plot.title}</Title>
        <PlotlyChart data={plot.data} height={430} layout={{
          xaxis: { title: { text: plot.projection === 'pca' ? `PC 1 (${(results.metrics.pca_2d_variance[0] * 100).toFixed(1)} %)` : 'UMAP 1' } },
          yaxis: { title: { text: plot.projection === 'pca' ? `PC 2 (${(results.metrics.pca_2d_variance[1] * 100).toFixed(1)} %)` : 'UMAP 2' } },
          legend: { orientation: 'h', y: -0.2 }, margin: { b: 100 },
        }} config={{ toImageButtonOptions: { format: 'png', filename: `representation-${run.id}-${plot.id}`, scale: 2 } }} />
      </Paper>)}</SimpleGrid>
    </>}
  </Stack>;
}
