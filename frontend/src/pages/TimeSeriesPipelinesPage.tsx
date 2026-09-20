import { useEffect, useRef, useState } from 'react';
import { Alert, Badge, Button, Code, Group, Loader, Modal, NumberInput, Paper, Progress, Select, Stack, Table, Text, TextInput, Title } from '@mantine/core';
import { sensorApi, timeSeriesApi } from '../api';
import { fieldDefaults, Flow, ModelFields } from '../timeSeries/ModelFields';
import { pipelineBlockers, watchPipelinePreview, type PreviewState } from '../timeSeries/pipelinePreview';
import type { ModelDefinition, Parameters, SensorModel, SensorPipeline, SensorRun, TimeSeriesDataset, TimeSeriesSplit } from '../timeSeries/types';

export function TimeSeriesPipelinesPage({ active }: { active: boolean }) {
  const [pipelines, setPipelines] = useState<SensorPipeline[]>([]);
  const [datasets, setDatasets] = useState<TimeSeriesDataset[]>([]);
  const [splits, setSplits] = useState<TimeSeriesSplit[]>([]);
  const [models, setModels] = useState<SensorModel[]>([]);
  const [definitions, setDefinitions] = useState<ModelDefinition[]>([]);
  const [runs, setRuns] = useState<SensorRun[]>([]);
  const [id, setId] = useState<number>();
  const [name, setName] = useState('Neue Pipeline');
  const [dataset, setDataset] = useState<string | null>(null);
  const [split, setSplit] = useState<string | null>(null);
  const [modelId, setModelId] = useState<string | null>(null);
  const [length, setLength] = useState(36);
  const lengthTouched = useRef(false);
  const [training, setTraining] = useState<Parameters>({});
  const [previewState, setPreviewState] = useState<PreviewState>();
  const [previewRetry, setPreviewRetry] = useState(0);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [logs, setLogs] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<{ kind: 'pipeline' | 'run'; id: number }>();
  const model = models.find(m => m.id === Number(modelId));
  const definition = definitions.find(d => d.kind === model?.kind);
  const selectedDataset = datasets.find(d => d.id === Number(dataset));
  const selectedSplit = splits.find(s => s.id === Number(split) && s.dataset_id === selectedDataset?.id);
  const previewKey = JSON.stringify({ dataset, split, modelId, length: lengthTouched.current ? length : null, training, previewRetry });
  const currentPreviewState = previewState?.key === previewKey ? previewState : undefined;
  const preview = currentPreviewState?.status === 'ready' ? currentPreviewState.preview : undefined;
  const blockers = pipelineBlockers({ name, datasetSelected: !!selectedDataset, splitSelected: !!selectedSplit, modelSelected: !!model, state: currentPreviewState });
  const checking = !!selectedDataset && !!selectedSplit && !!model && (!currentPreviewState || currentPreviewState.status === 'loading');
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    Promise.all([sensorApi.pipelines(), timeSeriesApi.datasets(), timeSeriesApi.splits(), sensorApi.models(), sensorApi.definitions()]).then(([p, d, s, m, defs]) => {
      if (!cancelled) { setPipelines(p); setDatasets(d); setSplits(s); setModels(m); setDefinitions(defs); }
    }).catch(e => !cancelled && setError(String(e)));
    const refresh = () => sensorApi.runs().then(rows => !cancelled && setRuns(rows)).catch(e => !cancelled && setError(String(e)));
    void refresh(); const timer = setInterval(refresh, 3000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [active]);
  useEffect(() => {
    if (!active || !selectedSplit || !model) { setPreviewState(undefined); return; }
    return watchPipelinePreview(previewKey,
      signal => sensorApi.preview({ split_id: Number(split), model_id: Number(modelId), window_length: lengthTouched.current ? length : undefined, training }, signal),
      state => { setPreviewState(state); if (state.preview && !lengthTouched.current) setLength(state.preview.window_length); });
  }, [active, previewKey, selectedSplit, model]);
  function edit(p: SensorPipeline, clone = false) { setId(clone ? undefined : p.id); setName(p.name + (clone ? ' – Kopie' : '')); setDataset(String(p.dataset_id)); setSplit(String(p.split_id)); setModelId(String(p.model_id)); lengthTouched.current = true; setLength(p.window_length); setTraining(p.training); }
  async function save(start = false) {
    if (busy || blockers.length) return;
    setBusy(true); setError('');
    try { const row = await sensorApi.savePipeline({ name, split_id: Number(split), model_id: Number(modelId), window_length: length, training }, id); setId(row.id); lengthTouched.current = true; setPipelines(await sensorApi.pipelines()); if (start) { await sensorApi.start(row.id); setRuns(await sensorApi.runs()); } }
    catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  if (!active) return null;
  return <Stack p="md"><Group justify="space-between"><div><Title order={2}>Training Pipelines</Title><Text c="dimmed">Ein gemeinsamer Datenvertrag – vom gespeicherten Split bis zum ausgewählten Checkpoint.</Text></div><Button variant="light" onClick={() => { setId(undefined); setName('Neue Pipeline'); setSplit(null); setDataset(null); setModelId(null); setTraining({}); lengthTouched.current = false; setLength(36); }}>Neue Pipeline</Button></Group>
    {error && <Alert color="red" withCloseButton onClose={() => setError('')}>{error}</Alert>}
    <Group align="stretch">{pipelines.map(p => <Paper withBorder p="md" radius="md" key={p.id} style={{ flex: '1 1 260px' }}><Stack gap="xs"><Text fw={600}>{p.name}</Text><Text size="sm" c="dimmed">{p.snapshot.model.name} · L={p.window_length} · {p.snapshot.split.name}</Text><Group><Button size="xs" onClick={() => edit(p)}>Bearbeiten</Button><Button size="xs" variant="subtle" onClick={() => edit(p, true)}>Duplizieren</Button><Button size="xs" variant="light" onClick={async () => { try { await sensorApi.start(p.id); setRuns(await sensorApi.runs()); } catch (e) { setError(String(e)); } }}>Gespeicherten Stand starten</Button><Button size="xs" variant="subtle" color="red" onClick={() => setDeleting({ kind: 'pipeline', id: p.id })}>Löschen</Button></Group></Stack></Paper>)}</Group>
    <Paper withBorder radius="md" p="lg"><Stack><Title order={3}>{id ? 'Pipeline bearbeiten' : 'Pipeline konfigurieren'}</Title><TextInput label="Name" value={name} onChange={e => setName(e.currentTarget.value)} />
      <Group grow align="start"><Select searchable label="Datenbasis" data={datasets.map(d => ({ value: String(d.id), label: d.name }))} value={dataset} onChange={v => { setDataset(v); setSplit(null); }} /><Select searchable label="Gespeicherter Split" data={splits.filter(s => s.dataset_id === Number(dataset)).map(s => ({ value: String(s.id), label: s.name }))} value={split} onChange={setSplit} /><Select searchable label="Modell" data={models.map(m => ({ value: String(m.id), label: m.name }))} value={modelId} onChange={v => { setModelId(v); const kind = models.find(m => String(m.id) === v)?.kind; const d = definitions.find(d => d.kind === kind); setTraining(d ? fieldDefaults(d.training) : {}); }} /></Group>
      <Group grow align="start"><Select label="Skalierung" value="minmax" data={[{ value: 'minmax', label: 'Min-Max pro Sensor · ausschließlich Train · kein Clipping' }]} /><NumberInput label="Fensterlänge L (Samples)" description="MLTrace-Standard: 3 h nominale Historie. Modellwechsel erhält L." min={1} max={100000} allowDecimal={false} value={length} onChange={v => { lengthTouched.current = true; setLength(Number(v)); }} /></Group>
      <Flow steps={[datasets.find(d => String(d.id) === dataset)?.name ?? 'Datenbasis', splits.find(s => String(s.id) === split)?.name ?? 'Split', 'Train-only MinMax', `${length} × ${preview?.sensor_count ?? 'D'} · Schritt 1`, model?.name ?? 'Modell', 'Training → Validation-Checkpoint', 'Scores + Rekonstruktion + z_sensor']} />
      <Text size="sm" c="dimmed">Tags sind Annotationen. Alle Train-Zeilen des Splits werden verwendet. Kein Resampling, Auffüllen oder Clipping. Speichern übernimmt den aktuell angezeigten Split und die aktuelle Architektur; gespeicherte Läufe bleiben unverändert.</Text>
      {checking && <Group role="status"><Loader size="sm" /><Text size="sm">Vorschau wird berechnet …</Text></Group>}
      {preview && <Paper withBorder p="md"><Stack gap="sm"><Group><Badge>{preview.sensor_count} Sensoren</Badge><Text>Samplingintervall: {preview.sampling_interval_seconds / 60} min</Text><Text>Fenster: {preview.window_length} Samples @ {preview.sampling_interval_seconds / 60} min</Text></Group><Group><Text>Nominale Historie: {preview.nominal_history_seconds / 60} min</Text><Text>Zeitstempelspanne: {preview.timestamp_span_seconds / 60} min</Text><Text>Schrittweite: 1 Sample = {preview.sampling_interval_seconds / 60} min</Text><Text>Erkannte Zeitlücken: {preview.detected_gaps}</Text></Group>
        {preview.timestamp_span_min_seconds !== preview.timestamp_span_max_seconds && <Text size="sm">Tatsächliche Zeitstempelspannen: {(preview.timestamp_span_min_seconds ?? 0) / 60}–{(preview.timestamp_span_max_seconds ?? 0) / 60} min (oben: Median).</Text>}
        {preview.representation && <Text size="sm">Encoderrepräsentation: {preview.representation.shape.join(' × ')} = {preview.representation.dimension} Werte{preview.representation.pooling_factor ? ` · Poolingfaktor ${preview.representation.pooling_factor} (MLTrace-Anpassung)` : ''}</Text>}
        <Table><Table.Thead><Table.Tr><Table.Th>Gruppe</Table.Th><Table.Th>Zugeordnete Zeilen</Table.Th><Table.Th>Vollständige Fenster</Table.Th><Table.Th>Anlaufpunkte</Table.Th><Table.Th>Sequenzen</Table.Th></Table.Tr></Table.Thead><Table.Tbody>{Object.entries(preview.counts).map(([group, counts]) => <Table.Tr key={group}><Table.Td>{group}</Table.Td><Table.Td>{counts.rows}</Table.Td><Table.Td>{counts.windows}</Table.Td><Table.Td>{counts.warmup_rows}</Table.Td><Table.Td>{counts.segments}</Table.Td></Table.Tr>)}</Table.Tbody></Table>
        {preview.errors.map(e => <Alert key={e} color="red">{e}</Alert>)}{preview.scaler.constant_columns.length > 0 && <Alert color="yellow">Konstante Trainingssensoren (Nenner = 1): {preview.scaler.constant_columns.join(', ')}</Alert>}
      </Stack></Paper>}
      {definition && <><Title order={4}>Trainingsparameter</Title><ModelFields fields={definition.training} values={training} onChange={setTraining} /><Text size="sm">Checkpoint: mit Validation {model?.kind === 'usad' ? 'mittlerer Window anomaly score mit festen α/β' : model?.kind === 'tcn_ae' ? 'mittlerer Log-Cosh-Loss' : 'deterministische, rauschfreie Gauß-NLL + gewichteter KL-Term'}; ohne Validation letzte Epoche. Test beeinflusst die Auswahl nie.</Text></>}
      <Alert color={currentPreviewState?.status === 'error' || !!preview?.errors.length ? 'red' : blockers.length ? 'blue' : 'green'} title={blockers.length ? 'Speichern noch nicht möglich' : 'Bereit zum Speichern und Trainieren'} aria-live="polite">
        {blockers.length ? blockers.map(message => <Text key={message} size="sm">{message}</Text>) : <Text size="sm">Die aktuelle Auswahl wurde erfolgreich geprüft.</Text>}
        {!!selectedSplit && !!model && <Button mt="xs" size="xs" variant="light" disabled={busy} onClick={() => setPreviewRetry(value => value + 1)}>{checking ? 'Prüfung neu starten' : 'Erneut prüfen'}</Button>}
      </Alert>
      <Group><Button loading={busy} disabled={busy || !!blockers.length} onClick={() => save()}>Pipeline speichern</Button><Button loading={busy} variant="light" disabled={busy || !!blockers.length} onClick={() => save(true)}>Speichern und Training starten</Button></Group>
    </Stack></Paper>
    <Title order={3}>Läufe</Title>{runs.map(run => <Paper key={run.id} withBorder p="md"><Stack gap="xs"><Group justify="space-between"><Text fw={600}>#{run.id} · {run.name}</Text><Badge color={run.status === 'failed' ? 'red' : run.status === 'finished' ? 'green' : 'blue'}>{run.status} · {run.current_step}</Badge></Group><Progress value={100 * run.epoch / run.epochs} /><Text size="sm">Epoche {run.epoch}/{run.epochs} · Loss {run.train_loss?.toPrecision(5) ?? '–'} · Validation {run.val_loss?.toPrecision(5) ?? '–'} · {Math.round(run.duration_seconds ?? 0)} s · {run.device ?? 'wartet auf Scheduler'}</Text>{run.error_message && <Alert color="red">{run.error_message}</Alert>}<Group><Button variant="subtle" size="xs" onClick={() => sensorApi.logs(run.id).then(l => setLogs(l.text || 'Noch keine Logs.')).catch(e => setError(String(e)))}>Logs</Button>{['queued', 'running'].includes(run.status) ? <Button variant="subtle" color="red" size="xs" onClick={() => sensorApi.abort(run.id).then(() => sensorApi.runs()).then(setRuns).catch(e => setError(String(e)))}>Abbrechen</Button> : <Button variant="subtle" color="red" size="xs" onClick={() => setDeleting({ kind: 'run', id: run.id })}>Löschen</Button>}</Group></Stack></Paper>)}
    <Modal opened={logs !== null} onClose={() => setLogs(null)} title="Trainingslogs" size="xl"><Code block style={{ whiteSpace: 'pre-wrap' }}>{logs}</Code></Modal>
    <Modal opened={!!deleting} onClose={() => setDeleting(undefined)} title="Eintrag löschen"><Stack><Text>Eintrag und zugehörige Ergebnisartefakte löschen? Referenzierte Pipelines sind geschützt.</Text><Button color="red" onClick={async () => { try { if (deleting?.kind === 'run') await sensorApi.deleteRun(deleting.id); else if (deleting) await sensorApi.deletePipeline(deleting.id); setRuns(await sensorApi.runs()); setPipelines(await sensorApi.pipelines()); } catch (e) { setError(String(e)); } setDeleting(undefined); }}>Löschen</Button></Stack></Modal>
  </Stack>;
}
