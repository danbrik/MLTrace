import { Alert, Badge, Button, Group, Image, Loader, NumberInput, Paper, Progress, Select, SimpleGrid, Stack, Table, Text, Title } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { TimeRangePresetPicker } from '../timeRangePresets/TimeRangePresetPicker';
import { DateTime24Input } from '../components/DateTime24Input';
import {
  abortMeanVarianceRun, createMeanVarianceRun, deleteMeanVarianceRun, getMeanVarianceLog,
  getMeanVarianceResults, getMeanVarianceRun, listPreprocessingPipelines, listMeanVarianceRuns,
  listTrainingDatasets, previewMeanVariance, meanVarianceArtifactUrl,
} from '../api';
import { displayTime, emptyPair, initialConfig, phaseLabel, templateConfig, validateConfig } from '../meanVariance/helpers';
import type { PreprocessingPipeline, TrainingDataset } from '../types';
import { isPairConfig, isPairResults, type MeanVarianceConfig, type MeanVariancePreview, type MeanVarianceResults, type MeanVarianceRun, type HeatmapScale } from '../meanVariance/types';
const activeRun = (run: MeanVarianceRun | null) => !!run && ['queued', 'running'].includes(run.status);
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const periodText = (period: {start: string; end: string}) => `${displayTime(period.start)} – ${displayTime(period.end)}`;
const scaleText = (scale: HeatmapScale) => scale.mode === 'auto' ? 'Automatisch' : `${scale.limit} gray value²`;

export function MeanVariancePage({ active, projectId }: { active: boolean; projectId: string }) {
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [config, setConfig] = useState<MeanVarianceConfig>(() => structuredClone(initialConfig));
  const [preview, setPreview] = useState<{ signature: string; value: MeanVariancePreview } | null>(null);
  const [runs, setRuns] = useState<MeanVarianceRun[]>([]);
  const [run, setRun] = useState<MeanVarianceRun | null>(null);
  const [results, setResults] = useState<MeanVarianceResults | null>(null);
  const [log, setLog] = useState<string | null>(null);
  const [busy, setBusy] = useState(false), [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null), [pollError, setPollError] = useState<string | null>(null);
  const initialized = useRef(false);
  const viewRevision = useRef(0);
  const signature = JSON.stringify(config);
  const currentPreview = preview?.signature === signature ? preview.value : null;
  const dataset = datasets.find(item => item.id === config.training_dataset_id);
  const validation = validateConfig(config, dataset?.start_timestamp ?? undefined, dataset?.end_timestamp ?? undefined)
    ?? (!datasets.some(item => item.id === config.training_dataset_id && !item.invalid_rule_count) ? 'Bitte einen verfügbaren Datensatz auswählen.' : null)
    ?? (!pipelines.some(item => item.id === config.preprocessing_pipeline_id) ? 'Bitte eine verfügbare Preprocessing-Pipeline auswählen.' : null);
  function openRun(next: MeanVarianceRun | null) {
    viewRevision.current++; setRun(next); setResults(null); setPreview(null); setLog(null); setPollError(null); setError(null);
  }
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    setLoading(true);
    Promise.all([listTrainingDatasets(), listPreprocessingPipelines(), listMeanVarianceRuns(projectId)])
      .then(([nextDatasets, nextPipelines, nextRuns]) => {
        if (cancelled) return;
        setDatasets(nextDatasets); setPipelines(nextPipelines); setRuns(nextRuns);
        if (!initialized.current) {
          initialized.current = true;
          openRun(nextRuns.find(item => activeRun(item)) ?? nextRuns[0] ?? null);
        }
      }).catch(reason => { if (!cancelled) setError(errorText(reason)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [active, projectId]);
  useEffect(() => {
    if (!active || !run || !activeRun(run)) return;
    let cancelled = false, pending = false;
    const revision = viewRevision.current;
    const timer = window.setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const next = await getMeanVarianceRun(run.id, projectId);
        if (!cancelled && revision === viewRevision.current) { setPollError(null); setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item)); }
      } catch (reason) { if (!cancelled && revision === viewRevision.current) setPollError(errorText(reason)); }
      finally { pending = false; }
    }, 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active, projectId, run?.id, run?.status]);
  useEffect(() => {
    if (!active || run?.status !== 'finished') return;
    let cancelled = false;
    const revision = viewRevision.current;
    getMeanVarianceResults(run.id, projectId).then(next => {
      if (!cancelled && revision === viewRevision.current) setResults(next);
    }).catch(reason => { if (!cancelled && revision === viewRevision.current) setError(errorText(reason)); });
    return () => { cancelled = true; };
  }, [active, projectId, run?.id, run?.status]);
  async function action(work: () => Promise<void>) {
    setBusy(true); setError(null);
    try { await work(); } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  }
  function update(values: Partial<MeanVarianceConfig>) { setPreview(null); setConfig(current => ({ ...current, ...values })); }
  function updatePeriod(index: number, role: 'normal' | 'anomaly', values: Partial<{ start: string; end: string }>) {
    setPreview(null);
    setConfig(current => ({ ...current, pairs: current.pairs.map((pair, i) => i === index ? { ...pair, [role]: { ...pair[role], ...values } } : pair) }));
  }
  function newDraft(template?: MeanVarianceRun) {
    initialized.current = true;
    setConfig(template ? templateConfig(template.config) : structuredClone(initialConfig)); openRun(null);
  }
  return <Stack gap="lg">
    <Group justify="space-between"><div><Title order={2}>Varianzvergleich</Title><Text c="dimmed">Zeitliche Varianzen für bis zu sechs Paare aus Normalzustand und Anomaliephase.</Text></div>
      <Button variant="light" disabled={busy || loading} onClick={() => newDraft()}>Neuer Vergleich</Button></Group>
    {error && <Alert color="red" withCloseButton onClose={() => setError(null)}>{error}</Alert>}
    {loading && <Loader size="sm" />}
    <Select label="Lauf öffnen" searchable disabled={busy || loading} value={run ? String(run.id) : null} allowDeselect={false}
      data={runs.map(item => ({ value: String(item.id), label: `#${item.id} · ${item.training_dataset_name} · ${phaseLabel(item.status)}` }))}
      onChange={value => openRun(runs.find(item => String(item.id) === value) ?? null)} />
    {run ? <Paper withBorder p="lg"><Stack>
      <Group><Title order={4}>Gespeicherte Einstellungen · Lauf #{run.id}</Title><Badge>{phaseLabel(run.status)}</Badge></Group>
      <Text>Datensatz: {run.training_dataset_name} · Preprocessing: {run.pipeline_snapshot.name}</Text>
      <Text size="sm" c="dimmed">Schreibgeschützt. Änderungen über „Als Vorlage übernehmen“ erzeugen einen neuen Lauf.</Text>
      {isPairConfig(run.config) ? <>
        <Text>Gemeinsames Sampling: jedes {run.config.sampling_rate}. Bild</Text>
        <Text size="sm">Varianzskala: {scaleText(run.config.variance_scale)} · Differenzskala: {scaleText(run.config.difference_scale)}</Text>
        <Table><Table.Thead><Table.Tr><Table.Th>Paar</Table.Th><Table.Th>Normalzustand (einschließlich)</Table.Th><Table.Th>Anomaliephase (einschließlich)</Table.Th></Table.Tr></Table.Thead>
          <Table.Tbody>{run.config.pairs.map((pair, i) => <Table.Tr key={i}><Table.Td>u{i + 1}</Table.Td><Table.Td>{periodText(pair.normal)}</Table.Td><Table.Td>{periodText(pair.anomaly)}</Table.Td></Table.Tr>)}</Table.Tbody></Table>
      </> : <>
        <Alert color="blue">Bisheriger Einzelvergleich. Die gespeicherte Varianzdifferenz bleibt unverändert; es erfolgt keine Neuberechnung.</Alert>
        <Text>Normalzustand: {periodText(run.config.reference)}</Text>
        <Text size="sm">{run.config.reference.mode === 'random' ? `Zufall: ${run.config.reference.count} Bilder · Seed ${run.config.reference.seed}` : `Sampling: jedes ${run.config.reference.sampling_rate}. Bild`}</Text>
        <Text>Anomaliephase: {periodText(run.config.anomaly)}</Text><Text size="sm">Sampling: jedes {run.config.anomaly.sampling_rate}. Bild · Differenzskala: {scaleText(run.config.variance_scale)}</Text>
      </>}
      <Group><Button disabled={busy} onClick={() => newDraft(run)}>Als Vorlage übernehmen</Button>
        <Button variant="subtle" disabled={busy} onClick={() => void action(async () => {
          const revision = viewRevision.current, next = await getMeanVarianceLog(run.id, projectId);
          if (revision === viewRevision.current) setLog(next.log);
        })}>Log anzeigen</Button>
        {activeRun(run) ? <Button color="orange" disabled={busy || run.cancel_requested} onClick={() => void action(async () => {
          const next = await abortMeanVarianceRun(run.id, projectId); setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item));
        })}>{run.cancel_requested ? 'Abbruch angefordert' : 'Abbrechen'}</Button> : <Button color="red" variant="subtle" disabled={busy} onClick={() => void action(async () => {
          await deleteMeanVarianceRun(run.id, projectId); setRuns(current => current.filter(item => item.id !== run.id)); openRun(null); setConfig(structuredClone(initialConfig));
        })}>Lauf löschen</Button>}
      </Group>
      {activeRun(run) && <><Text size="sm">{phaseLabel(run.current_step)} · {run.processed_images}/{run.total_images ?? '…'}</Text><Progress value={run.total_images ? 100 * run.processed_images / run.total_images : 0} animated /></>}
      {pollError && <Alert color="orange">{pollError}</Alert>}{run.error_message && <Alert color="red">{run.error_message}</Alert>}
      {log !== null && <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 240, overflow: 'auto' }}>{log || 'Noch keine Log-Einträge.'}</pre>}
    </Stack></Paper> : <>
      <Paper withBorder p="lg"><Stack>
        <Title order={4}>Gemeinsame Einstellungen</Title>
        <SimpleGrid cols={{ base: 1, md: 3 }}>
          <Select label="Train/Test Dataset" searchable value={config.training_dataset_id ? String(config.training_dataset_id) : null} disabled={busy}
            data={datasets.map(item => ({ value: String(item.id), label: item.name, disabled: item.invalid_rule_count > 0 }))}
            onChange={value => update({ training_dataset_id: Number(value) })} />
          <Select label="Preprocessing-Pipeline" searchable value={config.preprocessing_pipeline_id ? String(config.preprocessing_pipeline_id) : null} disabled={busy}
            data={pipelines.map(item => ({ value: String(item.id), label: item.name }))} onChange={value => update({ preprocessing_pipeline_id: Number(value) })} />
          <NumberInput label="Gemeinsames Sampling: jedes n-te Bild" min={1} allowDecimal={false} value={config.sampling_rate || ''} disabled={busy}
            onChange={value => update({ sampling_rate: Number(value) })} />
        </SimpleGrid>
        {config.sampling_rate === 0 && <Alert color="yellow">Die bisherige Auswahl verwendet unterschiedliche Sampling-Werte oder Zufall. Bitte ein gemeinsames n ausdrücklich auswählen.</Alert>}
        {dataset && <Text size="sm">Datensatzgrenzen: {displayTime(dataset.start_timestamp ?? '—')} bis {displayTime(dataset.end_timestamp ?? '—')}</Text>}
        <Text size="sm" c="dimmed">Alle Paare teilen Datensatz, Pipeline und Sampling. Beide Zeitgrenzen zählen mit. Datensatzsampling gilt zuerst; jedes n-te Bild beginnt je Zeitraum neu. Überlappungen sind erlaubt.</Text>
      </Stack></Paper>
      {config.pairs.map((pair, index) => <Paper key={index} withBorder p="lg"><Stack>
        <Group justify="space-between"><Title order={4}>u{index + 1}</Title><Button color="red" variant="subtle" disabled={busy || config.pairs.length === 1}
          onClick={() => update({ pairs: config.pairs.filter((_, i) => i !== index) })}>Paar entfernen</Button></Group>
        <SimpleGrid cols={{ base: 1, lg: 2 }}>{(['normal', 'anomaly'] as const).map(role => <Stack key={role}>
          <Text fw={600}>{role === 'normal' ? 'Normalzustand' : 'Anomaliephase'}</Text>
          <TimeRangePresetPicker projectId={projectId} active={active} value={pair[role]} disabled={busy}
            min={dataset?.start_timestamp ?? undefined} max={dataset?.end_timestamp ?? undefined}
            applyDisabledReason={!dataset ? 'Bitte zuerst einen Datensatz auswählen.' : undefined} onApply={range => updatePeriod(index, role, range)} />
          <DateTime24Input label={`u${index + 1} ${role === 'normal' ? 'Normalzustand' : 'Anomaliephase'} Beginn (einschließlich)`} value={pair[role].start} disabled={busy}
            min={dataset?.start_timestamp ?? undefined} max={dataset?.end_timestamp ?? undefined} onChange={start => updatePeriod(index, role, { start })} />
          <DateTime24Input label={`u${index + 1} ${role === 'normal' ? 'Normalzustand' : 'Anomaliephase'} Ende (einschließlich)`} value={pair[role].end} disabled={busy}
            min={dataset?.start_timestamp ?? undefined} max={dataset?.end_timestamp ?? undefined} onChange={end => updatePeriod(index, role, { end })} />
        </Stack>)}</SimpleGrid>
      </Stack></Paper>)}
      <Button variant="light" w="fit-content" disabled={busy || config.pairs.length >= 6} onClick={() => update({ pairs: [...config.pairs, emptyPair()] })}>Paar hinzufügen</Button>
      <Paper withBorder p="lg"><Stack>
        <Title order={4}>Darstellung und Berechnung</Title>
        <Text size="sm">Differenz = Varianz Anomaliephase − Varianz Normalzustand (Populationsvarianz). Blau bedeutet geringere, Rot höhere Varianz. Die Farbskalen gelten für alle Paare.</Text>
        <SimpleGrid cols={{ base: 1, md: 2 }}>{(['variance_scale', 'difference_scale'] as const).map(key => <Stack key={key} gap="xs">
          <Select label={key === 'variance_scale' ? 'Farbskala Varianz' : 'Farbskala Differenz'} value={config[key].mode} disabled={busy} allowDeselect={false}
            data={[{ value: 'auto', label: 'Automatisch' }, { value: 'manual', label: 'Manueller Grenzwert' }]}
            onChange={mode => update({ [key]: { ...config[key], mode: mode as 'auto' | 'manual' } })} />
          {config[key].mode === 'manual' && <NumberInput label={key === 'variance_scale' ? 'Obergrenze Varianz' : 'Grenze ± der Differenz'} description="gray value²; begrenzt nur die Farben."
            value={config[key].limit ?? ''} min={0} disabled={busy} onChange={value => update({ [key]: { ...config[key], limit: value === '' ? null : Number(value) } })} />}
        </Stack>)}</SimpleGrid>
        {validation && <Text size="sm" c="dimmed">{validation}</Text>}
        <Group><Button variant="light" disabled={busy || !!validation} onClick={() => void action(async () => setPreview({ signature, value: await previewMeanVariance(config, projectId) }))}>Auswahl prüfen</Button>
          <Button disabled={busy || !!validation || !currentPreview || !!currentPreview.errors.length} onClick={() => void action(async () => {
            const next = await createMeanVarianceRun(config, projectId); setRuns(current => [next, ...current]); openRun(next);
          })}>Berechnung starten</Button></Group>
        {currentPreview && <>
          <Table><Table.Thead><Table.Tr><Table.Th>Paar / Zeitraum</Table.Th><Table.Th>Verfügbar</Table.Th><Table.Th>Ausgewählt</Table.Th><Table.Th>Restblock</Table.Th></Table.Tr></Table.Thead>
            <Table.Tbody>{currentPreview.pairs.flatMap((pair, index) => (['reference', 'anomaly'] as const).map(role => <Table.Tr key={`${index}-${role}`}><Table.Td>u{index + 1} {role === 'reference' ? 'Normalzustand' : 'Anomaliephase'}</Table.Td><Table.Td>{pair[role].available}</Table.Td><Table.Td>{pair[role].selected}</Table.Td><Table.Td>{pair[role].remainder}</Table.Td></Table.Tr>))}</Table.Tbody></Table>
          {currentPreview.pairs.flatMap((pair, index) => (['reference', 'anomaly'] as const).filter(role => pair[role].selected === 1).map(role => <Alert key={`${index}-${role}`} color="yellow">u{index + 1} {role === 'reference' ? 'Normalzustand' : 'Anomaliephase'}: Nur ein Bild; Varianz 0, keine zeitliche Vergleichsbasis.</Alert>))}
          {currentPreview.errors.map((message, i) => <Alert key={i} color="orange">{message}</Alert>)}
        </>}
      </Stack></Paper>
    </>}
    {results && run?.status === 'finished' && <Paper withBorder p="lg"><Stack>
      <Title order={4}>Ergebnis · Lauf #{run.id}</Title>
      <Text size="sm">{results.width} × {results.height} Pixel · Populationsvarianz · {run.training_dataset_name} · {run.pipeline_snapshot.name}</Text>
      {results.warnings.map((message, i) => <Alert color="yellow" key={i}>{message}</Alert>)}
      {isPairResults(results) ? <>
        {results.pairs.map(pair => <Text key={pair.label} size="sm">{pair.label}: {pair.counts.normal} Normalbilder ({periodText(pair.periods.normal)}) · {pair.counts.anomaly} Anomaliebilder ({periodText(pair.periods.anomaly)})</Text>)}
        <Image src={meanVarianceArtifactUrl(run.id, results.filename, projectId)} fit="contain" alt="Varianzvergleich: Normalzustand, Unruhe und Differenz für alle Paare" />
        <Button component="a" w="fit-content" href={meanVarianceArtifactUrl(run.id, results.filename, projectId, true)}>Vergleich als PNG herunterladen</Button>
      </> : <>
        <Text size="sm">{results.reference_count} Normalbilder · {results.anomaly_count} Anomaliebilder</Text>
        <Image src={meanVarianceArtifactUrl(run.id, results.maps.variance.filename, projectId)} fit="contain" alt="Gespeicherte Varianzdifferenz des bisherigen Einzelvergleichs" />
        <Button component="a" w="fit-content" href={meanVarianceArtifactUrl(run.id, results.maps.variance.filename, projectId, true)}>Varianzdifferenz als PNG herunterladen</Button>
      </>}
    </Stack></Paper>}
  </Stack>;
}
