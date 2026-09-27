import { Alert, Badge, Button, Group, Image, Loader, NumberInput, Paper, Progress, Select, SimpleGrid, Stack, Table, Text, Title } from '@mantine/core';
import { useEffect, useState } from 'react';
import { TimeRangePresetPicker } from '../timeRangePresets/TimeRangePresetPicker';
import { copyTimeRange } from '../timeRangePresets/helpers';
import { DateTime24Input } from '../components/DateTime24Input';
import {
  abortMeanVarianceRun, createMeanVarianceRun, deleteMeanVarianceRun, getMeanVarianceLog,
  getMeanVarianceResults, getMeanVarianceRun, listPreprocessingPipelines, listMeanVarianceRuns,
  listTrainingDatasets, previewMeanVariance, meanVarianceArtifactUrl,
} from '../api';
import { displayTime, initialConfig, PHASES, validateConfig } from '../meanVariance/helpers';
import type { PreprocessingPipeline, TrainingDataset } from '../types';
import type { MeanVarianceConfig, MeanVariancePreview, MeanVarianceResults, MeanVarianceRun } from '../meanVariance/types';

const activeRun = (run: MeanVarianceRun | null) => !!run && ['queued', 'running'].includes(run.status);
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);

export function MeanVariancePage({ active, projectId }: { active: boolean; projectId: string }) {
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [config, setConfig] = useState<MeanVarianceConfig>(initialConfig);
  const [preview, setPreview] = useState<{ signature: string; value: MeanVariancePreview } | null>(null);
  const [runs, setRuns] = useState<MeanVarianceRun[]>([]);
  const [run, setRun] = useState<MeanVarianceRun | null>(null);
  const [results, setResults] = useState<MeanVarianceResults | null>(null);
  const [log, setLog] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const signature = JSON.stringify(config);
  const currentPreview = preview?.signature === signature ? preview.value : null;
  const validation = validateConfig(config);
  const dataset = datasets.find(item => item.id === config.training_dataset_id);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    setLoading(true);
    Promise.all([listTrainingDatasets(), listPreprocessingPipelines(), listMeanVarianceRuns(projectId)])
      .then(([nextDatasets, nextPipelines, nextRuns]) => {
        if (cancelled) return;
        setDatasets(nextDatasets); setPipelines(nextPipelines); setRuns(nextRuns);
        setRun(current => nextRuns.find(item => item.id === current?.id) ?? nextRuns.find(item => activeRun(item)) ?? nextRuns[0] ?? null);
      }).catch(reason => { if (!cancelled) setError(errorText(reason)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [active, projectId]);

  useEffect(() => {
    if (!active || !run || !activeRun(run)) return;
    let cancelled = false;
    let pending = false;
    const timer = window.setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const next = await getMeanVarianceRun(run.id, projectId);
        if (!cancelled) { setPollError(null); setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item)); }
      } catch (reason) { if (!cancelled) setPollError(errorText(reason)); }
      finally { pending = false; }
    }, 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active, projectId, run?.id, run?.status]);

  useEffect(() => {
    setResults(null); setLog(null);
    if (!active || run?.status !== 'finished') return;
    let cancelled = false;
    getMeanVarianceResults(run.id, projectId).then(next => {
      if (cancelled) return;
      setResults(next);
    }).catch(reason => { if (!cancelled) setError(errorText(reason)); });
    return () => { cancelled = true; };
  }, [active, projectId, run?.id, run?.status]);

  async function action(work: () => Promise<void>) {
    setBusy(true); setError(null);
    try { await work(); } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  }
  function update(values: Partial<MeanVarianceConfig>) { setPreview(null); setConfig(current => ({ ...current, ...values })); }
  function useTemplate() {
    if (!run) return;
    setConfig(structuredClone(run.config)); setPreview(null);
  }

  return <Stack gap="lg">
    <div><Title order={2}>Mittelwert-/Varianzvergleich</Title><Text c="dimmed">Vergleich zweier Zeiträume anhand ihrer pixelweisen Mittelwerte und zeitlichen Varianzen.</Text></div>
    {error && <Alert color="red" withCloseButton onClose={() => setError(null)}>{error}</Alert>}
    {loading && <Loader size="sm" />}
    <Paper withBorder p="lg"><Stack>
      <Title order={4}>1 · Datensatz und Preprocessing</Title>
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        <Select label="Train/Test Dataset" searchable value={config.training_dataset_id ? String(config.training_dataset_id) : null} disabled={busy}
          data={datasets.map(item => ({ value: String(item.id), label: item.name, disabled: item.invalid_rule_count > 0 }))}
          onChange={value => {
            const selected = datasets.find(item => item.id === Number(value));
            const range = { start: selected?.start_timestamp ?? '', end: selected?.end_timestamp ?? '' };
            update({ training_dataset_id: Number(value), reference: { ...config.reference, ...range }, anomaly: { ...config.anomaly, ...range } });
          }} />
        <Select label="Preprocessing-Pipeline" searchable value={config.preprocessing_pipeline_id ? String(config.preprocessing_pipeline_id) : null} disabled={busy}
          data={pipelines.map(item => ({ value: String(item.id), label: item.name }))} onChange={value => update({ preprocessing_pipeline_id: Number(value) })}
          description="Für beide Zeiträume identisch; Ausgabe muss ein Graustufenbild sein." />
      </SimpleGrid>
      {dataset && <Text size="sm">Verfügbarer Zeitraum: {displayTime(dataset.start_timestamp ?? '—')} bis {displayTime(dataset.end_timestamp ?? '—')}</Text>}
      <Text size="sm" c="dimmed">Das gespeicherte Datensatzsampling gilt zuerst. Zeiträume sind einschließlich Beginn und Ende; Lücken werden nicht aufgefüllt.</Text>
    </Stack></Paper>
    <SimpleGrid cols={{ base: 1, lg: 2 }}>
      {(['reference', 'anomaly'] as const).map(role => <Paper key={role} withBorder p="lg"><Stack>
        <Title order={4}>{role === 'reference' ? '2 · Normalphase' : '3 · Anomaliezeitraum'}</Title>
        <TimeRangePresetPicker projectId={projectId} active={active} value={config[role]} disabled={busy}
          min={dataset?.start_timestamp ?? undefined} max={dataset?.end_timestamp ?? undefined}
          applyDisabledReason={!dataset ? 'Bitte zuerst einen Datensatz auswählen.' : undefined}
          onApply={range => {
            setPreview(null);
            setConfig(current => ({ ...current, [role]: copyTimeRange(current[role], range) }));
          }} />
        <DateTime24Input label="Beginn (einschließlich)" value={config[role].start} disabled={busy} min={dataset?.start_timestamp ?? undefined} max={dataset?.end_timestamp ?? undefined}
          onChange={start => update({ [role]: { ...config[role], start } })} />
        <DateTime24Input label="Ende (einschließlich)" value={config[role].end} disabled={busy} min={dataset?.start_timestamp ?? undefined} max={dataset?.end_timestamp ?? undefined}
          onChange={end => update({ [role]: { ...config[role], end } })} />
        {role === 'reference' && <Select label="Auswahlverfahren" value={config.reference.mode} allowDeselect={false} disabled={busy}
          data={[{ value: 'regular', label: 'Jedes n-te Bild' }, { value: 'random', label: 'Zufällige Anzahl' }]}
          onChange={mode => update({ reference: { ...config.reference, mode: mode as 'regular' | 'random' } })} />}
        {role === 'reference' && config.reference.mode === 'random' ? <SimpleGrid cols={2}>
          <NumberInput label="Anzahl Bilder" min={1} allowDecimal={false} value={config.reference.count} disabled={busy}
            onChange={value => update({ reference: { ...config.reference, count: Number(value) } })} />
          <NumberInput label="Seed" min={0} max={4294967295} allowDecimal={false} value={config.reference.seed} disabled={busy}
            onChange={value => update({ reference: { ...config.reference, seed: Number(value) } })} />
        </SimpleGrid> : <NumberInput label="Sampling: jedes n-te Bild" min={1} allowDecimal={false} value={config[role].sampling_rate} disabled={busy}
          description="15 wählt Bild 15, 30, 45 …; unvollständige Restblöcke entfallen."
          onChange={value => update({ [role]: { ...config[role], sampling_rate: Number(value) } })} />}
      </Stack></Paper>)}
    </SimpleGrid>
    <Paper withBorder p="lg"><Stack>
      <Title order={4}>4 · Darstellung und Berechnung</Title>
      <Text size="sm">Mittelwertunterschied = |Mittelwert Normalphase − Mittelwert Anomaliephase|. Varianzdifferenz = Varianz Anomaliephase − Varianz Normalphase (Division durch die Bildanzahl).</Text>
      <Text size="sm" c="dimmed">Blau = geringere Varianz · Weiß = unverändert · Rot = höhere Varianz in der Anomaliephase.</Text>
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        {(['mean_scale', 'variance_scale'] as const).map(key => <Stack key={key} gap="xs">
          <Select label={key === 'mean_scale' ? 'Farbskala Mittelwertunterschied' : 'Farbskala Varianzdifferenz'}
            value={config[key].mode} disabled={busy} allowDeselect={false}
            data={[{ value: 'auto', label: 'Automatisch' }, { value: 'manual', label: 'Manueller Grenzwert' }]}
            onChange={mode => update({ [key]: { ...config[key], mode: mode as 'auto' | 'manual' } })} />
          {config[key].mode === 'manual' && <NumberInput label={key === 'mean_scale' ? 'Obergrenze Mittelwertunterschied' : 'Grenze ± für die Varianzdifferenz'}
            description={key === 'mean_scale' ? 'Pipeline-Einheiten; begrenzt nur die Farbdarstellung.' : 'Pipeline-Einheiten²; begrenzt nur die Farbdarstellung.'}
            value={config[key].limit ?? ''} min={0} disabled={busy}
            onChange={limit => update({ [key]: { ...config[key], limit: limit === '' ? null : Number(limit) } })} />}
        </Stack>)}
      </SimpleGrid>
      {validation && <Text size="sm" c="dimmed">{validation}</Text>}
      <Group>
        <Button variant="light" disabled={!!validation || busy} onClick={() => void action(async () => setPreview({ signature, value: await previewMeanVariance(config, projectId) }))}>Auswahl prüfen</Button>
        <Button disabled={busy || !!validation || !currentPreview || !!currentPreview.errors.length} onClick={() => void action(async () => {
          const next = await createMeanVarianceRun(config, projectId); setRun(next); setRuns(current => [next, ...current]);
        })}>Berechnung starten</Button>
      </Group>
      {currentPreview && <>
        <Table><Table.Thead><Table.Tr><Table.Th>Zeitraum</Table.Th><Table.Th>Verfügbare Bilder</Table.Th><Table.Th>Ausgewählt</Table.Th><Table.Th>Restblock</Table.Th></Table.Tr></Table.Thead>
          <Table.Tbody>{(['reference', 'anomaly'] as const).map(role => <Table.Tr key={role}><Table.Td>{role === 'reference' ? 'Normalphase' : 'Anomaliephase'}</Table.Td><Table.Td>{currentPreview[role].available}</Table.Td><Table.Td>{currentPreview[role].selected}</Table.Td><Table.Td>{currentPreview[role].remainder}</Table.Td></Table.Tr>)}</Table.Tbody></Table>
        {(['reference', 'anomaly'] as const).filter(role => currentPreview[role].selected === 1).map(role => <Alert key={role} color="yellow">
          {role === 'reference' ? 'Normalphase' : 'Anomaliephase'}: Nur ein Bild; Varianz 0, keine zeitliche Vergleichsbasis.
        </Alert>)}
        {currentPreview.errors.map(message => <Alert color="orange" key={message}>{message}</Alert>)}
      </>}
    </Stack></Paper>
    <Paper withBorder p="lg"><Stack>
      <Title order={4}>Gespeicherte Analysen</Title>
      <Select label="Lauf öffnen" searchable disabled={busy} value={run ? String(run.id) : null} allowDeselect={false}
        data={runs.map(item => ({ value: String(item.id), label: `#${item.id} · ${item.training_dataset_name} · ${PHASES[item.status] ?? item.status}` }))}
        onChange={value => { setRun(runs.find(item => String(item.id) === value) ?? null); setPollError(null); }} />
      {run && <>
        <Group><Badge>{PHASES[run.status] ?? run.status}</Badge><Text>{run.training_dataset_name} · {run.pipeline_snapshot.name}</Text></Group>
        <Group>
          <Button variant="light" disabled={busy} onClick={useTemplate}>Als Vorlage übernehmen</Button>
          <Button variant="subtle" disabled={busy} onClick={() => void action(async () => setLog((await getMeanVarianceLog(run.id, projectId)).log))}>Log anzeigen</Button>
          {activeRun(run) ? <Button color="orange" disabled={busy || run.cancel_requested} onClick={() => void action(async () => {
            const next = await abortMeanVarianceRun(run.id, projectId); setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item));
          })}>{run.cancel_requested ? 'Abbruch angefordert' : 'Abbrechen'}</Button>
            : <Button color="red" variant="subtle" disabled={busy} onClick={() => void action(async () => {
              await deleteMeanVarianceRun(run.id, projectId); setRuns(current => current.filter(item => item.id !== run.id)); setRun(null);
            })}>Lauf löschen</Button>}
        </Group>
        {activeRun(run) && <><Text size="sm">{PHASES[run.current_step] ?? run.current_step} · {run.processed_images}/{run.total_images ?? '…'}</Text>
          <Progress value={run.total_images ? 100 * run.processed_images / run.total_images : 0} animated /></>}
        {pollError && <Alert color="orange">Status konnte nicht aktualisiert werden: {pollError}</Alert>}
        {run.error_message && <Alert color="red">{run.error_message}</Alert>}
        {log !== null && <Paper p="sm" withBorder><pre style={{ whiteSpace: 'pre-wrap', maxHeight: 240, overflow: 'auto' }}>{log || 'Noch keine Log-Einträge.'}</pre></Paper>}
        {results && run.status === 'finished' && <>
          <Text size="sm">{results.reference_count} Normalbilder · {results.anomaly_count} Anomaliebilder · {results.width} × {results.height} Pixel · Populationsvarianz</Text>
          <Text size="sm">Normalphase: {displayTime(run.config.reference.start)} – {displayTime(run.config.reference.end)}</Text>
          <Text size="sm">Anomaliephase: {displayTime(run.config.anomaly.start)} – {displayTime(run.config.anomaly.end)}</Text>
          {results.warnings.map(message => <Alert color="yellow" key={message}>{message}</Alert>)}
          <SimpleGrid cols={{ base: 1, xl: 2 }}>
            {(['mean', 'variance'] as const).map(key => {
              const map = results.maps[key];
              return <Stack key={key}>
                <Title order={4}>{map.title}</Title>
                {map.all_zero && <Alert color="blue">Keine Unterschiede: alle Pixelwerte sind 0.</Alert>}
                <Image src={meanVarianceArtifactUrl(run.id, map.filename, projectId)} fit="contain" alt={map.title} />
                <Text size="sm">Werte: {map.minimum.toPrecision(6)} bis {map.maximum.toPrecision(6)} {map.unit}</Text>
                <Button component="a" w="fit-content" href={meanVarianceArtifactUrl(run.id, map.filename, projectId, true)}>{key === 'mean' ? 'Mittelwert-Heatmap als PNG herunterladen' : 'Varianz-Heatmap als PNG herunterladen'}</Button>
              </Stack>;
            })}
          </SimpleGrid>
        </>}
      </>}
    </Stack></Paper>
  </Stack>;
}
