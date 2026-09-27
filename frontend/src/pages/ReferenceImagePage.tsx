import { Alert, Badge, Button, Group, Image, Loader, NumberInput, Paper, Progress, Select, SimpleGrid, Stack, Table, Text, Title } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { DateTime24Input } from '../components/DateTime24Input';
import {
  abortReferenceImageRun, createReferenceImageRun, deleteReferenceImageRun, getReferenceImageLog,
  getReferenceImageResults, getReferenceImageRun, listPreprocessingPipelines, listReferenceImageRuns,
  listTrainingDatasets, lookupReferenceImageFrame, previewReferenceImage, referenceImageArtifactUrl,
} from '../api';
import { displayTime, frameFilename, initialConfig, PHASES, validateConfig } from '../referenceImage/helpers';
import type { PreprocessingPipeline, ReferenceImageConfig, ReferenceImageLookup, ReferenceImagePreview, ReferenceImageResults, ReferenceImageRun, TrainingDataset } from '../types';

const activeRun = (run: ReferenceImageRun | null) => !!run && ['queued', 'running'].includes(run.status);
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);

export function ReferenceImagePage({ active }: { active: boolean }) {
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [config, setConfig] = useState<ReferenceImageConfig>(initialConfig);
  const [preview, setPreview] = useState<{ signature: string; value: ReferenceImagePreview } | null>(null);
  const [runs, setRuns] = useState<ReferenceImageRun[]>([]);
  const [run, setRun] = useState<ReferenceImageRun | null>(null);
  const [results, setResults] = useState<ReferenceImageResults | null>(null);
  const [lookup, setLookup] = useState<ReferenceImageLookup | null>(null);
  const [timestamp, setTimestamp] = useState('');
  const [log, setLog] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const signature = JSON.stringify(config);
  const currentPreview = preview?.signature === signature ? preview.value : null;
  const validation = validateConfig(config);
  const dataset = datasets.find(item => item.id === config.training_dataset_id);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    setLoading(true);
    Promise.all([listTrainingDatasets(), listPreprocessingPipelines(), listReferenceImageRuns()])
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
        const next = await getReferenceImageRun(run.id);
        if (!cancelled) { setPollError(null); setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item)); }
      } catch (reason) { if (!cancelled) setPollError(errorText(reason)); }
      finally { pending = false; }
    }, 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active, run?.id, run?.status]);

  useEffect(() => {
    setResults(null); setLookup(null); setLog(null);
    if (!active || run?.status !== 'finished') return;
    let cancelled = false;
    getReferenceImageResults(run.id).then(next => {
      if (cancelled) return;
      setResults(next);
      const first = next.frames[0];
      if (first) { setTimestamp(first.timestamp); setLookup({ requested_timestamp: first.timestamp, exact: true, frame: first }); }
    }).catch(reason => { if (!cancelled) setError(errorText(reason)); });
    return () => { cancelled = true; };
  }, [active, run?.id, run?.status]);

  async function action(work: () => Promise<void>) {
    setBusy(true); setError(null);
    try { await work(); } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  }
  function update(values: Partial<ReferenceImageConfig>) { setConfig(current => ({ ...current, ...values })); }
  function useTemplate() {
    if (!run) return;
    setConfig({ ...initialConfig, ...structuredClone(run.config), processing_mode: 'shift_clip', scale_mode: 'auto', scale_limit: null }); setPreview(null);
  }

  return <Stack gap="lg">
    <div><Title order={2}>Referenzbild-Analyse</Title><Text c="dimmed">Veränderungen gegenüber einem gemeinsamen Normalbild als Graustufenvideo.</Text></div>
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
        <Title order={4}>{role === 'reference' ? '2 · Referenzzeitraum' : '3 · Anomaliezeitraum'}</Title>
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
      <Select label="Bildverarbeitung" value={config.processing_mode} allowDeselect={false} disabled={busy}
        data={[{ value: 'shift_clip', label: 'Shift und Clipping · 16-Bit-PNG' }, { value: 'signed', label: 'Bisherige Vorzeichen-Darstellung · 8-Bit-PNG' }]}
        onChange={value => update({ processing_mode: value as 'shift_clip' | 'signed', scale_mode: 'auto', scale_limit: null })} />
      {config.processing_mode === 'shift_clip' ? <>
        <Text size="sm">Ergebnis = clip(Bild − Referenzbild + Shift, Minimum, Maximum). Der PNG-Download speichert diese Werte als 16-Bit-Graustufenbild.</Text>
        <SimpleGrid cols={{ base: 1, md: 3 }}>
          <NumberInput label="Shift" value={Number.isFinite(config.shift) ? config.shift : ''} disabled={busy} onChange={value => update({ shift: value === '' ? NaN : Number(value) })} />
          <NumberInput label="Clip-Minimum" min={0} max={65535} allowDecimal={false} value={Number.isFinite(config.clip_min) ? config.clip_min : ''} disabled={busy} onChange={value => update({ clip_min: value === '' ? NaN : Number(value) })} />
          <NumberInput label="Clip-Maximum" min={0} max={65535} allowDecimal={false} value={Number.isFinite(config.clip_max) ? config.clip_max : ''} disabled={busy} onChange={value => update({ clip_max: value === '' ? NaN : Number(value) })} />
        </SimpleGrid>
        <Text size="sm" c="dimmed">Vorschau und MP4 bilden den Clip-Bereich auf Schwarz bis Weiß ab (8 Bit). Unveränderte Pixel haben vor Clipping den Wert des Shifts; der Zeitstempel wird oben rechts eingeblendet.</Text>
      </> : <Text size="sm">Grau = unverändert · Weiß = heller · Schwarz = dunkler als die Referenz. Eine gemeinsame Skala gilt für das gesamte Video.</Text>}
      <SimpleGrid cols={{ base: 1, md: 3 }}>
        {config.processing_mode === 'signed' && <Select label="Kontrast" value={config.scale_mode} allowDeselect={false} disabled={busy}
          data={[{ value: 'auto', label: 'Automatisch für den gesamten Lauf' }, { value: 'manual', label: 'Manueller Grenzwert' }]}
          onChange={value => update({ scale_mode: value as 'auto' | 'manual', scale_limit: value === 'auto' ? null : config.scale_limit })} />}
        {config.processing_mode === 'signed' && config.scale_mode === 'manual' && <NumberInput label="Differenz für Schwarz / Weiß" description="In Einheiten der Pipeline-Ausgabe; größere Werte werden abgeschnitten."
          min={0} value={config.scale_limit ?? ''} disabled={busy} onChange={value => update({ scale_limit: value === '' ? null : Number(value) })} />}
        <NumberInput label="Videogeschwindigkeit (FPS)" min={1} max={120} allowDecimal={false} value={config.fps} disabled={busy} onChange={value => update({ fps: Number(value) })} />
      </SimpleGrid>
      {validation && <Text size="sm" c="dimmed">{validation}</Text>}
      <Group>
        <Button variant="light" disabled={!!validation || busy} onClick={() => void action(async () => setPreview({ signature, value: await previewReferenceImage(config) }))}>Auswahl prüfen</Button>
        <Button disabled={busy || !!validation || !currentPreview || !!currentPreview.errors.length} onClick={() => void action(async () => {
          const next = await createReferenceImageRun(config); setRun(next); setRuns(current => [next, ...current]);
        })}>Berechnung starten</Button>
      </Group>
      {currentPreview && <>
        <Table><Table.Thead><Table.Tr><Table.Th>Zeitraum</Table.Th><Table.Th>Verfügbare Bilder</Table.Th><Table.Th>Ausgewählt</Table.Th><Table.Th>Restblock</Table.Th></Table.Tr></Table.Thead>
          <Table.Tbody>{(['reference', 'anomaly'] as const).map(role => <Table.Tr key={role}><Table.Td>{role === 'reference' ? 'Referenz' : 'Anomalie'}</Table.Td><Table.Td>{currentPreview[role].available}</Table.Td><Table.Td>{currentPreview[role].selected}</Table.Td><Table.Td>{currentPreview[role].remainder}</Table.Td></Table.Tr>)}</Table.Tbody></Table>
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
          <Button variant="subtle" disabled={busy} onClick={() => void action(async () => setLog((await getReferenceImageLog(run.id)).log))}>Log anzeigen</Button>
          {activeRun(run) ? <Button color="orange" disabled={busy || run.cancel_requested} onClick={() => void action(async () => {
            const next = await abortReferenceImageRun(run.id); setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item));
          })}>{run.cancel_requested ? 'Abbruch angefordert' : 'Abbrechen'}</Button>
            : <Button color="red" variant="subtle" disabled={busy} onClick={() => void action(async () => {
              await deleteReferenceImageRun(run.id); setRuns(current => current.filter(item => item.id !== run.id)); setRun(null);
            })}>Lauf löschen</Button>}
        </Group>
        {activeRun(run) && <><Text size="sm">{PHASES[run.current_step] ?? run.current_step} · {run.processed_images}/{run.total_images ?? '…'}</Text>
          <Progress value={run.total_images ? 100 * run.processed_images / run.total_images : 0} animated /></>}
        {pollError && <Alert color="orange">Status konnte nicht aktualisiert werden: {pollError}</Alert>}
        {run.error_message && <Alert color="red">{run.error_message}</Alert>}
        {log !== null && <Paper p="sm" withBorder><pre style={{ whiteSpace: 'pre-wrap', maxHeight: 240, overflow: 'auto' }}>{log || 'Noch keine Log-Einträge.'}</pre></Paper>}
        {results && run.status === 'finished' && <>
          <Text size="sm">{results.summary.reference_count} Referenzbilder · {results.summary.frame_count} Frames · {results.summary.fps} FPS · {results.summary.output_bit_depth === 16 ? `16 Bit · Shift ${results.summary.shift} · Clip ${results.summary.clip_min} bis ${results.summary.clip_max}` : `Kontrastgrenze ±${results.summary.scale_limit.toPrecision(5)}`}</Text>
          <video key={run.id} ref={video} controls preload="metadata" src={referenceImageArtifactUrl(run.id, 'video.mp4')} style={{ width: '100%', maxHeight: 600 }} />
          <Group><Button component="a" href={referenceImageArtifactUrl(run.id, 'video.mp4', true)}>MP4 herunterladen</Button></Group>
          <Title order={4}>Einzelbild nach Aufnahmezeitpunkt</Title>
          <Text size="sm" c="dimmed">Ohne exakten Treffer erscheint der nächste spätere Frame, hinter dem Videoende der letzte Frame.</Text>
          <DateTime24Input label="Gewünschter Aufnahmezeitpunkt" value={timestamp} disabled={busy} onChange={setTimestamp} />
          <Button w="fit-content" disabled={busy || !timestamp} onClick={() => void action(async () => {
            const next = await lookupReferenceImageFrame(run.id, timestamp); setLookup(next);
            if (video.current) video.current.currentTime = next.frame.index / results.summary.fps;
          })}>Bild anzeigen</Button>
          {lookup && <>
            {!lookup.exact && <Alert color="blue">Für {displayTime(lookup.requested_timestamp)} liegt kein Frame vor. Angezeigt wird {displayTime(lookup.frame.timestamp)}.</Alert>}
            <Text>Aufnahmezeitpunkt: {displayTime(lookup.frame.timestamp)} · Mittlere absolute Differenz: {lookup.frame.distance.toPrecision(6)}</Text>
            <Image src={referenceImageArtifactUrl(run.id, `${results.summary.output_bit_depth === 16 ? 'preview_' : ''}${frameFilename(lookup.frame.index)}`)} fit="contain" mah={600} alt={`Differenzbild ${displayTime(lookup.frame.timestamp)}`} />
            <Button component="a" w="fit-content" href={referenceImageArtifactUrl(run.id, frameFilename(lookup.frame.index), true)}>{results.summary.output_bit_depth === 16 ? 'Einzelbild als 16-Bit-PNG herunterladen' : 'Einzelbild als PNG herunterladen'}</Button>
          </>}
          <Title order={4}>Referenzbild</Title>
          <Image src={referenceImageArtifactUrl(run.id, 'reference.png')} fit="contain" mah={360} alt="Gemitteltes Referenzbild" />
          <Button component="a" w="fit-content" variant="light" href={referenceImageArtifactUrl(run.id, 'reference.png', true)}>Referenzbild als PNG herunterladen</Button>
        </>}
      </>}
    </Stack></Paper>
  </Stack>;
}
