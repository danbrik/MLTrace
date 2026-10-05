import { Alert, Badge, Button, Group, Loader, NumberInput, Paper, Progress, Select, SimpleGrid, Stack, Table, Text, Title } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { DateTime24Input } from '../components/DateTime24Input';
import { TimeRangePresetPicker } from '../timeRangePresets/TimeRangePresetPicker';
import { abortTemporalDifferenceRun, createTemporalDifferenceRun, deleteTemporalDifferenceRun, getTemporalDifferenceLog, getTemporalDifferenceRun,
  listPreprocessingPipelines, listTemporalDifferenceRuns, listTrainingDatasets, previewTemporalDifference } from '../api';
import type { PreprocessingPipeline, TrainingDataset } from '../types';
import type { Config, Preview, Run } from '../temporalDifference/types';
import { addDelta, defaultConfig, displayTime, labels, phaseLabel, roles, validDelta, validateConfig } from '../temporalDifference/helpers';
import { MatrixResults } from '../temporalDifference/MatrixResults';
import { TemporalResults } from '../temporalDifference/Results';
const activeRun = (run: Run | null) => !!run && ['queued', 'running'].includes(run.status);
const errorText = (value: unknown) => value instanceof Error ? value.message : String(value);

export function TemporalDifferencePage({ active, projectId }: { active: boolean; projectId: string }) {
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]), [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [config, setConfig] = useState<Config>(() => structuredClone(defaultConfig));
  const [preview, setPreview] = useState<{ signature: string; value: Preview } | null>(null);
  const [runs, setRuns] = useState<Run[]>([]), [run, setRun] = useState<Run | null>(null);
  const [templateNotice, setTemplateNotice] = useState(false);
  const [delta, setDelta] = useState<number | string>('');
  const [error, setError] = useState<string | null>(null), [pollError, setPollError] = useState<string | null>(null);
  const [log, setLog] = useState<string | null>(null);
  const [busy, setBusy] = useState(false), [loading, setLoading] = useState(false);
  const initialized = useRef(false), revision = useRef(0);
  const signature = JSON.stringify(config);
  const currentPreview = preview?.signature === signature ? preview.value : null;
  const dataset = datasets.find(item => item.id === config.training_dataset_id);
  const validation = validateConfig(config, dataset?.start_timestamp ?? undefined, dataset?.end_timestamp ?? undefined)
    ?? (!dataset || dataset.invalid_rule_count > 0 ? 'Bitte einen verfügbaren Datensatz auswählen.' : null)
    ?? (!pipelines.some(item => item.id === config.preprocessing_pipeline_id) ? 'Bitte eine verfügbare Preprocessing-Pipeline auswählen.' : null);
  function openRun(value: Run | null) {
    revision.current++; setRun(value); setPreview(null); setError(null); setPollError(null); setLog(null);
  }
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    setLoading(true);
    Promise.all([listTrainingDatasets(), listPreprocessingPipelines(), listTemporalDifferenceRuns(projectId)])
      .then(([nextDatasets, nextPipelines, nextRuns]) => {
        if (cancelled) return;
        setDatasets(nextDatasets); setPipelines(nextPipelines); setRuns(nextRuns);
        if (!initialized.current) { initialized.current = true; openRun(nextRuns.find(activeRun) ?? nextRuns[0] ?? null); }
      }).catch(reason => { if (!cancelled) setError(errorText(reason)); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [active, projectId]);
  useEffect(() => {
    if (!active || !run || !activeRun(run)) return;
    let cancelled = false, pending = false;
    const view = revision.current;
    const timer = window.setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const next = await getTemporalDifferenceRun(run.id, projectId);
        if (!cancelled && view === revision.current) { setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item)); setPollError(null); }
      } catch (reason) { if (!cancelled && view === revision.current) setPollError(errorText(reason)); }
      finally { pending = false; }
    }, 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active, projectId, run?.id, run?.status]);
  async function action(work: () => Promise<void>) {
    setBusy(true); setError(null);
    try { await work(); } catch (reason) { setError(errorText(reason)); }
    finally { setBusy(false); }
  }
  function update(values: Partial<Config>) { setPreview(null); setConfig(current => ({ ...current, ...values })); }
  function draft(template?: Run) {
    initialized.current = true;
    setConfig({ ...structuredClone(template?.config ?? defaultConfig), selection_version: 2, block_seconds: template?.config.block_seconds ?? 300, seed: template?.config.seed ?? 42 });
    if (template && template.config.selection_version !== 2) setTemplateNotice(true); else setTemplateNotice(false); setDelta(''); openRun(null);
  }
  return <Stack gap="lg">
    <Group justify="space-between"><div><Title order={2}>Zeitabstands-Analyse</Title><Text c="dimmed">Absolute Pixeländerung bei gemeinsamen, über den Zeitraum verteilten Startzeitpunkten.</Text></div>
      <Button variant="light" disabled={busy || loading} onClick={() => draft()}>Neue Analyse</Button></Group>
    {error && <Alert color="red" withCloseButton onClose={() => setError(null)}>{error}</Alert>}
    {loading && <Loader size="sm" />}
    <Select label="Lauf öffnen" placeholder="Gespeicherten Lauf auswählen" searchable allowDeselect={false} disabled={busy || loading}
      value={run ? String(run.id) : null} data={runs.map(item => ({ value: String(item.id), label: `#${item.id} · ${item.training_dataset_name} · ${phaseLabel(item.status)}` }))}
      onChange={value => openRun(runs.find(item => String(item.id) === value) ?? null)} />
    {run ? <Paper withBorder p="lg"><Stack>
      <Group><Title order={4}>Gespeicherte Einstellungen · Lauf #{run.id}</Title><Badge>{phaseLabel(run.status)}</Badge></Group>
      <Text>Datensatz: {run.training_dataset_name} · Preprocessing: {run.pipeline_snapshot.name}</Text>
      {roles.map(role => <Text key={role}>{labels[role]}: {displayTime(run.config[role].start)} – {displayTime(run.config[role].end)} (Europe/Berlin)</Text>)}
      <Text>Zeitabstände: {run.config.deltas_seconds.join(', ')} Sekunden · gespeichertes Dataset-Sampling</Text>
      <Text>{run.config.selection_version === 2 ? `Geschichtete Auswahl · Blockgröße ${run.config.block_seconds} s · Seed ${run.config.seed} · Partner-Toleranz ±0,5 s` : 'Bisherige vollständige Auswahl mit exakten Partnerzeitpunkten'}</Text>
      {run.result?.selection && roles.map(role => { const p = run.result!.selection.periods[role]; return p.start_range_start ? <Text key={role}>{labels[role]}: Startbereich {displayTime(p.start_range_start)} – {p.start_range_end ? displayTime(p.start_range_end) : 'ungültig'} · {p.selected_start_count} Startpunkte</Text> : null; })}
      <Text size="sm" c="dimmed">Berechnungseinstellungen sind eingefroren. „Als Vorlage übernehmen“ erstellt einen neuen Entwurf.</Text>
      <Group><Button disabled={busy} onClick={() => draft(run)}>Als Vorlage übernehmen</Button>
        <Button variant="subtle" disabled={busy} onClick={() => void action(async () => {
          const view = revision.current, next = await getTemporalDifferenceLog(run.id, projectId);
          if (view === revision.current) setLog(next.log);
        })}>Log anzeigen</Button>
        {activeRun(run) ? <Button color="orange" disabled={busy || run.cancel_requested} onClick={() => void action(async () => {
          const next = await abortTemporalDifferenceRun(run.id, projectId); setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item));
        })}>{run.cancel_requested ? 'Abbruch angefordert' : 'Abbrechen'}</Button> : <Button color="red" variant="subtle" disabled={busy} onClick={() => void action(async () => {
          await deleteTemporalDifferenceRun(run.id, projectId); setRuns(current => current.filter(item => item.id !== run.id)); draft();
        })}>Lauf löschen</Button>}
      </Group>
      {activeRun(run) && <><Text size="sm">{phaseLabel(run.current_step)} · {run.processed_images}/{run.total_images ?? '…'} Paare</Text><Progress animated value={run.total_images ? 100 * run.processed_images / run.total_images : 0} /></>}
      {pollError && <Alert color="orange">{pollError}</Alert>}{run.error_message && <Alert color="red">{run.error_message}</Alert>}
      {log !== null && <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 250, overflow: 'auto' }}>{log || 'Noch keine Log-Einträge.'}</pre>}
    </Stack></Paper> : <>
      <Paper withBorder p="lg"><Stack>
        <Title order={4}>1 · Datensatz und Preprocessing</Title>
        <SimpleGrid cols={{ base: 1, md: 2 }}>
          <Select label="Train/Test Dataset" searchable disabled={busy} value={config.training_dataset_id ? String(config.training_dataset_id) : null}
            data={datasets.map(item => ({ value: String(item.id), label: item.name, disabled: item.invalid_rule_count > 0 }))} onChange={value => update({ training_dataset_id: Number(value) })} />
          <Select label="Preprocessing-Pipeline" searchable disabled={busy} value={config.preprocessing_pipeline_id ? String(config.preprocessing_pipeline_id) : null}
            data={pipelines.map(item => ({ value: String(item.id), label: item.name }))} onChange={value => update({ preprocessing_pipeline_id: Number(value) })} />
        </SimpleGrid>
        <Text size="sm" c="dimmed">Das gespeicherte Dataset-Sampling gilt zuerst. Verglichen werden alle Pixel der fertigen Graustufenbilder; ein ROI-Zuschnitt wird aus dem Preprocessing übernommen.</Text>
        {dataset && <Text size="sm">Datensatzgrenzen: {displayTime(dataset.start_timestamp ?? '—')} bis {displayTime(dataset.end_timestamp ?? '—')}</Text>}
      </Stack></Paper>
      <Paper withBorder p="lg"><Stack>
        <Title order={4}>2 · Zeiträume</Title>
        <SimpleGrid cols={{ base: 1, lg: 2 }}>{roles.map(role => <Stack key={role}>
          <Text fw={600}>{labels[role]}zeitraum</Text>
          <TimeRangePresetPicker projectId={projectId} active={active} disabled={busy} value={config[role]}
            min={dataset?.start_timestamp ?? undefined} max={dataset?.end_timestamp ?? undefined} onApply={value => update({ [role]: value })} />
          <DateTime24Input label="Beginn (einschließlich)" value={config[role].start} disabled={busy} onChange={value => update({ [role]: { ...config[role], start: value } })} />
          <DateTime24Input label="Ende (einschließlich)" value={config[role].end} disabled={busy} onChange={value => update({ [role]: { ...config[role], end: value } })} />
        </Stack>)}</SimpleGrid>
        <Text size="sm" c="dimmed">Aufnahmezeiten in Europe/Berlin. Beide Bilder eines Paares müssen im jeweiligen Zeitraum liegen. Die Referenzdauer ist frei wählbar.</Text>
      </Stack></Paper>
      <Paper withBorder p="lg"><Stack>
        <Title order={4}>3 · Zeitabstände und Berechnung</Title>
        {templateNotice && <Alert color="blue">Diese Vorlage verwendet jetzt die geschichtete Auswahl statt aller exakten Paare.</Alert>}
        <Group><NumberInput label="Blockgröße (Sekunden)" min={1} allowDecimal={false} value={config.block_seconds ?? 300} disabled={busy} onChange={value => update({ block_seconds: Number(value) })} />
          <NumberInput label="Zufallsseed" min={0} allowDecimal={false} value={config.seed ?? 42} disabled={busy} onChange={value => update({ seed: value === '' ? NaN : Number(value) })} /></Group>
        <Group align="end"><NumberInput label="Zeitabstand (Sekunden)" min={1} allowDecimal={false} disabled={busy} value={delta} onChange={setDelta} />
          <Button variant="light" disabled={busy || !validDelta(Number(delta)) || config.deltas_seconds.includes(Number(delta))} onClick={() => { update({ deltas_seconds: addDelta(config.deltas_seconds, Number(delta)) }); setDelta(''); }}>Hinzufügen</Button></Group>
        <Group>{config.deltas_seconds.map(value => <Button key={value} size="xs" variant="light" disabled={busy} aria-label={`${value} Sekunden entfernen`}
          onClick={() => update({ deltas_seconds: config.deltas_seconds.filter(item => item !== value) })}>{value} s ×</Button>)}</Group>
        <Text size="sm" c="dimmed">Pro Paar: Mittelwert der absoluten Pixeldifferenzen. Pro Zeitabstand: Median und IQR. Pro Block wird ein Startpunkt mit Partnern für alle Abstände gezogen (±0,5 s). Der Startbereich endet um den größten Abstand vor dem Zeitraumende.</Text>
        {validation && <Text size="sm" c="dimmed">{validation}</Text>}
        <Group><Button variant="light" disabled={busy || loading || !!validation} onClick={() => void action(async () => {
          const next = await previewTemporalDifference(config, projectId); setPreview({ signature, value: next });
        })}>Auswahl prüfen</Button>
          <Button disabled={busy || loading || !!validation || !currentPreview || !!currentPreview.errors.length} onClick={() => void action(async () => {
            const next = await createTemporalDifferenceRun(config, projectId); setRuns(current => [next, ...current]); openRun(next);
          })}>Berechnung starten</Button></Group>
        {currentPreview && <>
          <Text>{roles.map(role => `${labels[role]}: ${currentPreview.periods[role].image_count} Bilder`).join(' · ')}</Text>
          <Table><Table.Thead><Table.Tr>{['Zeitraum', 'Δt (s)', 'Paare', 'Fehlende Zielbilder'].map(label => <Table.Th key={label}>{label}</Table.Th>)}</Table.Tr></Table.Thead>
            <Table.Tbody>{roles.flatMap(role => currentPreview.periods[role].deltas.map(row => <Table.Tr key={`${role}-${row.delta_seconds}`}><Table.Td>{labels[role]}</Table.Td><Table.Td>{row.delta_seconds}</Table.Td><Table.Td>{row.pair_count || 'Keine Paare'}</Table.Td><Table.Td>{row.missing_targets}</Table.Td></Table.Tr>))}</Table.Tbody></Table>
          {roles.map(role => { const p = currentPreview.periods[role]; return <Text key={role} size="sm">{labels[role]}: Startbereich {displayTime(p.start_range_start ?? '—')} – {displayTime(p.start_range_end ?? '—')} · {p.block_count} Blöcke · {p.valid_candidates}/{p.candidate_count} gültige Kandidaten · {p.selected_start_count} ausgewählt · {p.empty_blocks} leere Blöcke</Text>; })}
          <Text size="sm" c="dimmed">Fehlende Partner werden je Abstand unter allen Kandidaten im eingeschränkten Startbereich gezählt. Ein fehlender Partner schließt den Startpunkt für alle Abstände aus.</Text>
          {currentPreview.errors.map(message => <Alert color="orange" key={message}>{message}</Alert>)}
        </>}
      </Stack></Paper>
    </>}
    {run?.status === 'finished' && <MatrixResults key={`matrix-${projectId}-${run.id}`} run={run} projectId={projectId} />}
    {run?.status === 'finished' && <TemporalResults key={run.id} run={run} projectId={projectId} onPlotSaved={settings => {
      setRun(current => current?.id === run.id ? { ...current, plot_settings: settings } : current);
      setRuns(current => current.map(item => item.id === run.id ? { ...item, plot_settings: settings } : item));
    }} />}
  </Stack>;
}
