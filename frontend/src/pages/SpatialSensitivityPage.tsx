import { Accordion, Tabs, ThemeIcon, ActionIcon, Alert, Badge, Button, Group, Image, Loader, MultiSelect, NumberInput, Paper, Progress, ScrollArea, Select, SimpleGrid, Stack, Table, Text, TextInput, Title } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Copy, Download, Plus, Play, Save, Square, Trash2, Scan, Settings2, Images, BarChart3 } from 'lucide-react';
import { PointerEvent, useEffect, useMemo, useRef, useState } from 'react';
import { abortSpatialSensitivityRun, createSpatialSensitivityConfiguration, createSpatialSensitivityRun, deleteSpatialSensitivityConfiguration, getSpatialSensitivityRun, listSpatialSensitivityConfigurations, listSpatialSensitivityRuns, listTrainingDatasets, previewSpatialSensitivity, previewSpatialSensitivityWarp, spatialSensitivityArtifactUrl, updateSpatialSensitivityConfiguration } from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PointPickerControl } from '../preprocessing/controls/PointPickerControl';
import type { PreprocessingPreviewImage, SpatialSensitivityAnalysisConfig, SpatialSensitivityConfiguration, SpatialSensitivityEvent, SpatialSensitivityPreview, SpatialSensitivityRun, SpatialSensitivityWarpConfig, SpatialSensitivityWarpPreview, TrainingDataset } from '../types';

type Point = { x: number; y: number };
type DraftEvent = { key: string; id: string; datasetId: string | null; normalStart: string; normalManual: boolean; start: string; end: string };
const initialPoints: Point[] = [{ x: 160, y: 120 }, { x: 1120, y: 120 }, { x: 1120, y: 840 }, { x: 160, y: 840 }];
const fmt = (value: unknown, digits = 3) => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('de-DE', { maximumFractionDigits: digits }) : '—';
const runLabels: Record<string, string> = { queued: 'Wartet auf Start', running: 'Wird berechnet', finished: 'Abgeschlossen', failed: 'Fehlgeschlagen', aborted: 'Abgebrochen', indexing_candidates: 'Bildauswahl vorbereiten', loading_normal_sample: 'Normalbilder laden', loading_event_sample: 'Ereignisbilder laden', calculating_normal_statistics: 'Normalzustand auswerten', calculating_event_statistics: 'Ereignis auswerten', aggregating: 'Ergebnisse zusammenführen' };
function canonical(value: unknown): string {
  return JSON.stringify(value, (_key, item) => item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.entries(item).sort(([a],[b]) => a.localeCompare(b))) : item);
}
const uid = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
function shiftLocal(value: string, milliseconds: number): string {
  if (!value) return '';
  const date = new Date(new Date(value).getTime() + milliseconds);
  const part = (number: number) => String(number).padStart(2, '0');
  return `${date.getFullYear()}-${part(date.getMonth()+1)}-${part(date.getDate())}T${part(date.getHours())}:${part(date.getMinutes())}:${part(date.getSeconds())}`;
}

function RoiPicker({ preview, points, onChange }: { preview: SpatialSensitivityPreview; points: Point[]; onChange: (value: Point[]) => void }) {
  const ref = useRef<HTMLDivElement | null>(null); const dragging = useRef<number | null>(null);
  const polygon = points.map((p) => `${p.x},${p.y}`).join(' ');
  function move(event: PointerEvent<HTMLDivElement>) {
    if (dragging.current === null || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(preview.width - 1, (event.clientX - rect.left) / rect.width * preview.width));
    const y = Math.max(0, Math.min(preview.height - 1, (event.clientY - rect.top) / rect.height * preview.height));
    onChange(points.map((p, index) => index === dragging.current ? { x, y } : p));
  }
  return <Stack gap="xs"><div ref={ref} onPointerMove={move} onPointerUp={() => dragging.current = null} onPointerLeave={() => dragging.current = null} style={{ position: 'relative', maxWidth: 900, touchAction: 'none', margin: '0 auto', width: '100%' }}>
    <img src={preview.image_data_url} alt="ROI source" style={{ width: '100%', display: 'block' }} />
    <svg viewBox={`0 0 ${preview.width} ${preview.height}`} preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
      <polygon points={polygon} fill="rgba(255,165,0,.08)" stroke="orange" strokeWidth="5" vectorEffect="non-scaling-stroke" />
    </svg>
    {points.map((point, index) => <button key={index} type="button" aria-label={`ROI-Eckpunkt ${index + 1} verschieben`} onPointerDown={(event) => { dragging.current = index; event.currentTarget.setPointerCapture?.(event.pointerId); }} style={{ position: 'absolute', left: `${point.x / preview.width * 100}%`, top: `${point.y / preview.height * 100}%`, transform: 'translate(-50%,-50%)', borderRadius: '50%', width: 28, height: 28, border: '2px solid white', background: 'orange', cursor: 'grab' }}>{index + 1}</button>)}
  </div><Text size="xs" c="dimmed">Eckpunkte ziehen oder Koordinaten präzise eingeben. X = horizontal, Y = vertikal.</Text><SimpleGrid cols={{base:1,sm:2,lg:4}}>{points.map((p, i) => <Paper key={i} withBorder p="xs" radius="md"><Text size="xs" fw={700} mb={4}>Punkt {i + 1}</Text><Group grow wrap="nowrap"><NumberInput label="X" aria-label={`Punkt ${i+1} X`} value={p.x} min={0} max={preview.width-1} decimalScale={2} onChange={value => { if(typeof value === 'number') onChange(points.map((point,index) => index===i ? {...point,x:value} : point)); }}/><NumberInput label="Y" aria-label={`Punkt ${i+1} Y`} value={p.y} min={0} max={preview.height-1} decimalScale={2} onChange={value => { if(typeof value === 'number') onChange(points.map((point,index) => index===i ? {...point,y:value} : point)); }}/></Group></Paper>)}</SimpleGrid></Stack>;
}

export function SpatialSensitivityPage({ active }: { active: boolean }) {
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]); const [runs, setRuns] = useState<SpatialSensitivityRun[]>([]);
  const [configurations, setConfigurations] = useState<SpatialSensitivityConfiguration[]>([]); const [configurationId, setConfigurationId] = useState<string | null>(null);
  const [configurationName, setConfigurationName] = useState(''); const [configurationDescription, setConfigurationDescription] = useState('');
  const [savedSnapshot, setSavedSnapshot] = useState('');
  const [view, setView] = useState<string | null>('setup');
  const [datasetIds, setDatasetIds] = useState<string[]>([]); const [sourceDataset, setSourceDataset] = useState<string | null>(null); const [sourceTime, setSourceTime] = useState('');
  const [preview, setPreview] = useState<SpatialSensitivityPreview | null>(null); const [points, setPoints] = useState<Point[]>(initialPoints);
  const [events, setEvents] = useState<DraftEvent[]>([]); const [hours, setHours] = useState<number | string>(24); const [epsilon, setEpsilon] = useState<number | string>(1);
  const [normalSampleSize, setNormalSampleSize] = useState<number | string>(1000); const [eventSampleSize, setEventSampleSize] = useState<number | string>(1000); const [samplingSeed, setSamplingSeed] = useState<number | string>(42);
  const [exampleEvent, setExampleEvent] = useState<string | null>(null); const [normalExampleTime, setNormalExampleTime] = useState(''); const [eventExampleTime, setEventExampleTime] = useState('');
  const [normalPreview, setNormalPreview] = useState<SpatialSensitivityPreview | null>(null); const [eventPreview, setEventPreview] = useState<SpatialSensitivityPreview | null>(null);
  const [warpConfig, setWarpConfig] = useState<SpatialSensitivityWarpConfig>({ source_points: initialPoints, output_shape_mode: 'preserve_rectangle', output_width: 128, output_height: 128, interpolation: 'linear' });
  const [normalWarpPreview, setNormalWarpPreview] = useState<SpatialSensitivityWarpPreview | null>(null); const [eventWarpPreview, setEventWarpPreview] = useState<SpatialSensitivityWarpPreview | null>(null);
  const [run, setRun] = useState<SpatialSensitivityRun | null>(null); const [busy, setBusy] = useState(false); const [error, setError] = useState<string | null>(null);
  useEffect(() => { if (!active) return; Promise.all([listTrainingDatasets(), listSpatialSensitivityRuns(), listSpatialSensitivityConfigurations()]).then(([d, r, c]) => { setDatasets(d); setRuns(r); setConfigurations(c); if (r[0]) setRun(r[0]); }).catch(e => setError(String(e))); }, [active]);
  useEffect(() => { if (!active || !run || !['queued','running'].includes(run.status)) return; const timer = window.setInterval(() => getSpatialSensitivityRun(run.id).then(async next => { setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item)); if (next.status === 'finished') setConfigurations(await listSpatialSensitivityConfigurations()); }), 1500); return () => window.clearInterval(timer); }, [active, run?.id, run?.status]);
  const chosenEvent = events.find(item => item.key === exampleEvent);
  const loadedConfiguration = configurations.find(item => String(item.id) === configurationId) ?? null;
  const analysisConfig = useMemo<SpatialSensitivityAnalysisConfig | null>(() => {
    if (!sourceDataset || !(preview?.source_timestamp ?? sourceTime)) return null;
    return {
      training_dataset_ids: datasetIds.map(Number),
      events: events.map(e => ({ id: e.id.trim(), training_dataset_id: Number(e.datasetId), normal_start: e.normalStart || null, start: e.start, end: e.end })),
      normal_window_hours: Number(hours), epsilon: Number(epsilon), roi_points: points,
      normal_sample_size: Number(normalSampleSize), event_sample_size: Number(eventSampleSize), sampling_seed: Number(samplingSeed), sampling_mode: 'deterministic_uniform',
      roi_source_dataset_id: Number(sourceDataset), roi_source_timestamp: preview?.source_timestamp ?? sourceTime,
      example_event_id: chosenEvent?.id ?? null,
      example_normal_timestamp: normalPreview?.source_timestamp ?? (normalExampleTime || null),
      example_event_timestamp: eventPreview?.source_timestamp ?? (eventExampleTime || null),
      warp_preview_config: warpConfig,
    };
  }, [datasetIds, events, hours, epsilon, normalSampleSize, eventSampleSize, samplingSeed, points, sourceDataset, sourceTime, preview?.source_timestamp, chosenEvent?.id, normalExampleTime, eventExampleTime, normalPreview?.source_timestamp, eventPreview?.source_timestamp, warpConfig]);
  const currentSnapshot = analysisConfig ? canonical({ name: configurationName.trim(), description: configurationDescription, config: analysisConfig }) : '';
  const dirty = Boolean(savedSnapshot && currentSnapshot !== savedSnapshot);
  const contentMatchesLoaded = Boolean(analysisConfig && loadedConfiguration && canonical(analysisConfig) === canonical(loadedConfiguration.config));
  const overlapWarnings = useMemo(() => events.flatMap(event => events.filter(other => other.key !== event.key && other.datasetId === event.datasetId && event.start && event.normalStart && other.start && other.end && new Date(other.end) >= new Date(event.normalStart) && new Date(other.start) < new Date(event.start)).map(other => `${event.id || 'Ereignis'}: Normalfenster überlappt ${other.id || 'Ereignis'}`)), [events]);
  function addEvent() { setEvents(current => [...current, { key: uid(), id: `U${current.length + 1}`, datasetId: datasetIds[0] ?? null, normalStart: '', normalManual: false, start: '', end: '' }]); }
  function updateEvent(key: string, change: Partial<DraftEvent>) { setEvents(current => current.map(item => {
    if (item.key !== key) return item;
    const next = { ...item, ...change };
    if (change.start !== undefined && !item.normalManual) next.normalStart = shiftLocal(change.start, -Number(hours) * 3600000);
    return next;
  })); }
  function changeHours(value: number | string) {
    setHours(value); const numeric = Number(value);
    if (numeric > 0) setEvents(current => current.map(item => item.normalManual || !item.start ? item : { ...item, normalStart: shiftLocal(item.start, -numeric * 3600000) }));
  }
  async function loadConfiguration(value: string | null) {
    setConfigurationId(value); const selected = configurations.find(item => String(item.id) === value); if (!selected) return;
    const config = selected.config; const drafts = config.events.map((event, index) => ({ key: `saved-${selected.id}-${index}`, id: event.id, datasetId: String(event.training_dataset_id), normalStart: event.normal_start ?? shiftLocal(event.start, -config.normal_window_hours * 3600000), normalManual: Boolean(event.normal_start), start: event.start, end: event.end }));
    setConfigurationName(selected.name); setConfigurationDescription(selected.description ?? ''); setDatasetIds(config.training_dataset_ids.map(String));
    setEvents(drafts); setHours(config.normal_window_hours); setEpsilon(config.epsilon); setNormalSampleSize(config.normal_sample_size ?? 1000); setEventSampleSize(config.event_sample_size ?? 1000); setSamplingSeed(config.sampling_seed ?? 42); setPoints(config.roi_points); setSourceDataset(String(config.roi_source_dataset_id)); setSourceTime(config.roi_source_timestamp);
    const exampleKey = drafts.find(item => item.id === config.example_event_id)?.key ?? null; setExampleEvent(exampleKey);
    setNormalExampleTime(config.example_normal_timestamp ?? ''); setEventExampleTime(config.example_event_timestamp ?? '');
    setWarpConfig(config.warp_preview_config ?? { source_points: initialPoints, output_shape_mode: 'preserve_rectangle', output_width: 128, output_height: 128, interpolation: 'linear' });
    setNormalWarpPreview(null); setEventWarpPreview(null);
    setPreview(null); setNormalPreview(null); setEventPreview(null); setError(null);
    setSavedSnapshot(canonical({ name: selected.name, description: selected.description ?? '', config }));
    setBusy(true);
    try {
      const sourcePromise = previewSpatialSensitivity({ training_dataset_id: config.roi_source_dataset_id, target_timestamp: config.roi_source_timestamp });
      const event = config.events.find(item => item.id === config.example_event_id);
      const examplePromises = event && config.example_normal_timestamp && config.example_event_timestamp ? [
        previewSpatialSensitivity({ training_dataset_id: event.training_dataset_id, target_timestamp: config.example_normal_timestamp, range_start: event.normal_start ?? shiftLocal(event.start, -config.normal_window_hours * 3600000), range_end: shiftLocal(event.start, -1) }),
        previewSpatialSensitivity({ training_dataset_id: event.training_dataset_id, target_timestamp: config.example_event_timestamp, range_start: event.start, range_end: event.end }),
      ] : [];
      const [source, ...examples] = await Promise.all([sourcePromise, ...examplePromises]); setPreview(source); setNormalPreview(examples[0] ?? null); setEventPreview(examples[1] ?? null);
      if (selected.latest_finished_run_id) {
        const finished = await getSpatialSensitivityRun(selected.latest_finished_run_id); setRun(finished); setView('results');
        notifications.show({ color: 'green', title: 'Gespeichertes Ergebnis geladen', message: `Lauf #${finished.id} gehört exakt zu dieser Konfiguration.` });
      } else setRun(null);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  function newConfiguration() { setConfigurationId(null); setConfigurationName(''); setConfigurationDescription(''); setSavedSnapshot(''); setRun(null); }
  async function saveConfiguration(asCopy = false) {
    if (!analysisConfig || !configurationName.trim()) return;
    setBusy(true); setError(null);
    try {
      let saved: SpatialSensitivityConfiguration;
      if (asCopy) {
        const copyName = window.prompt('Name der neuen Konfiguration', `${configurationName.trim()} Kopie`);
        if (!copyName?.trim()) return;
        saved = await createSpatialSensitivityConfiguration({ name: copyName.trim(), description: configurationDescription, config: analysisConfig });
      } else if (configurationId) saved = await updateSpatialSensitivityConfiguration(Number(configurationId), { name: configurationName.trim(), description: configurationDescription, config: analysisConfig });
      else saved = await createSpatialSensitivityConfiguration({ name: configurationName.trim(), description: configurationDescription, config: analysisConfig });
      const next = await listSpatialSensitivityConfigurations(); setConfigurations(next); setConfigurationId(String(saved.id)); setConfigurationName(saved.name); setConfigurationDescription(saved.description ?? '');
      setSavedSnapshot(canonical({ name: saved.name, description: saved.description ?? '', config: saved.config }));
      if (saved.latest_finished_run_id) setRun(await getSpatialSensitivityRun(saved.latest_finished_run_id));
      notifications.show({ color: 'green', title: 'Konfiguration gespeichert', message: saved.name });
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  async function removeConfiguration() {
    if (!configurationId || !window.confirm(`Konfiguration "${configurationName}" löschen? Die berechneten Läufe bleiben erhalten.`)) return;
    setBusy(true); try { await deleteSpatialSensitivityConfiguration(Number(configurationId)); setConfigurations(await listSpatialSensitivityConfigurations()); newConfiguration(); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  }
  async function loadSource() { if (!sourceDataset || !sourceTime) return; setBusy(true); try { const value = await previewSpatialSensitivity({ training_dataset_id: Number(sourceDataset), target_timestamp: sourceTime }); setPreview(value); setSourceTime(value.source_timestamp); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); } }
  async function loadExamples() { if (!chosenEvent?.datasetId || !chosenEvent.start || !chosenEvent.end || !normalExampleTime || !eventExampleTime) return; setBusy(true); try {
    const [normal, event] = await Promise.all([
      previewSpatialSensitivity({ training_dataset_id: Number(chosenEvent.datasetId), target_timestamp: normalExampleTime, range_start: chosenEvent.normalStart || shiftLocal(chosenEvent.start, -Number(hours) * 3600000), range_end: shiftLocal(chosenEvent.start, -1) }),
      previewSpatialSensitivity({ training_dataset_id: Number(chosenEvent.datasetId), target_timestamp: eventExampleTime, range_start: chosenEvent.start, range_end: chosenEvent.end }),
    ]); setNormalPreview(normal); setEventPreview(event); setNormalExampleTime(normal.source_timestamp); setEventExampleTime(event.source_timestamp);
  } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); } }
  async function loadWarpPreviews() {
    if (!chosenEvent?.datasetId || !chosenEvent.start || !chosenEvent.end || !normalExampleTime || !eventExampleTime) return;
    setBusy(true); setError(null);
    try {
      const common = { training_dataset_id: Number(chosenEvent.datasetId), warp: warpConfig };
      const [normal, event] = await Promise.all([
        previewSpatialSensitivityWarp({ ...common, target_timestamp: normalExampleTime, range_start: chosenEvent.normalStart || shiftLocal(chosenEvent.start, -Number(hours) * 3600000), range_end: shiftLocal(chosenEvent.start, -1) }),
        previewSpatialSensitivityWarp({ ...common, target_timestamp: eventExampleTime, range_start: chosenEvent.start, range_end: chosenEvent.end }),
      ]);
      setNormalWarpPreview(normal); setEventWarpPreview(event);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  const warpInput: PreprocessingPreviewImage | null = normalPreview ? {
    node_id: `spatial-warp-${normalPreview.source_timestamp}`, step_type: 'load_image', label: 'Normal raw image',
    width: normalPreview.width, height: normalPreview.height, channels: 1, dtype: normalPreview.dtype,
    value_min: 0, value_max: 65535, image_data_url: normalPreview.image_data_url,
  } : null;
  const valid = preview && datasetIds.length > 0 && events.length > 0 && events.every(e => e.id.trim() && e.datasetId && e.normalStart && e.start && e.end && new Date(e.normalStart) < new Date(e.start) && new Date(e.end) >= new Date(e.start)) && Number(hours) > 0 && Number(epsilon) > 0 && Number.isInteger(Number(normalSampleSize)) && Number(normalSampleSize) > 0 && Number.isInteger(Number(eventSampleSize)) && Number(eventSampleSize) > 0 && Number.isInteger(Number(samplingSeed)) && Number(samplingSeed) >= 0;
  async function start(force = false) { if (!valid || !sourceDataset || !analysisConfig) return;
    if (!force && contentMatchesLoaded && loadedConfiguration?.latest_finished_run_id) { setRun(await getSpatialSensitivityRun(loadedConfiguration.latest_finished_run_id)); setView('results'); return; }
    setBusy(true); setError(null); try {
    const payloadEvents: SpatialSensitivityEvent[] = analysisConfig.events;
    const next = await createSpatialSensitivityRun({ ...analysisConfig, events: payloadEvents, configuration_id: contentMatchesLoaded ? Number(configurationId) : null });
    setRun(next); setView('results'); setRuns(current => [next, ...current]); notifications.show({ color: 'blue', title: 'Analyse eingeplant', message: `Lauf #${next.id} wartet im Scheduler.` });
  } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); } }
  async function abortRun() { if (!displayedRun || !['queued','running'].includes(displayedRun.status)) return; setBusy(true); setError(null); try {
    const next = await abortSpatialSensitivityRun(displayedRun.id); setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item));
    notifications.show({ color: 'orange', title: 'Abbruch angefordert', message: `Lauf #${next.id} wird beendet.` });
  } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); } }
  const contentDirty = Boolean(loadedConfiguration && !contentMatchesLoaded);
  const displayedRun = run; const result = contentDirty ? null : displayedRun?.result; const progress = displayedRun?.total_images ? displayedRun.processed_images / displayedRun.total_images * 100 : 0;
  return <Stack gap="lg" pb="xl">
    <Group justify="space-between" align="flex-start"><Group align="flex-start"><ThemeIcon size={52} radius="lg" variant="light" color="teal"><Scan size={28}/></ThemeIcon><div><Text size="xs" fw={700} c="teal" tt="uppercase" lts={1.5}>Räumliche Bildanalyse</Text><Title order={2}>Spatial ROI Sensitivity</Title><Text c="dimmed" size="sm" maw={680}>Wo verändert sich das Bild während eines Ereignisses? Vergleiche die feste ROI mit dem gesamten Bereich außerhalb.</Text></div></Group><Badge variant="outline" color="teal">Originalauflösung · 1280 × 960</Badge></Group>
    <SimpleGrid cols={{base:1,sm:3}} spacing="sm">{[
      ['Datensätze', String(datasetIds.length), 'Ausgewählte Train/Test-Sets'],
      ['Ereignisse', String(events.length), 'Mit lokalem Normalzustand'],
      ['Stichprobe je Ereignis', `${fmt(Number(normalSampleSize),0)} / ${fmt(Number(eventSampleSize),0)}`, 'Normalbilder / Ereignisbilder'],
    ].map(([label,value,hint]) => <Paper key={label} withBorder radius="md" p="md"><Text size="xs" c="dimmed" fw={600}>{label}</Text><Text size="xl" fw={700} mt={4}>{value}</Text><Text size="xs" c="dimmed">{hint}</Text></Paper>)}</SimpleGrid>
    <Tabs value={view} onChange={setView} keepMounted variant="outline" radius="md">
      <Tabs.List mb="lg"><Tabs.Tab value="setup" leftSection={<Settings2 size={16}/>}>Analyse einrichten</Tabs.Tab><Tabs.Tab value="preview" leftSection={<Images size={16}/>}>Bild- & Warp-Vorschau</Tabs.Tab><Tabs.Tab value="results" leftSection={<BarChart3 size={16}/>}>Läufe & Ergebnisse {run && <Badge size="xs" ml={6} variant="light">{runLabels[run.status] ?? run.status}</Badge>}</Tabs.Tab></Tabs.List>
      <Tabs.Panel value="setup"><Stack gap="lg">
    <Paper withBorder p="lg" radius="md"><Stack gap="sm"><Group justify="space-between"><Title order={4}>Gespeicherte Konfiguration</Title><Button variant="light" leftSection={<Plus size={16}/>} onClick={newConfiguration}>Neue Konfiguration</Button></Group>
      <SimpleGrid cols={{base:1,md:3}}><Select label="Konfiguration laden" searchable clearable data={configurations.map(item => ({ value:String(item.id), label:`${item.name}${item.latest_finished_run_id ? ' · berechnet' : ''}` }))} value={configurationId} onChange={loadConfiguration}/><TextInput label="Name" value={configurationName} onChange={event => setConfigurationName(event.currentTarget.value)}/><TextInput label="Beschreibung" value={configurationDescription} onChange={event => setConfigurationDescription(event.currentTarget.value)}/></SimpleGrid>
      <Group><Button leftSection={<Save size={16}/>} onClick={() => saveConfiguration(false)} disabled={!analysisConfig || !configurationName.trim() || busy}>Speichern</Button><Button variant="light" leftSection={<Copy size={16}/>} onClick={() => saveConfiguration(true)} disabled={!analysisConfig || !configurationName.trim() || busy}>Als neu speichern</Button>{configurationId && <Button variant="subtle" color="red" leftSection={<Trash2 size={16}/>} onClick={removeConfiguration}>Löschen</Button>}</Group>
      {dirty && <Alert color="yellow">Die sichtbare Konfiguration enthält ungespeicherte Änderungen. Ein früheres Ergebnis wird dafür nicht verwendet.</Alert>}
      {!dirty && contentMatchesLoaded && loadedConfiguration?.latest_finished_run_id && <Alert color="green">Für diese Konfiguration ist bereits ein fertiges Ergebnis vorhanden und wird direkt geladen.</Alert>}
    </Stack></Paper>
    <Paper withBorder p="lg" radius="md"><Stack gap="md"><Group><ThemeIcon variant="light" radius="xl">1</ThemeIcon><div><Title order={4}>Datensätze & feste ROI</Title><Text size="sm" c="dimmed">Lade ein Normalbild und positioniere die vier Eckpunkte in Originalkoordinaten.</Text></div></Group><MultiSelect label="Train/Test-Datensätze" data={datasets.map(d => ({ value: String(d.id), label: d.name, disabled: d.invalid_rule_count > 0 }))} value={datasetIds} onChange={value => { setDatasetIds(value); setEvents(current => current.filter(e => e.datasetId && value.includes(e.datasetId))); if (sourceDataset && !value.includes(sourceDataset)) setSourceDataset(null); }} searchable />
      <SimpleGrid cols={{ base: 1, md: 3 }}><Select label="Datensatz für ROI-Normalbild" data={datasets.filter(d => datasetIds.includes(String(d.id))).map(d => ({ value: String(d.id), label: d.name }))} value={sourceDataset} onChange={value => { setSourceDataset(value); setPreview(null); }} /><DateTime24Input label="Zielzeitpunkt Normalbild" value={sourceTime} onChange={value => { setSourceTime(value); setPreview(null); }} /><Button mt={25} onClick={loadSource} disabled={!sourceDataset || !sourceTime || busy}>Nächstes valides Bild laden</Button></SimpleGrid>
      {preview && <><Text size="sm">Geladen: {new Date(preview.source_timestamp).toLocaleString('de-DE')} · {preview.width}×{preview.height} · {preview.dtype}</Text><RoiPicker preview={preview} points={points} onChange={setPoints} /></>}
      {!preview && <Paper p="xl" radius="md" bg="var(--mantine-color-default-hover)" ta="center"><Scan size={32} color="var(--mantine-color-dimmed)"/><Text fw={600} mt="sm">Noch kein ROI-Bild geladen</Text><Text size="sm" c="dimmed">Wähle oben einen Datensatz und einen Normal-Zielzeitpunkt.</Text></Paper>}
      </Stack></Paper><Paper withBorder radius="md" p="lg"><Stack gap="md"><Group><ThemeIcon variant="light" radius="xl">2</ThemeIcon><Title order={4}>Normalfenster & Stichprobe</Title></Group>
      <SimpleGrid cols={{ base: 1, md: 2 }}><NumberInput label="Lokales Normalfenster (Stunden)" description="Vorgabe für den automatisch angelegten Normalstart neuer Ereignisse." min={0.01} value={hours} onChange={changeHours} /><NumberInput label="Epsilon" min={Number.EPSILON} value={epsilon} onChange={setEpsilon} /></SimpleGrid>
      <div><Title order={4}>Reproduzierbare Bildstichprobe</Title><Text size="sm" c="dimmed">Aus jedem Zeitfenster werden Bilder gleichverteilt, ohne Zurücklegen und anhand des Seeds ausgewählt. Ungültige Treffer werden übersprungen und deterministisch ersetzt. Median und MAD werden exakt über die tatsächlich geladenen Stichprobenbilder berechnet.</Text></div>
      <SimpleGrid cols={{ base: 1, md: 3 }}><NumberInput label="Normalbilder" description="Maximale Zahl valider Bilder aus dem Normalzustand." min={1} max={100000} step={100} allowDecimal={false} value={normalSampleSize} onChange={setNormalSampleSize}/><NumberInput label="Ereignisbilder" description="Maximale Zahl valider Bilder aus dem Ereigniszeitraum." min={1} max={100000} step={100} allowDecimal={false} value={eventSampleSize} onChange={setEventSampleSize}/><NumberInput label="Zufalls-Seed" description="Gleicher Seed und gleiche Konfiguration ergeben dieselbe Auswahl." min={0} max={Number.MAX_SAFE_INTEGER} allowDecimal={false} value={samplingSeed} onChange={setSamplingSeed}/></SimpleGrid>
      </Stack></Paper><Paper withBorder radius="md" p="lg"><Stack gap="md"><Group><ThemeIcon variant="light" radius="xl">3</ThemeIcon><Title order={4}>Ereignisintervalle</Title></Group>
      <Group justify="space-between"><div><Text size="sm" c="dimmed">Jedes Ereignis vergleicht sein geschlossenes Ereignisintervall mit dem direkt davor liegenden Normalzustand. Der Normalstart wird zunächst aus Ereignisstart minus lokalem Fenster berechnet und kann anschließend einzeln angepasst werden.</Text></div><Button variant="light" leftSection={<Plus size={16}/>} onClick={addEvent} disabled={!datasetIds.length}>Ereignis hinzufügen</Button></Group>
      {!events.length && <Text c="dimmed" size="sm" py="lg" ta="center">Noch keine Ereignisse. Wähle einen Datensatz und füge das erste Intervall hinzu.</Text>}
      {events.map(e => <Paper key={e.key} withBorder p="sm"><SimpleGrid cols={{ base: 1, sm: 2, xl: 3 }}><TextInput label="Event-ID" value={e.id} onChange={x => updateEvent(e.key,{id:x.currentTarget.value})}/><Select label="Datensatz" data={datasets.filter(d => datasetIds.includes(String(d.id))).map(d => ({value:String(d.id),label:d.name}))} value={e.datasetId} onChange={value => updateEvent(e.key,{datasetId:value})}/><DateTime24Input label="Normalstart (inklusive)" value={e.normalStart} onChange={value => updateEvent(e.key,{normalStart:value,normalManual:true})}/><DateTime24Input label="Ereignisstart (inklusive)" value={e.start} onChange={value => updateEvent(e.key,{start:value})}/><DateTime24Input label="Ereignisende (inklusive)" value={e.end} onChange={value => updateEvent(e.key,{end:value})}/><Group align="flex-end" justify="flex-end"><ActionIcon aria-label={`Ereignis ${e.id} entfernen`} color="red" variant="light" onClick={() => setEvents(current => current.filter(x => x.key !== e.key))}><Trash2 size={17}/></ActionIcon></Group></SimpleGrid><Text size="xs" c="dimmed" mt="xs">Normalzustand: [{e.normalStart || '—'}, {e.start || '—'}) · Ereignis: [{e.start || '—'}, {e.end || '—'}]</Text></Paper>)}
      {overlapWarnings.length > 0 && <Alert color="yellow" title="Überlappende Normalfenster">{[...new Set(overlapWarnings)].join(' · ')}</Alert>}
      </Stack></Paper>
      <Paper withBorder radius="md" p="md"><Group justify="space-between"><Text size="sm" c="dimmed">{valid ? 'Bereit für die Analyse. Die ROI bleibt für alle Ereignisse fest.' : 'Zum Start: Datensätze und ROI-Bild auswählen, vollständige Ereignisintervalle und gültige Parameter angeben.'}</Text><Button onClick={() => start(false)} disabled={!valid || busy} loading={busy} leftSection={<Play size={16}/>}>{contentMatchesLoaded && loadedConfiguration?.latest_finished_run_id ? 'Ergebnis öffnen' : 'Analyse starten'}</Button></Group></Paper>
      </Stack></Tabs.Panel><Tabs.Panel value="preview"><Paper withBorder radius="md" p="lg"><Stack gap="md">
      <Group justify="space-between"><Title order={3}>Beispielbilder vergleichen</Title><Badge variant="light" color="orange">Nur Vorschau</Badge></Group>
      <div><Title order={4}>Illustrative Rohbilder</Title><Text size="sm" c="dimmed">Diese zwei Bilder dienen nur der visuellen Einordnung und der Warp-Vorschau. Für jeden Zielzeitpunkt wird das nächstgelegene valide Bild innerhalb des zugehörigen Normal- beziehungsweise Ereignisintervalls geladen. Die eigentliche Analyse verwendet die oben konfigurierte reproduzierbare Zufallsstichprobe.</Text></div><SimpleGrid cols={{ base:1, md:3 }}><Select label="Beispielereignis" data={events.map(e => ({value:e.key,label:e.id}))} value={exampleEvent} onChange={value => { setExampleEvent(value); setNormalPreview(null); setEventPreview(null); setNormalWarpPreview(null); setEventWarpPreview(null); }}/><DateTime24Input label="Normal-Zielzeitpunkt" value={normalExampleTime} onChange={setNormalExampleTime}/><DateTime24Input label="Ereignis-Zielzeitpunkt" value={eventExampleTime} onChange={setEventExampleTime}/></SimpleGrid><Button variant="light" onClick={loadExamples} disabled={!chosenEvent || !normalExampleTime || !eventExampleTime || busy}>Beispielbilder laden</Button>
      {(normalPreview || eventPreview) && <SimpleGrid cols={{base:1,md:2}}>{normalPreview && <Image loading="lazy" fit="contain" radius="md" src={normalPreview.image_data_url} alt="Normal example"/>}{eventPreview && <Image loading="lazy" fit="contain" radius="md" src={eventPreview.image_data_url} alt="Event example"/>}</SimpleGrid>}
      <Accordion variant="separated" radius="md"><Accordion.Item value="warp"><Accordion.Control icon={<Scan size={18}/>}>Perspektive prüfen · Warp-Vorschau</Accordion.Control><Accordion.Panel><Stack gap="md">
      <div><Title order={4}>Warp-Perspective-Vorschau</Title><Text size="sm" c="dimmed">Die vier Punkte und Parameter entsprechen exakt dem Preprocessing-Schritt „Warp perspective“. Dieselbe Transformation wird nur auf die beiden Beispielbilder angewendet und verändert weder Analysebilder noch ROI-Auswertung oder Heatmaps.</Text></div>
      {warpInput && (<PointPickerControl inputImage={warpInput} config={warpConfig} onChange={partial => {
        setWarpConfig(current => ({ ...current, ...partial } as SpatialSensitivityWarpConfig));
        setNormalWarpPreview(null); setEventWarpPreview(null);
      }}/>)}
      <SimpleGrid cols={{base:1,md:4}}><Select label="Art der Transformation" data={[{value:'preserve_rectangle',label:'Perspektivisch · Rechteckgröße automatisch'},{value:'manual',label:'Perspektivisch · Ausgabegröße manuell'}]} value={warpConfig.output_shape_mode} onChange={value => setWarpConfig(current => ({...current,output_shape_mode:(value ?? 'preserve_rectangle') as SpatialSensitivityWarpConfig['output_shape_mode']}))}/><Select label="Interpolation" data={[{value:'nearest',label:'Nearest'},{value:'linear',label:'Linear'},{value:'area',label:'Area'},{value:'cubic',label:'Cubic'}]} value={warpConfig.interpolation} onChange={value => setWarpConfig(current => ({...current,interpolation:(value ?? 'linear') as SpatialSensitivityWarpConfig['interpolation']}))}/>{warpConfig.output_shape_mode==='manual'&&<><NumberInput label="Ausgabebreite" min={1} value={warpConfig.output_width} onChange={value => setWarpConfig(current => ({...current,output_width:Number(value)}))}/><NumberInput label="Ausgabehöhe" min={1} value={warpConfig.output_height} onChange={value => setWarpConfig(current => ({...current,output_height:Number(value)}))}/></>}</SimpleGrid>
      <Button variant="light" onClick={loadWarpPreviews} disabled={!normalPreview || !eventPreview || busy}>Warp auf Normal- und Ereignisbild anwenden</Button>
      {(normalWarpPreview || eventWarpPreview) && <SimpleGrid cols={{base:1,md:2}}>{normalWarpPreview && <Paper withBorder p="sm"><Text fw={600}>Gewarpter Normalzustand</Text><Text size="sm" c="dimmed">{normalWarpPreview.input_width}×{normalWarpPreview.input_height} → {normalWarpPreview.output_width}×{normalWarpPreview.output_height} · {normalWarpPreview.output_shape_mode} · {normalWarpPreview.interpolation}</Text><Image loading="lazy" fit="contain" radius="md" src={normalWarpPreview.image_data_url} alt="Warped normal preview"/></Paper>}{eventWarpPreview && <Paper withBorder p="sm"><Text fw={600}>Gewarptes Ereignisbild</Text><Text size="sm" c="dimmed">{eventWarpPreview.input_width}×{eventWarpPreview.input_height} → {eventWarpPreview.output_width}×{eventWarpPreview.output_height} · {eventWarpPreview.output_shape_mode} · {eventWarpPreview.interpolation}</Text><Image loading="lazy" fit="contain" radius="md" src={eventWarpPreview.image_data_url} alt="Warped event preview"/></Paper>}</SimpleGrid>}
      </Stack></Accordion.Panel></Accordion.Item></Accordion>
      <Group justify="flex-end">{contentMatchesLoaded && loadedConfiguration?.latest_finished_run_id && <Button variant="light" leftSection={<Play size={16}/>} disabled={!valid || busy} onClick={() => start(true)}>Neu berechnen</Button>}<Button leftSection={busy ? <Loader size={16}/> : <Play size={16}/>} disabled={!valid || busy} onClick={() => start(false)}>{contentMatchesLoaded && loadedConfiguration?.latest_finished_run_id ? 'Vorhandenes Ergebnis öffnen' : 'Analyse starten'}</Button></Group></Stack></Paper></Tabs.Panel>
    <Tabs.Panel value="results"><Stack gap="md">
    {!runs.length && <Paper withBorder radius="md" p="xl" ta="center"><BarChart3 size={36}/><Title order={4} mt="sm">Deine Ergebnisse erscheinen hier</Title><Text c="dimmed" size="sm" mt="xs">Richte die Analyse ein und starte einen Lauf. Abgeschlossene Läufe bleiben abrufbar.</Text><Button variant="light" mt="md" onClick={() => setView('setup')}>Zur Einrichtung</Button></Paper>}
    {error && <Alert color="red" title="Analyse nicht möglich">{error}</Alert>}
    {runs.length > 0 && <Select label="Gespeicherter Lauf" data={runs.map(r => ({value:String(r.id),label:`#${r.id} · ${r.dataset_snapshot.map(d=>d.name).join(', ')} · ${runLabels[r.status] ?? r.status}`}))} value={run ? String(run.id) : null} onChange={v => { const found=runs.find(r=>String(r.id)===v); if(found)setRun(found); }}/>}
    {displayedRun && <Paper withBorder p="md"><Group justify="space-between"><div><Text fw={600}>{runLabels[displayedRun.current_step] ?? displayedRun.current_step}</Text><Text size="sm" c="dimmed">{displayedRun.processed_images}/{displayedRun.total_images ?? '—'} Bilder · {displayedRun.failed_images} ungültig</Text></div><Group>{contentMatchesLoaded && loadedConfiguration?.latest_finished_run_id === displayedRun.id && <Badge color="green" variant="light">Gespeichertes Ergebnis</Badge>}<Badge color={displayedRun.status==='finished'?'green':displayedRun.status==='failed'?'red':displayedRun.status==='aborted'?'orange':'blue'}>{runLabels[displayedRun.status] ?? displayedRun.status}</Badge>{['queued','running'].includes(displayedRun.status)&&<Button color="red" variant="light" leftSection={<Square size={14}/>} loading={busy} onClick={abortRun}>Analyse abbrechen</Button>}</Group></Group>{['queued','running'].includes(displayedRun.status)&&<Progress value={Math.min(100, progress)} animated/>}{displayedRun.error_message&&<Alert mt="sm" color="red">{displayedRun.error_message}</Alert>}</Paper>}
    {result && <><Paper withBorder p="md"><Group justify="space-between"><div><Title order={4}>Ergebnis</Title><Text size="sm" c="dimmed">Gemeinsame Skalen: D ≤ {fmt(result.vmax_d)} · Z ≤ {fmt(result.vmax_z)} · ε={result.epsilon}</Text><Text size="sm" c="dimmed">Deterministische Stichprobe: Normal ≤ {fmt(result.normal_sample_size,0)} · Ereignis ≤ {fmt(result.event_sample_size,0)} · Seed {fmt(result.sampling_seed,0)}</Text></div><Button component="a" href={spatialSensitivityArtifactUrl(run!.id,'spatial_sensitivity_artifacts.zip')} leftSection={<Download size={16}/>}>Alle Artefakte</Button></Group></Paper>
      {result.events.some(row => Number(row.normal_sample_shortfall) > 0 || Number(row.event_sample_shortfall) > 0) && <Alert color="yellow" title="Kleinere Stichprobe">Mindestens ein Zeitfenster enthielt weniger valide Bilder als angefordert. Die tatsächlich verwendeten Anzahlen stehen in der Ergebnistabelle und im Manifest.</Alert>}
      <SimpleGrid cols={{base:1,sm:3}}>{[
        ['Änderungsanteil in der ROI', `${fmt(result.median.p_in,1)} %`, 'Median P in · Anteil der absoluten Gesamtänderung'],
        ['Relative Änderung pro Pixel', fmt(result.median.q_d), 'Median Q D · mittlere Änderung innen / außen'],
        ['Variabilitätsbereinigte Änderung', fmt(result.median.q_z), 'Median Q Z · normalisierte Änderung innen / außen'],
      ].map(([label,value,hint]) => <Paper key={label} withBorder radius="md" p="lg"><Text size="sm" fw={600}>{label}</Text><Text size="xl" fw={700} c="teal" my="xs">{value}</Text><Text size="xs" c="dimmed">{hint}</Text></Paper>)}</SimpleGrid>
      <Text size="sm" c="dimmed">P in beschreibt die Konzentration der Gesamtänderung. Q D und Q Z vergleichen die Änderung pro Pixel, jeweils mit Epsilon im Nenner. Die Werte sind deskriptiv; die ROI-Fläche ist bei der Einordnung mitzuberücksichtigen.</Text>
      <Image loading="lazy" fit="contain" radius="md" src={spatialSensitivityArtifactUrl(run!.id,'publication_overview.png')} alt="Publication overview"/>
      <SimpleGrid cols={{base:1,md:2}}><Image loading="lazy" fit="contain" radius="md" src={spatialSensitivityArtifactUrl(run!.id,'events_grid_D.png')} alt="D event grid"/><Image loading="lazy" fit="contain" radius="md" src={spatialSensitivityArtifactUrl(run!.id,'events_grid_Z.png')} alt="Z event grid"/><Image loading="lazy" fit="contain" radius="md" src={spatialSensitivityArtifactUrl(run!.id,'aggregate_D.png')} alt="Aggregated D"/><Image loading="lazy" fit="contain" radius="md" src={spatialSensitivityArtifactUrl(run!.id,'aggregate_Z.png')} alt="Aggregated Z"/><Image loading="lazy" fit="contain" radius="md" src={spatialSensitivityArtifactUrl(run!.id,'z_inside_outside.png')} alt="Inside outside Z"/></SimpleGrid>
      <ScrollArea><Table striped miw={1500}><Table.Thead><Table.Tr>{['Event','Datensatz','N Kandidaten','N genutzt','N Versuche','U Kandidaten','U genutzt','U Versuche','D in','D out','Q D','Z in','Z out','Q Z','P in %'].map(x=><Table.Th key={x}>{x}</Table.Th>)}</Table.Tr></Table.Thead><Table.Tbody>{[...result.events,result.median].map((r,i)=><Table.Tr key={i}><Table.Td>{String(r.event_id)}</Table.Td><Table.Td>{String(r.training_dataset??'')}</Table.Td><Table.Td>{fmt(r.normal_candidate_count,0)}</Table.Td><Table.Td>{fmt(r.normal_image_count,0)}</Table.Td><Table.Td>{fmt(r.normal_attempted_count,0)}</Table.Td><Table.Td>{fmt(r.event_candidate_count,0)}</Table.Td><Table.Td>{fmt(r.event_image_count,0)}</Table.Td><Table.Td>{fmt(r.event_attempted_count,0)}</Table.Td><Table.Td>{fmt(r.mean_d_in)}</Table.Td><Table.Td>{fmt(r.mean_d_out)}</Table.Td><Table.Td>{fmt(r.q_d)}</Table.Td><Table.Td>{fmt(r.mean_z_in)}</Table.Td><Table.Td>{fmt(r.mean_z_out)}</Table.Td><Table.Td>{fmt(r.q_z)}</Table.Td><Table.Td>{fmt(r.p_in,1)}</Table.Td></Table.Tr>)}</Table.Tbody></Table></ScrollArea></>}
    </Stack></Tabs.Panel></Tabs>
    {error && view !== 'results' && <Alert color="red" title="Bitte Eingaben prüfen" withCloseButton onClose={() => setError(null)}>{error}</Alert>}
    {run && ['queued','running'].includes(run.status) && view !== 'results' && <Paper withBorder radius="md" p="md" style={{position:'sticky', bottom:16, zIndex:20, background:'var(--mantine-color-body)', boxShadow:'var(--mantine-shadow-md)'}}><Group justify="space-between"><Group><Loader size="sm"/><div><Text fw={600}>Lauf #{run.id} · {runLabels[run.status]}</Text><Text size="xs" c="dimmed">{runLabels[run.current_step] ?? run.current_step}</Text></div></Group><Group><Button variant="subtle" onClick={() => setView('results')}>Fortschritt ansehen</Button><Button color="red" variant="light" onClick={abortRun} leftSection={<Square size={14}/>}>Abbrechen</Button></Group></Group></Paper>}
  </Stack>;
}
