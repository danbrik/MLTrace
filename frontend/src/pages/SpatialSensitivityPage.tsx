import { ActionIcon, Alert, Badge, Button, Group, Image, Loader, MultiSelect, NumberInput, Paper, Progress, ScrollArea, Select, SimpleGrid, Stack, Table, Text, TextInput, Title } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Download, Plus, Play, Trash2 } from 'lucide-react';
import { PointerEvent, useEffect, useMemo, useRef, useState } from 'react';
import { createSpatialSensitivityRun, getSpatialSensitivityRun, listSpatialSensitivityRuns, listTrainingDatasets, previewSpatialSensitivity, spatialSensitivityArtifactUrl } from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import type { SpatialSensitivityEvent, SpatialSensitivityPreview, SpatialSensitivityRun, TrainingDataset } from '../types';

type Point = { x: number; y: number };
type DraftEvent = { key: string; id: string; datasetId: string | null; start: string; end: string };
const initialPoints: Point[] = [{ x: 160, y: 120 }, { x: 1120, y: 120 }, { x: 1120, y: 840 }, { x: 160, y: 840 }];
const fmt = (value: unknown, digits = 3) => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('de-DE', { maximumFractionDigits: digits }) : '—';
const uid = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

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
  return <Stack gap="xs"><div ref={ref} onPointerMove={move} onPointerUp={() => dragging.current = null} onPointerLeave={() => dragging.current = null} style={{ position: 'relative', maxWidth: 900 }}>
    <img src={preview.image_data_url} alt="ROI source" style={{ width: '100%', display: 'block' }} />
    <svg viewBox={`0 0 ${preview.width} ${preview.height}`} preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
      <polygon points={polygon} fill="rgba(255,165,0,.08)" stroke="orange" strokeWidth="5" vectorEffect="non-scaling-stroke" />
    </svg>
    {points.map((point, index) => <button key={index} type="button" onPointerDown={(event) => { dragging.current = index; event.currentTarget.setPointerCapture?.(event.pointerId); }} style={{ position: 'absolute', left: `${point.x / preview.width * 100}%`, top: `${point.y / preview.height * 100}%`, transform: 'translate(-50%,-50%)', borderRadius: '50%', width: 28, height: 28, border: '2px solid white', background: 'orange', cursor: 'grab' }}>{index + 1}</button>)}
  </div><Group>{points.map((p, i) => <Badge key={i} variant="light">P{i + 1}: {Math.round(p.x)}, {Math.round(p.y)}</Badge>)}</Group></Stack>;
}

export function SpatialSensitivityPage({ active }: { active: boolean }) {
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]); const [runs, setRuns] = useState<SpatialSensitivityRun[]>([]);
  const [datasetIds, setDatasetIds] = useState<string[]>([]); const [sourceDataset, setSourceDataset] = useState<string | null>(null); const [sourceTime, setSourceTime] = useState('');
  const [preview, setPreview] = useState<SpatialSensitivityPreview | null>(null); const [points, setPoints] = useState<Point[]>(initialPoints);
  const [events, setEvents] = useState<DraftEvent[]>([]); const [hours, setHours] = useState<number | string>(24); const [epsilon, setEpsilon] = useState<number | string>(1);
  const [exampleEvent, setExampleEvent] = useState<string | null>(null); const [normalExampleTime, setNormalExampleTime] = useState(''); const [eventExampleTime, setEventExampleTime] = useState('');
  const [normalPreview, setNormalPreview] = useState<SpatialSensitivityPreview | null>(null); const [eventPreview, setEventPreview] = useState<SpatialSensitivityPreview | null>(null);
  const [run, setRun] = useState<SpatialSensitivityRun | null>(null); const [busy, setBusy] = useState(false); const [error, setError] = useState<string | null>(null);
  useEffect(() => { if (!active) return; Promise.all([listTrainingDatasets(), listSpatialSensitivityRuns()]).then(([d, r]) => { setDatasets(d); setRuns(r); if (r[0]) setRun(r[0]); }).catch(e => setError(String(e))); }, [active]);
  useEffect(() => { if (!active || !run || !['queued','running'].includes(run.status)) return; const timer = window.setInterval(() => getSpatialSensitivityRun(run.id).then(next => { setRun(next); setRuns(current => current.map(item => item.id === next.id ? next : item)); }), 1500); return () => window.clearInterval(timer); }, [active, run?.id, run?.status]);
  const chosenEvent = events.find(item => item.key === exampleEvent);
  const overlapWarnings = useMemo(() => events.flatMap(event => events.filter(other => other.key !== event.key && other.datasetId === event.datasetId && event.start && other.start && other.end && new Date(other.end) >= new Date(new Date(event.start).getTime() - Number(hours) * 3600000) && new Date(other.start) < new Date(event.start)).map(other => `${event.id || 'Ereignis'}: Normalfenster überlappt ${other.id || 'Ereignis'}`)), [events, hours]);
  function addEvent() { setEvents(current => [...current, { key: uid(), id: `U${current.length + 1}`, datasetId: datasetIds[0] ?? null, start: '', end: '' }]); }
  function updateEvent(key: string, change: Partial<DraftEvent>) { setEvents(current => current.map(item => item.key === key ? { ...item, ...change } : item)); }
  async function loadSource() { if (!sourceDataset || !sourceTime) return; setBusy(true); try { const value = await previewSpatialSensitivity({ training_dataset_id: Number(sourceDataset), target_timestamp: sourceTime }); setPreview(value); setSourceTime(value.source_timestamp); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); } }
  async function loadExamples() { if (!chosenEvent?.datasetId || !chosenEvent.start || !chosenEvent.end || !normalExampleTime || !eventExampleTime) return; setBusy(true); try {
    const [normal, event] = await Promise.all([
      previewSpatialSensitivity({ training_dataset_id: Number(chosenEvent.datasetId), target_timestamp: normalExampleTime, range_start: new Date(new Date(chosenEvent.start).getTime() - Number(hours) * 3600000).toISOString().slice(0,19), range_end: new Date(new Date(chosenEvent.start).getTime() - 1).toISOString() }),
      previewSpatialSensitivity({ training_dataset_id: Number(chosenEvent.datasetId), target_timestamp: eventExampleTime, range_start: chosenEvent.start, range_end: chosenEvent.end }),
    ]); setNormalPreview(normal); setEventPreview(event); setNormalExampleTime(normal.source_timestamp); setEventExampleTime(event.source_timestamp);
  } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); } }
  const valid = preview && datasetIds.length > 0 && events.length > 0 && events.every(e => e.id.trim() && e.datasetId && e.start && e.end && new Date(e.end) >= new Date(e.start)) && Number(hours) > 0 && Number(epsilon) > 0;
  async function start() { if (!valid || !sourceDataset) return; setBusy(true); setError(null); try {
    const payloadEvents: SpatialSensitivityEvent[] = events.map(e => ({ id: e.id.trim(), training_dataset_id: Number(e.datasetId), start: e.start, end: e.end }));
    const next = await createSpatialSensitivityRun({ training_dataset_ids: datasetIds.map(Number), events: payloadEvents, normal_window_hours: Number(hours), epsilon: Number(epsilon), roi_points: points, roi_source_dataset_id: Number(sourceDataset), roi_source_timestamp: preview.source_timestamp, example_event_id: chosenEvent?.id ?? null, example_normal_timestamp: normalPreview?.source_timestamp ?? null, example_event_timestamp: eventPreview?.source_timestamp ?? null });
    setRun(next); setRuns(current => [next, ...current]); notifications.show({ color: 'blue', title: 'Analyse eingeplant', message: `Lauf #${next.id} wartet im Scheduler.` });
  } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); } }
  const result = run?.result; const progress = run?.total_images ? run.processed_images / run.total_images * 100 : 0;
  return <Stack gap="md"><div><Title order={2}>Spatial ROI Sensitivity</Title><Text c="dimmed">Vollauflösende räumliche Ereignisanalyse mit einer festen, rein auswertenden ROI.</Text></div>
    <Paper withBorder p="md"><Stack gap="md"><MultiSelect label="Train/Test-Datensätze" data={datasets.map(d => ({ value: String(d.id), label: d.name, disabled: d.invalid_rule_count > 0 }))} value={datasetIds} onChange={value => { setDatasetIds(value); setEvents(current => current.filter(e => e.datasetId && value.includes(e.datasetId))); if (sourceDataset && !value.includes(sourceDataset)) setSourceDataset(null); }} searchable />
      <SimpleGrid cols={{ base: 1, md: 3 }}><Select label="Datensatz für ROI-Normalbild" data={datasets.filter(d => datasetIds.includes(String(d.id))).map(d => ({ value: String(d.id), label: d.name }))} value={sourceDataset} onChange={setSourceDataset} /><DateTime24Input label="Zielzeitpunkt Normalbild" value={sourceTime} onChange={setSourceTime} /><Button mt={25} onClick={loadSource} disabled={!sourceDataset || !sourceTime || busy}>Nächstes valides Bild laden</Button></SimpleGrid>
      {preview && <><Text size="sm">Geladen: {new Date(preview.source_timestamp).toLocaleString('de-DE')} · {preview.width}×{preview.height} · {preview.dtype}</Text><RoiPicker preview={preview} points={points} onChange={setPoints} /></>}
      <SimpleGrid cols={{ base: 1, md: 2 }}><NumberInput label="Lokales Normalfenster (Stunden)" min={0.01} value={hours} onChange={setHours} /><NumberInput label="Epsilon" min={Number.EPSILON} value={epsilon} onChange={setEpsilon} /></SimpleGrid>
      <Group justify="space-between"><Title order={4}>Ereignisse</Title><Button variant="light" leftSection={<Plus size={16}/>} onClick={addEvent} disabled={!datasetIds.length}>Ereignis hinzufügen</Button></Group>
      {events.map(e => <Paper key={e.key} withBorder p="sm"><SimpleGrid cols={{ base: 1, lg: 5 }}><TextInput label="Event-ID" value={e.id} onChange={x => updateEvent(e.key,{id:x.currentTarget.value})}/><Select label="Datensatz" data={datasets.filter(d => datasetIds.includes(String(d.id))).map(d => ({value:String(d.id),label:d.name}))} value={e.datasetId} onChange={value => updateEvent(e.key,{datasetId:value})}/><DateTime24Input label="Start (inklusive)" value={e.start} onChange={value => updateEvent(e.key,{start:value})}/><DateTime24Input label="Ende (inklusive)" value={e.end} onChange={value => updateEvent(e.key,{end:value})}/><Group align="flex-end" justify="flex-end"><ActionIcon color="red" variant="light" onClick={() => setEvents(current => current.filter(x => x.key !== e.key))}><Trash2 size={17}/></ActionIcon></Group></SimpleGrid></Paper>)}
      {overlapWarnings.length > 0 && <Alert color="yellow" title="Überlappende Normalfenster">{[...new Set(overlapWarnings)].join(' · ')}</Alert>}
      <Title order={4}>Illustrative Rohbilder</Title><SimpleGrid cols={{ base:1, md:3 }}><Select label="Beispielereignis" data={events.map(e => ({value:e.key,label:e.id}))} value={exampleEvent} onChange={setExampleEvent}/><DateTime24Input label="Normal-Zielzeitpunkt" value={normalExampleTime} onChange={setNormalExampleTime}/><DateTime24Input label="Ereignis-Zielzeitpunkt" value={eventExampleTime} onChange={setEventExampleTime}/></SimpleGrid><Button variant="light" onClick={loadExamples} disabled={!chosenEvent || !normalExampleTime || !eventExampleTime || busy}>Beispielbilder laden</Button>
      {(normalPreview || eventPreview) && <SimpleGrid cols={{base:1,md:2}}>{normalPreview && <Image src={normalPreview.image_data_url} alt="Normal example"/>}{eventPreview && <Image src={eventPreview.image_data_url} alt="Event example"/>}</SimpleGrid>}
      <Group justify="flex-end"><Button leftSection={busy ? <Loader size={16}/> : <Play size={16}/>} disabled={!valid || busy} onClick={start}>Analyse starten</Button></Group></Stack></Paper>
    {error && <Alert color="red" title="Analyse nicht möglich">{error}</Alert>}
    {runs.length > 0 && <Select label="Gespeicherter Lauf" data={runs.map(r => ({value:String(r.id),label:`#${r.id} · ${r.dataset_snapshot.map(d=>d.name).join(', ')} · ${r.status}`}))} value={run ? String(run.id) : null} onChange={v => { const found=runs.find(r=>String(r.id)===v); if(found)setRun(found); }}/>} 
    {run && <Paper withBorder p="md"><Group justify="space-between"><div><Text fw={600}>{run.current_step}</Text><Text size="sm" c="dimmed">{run.processed_images}/{run.total_images ?? '—'} Bilder · {run.failed_images} ungültig</Text></div><Badge color={run.status==='finished'?'green':run.status==='failed'?'red':'blue'}>{run.status}</Badge></Group>{['queued','running'].includes(run.status)&&<Progress value={progress} animated/>}{run.error_message&&<Alert mt="sm" color="red">{run.error_message}</Alert>}</Paper>}
    {result && <><Paper withBorder p="md"><Group justify="space-between"><div><Title order={4}>Ergebnis</Title><Text size="sm" c="dimmed">Gemeinsame Skalen: D ≤ {fmt(result.vmax_d)} · Z ≤ {fmt(result.vmax_z)} · ε={result.epsilon}</Text></div><Button component="a" href={spatialSensitivityArtifactUrl(run!.id,'spatial_sensitivity_artifacts.zip')} leftSection={<Download size={16}/>}>Alle Artefakte</Button></Group></Paper>
      <Image src={spatialSensitivityArtifactUrl(run!.id,'publication_overview.png')} alt="Publication overview"/>
      <SimpleGrid cols={{base:1,md:2}}><Image src={spatialSensitivityArtifactUrl(run!.id,'events_grid_D.png')} alt="D event grid"/><Image src={spatialSensitivityArtifactUrl(run!.id,'events_grid_Z.png')} alt="Z event grid"/><Image src={spatialSensitivityArtifactUrl(run!.id,'aggregate_D.png')} alt="Aggregated D"/><Image src={spatialSensitivityArtifactUrl(run!.id,'aggregate_Z.png')} alt="Aggregated Z"/><Image src={spatialSensitivityArtifactUrl(run!.id,'z_inside_outside.png')} alt="Inside outside Z"/></SimpleGrid>
      <ScrollArea><Table striped miw={1100}><Table.Thead><Table.Tr>{['Event','Datensatz','N','U','D in','D out','Q D','Z in','Z out','Q Z','P in %'].map(x=><Table.Th key={x}>{x}</Table.Th>)}</Table.Tr></Table.Thead><Table.Tbody>{[...result.events,result.median].map((r,i)=><Table.Tr key={i}><Table.Td>{String(r.event_id)}</Table.Td><Table.Td>{String(r.training_dataset??'')}</Table.Td><Table.Td>{fmt(r.normal_image_count,0)}</Table.Td><Table.Td>{fmt(r.event_image_count,0)}</Table.Td><Table.Td>{fmt(r.mean_d_in)}</Table.Td><Table.Td>{fmt(r.mean_d_out)}</Table.Td><Table.Td>{fmt(r.q_d)}</Table.Td><Table.Td>{fmt(r.mean_z_in)}</Table.Td><Table.Td>{fmt(r.mean_z_out)}</Table.Td><Table.Td>{fmt(r.q_z)}</Table.Td><Table.Td>{fmt(r.p_in,1)}</Table.Td></Table.Tr>)}</Table.Tbody></Table></ScrollArea></>}
  </Stack>;
}
