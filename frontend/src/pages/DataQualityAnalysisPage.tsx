import { useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Badge, Button, Card, Checkbox, FileInput, Group, Loader, MultiSelect, NumberInput, Pagination, Paper, Progress, ScrollArea, Select, SimpleGrid, Stack, Table, Text, TextInput, Title } from '@mantine/core';
import { dataQualityAction, deleteDataQuality, getDataQuality, getDataQualityHeatmap, listRedundancySources, lookupDataQuality, startDataQuality, uploadRedundancySource } from '../api';
import type { DataQualityAnalysis, DataQualityHeatmap, DataQualityParameters, RedundancySource } from '../types';
import { PlotlyChart } from '../components/PlotlyChart';
import type { Data } from '../lib/plotly';
import { qualityFlags, visibleSensors } from '../dataQuality/helpers';

const fmt = (value: unknown) => typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : value == null ? '—' : String(value);
const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const summaryLabels: Record<string, string> = {
  start_timestamp: 'Start', end_timestamp: 'End', interval_seconds: 'Expected sampling interval (seconds)',
  expected_timepoints: 'Expected timepoints', present_timepoints: 'Present timepoints', missing_timepoints: 'Missing timepoints',
  coverage_percent: 'Temporal coverage (%)', duplicate_timestamps: 'Duplicate timestamps', non_monotone_timestamps: 'Non-monotone timestamps',
  invalid_timestamps: 'Invalid timestamps (entire CSV)', off_grid_rows: 'Off-grid rows', variable_count: 'Variables',
  variables_with_missing: 'Variables with missing values', constant_variables: 'Constant variables', duplicate_conflicts: 'Sensor/timepoint duplicate conflicts',
};

function Results({ analysis }: { analysis: DataQualityAnalysis }) {
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState('name');
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const [page, setPage] = useState(1);
  const [missing, setMissing] = useState(1);
  const [gap, setGap] = useState(60);
  const [heatmap, setHeatmap] = useState<DataQualityHeatmap | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [range, setRange] = useState<[string, string] | null>(null);
  const result = analysis.result!;
  const sensors = useMemo(() => visibleSensors(result.quality, search, sort, flaggedOnly, missing, gap), [result, search, sort, flaggedOnly, missing, gap]);
  const flagged = result.quality.filter(s => qualityFlags(s, missing, gap).length > 0);
  useEffect(() => setPage(1), [search, sort, flaggedOnly, missing, gap]);
  useEffect(() => {
    let alive = true;
    setLoading(true); setError('');
    const timer = window.setTimeout(() => {
      void getDataQualityHeatmap(analysis.id, range?.[0], range?.[1]).then(value => { if (alive) setHeatmap(value); })
        .catch(e => { if (alive) setError(message(e)); }).finally(() => { if (alive) setLoading(false); });
    }, 200);
    return () => { alive = false; window.clearTimeout(timer); };
  }, [analysis.id, range]);
  return <Stack>
    <Paper withBorder p="md"><Title order={3}>Dataset overview</Title><SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} mt="sm">
      {Object.entries(result.summary).map(([key, value]) => <Card withBorder key={key} p="sm"><Text size="xs" c="dimmed">{summaryLabels[key] ?? key}</Text><Text fw={600}>{fmt(value)}</Text></Card>)}
    </SimpleGrid><Text size="xs" c="dimmed" mt="sm">Off-grid and invalid timestamps are excluded. Duplicate timestamps use the first valid value per sensor; conflicts are reported. Missingness includes entirely absent timepoints.</Text></Paper>
    <Paper withBorder p="md"><Group justify="space-between"><Title order={3}>Missingness over time</Title><Button variant="light" onClick={() => setRange(null)}>Full range</Button></Group>
      <Text size="sm" c="dimmed">{heatmap?.aggregated ? 'Aggregated view: color shows the missing fraction in each time bin. Zoom in for finer detail.' : 'Each cell is an expected observation: 0 = present, 1 = missing.'}</Text>
      {loading && <Group><Loader size="xs" /><Text size="sm">Loading heatmap…</Text></Group>}{error && <Alert color="red">{error}</Alert>}
      {heatmap && heatmap.x.length > 0 ? <ScrollArea h={Math.min(650, 130 + heatmap.y.length * 22)}><PlotlyChart height={Math.max(320, heatmap.y.length * 22 + 100)}
        data={[{ type: 'heatmap', x: heatmap.x, y: heatmap.y, z: heatmap.z, zmin: 0, zmax: 1,
          customdata: heatmap.y.map(() => heatmap.bin_end), colorscale: [[0, '#edf6ff'], [1, '#084594']],
          colorbar: { title: { text: heatmap.aggregated ? 'Missing fraction' : 'Missing' } },
          hovertemplate: '%{y}<br>%{x} – %{customdata}<br>Missing: %{z:.2%}<extra></extra>' } as Data]}
        layout={{ uirevision: analysis.id, margin: { l: 160, r: 90, t: 15, b: 60 }, xaxis: { type: 'date', ...(range ? { range } : { autorange: true }) }, yaxis: { autorange: 'reversed' } }}
        onRelayout={event => {
          const e = event as Record<string, unknown>;
          if (e['xaxis.autorange']) { setRange(null); return; }
          const pair = e['xaxis.range'] as string[] | undefined;
          const start = pair?.[0] ?? e['xaxis.range[0]']; const end = pair?.[1] ?? e['xaxis.range[1]'];
          if (typeof start === 'string' && typeof end === 'string') setRange(current => current?.[0] === start && current?.[1] === end ? current : [start, end]);
        }} /></ScrollArea> : !loading && <Text>No expected observations in this viewport.</Text>}
    </Paper>
    <Paper withBorder p="md"><Title order={3}>Quality thresholds</Title><Group align="end" mt="sm">
      <NumberInput label="Missing greater than (%)" value={missing} min={0} max={100} onChange={v => setMissing(Number(v) || 0)} />
      <NumberInput label="Longest gap greater than (min)" value={gap} min={0} onChange={v => setGap(Number(v) || 0)} />
    </Group><Text size="xs" c="dimmed" mt="sm">Thresholds only change flags. Low unique counts alone are informational. Constant and empty sensors are always flagged.</Text></Paper>
    <Paper withBorder p="md"><Title order={3}>Sensors requiring attention ({flagged.length})</Title>
      {flagged.length ? <ScrollArea h={Math.min(350, 50 + flagged.length * 42)}><Table striped><Table.Thead><Table.Tr>{['Sensor', 'Missing %', 'Longest gap (min)', 'Unique', 'Reasons'].map(h => <Table.Th key={h}>{h}</Table.Th>)}</Table.Tr></Table.Thead><Table.Tbody>{flagged.map(s => <Table.Tr key={s.sensor}><Table.Td>{s.sensor}</Table.Td><Table.Td>{fmt(s.missing_percent)}</Table.Td><Table.Td>{fmt(s.longest_gap_minutes)}</Table.Td><Table.Td>{fmt(s.unique)}</Table.Td><Table.Td>{qualityFlags(s, missing, gap).join(', ')}</Table.Td></Table.Tr>)}</Table.Tbody></Table></ScrollArea> : <Text c="dimmed">No sensors exceed these criteria.</Text>}
    </Paper>
    <Paper withBorder p="md"><Title order={3}>Sensor quality</Title><Group my="md"><TextInput placeholder="Search sensors" value={search} onChange={e => setSearch(e.currentTarget.value)} /><Select value={sort} onChange={v => setSort(v ?? 'name')} data={[{ value: 'name', label: 'Sensor name' }, { value: 'missing', label: 'Missing % descending' }, { value: 'gap', label: 'Longest gap descending' }]} /><Checkbox label="Flagged only" checked={flaggedOnly} onChange={e => setFlaggedOnly(e.currentTarget.checked)} /></Group>
      <Text size="xs" c="dimmed" mb="sm">Longest gap = consecutive missing expected observations × sampling interval. Three missing points on a one-minute grid equal 3 min.</Text>
      <SimpleGrid cols={{ base: 1, md: 2, xl: 3 }}>{sensors.slice((page - 1) * 20, page * 20).map(s => <Card withBorder key={s.sensor}>
        <Group justify="space-between"><Text fw={700} style={{ overflowWrap: 'anywhere' }}>{s.sensor}</Text><Badge variant="light">{s.data_type}</Badge></Group>
        <Group gap={4} my="xs">{qualityFlags(s, missing, gap).map(flag => <Badge key={flag} color="orange" size="xs">{flag}</Badge>)}</Group>
        <SimpleGrid cols={2}>{([['N valid', s.valid_n], ['Missing %', s.missing_percent], ['Longest gap (min)', s.longest_gap_minutes], ['Unique', s.unique], ['Min', s.min], ['Q01', s.q01], ['Median', s.median], ['Q99', s.q99], ['Max', s.max], ['IQR', s.iqr], ['Std (sample)', s.std]] as const).map(([label, value], index) => <div key={label}><Text size="xs" c="dimmed">{label}</Text><Text size="sm" fw={500}>{s.data_type === 'text' && index >= 4 ? 'Not applicable' : fmt(value)}</Text></div>)}</SimpleGrid>
        {(s.invalid_n > 0 || s.conflict_n > 0) && <Text size="xs" c="orange" mt="xs">Invalid numeric values: {s.invalid_n}; conflicting duplicate timepoints: {s.conflict_n}</Text>}
      </Card>)}</SimpleGrid>{!sensors.length && <Text>No matching sensors.</Text>}<Pagination total={Math.max(1, Math.ceil(sensors.length / 20))} value={page} onChange={setPage} mt="md" />
    </Paper>
  </Stack>;
}

export function DataQualityAnalysisPage({ active }: { active: boolean }) {
  const [sources, setSources] = useState<RedundancySource[]>([]);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [timeColumn, setTimeColumn] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [types, setTypes] = useState<Record<string, 'numeric' | 'text'>>({});
  const [start, setStart] = useState(''); const [end, setEnd] = useState('');
  const [interval, setIntervalSeconds] = useState(60);
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<DataQualityAnalysis | null>(null);
  const [busy, setBusy] = useState(false); const [checking, setChecking] = useState(false);
  const [error, setError] = useState(''); const [cacheHit, setCacheHit] = useState(false);
  const selectionGeneration = useRef(0);
  const source = sources.find(s => String(s.id) === sourceId);
  const payload = useMemo<DataQualityParameters | null>(() => sourceId && timeColumn && columns.length && start && end && interval >= 1 ? {
    source_id: Number(sourceId), time_column: timeColumn, selected_columns: [...columns].sort(),
    data_types: Object.fromEntries([...columns].sort().map(c => [c, types[c] ?? 'text'])),
    start_timestamp: start, end_timestamp: end, interval_seconds: interval,
  } : null, [sourceId, timeColumn, columns, types, start, end, interval]);
  const payloadKey = JSON.stringify(payload);
  useEffect(() => { if (active) void listRedundancySources().then(setSources).catch(e => setError(message(e))); }, [active]);
  useEffect(() => {
    const generation = ++selectionGeneration.current;
    setAnalysis(null); setCacheHit(false); setError('');
    if (!payload || !active) { setChecking(false); return; }
    setChecking(true);
    const timer = window.setTimeout(() => {
      void lookupDataQuality(payload).then(row => { if (generation === selectionGeneration.current) { setAnalysis(row); setCacheHit(row?.job_status === 'ready'); } })
        .catch(e => { if (generation === selectionGeneration.current) setError(message(e)); })
        .finally(() => { if (generation === selectionGeneration.current) setChecking(false); });
    }, 350);
    return () => { window.clearTimeout(timer); selectionGeneration.current++; };
  }, [payloadKey, active]);
  useEffect(() => {
    if (!active || !analysis || !['queued', 'running'].includes(analysis.job_status)) return;
    let alive = true;
    const timer = window.setInterval(() => { void getDataQuality(analysis.id).then(row => { if (alive) setAnalysis(row); }).catch(e => { if (alive) setError(message(e)); }); }, 1000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [active, analysis?.id, analysis?.job_status]);
  function chooseSource(s: RedundancySource | undefined) {
    setSourceId(s ? String(s.id) : null);
    const time = s?.column_profiles.find(p => p.timestamp_fraction >= .8)?.name ?? null;
    setTimeColumn(time); setColumns(s?.headers.filter(c => c !== time) ?? []);
    setTypes(Object.fromEntries((s?.column_profiles ?? []).map(p => [p.name, p.numeric_fraction >= .8 ? 'numeric' : 'text'])));
    const profile = s?.column_profiles.find(p => p.name === time);
    setStart(profile?.timestamp_start ?? ''); setEnd(profile?.timestamp_end ?? '');
  }
  function chooseTime(time: string | null) {
    setTimeColumn(time); setColumns(source?.headers.filter(c => c !== time) ?? []);
    const profile = source?.column_profiles.find(p => p.name === time);
    setStart(profile?.timestamp_start ?? ''); setEnd(profile?.timestamp_end ?? '');
  }
  async function upload() {
    if (!file) return;
    setBusy(true); setError('');
    try { const next = await uploadRedundancySource(file); setSources(await listRedundancySources()); chooseSource(next); setFile(null); }
    catch (e) { setError(message(e)); } finally { setBusy(false); }
  }
  async function run(action?: 'retry' | 'cancel') {
    if (!payload) return;
    const generation = selectionGeneration.current;
    setBusy(true); setError('');
    try { const row = action && analysis ? await dataQualityAction(analysis.id, action) : await startDataQuality(payload); if (generation === selectionGeneration.current) { setAnalysis(row); setCacheHit(row.job_status === 'ready'); } }
    catch (e) { setError(message(e)); } finally { setBusy(false); }
  }
  const running = analysis && ['queued', 'running'].includes(analysis.job_status);
  return <Stack><Title order={2}>Data Quality Analysis</Title><Text c="dimmed">Inspect time coverage and sensor quality. CSV resources are shared with Redundancy Analysis.</Text>
    <Paper withBorder p="md"><Group align="end"><FileInput label="Upload CSV" accept=".csv,text/csv" value={file} onChange={setFile} /><Button disabled={!file} loading={busy} onClick={() => void upload()}>Upload</Button></Group>
      {busy && file && <Text size="sm" mt="xs">Uploading and profiling CSV…</Text>}
      <Select mt="md" label="Stored Resource" searchable value={sourceId} onChange={id => chooseSource(sources.find(s => String(s.id) === id))} data={sources.map(s => ({ value: String(s.id), label: `${s.name} · ${s.original_filename}` }))} />
      {source && <Stack mt="md"><Select label="Time column" searchable value={timeColumn} onChange={chooseTime} data={source.headers} />
        <Group><Button size="xs" variant="light" onClick={() => setColumns(source.headers.filter(c => c !== timeColumn))}>Select all sensors</Button><Button size="xs" variant="subtle" onClick={() => setColumns([])}>Deselect all</Button><Text size="sm">{columns.length} selected</Text></Group>
        <MultiSelect label="Columns included in analysis" searchable value={columns} onChange={setColumns} data={source.headers.filter(c => c !== timeColumn)} />
        <ScrollArea mah={220}><SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }}>{columns.map(c => <Select key={c} label={c} value={types[c] ?? 'text'} allowDeselect={false} data={[{ value: 'numeric', label: 'Numeric' }, { value: 'text', label: 'Text' }]} onChange={v => setTypes(t => ({ ...t, [c]: v === 'numeric' ? 'numeric' : 'text' }))} />)}</SimpleGrid></ScrollArea>
        <Group align="end"><TextInput type="datetime-local" step="any" label="Start (inclusive)" value={start} onChange={e => setStart(e.currentTarget.value)} /><TextInput type="datetime-local" step="any" label="End (inclusive)" value={end} onChange={e => setEnd(e.currentTarget.value)} /><NumberInput label="Sampling interval (seconds)" min={1} allowDecimal={false} value={interval} onChange={v => setIntervalSeconds(Number(v) || 0)} /></Group>
        <Group><Button disabled={!payload || !!running || checking || analysis?.job_status === 'ready'} loading={busy} onClick={() => void run(analysis && ['failed', 'cancelled'].includes(analysis.job_status) ? 'retry' : undefined)}>Load summary</Button>{checking && <Text size="sm">Checking saved results…</Text>}{cacheHit && <Badge color="green">Loaded saved result</Badge>}</Group>
      </Stack>}
    </Paper>
    {error && <Alert color="red">{error}</Alert>}
    {analysis && <Paper withBorder p="md"><Group justify="space-between"><Text fw={600}>{analysis.stage}</Text><Badge>{analysis.job_status}</Badge></Group>
      {running && <><Progress value={analysis.progress * 100} animated mt="sm" /><Text size="sm" mt="xs">{Math.round(analysis.progress * 100)}% · Elapsed: {Math.round(analysis.elapsed_seconds)} s · {analysis.eta_seconds == null ? 'Estimating remaining time…' : `About ${Math.ceil(analysis.eta_seconds)} s remaining`}</Text><Button mt="sm" color="orange" variant="light" onClick={() => void run('cancel')}>Cancel</Button></>}
      {analysis.error_message && <Alert color="red" mt="sm">{analysis.error_message}</Alert>}
      {!running && <Group mt="sm">{['failed', 'cancelled'].includes(analysis.job_status) && <Button loading={busy} onClick={() => void run('retry')}>Retry</Button>}<Button color="red" variant="subtle" onClick={() => { if (window.confirm('Delete this saved data quality analysis?')) void deleteDataQuality(analysis.id).then(() => { setAnalysis(null); setCacheHit(false); }).catch(e => setError(message(e))); }}>Delete analysis</Button></Group>}
    </Paper>}
    {analysis?.job_status === 'ready' && analysis.result && <Results key={analysis.id} analysis={analysis} />}
  </Stack>;
}
