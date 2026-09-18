import { useEffect, useMemo, useState } from 'react';
import { Alert, Badge, Button, Group, Loader, Pagination, Paper, Progress, ScrollArea, SegmentedControl, Select, SimpleGrid, Stack, Table, Text, TextInput, Title } from '@mantine/core';
import { characterizationAction, characterizationExportUrl, getCharacterization, getCharacterizationDetail, getCharacterizationSeries } from '../api';
import { PlotlyChart } from '../components/PlotlyChart';
import type { Data } from '../lib/plotly';
import type { CharacterizationDetail, CharacterizationRun, CharacterizationSeries, CharacterizationSummary } from '../types';
import { histogramTrace, summaryColumns, summaryRows, temporalTraces } from './characterization';

const fmt = (value: number | string | null) => value == null ? 'Not applicable' : typeof value === 'number' ? value.toLocaleString(undefined, { maximumSignificantDigits: 8 }) : value;
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);

function Detail({ id, sensor }: { id: number; sensor: string }) {
  const [detail, setDetail] = useState<CharacterizationDetail | null>(null);
  const [series, setSeries] = useState<CharacterizationSeries | null>(null);
  const [range, setRange] = useState<[string, string] | null>(null);
  const [detailError, setDetailError] = useState('');
  const [seriesError, setSeriesError] = useState('');
  const [seriesLoading, setSeriesLoading] = useState(true);
  const [absolute, setAbsolute] = useState('signed');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let alive = true;
    setDetail(null); setDetailError('');
    void getCharacterizationDetail(id, sensor).then(data => { if (alive) setDetail(data); })
      .catch(error => { if (alive) setDetailError(errorText(error)); });
    return () => { alive = false; };
  }, [id, sensor, attempt]);
  useEffect(() => {
    let alive = true;
    setSeriesLoading(true); setSeriesError('');
    const timer = window.setTimeout(() => {
      void getCharacterizationSeries(id, sensor, range).then(data => { if (alive) setSeries(data); })
        .catch(error => { if (alive) setSeriesError(errorText(error)); }).finally(() => { if (alive) setSeriesLoading(false); });
    }, 200);
    return () => { alive = false; window.clearTimeout(timer); };
  }, [id, sensor, range, attempt]);
  const traces = useMemo(() => series ? temporalTraces(series) : [], [series]);
  return <Stack>
    <Paper withBorder p="md"><Group justify="space-between"><Title order={4}>Temporal View</Title><Button size="xs" variant="light" onClick={() => setRange(null)}>Full range</Button></Group>
      <Text size="sm" c="dimmed">{series?.aggregated ? `Aggregated view (${fmt(series.interval_seconds / 60)} min): median and Q25–Q75. Zoom for original values.` : 'Original values on the expected time grid. Missing observations are not connected.'}</Text>
      {series?.aggregated && <Text size="xs" c="dimmed">Bins containing internal gaps are isolated; vertical whiskers show Q25–Q75 without bridging an outage.</Text>}
      {seriesLoading && <Group gap="xs"><Loader size="xs" /><Text size="sm">Loading time range…</Text></Group>}
      {seriesError && <Alert color="red">{seriesError}<Button variant="subtle" onClick={() => setAttempt(a => a + 1)}>Retry</Button></Alert>}
      {series && <PlotlyChart data={traces} height={380} layout={{ uirevision: `${id}:${sensor}`, xaxis: { type: 'date', ...(range ? { range } : { autorange: true }) }, yaxis: { title: { text: sensor } }, hovermode: 'closest' }} onRelayout={event => {
        const e = event as Record<string, unknown>;
        if (e['xaxis.autorange']) { setRange(null); return; }
        const pair = e['xaxis.range'] as string[] | undefined;
        const start = pair?.[0] ?? e['xaxis.range[0]'], end = pair?.[1] ?? e['xaxis.range[1]'];
        if (typeof start === 'string' && typeof end === 'string') setRange(current => current?.[0] === start && current?.[1] === end ? current : [start, end]);
      }} />}
      {!seriesLoading && series && !series.points.some(p => p.valid_n) && <Text c="dimmed">No valid observations in this time range.</Text>}
    </Paper>
    <Text size="sm" c="dimmed">Distribution and short-term dynamics use the entire analysis period, independently of the time-view zoom.</Text>
    {detailError && <Alert color="red">{detailError}<Button variant="subtle" onClick={() => setAttempt(a => a + 1)}>Retry</Button></Alert>}
    {!detail && !detailError && <Group><Loader size="sm" /><Text>Loading sensor distributions…</Text></Group>}
    {detail && <><SimpleGrid cols={{ base: 1, lg: 2 }}>
      <Paper withBorder p="md"><Title order={4}>Distribution</Title>{detail.histogram.counts.length ? <PlotlyChart data={histogramTrace(detail.histogram, sensor)} height={320} layout={{ xaxis: { title: { text: 'Value' } }, yaxis: { title: { text: 'Count' } }, bargap: .03 }} /> : <Text>No valid observations.</Text>}</Paper>
      <Paper withBorder p="md"><Title order={4}>Boxplot</Title>{detail.box ? <><PlotlyChart height={320} data={[{ type: 'box', name: sensor,
        q1: [detail.box.q25], median: [detail.box.median], q3: [detail.box.q75], lowerfence: [detail.box.lower], upperfence: [detail.box.upper], boxpoints: false } as Data]} />
        <Text size="xs" c="dimmed">Tukey whiskers (1.5 × IQR). {detail.box.outlier_count.toLocaleString()} observations outside the whiskers; all are retained in calculations.</Text></> : <Text>No valid observations.</Text>}</Paper>
    </SimpleGrid>
      <Paper withBorder p="md"><Group justify="space-between"><Title order={4}>Short-term Dynamics (Δx)</Title><SegmentedControl value={absolute} onChange={setAbsolute} data={[{ value: 'signed', label: 'Δx' }, { value: 'absolute', label: '|Δx|' }]} /></Group>
        <Text size="sm" c="dimmed">Differences between valid observations at directly adjacent grid points. % unchanged means Δx exactly zero.</Text>
        <SimpleGrid cols={{ base: 2, md: 3, xl: 6 }} my="md">{([
          ['Valid pairs', detail.dynamics.pair_count], ['Median |Δx|', detail.dynamics.median_abs], ['Q95 |Δx|', detail.dynamics.q95_abs], ['Q99 |Δx|', detail.dynamics.q99_abs], ['Max |Δx|', detail.dynamics.max_abs], ['% unchanged', detail.dynamics.unchanged_percent],
        ] as const).map(([label, value]) => <div key={label}><Text size="xs" c="dimmed">{label}</Text><Text fw={600}>{fmt(value)}</Text></div>)}</SimpleGrid>
        {detail.dynamics.non_finite_pair_count > 0 && <Alert color="orange">{detail.dynamics.non_finite_pair_count} differences exceed numeric precision and cannot be plotted.</Alert>}
        {detail.dynamics.pair_count ? <PlotlyChart height={330} data={histogramTrace(absolute === 'absolute' ? detail.absolute_delta_histogram : detail.delta_histogram, absolute === 'absolute' ? '|Δx|' : 'Δx')} layout={{ xaxis: { title: { text: absolute === 'absolute' ? '|Δx|' : 'Δx' } }, yaxis: { title: { text: 'Count' } }, bargap: .03 }} /> : <Text>No directly adjacent valid observations; dynamics are not available.</Text>}
      </Paper></>}
  </Stack>;
}

export function CharacterizationPanel({ analysisId, selectedSensor, onSelectSensor, onRunningChange }: {
  analysisId: number; selectedSensor: string | null; onSelectSensor: (sensor: string | null) => void; onRunningChange: (running: boolean) => void;
}) {
  const [run, setRun] = useState<CharacterizationRun | null>(null);
  const [loading, setLoading] = useState(true), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const [search, setSearch] = useState(''), [type, setType] = useState('all');
  const [sort, setSort] = useState<keyof CharacterizationSummary>('variable');
  const [descending, setDescending] = useState(false), [page, setPage] = useState(1), [size, setSize] = useState('20');
  const running = !!run && ['queued', 'running'].includes(run.job_status);
  useEffect(() => { onRunningChange(running); }, [running, onRunningChange]);
  useEffect(() => {
    let alive = true;
    setLoading(true); setRun(null); setError('');
    void getCharacterization(analysisId).then(row => { if (alive) setRun(row); }).catch(e => { if (alive) setError(errorText(e)); }).finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [analysisId]);
  useEffect(() => {
    if (!running) return;
    let alive = true;
    const timer = window.setInterval(() => { void getCharacterization(analysisId).then(row => { if (alive) setRun(row); }).catch(e => { if (alive) setError(errorText(e)); }); }, 1000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [analysisId, running]);
  const rows = useMemo(() => summaryRows(run?.result?.summary ?? [], search, type, sort, descending), [run, search, type, sort, descending]);
  const numeric = useMemo(() => run?.result?.summary.filter(s => s.data_type === 'numeric').map(s => s.variable) ?? [], [run]);
  useEffect(() => { if (run?.result && (!selectedSensor || !numeric.includes(selectedSensor))) onSelectSensor(numeric[0] ?? null); }, [numeric, selectedSensor, onSelectSensor, run]);
  useEffect(() => setPage(1), [search, type, sort, descending, size]);
  async function action(kind: 'start' | 'retry' | 'cancel') {
    setBusy(true); setError('');
    try { setRun(await characterizationAction(analysisId, kind)); } catch (e) { setError(errorText(e)); } finally { setBusy(false); }
  }
  return <Stack mt="md">
    <Group justify="space-between"><Title order={3}>Statistical &amp; Temporal Characterization</Title>{run?.job_status === 'ready' && <Badge color="green">Saved characterization</Badge>}</Group>
    {loading && <Loader />}{error && <Alert color="red">{error}</Alert>}
    {!loading && (!run || ['failed', 'cancelled'].includes(run.job_status)) && <Button loading={busy} onClick={() => void action(run ? 'retry' : 'start')}>{run ? 'Retry characterization' : 'Load characterization'}</Button>}
    {run?.error_message && <Alert color="red">{run.error_message}</Alert>}
    {running && <Paper withBorder p="md"><Text>{run!.stage}</Text><Progress value={run!.progress * 100} animated my="sm" /><Text size="sm">{Math.round(run!.progress * 100)}% · Elapsed: {Math.round(run!.elapsed_seconds)} s · {run!.eta_seconds == null ? 'Estimating remaining time…' : `About ${Math.ceil(run!.eta_seconds)} s remaining`}</Text><Button color="orange" variant="light" mt="sm" loading={busy} onClick={() => void action('cancel')}>Cancel characterization</Button></Paper>}
    {run?.job_status === 'ready' && run.result && <>
      <Paper withBorder p="md"><Title order={3}>Statistical Summary – All Variables</Title><Text size="sm" c="dimmed">Valid, deduplicated observations within the analysis period. N valid is an observation count.</Text>
        <Group my="md"><TextInput placeholder="Search variables…" value={search} onChange={e => setSearch(e.currentTarget.value)} /><Select aria-label="Variable type" value={type} allowDeselect={false} onChange={v => setType(v ?? 'all')} data={[{ value: 'all', label: 'All types' }, { value: 'numeric', label: 'Numeric' }, { value: 'text', label: 'Text' }]} /><Button component="a" href={characterizationExportUrl(analysisId, search, type, sort, descending)} variant="light">Export filtered CSV</Button></Group>
        <ScrollArea><Table striped highlightOnHover miw={1500}><Table.Thead><Table.Tr>{summaryColumns.map(c => <Table.Th key={c.key} aria-sort={sort === c.key ? descending ? 'descending' : 'ascending' : 'none'}><Button size="compact-xs" variant="subtle" onClick={() => { if (sort === c.key) setDescending(v => !v); else { setSort(c.key); setDescending(false); } }}>{c.label}{sort === c.key ? descending ? ' ↓' : ' ↑' : ''}</Button></Table.Th>)}</Table.Tr></Table.Thead>
          <Table.Tbody>{rows.slice((page - 1) * Number(size), page * Number(size)).map(row => <Table.Tr key={row.variable} bg={row.variable === selectedSensor ? 'var(--mantine-color-blue-light)' : undefined} onClick={() => { if (row.data_type === 'numeric') onSelectSensor(row.variable); }} style={{ cursor: row.data_type === 'numeric' ? 'pointer' : 'default' }}>{summaryColumns.map(c => <Table.Td key={c.key}>{c.key === 'variable' && row.data_type === 'numeric' ? <Button variant="subtle" size="compact-xs" onClick={() => onSelectSensor(row.variable)}>{row.variable}</Button> : fmt(row[c.key])}</Table.Td>)}</Table.Tr>)}</Table.Tbody>
        </Table></ScrollArea>{!rows.length && <Text>No matching variables.</Text>}
        <Group justify="space-between" mt="md"><Text size="sm">{rows.length} variables</Text><Pagination value={page} onChange={setPage} total={Math.max(1, Math.ceil(rows.length / Number(size)))} /><Select label="Rows per page" w={130} value={size} allowDeselect={false} onChange={v => setSize(v ?? '20')} data={['10', '20', '50']} /></Group>
      </Paper>
      <Paper withBorder p="md"><Group justify="space-between" mb="md"><Title order={3}>Variable Detail</Title><Group><Select label="Variable" searchable w={260} data={numeric} value={selectedSensor} onChange={onSelectSensor} allowDeselect={false} /><Button variant="light" aria-label="Previous variable" disabled={!selectedSensor || numeric.indexOf(selectedSensor) <= 0} onClick={() => onSelectSensor(numeric[numeric.indexOf(selectedSensor!) - 1])}>←</Button><Button variant="light" aria-label="Next variable" disabled={!selectedSensor || numeric.indexOf(selectedSensor) >= numeric.length - 1} onClick={() => onSelectSensor(numeric[numeric.indexOf(selectedSensor!) + 1])}>→</Button></Group></Group>
        {selectedSensor && numeric.includes(selectedSensor) ? <Detail key={`${analysisId}:${selectedSensor}`} id={analysisId} sensor={selectedSensor} /> : <Text c="dimmed">Detail plots are available for numeric variables only.</Text>}
      </Paper>
    </>}
  </Stack>;
}
