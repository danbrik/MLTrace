import {
  Accordion, Alert, Badge, Box, Button, Card, Collapse, FileInput, Group,
  Loader, MultiSelect, NumberInput, Paper, Progress, ScrollArea, SegmentedControl,
  Select, SimpleGrid, Slider, Stack, Table, Text, TextInput, Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Activity, Calculator, Copy, Download, Save, Trash2, Upload } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import {
  calculateRedundancyAnalysis, cancelRedundancyAnalysis, createRedundancyAnalysis,
  deleteRedundancyAnalysis, deleteRedundancySource, duplicateRedundancyAnalysis,
  finalizeRedundancyAnalysis, getFullRedundancySeries, getRedundancyAnalysis, getRedundancySeries,
  listRedundancyAnalyses, listRedundancySources, previewRedundancyClusterCut,
  redundancyExportUrl, retryRedundancyAnalysis, uploadRedundancySource,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart } from '../components/PlotlyChart';
import { buildPlotExportTable } from '../lib/plotExport';
import { withLineGapPolicy } from '../lib/plotGaps';
import type { Data } from '../lib/plotly';
import type {
  RedundancyAnalysis, RedundancyClusterCut, RedundancyConfig,
  RedundancySeries, RedundancySource,
} from '../types';
import { correlationMatrixView, filteredPairRows, INCOMPLETE_CORRELATION_MESSAGE } from '../redundancy/helpers';

const INCOMPLETE_MESSAGE = INCOMPLETE_CORRELATION_MESSAGE;
const DEFAULT_CONFIG: RedundancyConfig = {
  high_missing_fraction: 0.3,
  nearly_constant_fraction: 0.95,
  min_valid_values: 10,
  min_pair_values: 10,
  numeric_candidate_fraction: 0.8,
  missing_tokens: ['', 'na', 'n/a', 'null', 'none', 'nan'],
  linkage_method: 'average',
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown error';
}

function fmt(value: number | null | undefined, digits = 3): string {
  return value == null || !Number.isFinite(value) ? 'N/A' : value.toFixed(digits);
}

function isoLocal(value: string | null | undefined): string {
  return value ? value.slice(0, 19) : '';
}

function sourceProfile(source: RedundancySource | null, name: string) {
  return source?.column_profiles.find((item) => item.name === name) ?? null;
}

function seriesTraces(series: RedundancySeries, columns: string[], zScore: boolean): Data[] {
  const continuity = series.points.map((point) => point.continuity_segment);
  return columns.map((column) => {
    const raw = series.points.map((point) => point.values[column] ?? null);
    const finite = raw.filter((value): value is number => value != null && Number.isFinite(value));
    const mean = finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : 0;
    const variance = finite.length > 1
      ? finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (finite.length - 1)
      : 0;
    const std = Math.sqrt(variance);
    const y = zScore ? raw.map((value) => value == null ? null : (std > 0 ? (value - mean) / std : 0)) : raw;
    return withLineGapPolicy({
      type: 'scatter', mode: 'lines', name: column,
      x: series.points.map((point) => point.timestamp), y, connectgaps: false,
    } as Data, { continuity });
  });
}

function DownloadLinks({ analysisId }: { analysisId: number }) {
  const exports = [
    ['quality', 'Quality CSV'], ['pairs', 'Pairs CSV'], ['clusters', 'Clusters CSV'],
    ['correlation', 'Matrix CSV'], ['parameters', 'Parameters JSON'],
  ] as const;
  return <Group gap="xs">{exports.map(([kind, label]) => (
    <Button key={kind} component="a" href={redundancyExportUrl(analysisId, kind)} size="xs" variant="light" leftSection={<Download size={14} />}>
      {label}
    </Button>
  ))}</Group>;
}

function Results({ analysis, onRefresh }: { analysis: RedundancyAnalysis; onRefresh: (row: RedundancyAnalysis) => void }) {
  const result = analysis.result;
  const [heatmapValue, setHeatmapValue] = useState<'signed' | 'absolute'>('signed');
  const [heatmapOrder, setHeatmapOrder] = useState<'original' | 'clustered'>('original');
  const [cutoff, setCutoff] = useState(analysis.active_cutoff);
  const [cut, setCut] = useState<RedundancyClusterCut | null>(result?.cluster_cut ?? null);
  const [qualityFilter, setQualityFilter] = useState<'all' | 'excluded'>('all');
  const [qualitySort, setQualitySort] = useState<'variable' | 'missing' | 'valid'>('variable');
  const [pairThreshold, setPairThreshold] = useState(0);
  const [clusterId, setClusterId] = useState<string | null>(result?.cluster_cut.clusters[0]?.cluster_id.toString() ?? null);
  const [pairKey, setPairKey] = useState<string | null>(result?.pairs[0] ? `${result.pairs[0].variable_a}\u0000${result.pairs[0].variable_b}` : null);
  const [seriesMode, setSeriesMode] = useState<'raw' | 'z'>('raw');
  const [series, setSeries] = useState<RedundancySeries | null>(null);
  const [loadingSeries, setLoadingSeries] = useState(false);
  const [exclusionsOpen, setExclusionsOpen] = useState(false);

  useEffect(() => {
    setCut(result?.cluster_cut ?? null);
    setCutoff(analysis.active_cutoff);
  }, [analysis.active_cutoff, result]);

  const activeCluster = cut?.clusters.find((item) => item.cluster_id.toString() === clusterId) ?? cut?.clusters[0] ?? null;
  const activePair = useMemo(() => {
    if (!result || !pairKey) return null;
    const [left, right] = pairKey.split('\u0000');
    return result.pairs.find((item) => item.variable_a === left && item.variable_b === right) ?? null;
  }, [pairKey, result]);
  const detailColumns = useMemo(() => {
    const values = new Set<string>();
    activeCluster?.variables.forEach((item) => values.add(item));
    if (activePair) { values.add(activePair.variable_a); values.add(activePair.variable_b); }
    return [...values];
  }, [activeCluster, activePair]);

  useEffect(() => {
    if (!result || detailColumns.length === 0) { setSeries(null); return; }
    let cancelled = false;
    setLoadingSeries(true);
    getRedundancySeries(analysis.id, detailColumns, 8000)
      .then((value) => { if (!cancelled) setSeries(value); })
      .catch((error) => notifications.show({ color: 'red', title: 'Time series failed', message: errorMessage(error) }))
      .finally(() => { if (!cancelled) setLoadingSeries(false); });
    return () => { cancelled = true; };
  }, [analysis.id, detailColumns.join('\u0000'), result]);

  if (!result) return null;
  const excluded = result.clustering_exclusions;
  const heatmapView = correlationMatrixView(result, heatmapOrder, heatmapValue);
  const order = heatmapView.names;
  const heatmap = heatmapView.values;
  const heatmapN = heatmapView.commonN;
  const filteredQuality = result.quality
    .filter((item) => qualityFilter === 'all' || item.statuses.includes(INCOMPLETE_MESSAGE))
    .sort((left, right) => qualitySort === 'missing'
      ? right.missing_fraction - left.missing_fraction
      : qualitySort === 'valid' ? right.valid_n - left.valid_n : left.variable.localeCompare(right.variable));
  const filteredPairs = filteredPairRows(result.pairs, pairThreshold);
  const visibleSeries = series && activeCluster ? seriesTraces(series, activeCluster.variables, seriesMode === 'z') : [];
  const pairSeries = series && activePair ? seriesTraces(series, [activePair.variable_a, activePair.variable_b], seriesMode === 'z') : [];
  const pairPoints = series && activePair ? series.points.flatMap((point) => {
    const x = point.values[activePair.variable_a]; const y = point.values[activePair.variable_b];
    return x == null || y == null ? [] : [{ x, y }];
  }) : [];
  const clusterMatrix = activeCluster?.variables.map((left) => activeCluster.variables.map((right) => (
    result.spearman[result.variables.indexOf(left)][result.variables.indexOf(right)]
  ))) ?? [];

  async function applyCut(next: number) {
    setCutoff(next);
    try {
      const updated = await previewRedundancyClusterCut(analysis.id, next);
      setCut(updated);
      setClusterId(updated.clusters[0]?.cluster_id.toString() ?? null);
    } catch (error) {
      notifications.show({ color: 'red', title: 'Cutoff preview failed', message: errorMessage(error) });
    }
  }

  async function saveSnapshot() {
    try {
      const row = await finalizeRedundancyAnalysis(analysis.id, cutoff);
      onRefresh(row);
      notifications.show({ color: 'green', message: 'Immutable redundancy snapshot saved.' });
    } catch (error) {
      notifications.show({ color: 'red', title: 'Snapshot failed', message: errorMessage(error) });
    }
  }

  return <Stack gap="lg">
    <SimpleGrid cols={{ base: 2, md: 5 }}>
      <Card withBorder><Text size="xs" c="dimmed">Timepoints</Text><Text fw={700}>{result.summary.timepoint_count.toLocaleString()}</Text></Card>
      <Card withBorder><Text size="xs" c="dimmed">Variables</Text><Text fw={700}>{result.summary.numeric_variable_count}</Text></Card>
      <Card withBorder><Text size="xs" c="dimmed">Clusterable</Text><Text fw={700}>{result.summary.clusterable_variable_count}</Text></Card>
      <Card withBorder><Text size="xs" c="dimmed">Excluded</Text><Text fw={700}>{result.summary.excluded_from_clustering_count}</Text></Card>
      <Card withBorder><Text size="xs" c="dimmed">Invalid times</Text><Text fw={700}>{result.summary.invalid_time_row_count}</Text></Card>
    </SimpleGrid>

    {excluded.length > 0 && <Alert color="yellow" title={`${excluded.length} variables excluded from clustering`}>
      <Text size="sm">{INCOMPLETE_MESSAGE}</Text>
      <Button variant="subtle" size="xs" px={0} onClick={() => setExclusionsOpen((value) => !value)}>
        {exclusionsOpen ? 'Hide affected pairs' : 'Show variables and non-computable pairs'}
      </Button>
      <Collapse in={exclusionsOpen}><Stack gap={4} mt="xs">{excluded.map((item) => (
        <Text key={item.variable} size="xs"><b>{item.variable}</b>: {item.missing_with.join(', ')}</Text>
      ))}</Stack></Collapse>
    </Alert>}

    <Paper withBorder p="md">
      <Group justify="space-between" mb="sm"><Title order={4}>Quality check</Title><Group>
        <Select value={qualitySort} onChange={(value) => setQualitySort((value ?? 'variable') as 'variable' | 'missing' | 'valid')} allowDeselect={false}
          data={[{ value: 'variable', label: 'Sort: Variable' }, { value: 'missing', label: 'Sort: Missing' }, { value: 'valid', label: 'Sort: Valid N' }]} />
        <Select value={qualityFilter} onChange={(value) => setQualityFilter(value === 'excluded' ? 'excluded' : 'all')} allowDeselect={false}
          data={[{ value: 'all', label: 'All variables' }, { value: 'excluded', label: 'Excluded from clustering' }]} />
      </Group></Group>
      <ScrollArea><Table striped highlightOnHover miw={1000}><Table.Thead><Table.Tr>
        {['Variable', 'Valid', 'Missing', 'Mean', 'Median', 'Sample std', 'Min', 'Max', 'Unique', 'Status'].map((name) => <Table.Th key={name}>{name}</Table.Th>)}
      </Table.Tr></Table.Thead><Table.Tbody>{filteredQuality.map((item) => <Table.Tr key={item.variable}>
        <Table.Td>{item.variable}</Table.Td><Table.Td>{item.valid_n}</Table.Td><Table.Td>{item.missing_n} · {(item.missing_fraction * 100).toFixed(1)}%</Table.Td>
        <Table.Td>{fmt(item.mean)}</Table.Td><Table.Td>{fmt(item.median)}</Table.Td><Table.Td>{fmt(item.std)}</Table.Td>
        <Table.Td>{fmt(item.min)}</Table.Td><Table.Td>{fmt(item.max)}</Table.Td><Table.Td>{item.unique_n}</Table.Td>
        <Table.Td><Stack gap={2}>{item.statuses.map((status) => <Badge key={status} color={status === 'OK' ? 'green' : status === INCOMPLETE_MESSAGE ? 'yellow' : 'orange'} variant="light">{status}</Badge>)}</Stack></Table.Td>
      </Table.Tr>)}</Table.Tbody></Table></ScrollArea>
    </Paper>

    <Paper withBorder p="md">
      <Group justify="space-between" mb="sm"><Title order={4}>Correlation matrix</Title><Group>
        <SegmentedControl value={heatmapValue} onChange={(value) => setHeatmapValue(value as 'signed' | 'absolute')} data={[{ value: 'signed', label: 'Signed' }, { value: 'absolute', label: 'Absolute' }]} />
        <SegmentedControl value={heatmapOrder} onChange={(value) => setHeatmapOrder(value as 'original' | 'clustered')} data={[{ value: 'original', label: 'Original order' }, { value: 'clustered', label: 'Clustered order' }]} />
      </Group></Group>
      {heatmapOrder === 'clustered' && excluded.length > 0 && <Alert color="yellow" mb="sm">
        {excluded.length} variables are not in clustered order. {INCOMPLETE_MESSAGE} <Button variant="subtle" size="compact-xs" onClick={() => setExclusionsOpen(true)}>View list</Button>
      </Alert>}
      <PlotlyChart height={Math.max(420, order.length * 16)} data={[{
        type: 'heatmap', name: heatmapValue === 'signed' ? 'Spearman ρ' : '|Spearman ρ|', x: order, y: order, z: heatmap,
        customdata: heatmapN, zmin: heatmapValue === 'signed' ? -1 : 0, zmax: 1,
        colorscale: heatmapValue === 'signed' ? 'RdBu' : 'Viridis', reversescale: heatmapValue === 'signed',
        hovertemplate: '%{y} × %{x}<br>ρ=%{z:.4f}<br>common N=%{customdata}<extra></extra>',
      } as Data]} layout={{ margin: { l: 120, b: 120, r: 30, t: 10 } }} />
      <Text size="xs" c="dimmed">N/A cells have insufficient pairwise-complete observations or a constant variable. Hover reports the shared N.</Text>
    </Paper>

    <Paper withBorder p="md">
      <Title order={4}>Hierarchical clustering</Title>
      <Text size="sm" c="dimmed" mb="md">Distance is 1 − |ρ| with SciPy average linkage and optimal leaf ordering. The average-linkage cutoff does not guarantee that every individual pair inside a cluster exceeds the cutoff.</Text>
      <Text size="sm" fw={600}>Correlation cutoff |ρ| = {cutoff.toFixed(3)} (distance {(1 - cutoff).toFixed(3)})</Text>
      <Slider min={0} max={1} step={0.01} value={cutoff} onChange={setCutoff} onChangeEnd={(value) => void applyCut(value)} marks={[{ value: 0.5, label: '0.50' }, { value: 0.9, label: '0.90' }, { value: 1, label: '1.00' }]} mb={32} />
      {result.dendrogram.icoord.length > 0 ? <PlotlyChart height={360} data={result.dendrogram.icoord.map((x, index) => ({
        type: 'scatter', mode: 'lines', x, y: result.dendrogram.dcoord[index], line: { color: '#228be6' }, hoverinfo: 'skip', showlegend: false,
      } as Data))} layout={{ xaxis: { tickmode: 'array', tickvals: result.dendrogram.labels.map((_, index) => 5 + index * 10), ticktext: result.dendrogram.labels }, yaxis: { title: { text: '1 − |ρ|' } }, shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1 - cutoff, y1: 1 - cutoff, line: { color: 'red', dash: 'dash' } }] }} /> : <Text c="dimmed">A dendrogram requires at least two clusterable variables.</Text>}
      <SimpleGrid cols={{ base: 1, md: 2 }} mt="md">
        <Box><Select label="Cluster detail" value={clusterId} onChange={setClusterId} data={(cut?.clusters ?? []).map((item) => ({ value: item.cluster_id.toString(), label: `Cluster ${item.cluster_id} · ${item.variable_count} variables` }))} />
          {activeCluster && <Table mt="sm"><Table.Tbody>
            <Table.Tr><Table.Th>Variables</Table.Th><Table.Td>{activeCluster.variables.join(', ')}</Table.Td></Table.Tr>
            <Table.Tr><Table.Th>Mean |ρ|</Table.Th><Table.Td>{fmt(activeCluster.mean_abs_rho)}</Table.Td></Table.Tr>
            <Table.Tr><Table.Th>Min / Max |ρ|</Table.Th><Table.Td>{fmt(activeCluster.min_abs_rho)} / {fmt(activeCluster.max_abs_rho)}</Table.Td></Table.Tr>
          </Table.Tbody></Table>}
        </Box>
        <ScrollArea h={210}><Table striped><Table.Thead><Table.Tr><Table.Th>Cluster</Table.Th><Table.Th>N</Table.Th><Table.Th>Mean |ρ|</Table.Th></Table.Tr></Table.Thead><Table.Tbody>{(cut?.clusters ?? []).map((item) => <Table.Tr key={item.cluster_id} onClick={() => setClusterId(item.cluster_id.toString())} style={{ cursor: 'pointer' }}><Table.Td>{item.cluster_id}</Table.Td><Table.Td>{item.variable_count}</Table.Td><Table.Td>{fmt(item.mean_abs_rho)}</Table.Td></Table.Tr>)}</Table.Tbody></Table></ScrollArea>
      </SimpleGrid>
    </Paper>

    {activeCluster && <Paper withBorder p="md">
      <Group justify="space-between"><Title order={4}>Cluster time series</Title><SegmentedControl value={seriesMode} onChange={(value) => setSeriesMode(value as 'raw' | 'z')} data={[{ value: 'raw', label: 'Raw' }, { value: 'z', label: 'Z-score' }]} /></Group>
      {loadingSeries ? <Loader my="xl" /> : series && <><Badge variant="light" my="xs">Preview {series.points.length.toLocaleString()} / {series.total.toLocaleString()} points</Badge><PlotlyChart data={visibleSeries} rescaleYOnVisibleX height={430} layout={{ hovermode: 'x unified', xaxis: { type: 'date', rangeslider: { visible: true } }, yaxis: { title: { text: seriesMode === 'z' ? 'Z-score' : 'Value' } } }} fullResolutionExport={async () => buildPlotExportTable(seriesTraces(await getFullRedundancySeries(analysis.id, activeCluster.variables), activeCluster.variables, seriesMode === 'z'))} /></>}
      <Title order={5} mt="md">Within-cluster correlation</Title>
      <PlotlyChart height={Math.max(300, activeCluster.variables.length * 24)} data={[{ type: 'heatmap', name: 'Spearman ρ', x: activeCluster.variables, y: activeCluster.variables, z: clusterMatrix, zmin: -1, zmax: 1, colorscale: 'RdBu', reversescale: true, hovertemplate: '%{y} × %{x}<br>ρ=%{z:.4f}<extra></extra>' } as Data]} layout={{ margin: { l: 110, b: 90, t: 10, r: 20 } }} />
    </Paper>}

    <Paper withBorder p="md">
      <Group justify="space-between"><Title order={4}>Pairwise detail</Title><Text size="sm">Show pairs with |ρ| ≥ {pairThreshold.toFixed(2)}</Text></Group>
      <Slider value={pairThreshold} onChange={setPairThreshold} min={0} max={1} step={0.01} mb="md" />
      <Select searchable label="Selected pair" value={pairKey} onChange={setPairKey} data={filteredPairs.map((item) => ({ value: `${item.variable_a}\u0000${item.variable_b}`, label: `${item.variable_a} × ${item.variable_b} · |ρ| ${item.absolute_rho.toFixed(3)}` }))} />
      {activePair && <>
        <SimpleGrid cols={{ base: 1, md: 3 }} my="sm"><Card withBorder><Text size="xs">Spearman ρ</Text><Text fw={700}>{fmt(activePair.spearman_rho, 4)}</Text></Card><Card withBorder><Text size="xs">Pearson r</Text><Text fw={700}>{fmt(activePair.pearson_r, 4)}</Text></Card><Card withBorder><Text size="xs">Common N</Text><Text fw={700}>{activePair.common_n}</Text></Card></SimpleGrid>
        {series && <SimpleGrid cols={{ base: 1, xl: 2 }}><PlotlyChart data={pairSeries} rescaleYOnVisibleX height={360} layout={{ xaxis: { type: 'date' }, hovermode: 'x unified' }} fullResolutionExport={async () => buildPlotExportTable(seriesTraces(await getFullRedundancySeries(analysis.id, [activePair.variable_a, activePair.variable_b]), [activePair.variable_a, activePair.variable_b], seriesMode === 'z'))} /><PlotlyChart height={360} data={[{ type: 'scatter', mode: 'markers', name: `${activePair.variable_a} × ${activePair.variable_b}`, x: pairPoints.map((item) => item.x), y: pairPoints.map((item) => item.y), marker: { size: 5, opacity: 0.55 } } as Data]} layout={{ xaxis: { title: { text: activePair.variable_a } }, yaxis: { title: { text: activePair.variable_b } } }} /></SimpleGrid>}
      </>}
      <ScrollArea h={280} mt="md"><Table striped highlightOnHover><Table.Thead><Table.Tr><Table.Th>Variable A</Table.Th><Table.Th>Variable B</Table.Th><Table.Th>Spearman ρ</Table.Th><Table.Th>|ρ|</Table.Th><Table.Th>N</Table.Th></Table.Tr></Table.Thead><Table.Tbody>{filteredPairs.map((item) => <Table.Tr key={`${item.variable_a}\u0000${item.variable_b}`} onClick={() => setPairKey(`${item.variable_a}\u0000${item.variable_b}`)} style={{ cursor: 'pointer' }}><Table.Td>{item.variable_a}</Table.Td><Table.Td>{item.variable_b}</Table.Td><Table.Td>{fmt(item.spearman_rho, 4)}</Table.Td><Table.Td>{fmt(item.absolute_rho, 4)}</Table.Td><Table.Td>{item.common_n}</Table.Td></Table.Tr>)}</Table.Tbody></Table></ScrollArea>
    </Paper>

    <Group justify="space-between"><DownloadLinks analysisId={analysis.id} />{analysis.status === 'draft' && <Button leftSection={<Save size={16} />} onClick={() => void saveSnapshot()}>Save snapshot</Button>}</Group>
  </Stack>;
}

export function RedundancyAnalysisPage({ active }: { active: boolean }) {
  const [sources, setSources] = useState<RedundancySource[]>([]);
  const [analyses, setAnalyses] = useState<RedundancyAnalysis[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [timeColumn, setTimeColumn] = useState<string | null>(null);
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [variables, setVariables] = useState<string[]>([]);
  const [analysisName, setAnalysisName] = useState('Redundancy analysis');
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [busy, setBusy] = useState(false);

  const selectedSource = sources.find((item) => item.id.toString() === sourceId) ?? null;
  const selectedAnalysis = analyses.find((item) => item.id === selectedId) ?? null;
  const timeOptions = (selectedSource?.column_profiles ?? []).filter((item) => item.timestamp_fraction >= 0.8).map((item) => item.name);
  const numericOptions = (selectedSource?.column_profiles ?? []).filter((item) => item.numeric_fraction >= config.numeric_candidate_fraction && item.name !== timeColumn).map((item) => item.name);

  async function refresh(select?: number) {
    const [nextSources, nextAnalyses] = await Promise.all([listRedundancySources(), listRedundancyAnalyses()]);
    setSources(nextSources); setAnalyses(nextAnalyses);
    if (select !== undefined) setSelectedId(select);
  }

  useEffect(() => { if (active) void refresh().catch((error) => notifications.show({ color: 'red', message: errorMessage(error) })); }, [active]);
  useEffect(() => {
    if (!active || !selectedAnalysis || !['queued', 'running'].includes(selectedAnalysis.job_status)) return undefined;
    const timer = window.setInterval(() => {
      void getRedundancyAnalysis(selectedAnalysis.id).then((row) => {
        setAnalyses((current) => current.map((item) => item.id === row.id ? row : item));
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [active, selectedAnalysis?.id, selectedAnalysis?.job_status]);

  function chooseSource(value: string | null) {
    setSourceId(value); setVariables([]);
    const source = sources.find((item) => item.id.toString() === value) ?? null;
    const time = source?.column_profiles.find((item) => item.timestamp_fraction >= 0.8)?.name ?? null;
    setTimeColumn(time);
    const profile = sourceProfile(source, time ?? '');
    setStart(isoLocal(profile?.timestamp_start)); setEnd(isoLocal(profile?.timestamp_end));
  }

  function chooseTime(value: string | null) {
    setTimeColumn(value); setVariables((current) => current.filter((item) => item !== value));
    const profile = sourceProfile(selectedSource, value ?? '');
    setStart(isoLocal(profile?.timestamp_start)); setEnd(isoLocal(profile?.timestamp_end));
  }

  async function upload() {
    if (!file) return;
    setUploading(true);
    try {
      const source = await uploadRedundancySource(file, uploadName || undefined);
      await refresh();
      setSourceId(source.id.toString());
      setVariables([]);
      const time = source.column_profiles.find((item) => item.timestamp_fraction >= 0.8)?.name ?? null;
      setTimeColumn(time);
      const profile = sourceProfile(source, time ?? '');
      setStart(isoLocal(profile?.timestamp_start)); setEnd(isoLocal(profile?.timestamp_end));
      setFile(null); setUploadName('');
      notifications.show({ color: 'green', message: `Stored ${source.original_filename}.` });
    } catch (error) { notifications.show({ color: 'red', title: 'Upload failed', message: errorMessage(error) }); }
    finally { setUploading(false); }
  }

  async function calculate() {
    if (!selectedSource || !timeColumn || !start || !end || variables.length < 2) return;
    setBusy(true);
    try {
      const created = await createRedundancyAnalysis({ source_id: selectedSource.id, name: analysisName, time_column: timeColumn, start_timestamp: start, end_timestamp: end, selected_columns: variables, config, active_cutoff: 0.9 });
      const queued = await calculateRedundancyAnalysis(created.id);
      setAnalyses((current) => [queued, ...current.filter((item) => item.id !== queued.id)]); setSelectedId(queued.id);
    } catch (error) { notifications.show({ color: 'red', title: 'Calculation failed', message: errorMessage(error) }); }
    finally { setBusy(false); }
  }

  async function removeAnalysis(id: number) {
    if (!window.confirm('Delete this redundancy analysis?')) return;
    try { await deleteRedundancyAnalysis(id); setSelectedId((value) => value === id ? null : value); await refresh(); }
    catch (error) { notifications.show({ color: 'red', title: 'Delete failed', message: errorMessage(error) }); }
  }

  if (!active) return null;
  return <Stack gap="lg">
    <Box><Title order={2}>Redundancy Analysis</Title><Text c="dimmed">Diagnostic evidence for redundant variables in project-local multivariate CSV time series. MLTrace never removes or selects variables automatically.</Text></Box>

    <Accordion defaultValue="source"><Accordion.Item value="source"><Accordion.Control icon={<Upload size={18} />}>1 · Upload or reuse a CSV source</Accordion.Control><Accordion.Panel>
      <SimpleGrid cols={{ base: 1, md: 3 }}><FileInput label="CSV file" accept=".csv,text/csv" value={file} onChange={setFile} clearable /><TextInput label="Stored name (optional)" value={uploadName} onChange={(event) => setUploadName(event.currentTarget.value)} /><Button mt={25} loading={uploading} disabled={!file} onClick={() => void upload()}>Upload immutable source</Button></SimpleGrid>
      <Select mt="md" searchable label="Stored source" placeholder="Select a project CSV" value={sourceId} onChange={chooseSource} data={sources.map((item) => ({ value: item.id.toString(), label: `${item.name} · ${item.row_count.toLocaleString()} rows` }))} />
      {selectedSource && <><Group mt="xs"><Badge>{selectedSource.row_count.toLocaleString()} rows</Badge><Badge variant="light">{selectedSource.headers.length} columns</Badge><Text size="xs" c="dimmed">SHA-256 {selectedSource.sha256.slice(0, 12)}…</Text><Button ml="auto" size="xs" color="red" variant="subtle" leftSection={<Trash2 size={14} />} onClick={async () => { if (!window.confirm('Delete this stored CSV source?')) return; try { await deleteRedundancySource(selectedSource.id); chooseSource(null); await refresh(); } catch (error) { notifications.show({ color: 'red', message: errorMessage(error) }); } }}>Delete source</Button></Group>
        <ScrollArea mt="sm"><Table striped miw={700}><Table.Thead><Table.Tr>{selectedSource.headers.map((header) => <Table.Th key={header}>{header}</Table.Th>)}</Table.Tr></Table.Thead><Table.Tbody>{selectedSource.preview_rows.slice(0, 10).map((row, index) => <Table.Tr key={index}>{row.map((value, column) => <Table.Td key={column}>{value ?? <Text c="dimmed">N/A</Text>}</Table.Td>)}</Table.Tr>)}</Table.Tbody></Table></ScrollArea></>}
    </Accordion.Panel></Accordion.Item>

    <Accordion.Item value="configure"><Accordion.Control icon={<Activity size={18} />}>2 · Configure a diagnostic calculation</Accordion.Control><Accordion.Panel>
      {!selectedSource ? <Alert color="blue">Select a stored source first.</Alert> : <Stack>
        <SimpleGrid cols={{ base: 1, md: 2 }}><TextInput label="Analysis name" value={analysisName} onChange={(event) => setAnalysisName(event.currentTarget.value)} /><Select searchable label="Dataset-local time column" value={timeColumn} onChange={chooseTime} data={timeOptions} /></SimpleGrid>
        <SimpleGrid cols={{ base: 1, md: 2 }}><DateTime24Input label="Inclusive start" value={start} onChange={setStart} /><DateTime24Input label="Inclusive end" value={end} onChange={setEnd} /></SimpleGrid>
        <MultiSelect searchable label="Numeric variables" description={`Columns need at least ${(config.numeric_candidate_fraction * 100).toFixed(0)}% parseable non-empty values to be suggested.`} value={variables} onChange={setVariables} data={numericOptions} />
        <Paper withBorder p="sm"><Text fw={600} mb="xs">Quality-control parameters</Text><SimpleGrid cols={{ base: 1, sm: 2, xl: 5 }}>
          <NumberInput label="High missingness (%)" min={0} max={100} value={config.high_missing_fraction * 100} onChange={(value) => setConfig((current) => ({ ...current, high_missing_fraction: Number(value) / 100 }))} />
          <NumberInput label="Nearly constant (%)" min={50} max={100} value={config.nearly_constant_fraction * 100} onChange={(value) => setConfig((current) => ({ ...current, nearly_constant_fraction: Number(value) / 100 }))} />
          <NumberInput label="Min values / variable" min={3} value={config.min_valid_values} onChange={(value) => setConfig((current) => ({ ...current, min_valid_values: Number(value) }))} />
          <NumberInput label="Min values / pair" min={3} value={config.min_pair_values} onChange={(value) => setConfig((current) => ({ ...current, min_pair_values: Number(value) }))} />
          <NumberInput label="Numeric candidate (%)" min={0} max={100} value={config.numeric_candidate_fraction * 100} onChange={(value) => setConfig((current) => ({ ...current, numeric_candidate_fraction: Number(value) / 100 }))} />
        </SimpleGrid></Paper>
        <Button leftSection={<Calculator size={16} />} loading={busy} disabled={!analysisName.trim() || !timeColumn || !start || !end || variables.length < 2} onClick={() => void calculate()}>Calculate in background</Button>
      </Stack>}
    </Accordion.Panel></Accordion.Item></Accordion>

    <Paper withBorder p="md"><Title order={4} mb="sm">Saved analyses and drafts</Title>
      {analyses.length === 0 ? <Text c="dimmed">No redundancy analysis yet.</Text> : <ScrollArea><Table striped highlightOnHover><Table.Thead><Table.Tr><Table.Th>Name</Table.Th><Table.Th>Source</Table.Th><Table.Th>Status</Table.Th><Table.Th>Progress</Table.Th><Table.Th>Created</Table.Th><Table.Th /></Table.Tr></Table.Thead><Table.Tbody>{analyses.map((item) => <Table.Tr key={item.id} bg={selectedId === item.id ? 'var(--mantine-color-blue-light)' : undefined} onClick={() => setSelectedId(item.id)} style={{ cursor: 'pointer' }}><Table.Td>{item.name}</Table.Td><Table.Td>{sources.find((source) => source.id === item.source_id)?.name ?? item.source_id}</Table.Td><Table.Td><Badge color={item.status === 'finalized' ? 'green' : item.job_status === 'failed' ? 'red' : 'blue'}>{item.status === 'finalized' ? 'snapshot' : item.job_status}</Badge></Table.Td><Table.Td>{['queued', 'running'].includes(item.job_status) ? <Progress value={item.progress * 100} animated size="sm" w={140} /> : '—'}</Table.Td><Table.Td>{new Date(item.created_at).toLocaleString()}</Table.Td><Table.Td><Group wrap="nowrap" gap={4}>{['failed', 'cancelled', 'stale'].includes(item.job_status) && item.status === 'draft' && <Button size="compact-xs" onClick={(event) => { event.stopPropagation(); void retryRedundancyAnalysis(item.id).then((row) => { setAnalyses((current) => current.map((value) => value.id === row.id ? row : value)); setSelectedId(row.id); }); }}>Retry</Button>}{['queued', 'running'].includes(item.job_status) && <Button size="compact-xs" color="orange" onClick={(event) => { event.stopPropagation(); void cancelRedundancyAnalysis(item.id); }}>Cancel</Button>}<Button size="compact-xs" variant="subtle" onClick={(event) => { event.stopPropagation(); void duplicateRedundancyAnalysis(item.id).then((row) => refresh(row.id)); }}><Copy size={14} /></Button><Button size="compact-xs" color="red" variant="subtle" onClick={(event) => { event.stopPropagation(); void removeAnalysis(item.id); }}><Trash2 size={14} /></Button></Group></Table.Td></Table.Tr>)}</Table.Tbody></Table></ScrollArea>}
    </Paper>

    {selectedAnalysis && <Paper withBorder p="md"><Group justify="space-between"><Box><Title order={3}>{selectedAnalysis.name}</Title><Text size="sm" c="dimmed">{selectedAnalysis.start_timestamp} — {selectedAnalysis.end_timestamp} · {selectedAnalysis.selected_columns.length} variables</Text></Box><Badge size="lg">{selectedAnalysis.status === 'finalized' ? 'Immutable snapshot' : selectedAnalysis.job_status}</Badge></Group>
      {selectedAnalysis.error_message && <Alert color="red" mt="md" title="Calculation failed">{selectedAnalysis.error_message}</Alert>}
      {['queued', 'running'].includes(selectedAnalysis.job_status) && <Stack mt="md"><Progress value={selectedAnalysis.progress * 100} animated /><Text size="sm">CPU calculation: {(selectedAnalysis.progress * 100).toFixed(0)}%</Text></Stack>}
      {selectedAnalysis.job_status === 'stale' && <Alert color="yellow" mt="md">Inputs changed. Recalculate before saving a snapshot.</Alert>}
      {selectedAnalysis.job_status === 'ready' && <Box mt="lg"><Results analysis={selectedAnalysis} onRefresh={(row) => setAnalyses((current) => current.map((item) => item.id === row.id ? row : item))} /></Box>}
    </Paper>}
  </Stack>;
}
