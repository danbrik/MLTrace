import { Alert, Badge, Button, Group, Paper, Progress, Select, SimpleGrid, Stack, Switch, Table, Text, Title } from '@mantine/core';
import { Download, Eye, RefreshCw } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { listTestingRuns } from '../api';
import { InferenceSourceSelector } from '../components/InferenceSourceSelector';
import { PlotlyChart } from '../components/PlotlyChart';
import { DEFAULT_TABLE_PAGE_SIZE, TablePagination } from '../components/TablePagination';
import type { ExportOptions, SamplingMethod, SamplingRate } from '../inferenceExport/export';
import { loadInferenceExport } from '../inferenceExport/load';
import { inferenceExportPlotData } from '../inferenceExport/plot';
import type { PlotExportTable } from '../lib/plotExport';
import { tabularTableToCsv } from '../lib/tabularExport';
import type { TestingRun } from '../types';

export function InferenceExportPage({ active, projectId }: { active: boolean; projectId: string }) {
  const [runs, setRuns] = useState<TestingRun[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [options, setOptions] = useState<ExportOptions>({ samplingRate: 'original', method: 'mean', utc: false });
  const [table, setTable] = useState<PlotExportTable | null>(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [completed, setCompleted] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);
  const pending = useRef<AbortController | null>(null);

  const invalidate = useCallback(() => {
    generation.current += 1;
    pending.current?.abort();
    setTable(null);
    setError(null);
    setPage(1);
  }, []);

  const refresh = useCallback(async () => {
    invalidate();
    const request = generation.current;
    setRefreshing(true);
    setLoading(false);
    try {
      const available = (await listTestingRuns()).filter((run) => run.status === 'finished');
      if (request !== generation.current) return;
      setRuns(available);
      setSelectedIds((ids) => ids.filter((id) => available.some((run) => String(run.id) === id)));
    } catch (caught) {
      if (request === generation.current) setError(caught instanceof Error ? caught.message : 'Could not load inference sources.');
    } finally {
      if (request === generation.current) setRefreshing(false);
    }
  }, [invalidate]);

  useEffect(() => {
    if (active) void refresh();
    return () => {
      generation.current += 1;
      pending.current?.abort();
    };
  }, [active, refresh]);

  const preview = async () => {
    invalidate();
    const request = generation.current;
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true);
    setCompleted(0);
    try {
      const selected = selectedIds.map((id) => {
        const run = runs.find((item) => String(item.id) === id);
        if (!run) throw new Error(`Inference #${id} is no longer available. Refresh the source list.`);
        return run;
      });
      const result = await loadInferenceExport(selected, options, projectId, controller.signal, (count) => {
        if (request === generation.current) setCompleted(count);
      });
      if (request === generation.current) setTable(result);
    } catch (caught) {
      if (request === generation.current) setError(caught instanceof Error ? caught.message : 'Could not prepare the export.');
    } finally {
      if (request === generation.current) setLoading(false);
    }
  };

  const updateOptions = (update: Partial<ExportOptions>) => {
    invalidate();
    setOptions((current) => ({ ...current, ...update }));
  };
  const download = () => {
    if (!table) return;
    try {
      const url = URL.createObjectURL(new Blob([tabularTableToCsv(table)], { type: 'text/csv;charset=utf-8' }));
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `inference-export-${options.utc ? 'utc' : 'berlin'}-${options.samplingRate}${options.samplingRate === 'original' ? '' : `-${options.method}`}.csv`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not download the CSV.');
    }
  };
  const plotData = useMemo(() => table ? inferenceExportPlotData(table) : [], [table]);
  const busy = loading || refreshing;
  const firstRow = (page - 1) * DEFAULT_TABLE_PAGE_SIZE;

  return <Stack gap="lg">
    <Group justify="space-between">
      <div><Title order={2}>Inference CSV Export</Title><Text c="dimmed">Export the configured scores of finished inferences in one CSV.</Text></div>
      <Button variant="subtle" leftSection={<RefreshCw size={16} />} disabled={busy} onClick={() => void refresh()}>Refresh</Button>
    </Group>
    <InferenceSourceSelector runs={runs} selectedIds={selectedIds} multiple disabled={busy} onSelectionChange={(ids) => { invalidate(); setSelectedIds(ids); }} />
    <Paper withBorder p="md"><Stack gap="md">
      <Text fw={700}>2. Export settings</Text>
      <SimpleGrid cols={{ base: 1, sm: 2 }}>
        <Select label="Sampling rate" value={options.samplingRate} disabled={busy} allowDeselect={false}
          data={[{ value: 'original', label: 'Original' }, { value: '1min', label: '1 minute' }, { value: '5min', label: '5 minutes' }]}
          onChange={(value) => { if (value) updateOptions({ samplingRate: value as SamplingRate }); }} />
        <Select label="Sampling method" value={options.method} disabled={busy || options.samplingRate === 'original'} allowDeselect={false}
          data={['first', 'mean', 'median', 'min', 'max']} onChange={(value) => { if (value) updateOptions({ method: value as SamplingMethod }); }} />
      </SimpleGrid>
      <Switch label="Export in UTC" description="Convert all timestamps from Europe/Berlin to UTC, including daylight-saving time." checked={options.utc} disabled={busy} onChange={(event) => updateOptions({ utc: event.currentTarget.checked })} />
      <Text size="sm" c="dimmed">Entire stored period · {options.utc ? 'UTC' : 'Europe/Berlin local time'} · timestamp format: 2025.09.15 22:00:00 · comma-separated columns and unquoted numbers with a decimal point.</Text>
      <Group>
        <Button leftSection={<Eye size={16} />} disabled={!selectedIds.length || refreshing} loading={loading} onClick={() => void preview()}>Create preview</Button>
        <Button leftSection={<Download size={16} />} disabled={!table || busy} onClick={download}>Download CSV</Button>
      </Group>
      {loading && <Stack gap="xs"><Progress value={selectedIds.length ? completed / selectedIds.length * 100 : 0} animated /><Text size="sm">Loading complete inference data: {completed} / {selectedIds.length}</Text></Stack>}
      {refreshing && <Text size="sm">Loading inference sources…</Text>}
      {!table && !busy && selectedIds.length > 0 && !error && <Text size="sm" c="dimmed">Create a preview with the current selection and settings to enable download.</Text>}
      {error && <Alert color="red" title="Export could not be prepared">{error}</Alert>}
    </Stack></Paper>
    {table && <Paper withBorder p="md"><Stack gap="md">
      <Group><Text fw={700}>3. Export preview</Text><Badge>{table.rowCount.toLocaleString()} rows</Badge><Badge>{table.seriesCount} inferences</Badge><Badge>{options.utc ? 'UTC' : 'Europe/Berlin'}</Badge></Group>
      <PlotlyChart data={plotData} layout={{ xaxis: { type: 'date', title: { text: options.utc ? 'Timestamp (UTC)' : 'Timestamp (Europe/Berlin)' } }, yaxis: { title: { text: 'Configured score' } }, showlegend: true }} height={360} rescaleYOnVisibleX fullResolutionExport={async () => table} />
      <Text size="xs" c="dimmed">The chart may show fewer points. The table and CSV contain every export row. Missing values remain empty.</Text>
      <Table.ScrollContainer minWidth={Math.max(600, table.columns.length * 180)} maxHeight={440}>
        <Table striped withTableBorder withColumnBorders stickyHeader>
          <Table.Thead><Table.Tr>{table.columns.map((column) => <Table.Th key={column.name}>{column.name}</Table.Th>)}</Table.Tr></Table.Thead>
          <Table.Tbody>{Array.from({ length: Math.min(DEFAULT_TABLE_PAGE_SIZE, table.rowCount - firstRow) }, (_, offset) => <Table.Tr key={firstRow + offset}>
            {table.columns.map((column) => <Table.Td key={column.name}>{column.values[firstRow + offset] === null ? '' : String(column.values[firstRow + offset])}</Table.Td>)}
          </Table.Tr>)}</Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <TablePagination totalItems={table.rowCount} page={page} onChange={setPage} />
    </Stack></Paper>}
  </Stack>;
}
