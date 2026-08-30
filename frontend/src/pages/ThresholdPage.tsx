import {
  Alert, Badge, Box, Button, Card, FileInput, Group, MultiSelect, NumberInput,
  Paper, SegmentedControl, Select, SimpleGrid, Stack, Table, Text, Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Calculator, Plus, RotateCcw, Trash2, Upload } from 'lucide-react';
import { useMemo, useRef, useState } from 'react';

import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart, type PlotlyChartSelection } from '../components/PlotlyChart';
import { parseCsvFile, type CsvDocument } from '../csvMerge/csvMerge';
import { withLineGapPolicy } from '../lib/plotGaps';
import type { Data, Layout } from '../lib/plotly';
import {
  buildThresholdDataset, calculateQuantileThreshold, canonicalSelectionRange, decimateThresholdRows,
  numericCandidateColumns, thresholdAxisPlan, thresholdExportTable,
  timestampCandidateColumns, unionThresholdIntervals,
  type QuantileThresholdResult, type ThresholdScope,
} from '../threshold/threshold';

type ThresholdEntry = {
  id: string;
  column: string;
  quantile: number;
  scope: ThresholdScope;
  referenceStart: string;
  referenceEnd: string;
  stale: boolean;
  result: QuantileThresholdResult | null;
};

function identifier(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `threshold-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function resultValue(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? 'N/A' : value.toPrecision(8).replace(/0+$/, '').replace(/\.$/, '');
}

export function ThresholdPage({ active }: { active: boolean }) {
  const [file, setFile] = useState<File | null>(null);
  const [document, setDocument] = useState<CsvDocument | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [timeColumn, setTimeColumn] = useState<string | null>(null);
  const [graphColumns, setGraphColumns] = useState<string[]>([]);
  const [thresholds, setThresholds] = useState<ThresholdEntry[]>([]);
  const [selectionThresholdId, setSelectionThresholdId] = useState<string | null>(null);
  const loadId = useRef(0);

  const timeCandidates = useMemo(() => document ? timestampCandidateColumns(document) : [], [document]);
  const numericCandidates = useMemo(() => document ? numericCandidateColumns(document, timeColumn) : [], [document, timeColumn]);
  const dataset = useMemo(() => document && timeColumn
    ? buildThresholdDataset(document, timeColumn, graphColumns)
    : null, [document, graphColumns, timeColumn]);
  const previewRows = useMemo(() => dataset ? decimateThresholdRows(dataset.rows, graphColumns, 8000) : [], [dataset, graphColumns]);
  const axisPlan = useMemo(() => thresholdAxisPlan(graphColumns), [graphColumns]);
  const calculated = thresholds.filter((item) => item.result && !item.stale);
  const redIntervals = unionThresholdIntervals(calculated.flatMap((item) => item.result?.intervals ?? []));

  const data = useMemo<Data[]>(() => {
    if (!dataset) return [];
    const continuity = previewRows.map((row) => row.continuitySegment);
    const graphTraces = axisPlan.axes.map((axis) => withLineGapPolicy({
      type: 'scatter', mode: 'lines', name: axis.column,
      x: previewRows.map((row) => row.timestamp),
      y: previewRows.map((row) => row.values[axis.column]),
      yaxis: axis.traceAxis, line: { color: axis.color, width: 1.5 },
      connectgaps: false,
    } as Data, { continuity }));
    const thresholdTraces = calculated.flatMap((entry) => {
      const axis = axisPlan.axes.find((item) => item.column === entry.column);
      if (!axis || !entry.result || previewRows.length === 0) return [];
      return [withLineGapPolicy({
        type: 'scatter', mode: 'lines', name: `${entry.column} · q=${entry.quantile}`,
        x: previewRows.map((row) => row.timestamp),
        y: previewRows.map(() => entry.result!.value),
        yaxis: axis.traceAxis,
        line: { color: axis.color, width: 1.5, dash: 'dash' },
        connectgaps: false,
      } as Data, { continuity })];
    });
    return [...graphTraces, ...thresholdTraces];
  }, [axisPlan.axes, calculated, dataset, previewRows]);

  const layout = useMemo<Partial<Layout>>(() => {
    const axisLayout = Object.fromEntries(axisPlan.axes.map((axis, index) => [axis.layoutAxis, {
      title: { text: axis.column, font: { color: axis.color } },
      tickfont: { color: axis.color }, linecolor: axis.color, showline: true,
      side: axis.side, anchor: 'free', position: axis.position,
      ...(index === 0 ? {} : { overlaying: 'y' }),
    }]));
    const redShapes = redIntervals.map((interval) => interval.start === interval.end ? {
      type: 'line' as const, xref: 'x' as const, yref: 'paper' as const,
      x0: interval.start, x1: interval.end, y0: 0, y1: 1,
      line: { color: 'rgba(250, 82, 82, 0.6)', width: 6 }, layer: 'below' as const,
    } : {
      type: 'rect' as const, xref: 'x' as const, yref: 'paper' as const,
      x0: interval.start, x1: interval.end, y0: 0, y1: 1,
      fillcolor: 'rgba(250, 82, 82, 0.15)', line: { width: 0 }, layer: 'below' as const,
    });
    const referenceShapes = thresholds.filter((item) => item.scope === 'reference' && item.referenceStart && item.referenceEnd).map((item) => ({
      type: 'rect' as const, xref: 'x' as const, yref: 'paper' as const,
      x0: item.referenceStart, x1: item.referenceEnd, y0: 0, y1: 1,
      fillcolor: item.id === selectionThresholdId ? 'rgba(34, 139, 230, 0.16)' : 'rgba(34, 139, 230, 0.07)',
      line: { color: item.id === selectionThresholdId ? '#228be6' : 'rgba(34, 139, 230, 0.35)', width: 1 }, layer: 'below' as const,
    }));
    return {
      dragmode: selectionThresholdId ? 'select' : 'zoom', hovermode: 'x unified',
      xaxis: { type: 'date', title: { text: 'Dataset-local time' }, rangeslider: { visible: true, thickness: 0.1 }, domain: axisPlan.xDomain },
      ...axisLayout,
      shapes: [...redShapes, ...referenceShapes],
      legend: { orientation: 'h' }, uirevision: document?.fileName ?? 'threshold',
      margin: { l: graphColumns.length > 2 ? 105 : 70, r: graphColumns.length > 1 ? 105 : 45, b: 145 },
    } as Partial<Layout>;
  }, [axisPlan, document?.fileName, graphColumns.length, redIntervals, selectionThresholdId, thresholds]);

  function reset() {
    loadId.current += 1;
    setFile(null); setDocument(null); setErrors([]); setLoading(false);
    setTimeColumn(null); setGraphColumns([]); setThresholds([]); setSelectionThresholdId(null);
  }

  async function loadFile(next: File | null) {
    const id = ++loadId.current;
    setFile(next); setDocument(null); setErrors([]); setTimeColumn(null);
    setGraphColumns([]); setThresholds([]); setSelectionThresholdId(null);
    if (!next) { setLoading(false); return; }
    setLoading(true);
    const outcome = await parseCsvFile(next);
    if (id !== loadId.current) return;
    setLoading(false); setErrors(outcome.errors); setDocument(outcome.document);
    if (outcome.document) setTimeColumn(timestampCandidateColumns(outcome.document)[0] ?? null);
  }

  function changeTimeColumn(value: string | null) {
    if (thresholds.length > 0 && !window.confirm('Changing the time column removes all calculated thresholds. Continue?')) return;
    setTimeColumn(value); setGraphColumns((current) => current.filter((column) => column !== value));
    setThresholds([]); setSelectionThresholdId(null);
  }

  function changeGraphs(next: string[]) {
    const removed = graphColumns.filter((column) => !next.includes(column));
    const affected = thresholds.filter((threshold) => removed.includes(threshold.column));
    if (affected.length > 0 && !window.confirm(`Removing these graph columns also removes ${affected.length} threshold(s). Continue?`)) return;
    setGraphColumns(next.slice(0, 6));
    setThresholds((current) => current.filter((threshold) => !removed.includes(threshold.column)));
    if (selectionThresholdId && affected.some((item) => item.id === selectionThresholdId)) setSelectionThresholdId(null);
  }

  function addThreshold() {
    if (graphColumns.length === 0 || !dataset?.rows.length) return;
    setThresholds((current) => [...current, {
      id: identifier(), column: graphColumns[0], quantile: 0.99, scope: 'entire',
      referenceStart: dataset.rows[0].timestamp.slice(0, 19),
      referenceEnd: dataset.rows.at(-1)!.timestamp.slice(0, 19), stale: true, result: null,
    }]);
  }

  function updateThreshold(id: string, update: Partial<Omit<ThresholdEntry, 'id' | 'result' | 'stale'>>) {
    setThresholds((current) => current.map((item) => item.id === id ? { ...item, ...update, stale: true } : item));
  }

  function calculateThreshold(entry: ThresholdEntry) {
    if (!dataset) return;
    try {
      const result = calculateQuantileThreshold(
        dataset.rows, entry.column, entry.quantile, entry.scope,
        entry.referenceStart, entry.referenceEnd,
      );
      setThresholds((current) => current.map((item) => item.id === entry.id ? { ...item, result, stale: false } : item));
      setSelectionThresholdId(null);
      notifications.show({ color: 'green', message: `Threshold ${resultValue(result.value)} calculated from ${result.inputCount.toLocaleString()} values.` });
    } catch (error) {
      notifications.show({ color: 'red', title: 'Threshold calculation failed', message: error instanceof Error ? error.message : 'Unknown error' });
    }
  }

  function applySelection(selection: PlotlyChartSelection) {
    if (!selectionThresholdId) return;
    const range = canonicalSelectionRange(selection.start, selection.end);
    updateThreshold(selectionThresholdId, {
      referenceStart: range.start,
      referenceEnd: range.end,
    });
  }

  if (!active) return null;
  return <Stack gap="lg">
    <Group justify="space-between" align="flex-start">
      <Box><Title order={2}>Threshold</Title><Text c="dimmed">Overlay CSV time series on independent scales and highlight quantile exceedances. Files and calculations stay in this browser.</Text></Box>
      <Button variant="default" leftSection={<RotateCcw size={16} />} onClick={reset} disabled={!file}>Reset</Button>
    </Group>

    <Alert color="blue" title="Local workspace">The uploaded CSV is never sent to the MLTrace backend. Navigation keeps the current workspace; reloading the application discards it.</Alert>

    <Paper withBorder p="md"><Title order={4} mb="sm">1 · Load and configure the CSV</Title><Stack>
      <FileInput accept=".csv,text/csv,text/plain" clearable label="CSV file" placeholder="Choose a CSV file" leftSection={<Upload size={16} />} value={file} onChange={(value) => void loadFile(value)} disabled={loading} />
      {loading && <Text size="sm" c="dimmed">Parsing CSV in the browser…</Text>}
      {errors.length > 0 && <Alert color="red" title="CSV could not be loaded">{errors.map((error) => <Text size="sm" key={error}>{error}</Text>)}</Alert>}
      {document && <><Group><Badge>{document.rows.length.toLocaleString()} rows</Badge><Badge variant="light">{document.headers.length} columns</Badge><Badge variant="light">{document.delimiter === '\t' ? 'Tab' : document.delimiter} delimiter</Badge></Group>
        {timeCandidates.length === 0 && <Alert color="red">No dataset-local timestamp values were detected. A time column is required.</Alert>}
        <SimpleGrid cols={{ base: 1, md: 2 }}>
          <Select searchable label="Time column" description="Invalid timestamp rows are excluded and reported." value={timeColumn} onChange={changeTimeColumn} data={timeCandidates} />
          <MultiSelect searchable label="Graph columns" description="Select up to six numeric columns; every graph receives its own Y-axis." value={graphColumns} onChange={changeGraphs} maxValues={6} data={numericCandidates} />
        </SimpleGrid>
        <Table.ScrollContainer minWidth={700}><Table striped withTableBorder><Table.Thead><Table.Tr>{document.headers.map((header) => <Table.Th key={header}>{header}</Table.Th>)}</Table.Tr></Table.Thead><Table.Tbody>{document.rows.slice(0, 5).map((row, rowIndex) => <Table.Tr key={rowIndex}>{row.map((value, columnIndex) => <Table.Td key={columnIndex}>{value ?? <Text c="dimmed">N/A</Text>}</Table.Td>)}</Table.Tr>)}</Table.Tbody></Table></Table.ScrollContainer>
      </>}
    </Stack></Paper>

    {dataset && graphColumns.length > 0 && <>
      <Paper withBorder p="md"><Group justify="space-between" mb="sm"><Box><Title order={4}>2 · Time-series plot</Title><Text size="sm" c="dimmed">Zoom and pan only change the view. Each Y-axis rescales independently; blue areas are threshold reference ranges and red areas are exceedances.</Text></Box><Group><Badge variant="light">Preview {previewRows.length.toLocaleString()} / {dataset.rows.length.toLocaleString()}</Badge>{dataset.invalidTimestampRows > 0 && <Badge color="orange">{dataset.invalidTimestampRows} invalid timestamps</Badge>}</Group></Group>
        <Group gap="xs" mb="xs">{graphColumns.map((column) => <Badge key={column} color={dataset.invalidNumericByColumn[column] ? 'orange' : 'green'} variant="light">{column}: {dataset.invalidNumericByColumn[column] ?? 0} invalid numeric cells</Badge>)}</Group>
        <PlotlyChart data={data} layout={layout} height={560} config={{ scrollZoom: true, modeBarButtonsToAdd: ['select2d'] }} onSelected={applySelection} rescaleYOnVisibleX fullResolutionExport={async () => thresholdExportTable(dataset.rows, graphColumns, calculated.map((item) => ({ id: item.id, column: item.column, quantile: item.quantile, value: item.result!.value })))} />
        {selectionThresholdId && <Alert color="blue" mt="sm">Drag horizontally in the plot to replace this threshold's reference range. Use the mode bar to switch back to zoom when needed.</Alert>}
      </Paper>

      <Paper withBorder p="md"><Group justify="space-between" mb="md"><Box><Title order={4}>3 · Quantile thresholds</Title><Text size="sm" c="dimmed">Quantiles use every finite source value or an explicit inclusive reference range. A point exceeds the threshold only when it is strictly greater.</Text></Box><Button leftSection={<Plus size={16} />} onClick={addThreshold}>Add threshold</Button></Group>
        {thresholds.length === 0 ? <Text c="dimmed">Add a threshold to begin.</Text> : <Stack>{thresholds.map((entry, index) => <Card withBorder key={entry.id}><Group justify="space-between" mb="sm"><Group><Text fw={700}>Threshold {index + 1}</Text><Badge color={entry.stale ? 'yellow' : 'green'}>{entry.stale ? 'Not calculated / stale' : 'Current'}</Badge></Group><Button color="red" variant="subtle" size="xs" leftSection={<Trash2 size={14} />} onClick={() => { setThresholds((current) => current.filter((item) => item.id !== entry.id)); if (selectionThresholdId === entry.id) setSelectionThresholdId(null); }}>Remove</Button></Group>
          <SimpleGrid cols={{ base: 1, md: 3 }}><Select label="Graph column" value={entry.column} allowDeselect={false} data={graphColumns} onChange={(value) => value && updateThreshold(entry.id, { column: value })} /><NumberInput label="Quantile q" description="Linear interpolation; value from 0 to 1." min={0} max={1} step={0.001} decimalScale={6} value={entry.quantile} onChange={(value) => updateThreshold(entry.id, { quantile: Number(value) })} /><SegmentedControl mt={25} value={entry.scope} onChange={(value) => { const scope = value === 'reference' ? 'reference' : 'entire'; updateThreshold(entry.id, { scope }); if (scope === 'entire' && selectionThresholdId === entry.id) setSelectionThresholdId(null); }} data={[{ value: 'entire', label: 'Entire column' }, { value: 'reference', label: 'Reference range' }]} /></SimpleGrid>
          {entry.scope === 'reference' && <SimpleGrid cols={{ base: 1, md: 3 }} mt="sm"><DateTime24Input label="Inclusive reference start" value={entry.referenceStart} onChange={(value) => updateThreshold(entry.id, { referenceStart: value })} /><DateTime24Input label="Inclusive reference end" value={entry.referenceEnd} onChange={(value) => updateThreshold(entry.id, { referenceEnd: value })} /><Button mt={25} variant={selectionThresholdId === entry.id ? 'filled' : 'light'} onClick={() => setSelectionThresholdId((current) => current === entry.id ? null : entry.id)}>{selectionThresholdId === entry.id ? 'Stop plot selection' : 'Select range in plot'}</Button></SimpleGrid>}
          <Group mt="md"><Button leftSection={<Calculator size={15} />} onClick={() => calculateThreshold(entry)} disabled={!Number.isFinite(entry.quantile) || entry.quantile < 0 || entry.quantile > 1 || (entry.scope === 'reference' && (!entry.referenceStart || !entry.referenceEnd))}>Calculate threshold</Button>{entry.result && <><Badge variant="outline">Value {resultValue(entry.result.value)}</Badge><Badge variant="outline">Source N {entry.result.inputCount.toLocaleString()}</Badge><Badge color="red" variant="light">{entry.result.exceedancePointCount.toLocaleString()} points above</Badge><Badge color="red" variant="light">{entry.result.intervals.length.toLocaleString()} intervals</Badge></>}</Group>
        </Card>)}</Stack>}
      </Paper>
    </>}
  </Stack>;
}
