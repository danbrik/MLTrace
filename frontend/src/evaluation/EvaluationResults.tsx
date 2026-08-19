import {
  Alert,
  Badge,
  Group,
  MultiSelect,
  Paper,
  SegmentedControl,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core';
import { AlertTriangle, CheckCircle2, Clock3, Gauge, Target } from 'lucide-react';
import { useMemo, useState } from 'react';
import type { Data } from '../lib/plotly';
import type { Shape } from 'plotly.js';

import { PlotlyChart } from '../components/PlotlyChart';
import type {
  EvaluationDetectionEventResult,
  EvaluationLabelEvent,
  EvaluationScorePreview,
  EvaluationTimeRange,
  ModelEvaluation,
} from '../types';
import {
  calculationStatusColor,
  calculationStatusLabel,
  closestOperatingPoint,
  EVALUATION_QUANTILES,
  eventShapes,
  formatEvaluationMetric,
  metricValues,
  scoreTrace,
  statusIsCurrent,
} from './helpers';

function metricColor(group: 'A' | 'B' | 'C'): string {
  return group === 'A' ? 'violet' : group === 'B' ? 'teal' : 'orange';
}

function EvaluationMetricCards({ evaluation, quantile }: { evaluation: ModelEvaluation; quantile: number }) {
  const metrics = metricValues(evaluation, quantile);
  return (
    <>
      <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
        {metrics.map((metric) => (
          <Paper key={metric.key} withBorder p="md" radius="sm" style={{ borderTop: `3px solid var(--mantine-color-${metricColor(metric.group)}-5)` }}>
            <Group justify="space-between" align="start">
              <div>
                <Text size="xs" c="dimmed" tt="uppercase" fw={700}>{metric.group} · {metric.label}</Text>
                <Title order={3} mt={4}>{formatEvaluationMetric(metric.value, metric.unit)}</Title>
              </div>
              <Stack gap={4} align="end">
                {metric.group === 'A' ? <Target size={19} /> : metric.group === 'B' ? <Gauge size={19} /> : <Clock3 size={19} />}
                {(() => { const status = metric.group === 'A' ? evaluation.separation_status : metric.group === 'B' ? evaluation.drift_status : evaluation.detection_status; return <Badge size="xs" color={calculationStatusColor(status)} variant="light">{calculationStatusLabel(status)}</Badge>; })()}
              </Stack>
            </Group>
          </Paper>
        ))}
      </SimpleGrid>
      <Table withTableBorder striped>
        <Table.Thead><Table.Tr><Table.Th>Step</Table.Th><Table.Th>Metric</Table.Th><Table.Th>Value</Table.Th><Table.Th>Status</Table.Th></Table.Tr></Table.Thead>
        <Table.Tbody>
          {metrics.map((metric) => {
            const status = metric.group === 'A' ? evaluation.separation_status : metric.group === 'B' ? evaluation.drift_status : evaluation.detection_status;
            return <Table.Tr key={metric.key}><Table.Td>{metric.group}</Table.Td><Table.Td>{metric.label}</Table.Td><Table.Td>{formatEvaluationMetric(metric.value, metric.unit)}</Table.Td><Table.Td><Badge color={calculationStatusColor(status)} variant="light">{calculationStatusLabel(status)}</Badge></Table.Td></Table.Tr>;
          })}
        </Table.Tbody>
      </Table>
    </>
  );
}

function SeparationDiagnostics({
  evaluation,
  preview,
  events,
  categories,
  validity,
  onFocusRange,
}: {
  evaluation: ModelEvaluation;
  preview: EvaluationScorePreview | null;
  events: EvaluationLabelEvent[];
  categories: string[];
  validity: string;
  onFocusRange?: (range: EvaluationTimeRange) => void;
}) {
  const [focusedRange, setFocusedRange] = useState<EvaluationTimeRange | null>(null);
  const result = evaluation.separation_result;
  if (!result || !statusIsCurrent(evaluation.separation_status)) return evaluation.separation_error
    ? <Alert color="red" title="A is blocked">{evaluation.separation_error} No target event was silently omitted; correct its interval or normal window and recalculate A.</Alert>
    : <Alert color="gray">Calculate A to see event separation diagnostics.</Alert>;
  const rows = result.events.filter((event) => (
    (categories.length === 0 || categories.includes(event.category ?? 'Uncategorized'))
    && validity !== 'invalid'
  ));
  const shapes = [
    ...eventShapes(events),
    ...rows.map((event) => ({
      type: 'rect', xref: 'x', yref: 'paper', x0: event.normal_start, x1: event.normal_end,
      y0: 0, y1: 1, fillcolor: 'rgba(81,207,102,0.10)', line: { color: '#40c057', dash: 'dot', width: 1 }, layer: 'below',
    } as Shape)),
  ];
  const barData: Data[] = [
    { type: 'bar', name: 'Separation', x: rows.map((row) => row.name ?? row.event_id), y: rows.map((row) => row.separation), marker: { color: '#7950f2' } } as Data,
  ];
  if (rows.some((row) => row.separation_p95 != null)) {
    barData.push({ type: 'bar', name: 'Separation p95', x: rows.map((row) => row.name ?? row.event_id), y: rows.map((row) => row.separation_p95), marker: { color: '#b197fc' } } as Data);
  }
  return (
    <Stack>
      {preview && <PlotlyChart data={[scoreTrace(preview.points)]} layout={{ shapes, hovermode: 'x unified', xaxis: { type: 'date', rangeslider: { visible: true }, range: focusedRange ? [focusedRange.start_timestamp, focusedRange.end_timestamp] : undefined }, yaxis: { title: { text: preview.score_series } }, showlegend: false, uirevision: `sep-${evaluation.id}` }} height={370} />}
      <PlotlyChart data={barData} layout={{ barmode: 'group', xaxis: { title: { text: 'Target event' } }, yaxis: { title: { text: 'Robust separation' } }, legend: { orientation: 'h' } }} height={310} />
      <PlotlyChart data={[{ type: 'scatter', mode: 'markers', name: 'Normal MAD', x: rows.map((row) => row.name ?? row.event_id), y: rows.map((row) => row.normal_mad), marker: { color: '#40c057', size: 9 }, hovertemplate: '%{x}<br>MAD %{y:.6g}<extra></extra>' } as Data]} layout={{ xaxis: { title: { text: 'Target event' } }, yaxis: { title: { text: 'Local normal MAD' }, rangemode: 'tozero' }, showlegend: false }} height={240} />
      <Table striped highlightOnHover withTableBorder>
        <Table.Thead><Table.Tr><Table.Th>Event</Table.Th><Table.Th>Category</Table.Th><Table.Th>Separation</Table.Th><Table.Th>Sep95</Table.Th><Table.Th>Normal MAD</Table.Th><Table.Th>Normal window</Table.Th></Table.Tr></Table.Thead>
        <Table.Tbody>{rows.map((event) => (
          <Table.Tr key={event.event_id} style={{ cursor: 'pointer' }} onClick={() => { const range = { start_timestamp: event.normal_start, end_timestamp: event.event_end }; setFocusedRange(range); onFocusRange?.(range); }}>
            <Table.Td>{event.name ?? event.event_id}</Table.Td><Table.Td>{event.category ?? 'Uncategorized'}</Table.Td><Table.Td>{formatEvaluationMetric(event.separation)}</Table.Td><Table.Td>{formatEvaluationMetric(event.separation_p95 ?? null)}</Table.Td><Table.Td>{formatEvaluationMetric(event.normal_mad)}</Table.Td><Table.Td>{event.normal_start.replace('T', ' ')} – {event.normal_end.replace('T', ' ')}</Table.Td>
          </Table.Tr>
        ))}</Table.Tbody>
      </Table>
    </Stack>
  );
}

function DriftDiagnostics({ evaluation, onFocusRange }: { evaluation: ModelEvaluation; onFocusRange?: (range: EvaluationTimeRange) => void }) {
  const [focusedRange, setFocusedRange] = useState<EvaluationTimeRange | null>(null);
  const result = evaluation.drift_result;
  if (!result || !statusIsCurrent(evaluation.drift_status)) return <Alert color="gray">Calculate B to see drift diagnostics.</Alert>;
  const valid = result.windows.filter((window) => window.status === 'valid' && window.normalized_drift != null);
  const discarded = result.windows.filter((window) => window.status !== 'valid');
  const traces: Data[] = [
    { type: 'scatter', mode: 'lines+markers', name: 'Normalized drift Dₖ', x: valid.map((window) => window.start), y: valid.map((window) => window.normalized_drift), line: { color: '#12b886' } } as Data,
    { type: 'scatter', mode: 'lines+markers', name: 'Raw Wasserstein W₁', x: valid.map((window) => window.start), y: valid.map((window) => window.wasserstein_1), yaxis: 'y2', line: { color: '#15aabf', dash: 'dot' } } as Data,
  ];
  return (
    <Stack>
      <Group gap="xs"><Badge color="teal" variant="light">Reference IQR {formatEvaluationMetric(result.reference_iqr)}</Badge><Badge color={discarded.length ? 'yellow' : 'green'} variant="light">{valid.length} valid · {discarded.length} discarded windows</Badge></Group>
      <PlotlyChart
        data={traces}
        layout={{ hovermode: 'x unified', xaxis: { type: 'date', title: { text: 'Window start' }, range: focusedRange ? [focusedRange.start_timestamp, focusedRange.end_timestamp] : undefined }, yaxis: { title: { text: 'Normalized Dₖ' } }, yaxis2: { title: { text: 'Raw W₁' }, overlaying: 'y', side: 'right' }, legend: { orientation: 'h' } }}
        height={360}
        onClick={(point) => {
          const window = valid[point.pointNumber];
          if (window) { const range = { start_timestamp: window.start, end_timestamp: window.end }; setFocusedRange(range); onFocusRange?.(range); }
        }}
      />
      {discarded.length > 0 && <Table striped withTableBorder><Table.Thead><Table.Tr><Table.Th>Discarded window</Table.Th><Table.Th>Points</Table.Th><Table.Th>Reason</Table.Th></Table.Tr></Table.Thead><Table.Tbody>{discarded.map((window) => <Table.Tr key={`${window.start}-${window.end}`}><Table.Td>{window.start.replace('T', ' ')} – {window.end.replace('T', ' ')}</Table.Td><Table.Td>{window.point_count}</Table.Td><Table.Td>{window.exclusion_reason ?? 'Invalid window'}</Table.Td></Table.Tr>)}</Table.Tbody></Table>}
    </Stack>
  );
}

function sensitivityTrace(evaluation: ModelEvaluation, field: 'event_recall' | 'median_delay_seconds' | 'frame_fpr' | 'far_t0', label: string, percent = false): Data {
  const points = evaluation.detection_result?.operating_points ?? [];
  return {
    type: 'scatter', mode: 'lines+markers', name: label,
    x: points.map((point) => point.quantile),
    y: points.map((point) => {
      const value = point[field];
      return typeof value === 'number' && percent ? value * 100 : value;
    }),
    line: { color: '#fd7e14' },
  } as Data;
}

function DetectionDiagnostics({
  evaluation,
  preview,
  events,
  quantile,
  categories,
  validity,
  detectionFilter,
  onFocusRange,
}: {
  evaluation: ModelEvaluation;
  preview: EvaluationScorePreview | null;
  events: EvaluationLabelEvent[];
  quantile: number;
  categories: string[];
  validity: string;
  detectionFilter: string;
  onFocusRange?: (range: EvaluationTimeRange) => void;
}) {
  const [focusedRange, setFocusedRange] = useState<EvaluationTimeRange | null>(null);
  const point = closestOperatingPoint(evaluation, quantile);
  if (!point || !statusIsCurrent(evaluation.detection_status)) return <Alert color="gray">Calculate C to see threshold and event-detection diagnostics.</Alert>;
  const eventRows = point.events.filter((event: EvaluationDetectionEventResult) => (
    (categories.length === 0 || categories.includes(event.category ?? 'Uncategorized'))
    && validity !== 'invalid'
    && (detectionFilter === 'all' || (detectionFilter === 'detected') === event.detected)
  ));
  const falseAlarmShapes = (point.false_alarms ?? []).flatMap((alarm) => {
    const start = alarm.start ?? alarm.start_timestamp;
    const end = alarm.end ?? alarm.end_timestamp;
    if (!start || !end) return [];
    return [{ type: 'rect', xref: 'x', yref: 'paper', x0: start, x1: end, y0: 0, y1: 1, fillcolor: 'rgba(250,176,5,0.20)', line: { color: '#fab005', width: 1 }, layer: 'below' } as Shape];
  });
  const shapes: Shape[] = [
    ...eventShapes(events),
    { type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: point.threshold, y1: point.threshold, line: { color: '#e8590c', dash: 'dash', width: 2 } } as Shape,
    ...falseAlarmShapes,
  ];
  const detected = eventRows.filter((event) => event.detected && event.first_detection);
  const chartData: Data[] = preview ? [
    scoreTrace(preview.points),
    {
      type: 'scatter', mode: 'markers', name: 'First event detection',
      x: detected.map((event) => event.first_detection),
      y: detected.map((event) => preview.points.find((candidate) => candidate.timestamp === event.first_detection)?.value ?? point.threshold),
      marker: { color: '#2f9e44', size: 10, symbol: 'diamond' },
      hovertemplate: 'First detection<br>%{x}<extra></extra>',
    } as Data,
  ] : [];
  return (
    <Stack>
      <Group gap="xs"><Badge color="orange" variant="light">Threshold {formatEvaluationMetric(point.threshold)}</Badge><Badge color="blue" variant="light">{point.detected_event_count}/{point.event_count} events detected</Badge><Badge color="yellow" variant="light">{point.false_alarm_event_count} false-alarm phases</Badge></Group>
      {preview && <PlotlyChart data={chartData} layout={{ shapes, hovermode: 'x unified', xaxis: { type: 'date', rangeslider: { visible: true }, range: focusedRange ? [focusedRange.start_timestamp, focusedRange.end_timestamp] : undefined }, yaxis: { title: { text: preview.score_series } }, showlegend: true, legend: { orientation: 'h' }, uirevision: `det-${evaluation.id}-${point.quantile}` }} height={380} />}
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        {[
          ['event_recall', 'Event recall', true],
          ['median_delay_seconds', 'Median delay (s)', false],
          ['frame_fpr', 'Frame FPR', true],
          ['far_t0', 'FAR T₀', false],
        ].map(([field, label, percent]) => <PlotlyChart key={String(field)} data={[sensitivityTrace(evaluation, field as 'event_recall' | 'median_delay_seconds' | 'frame_fpr' | 'far_t0', String(label), Boolean(percent))]} layout={{ xaxis: { title: { text: 'Calibration quantile' }, tickformat: '.4f' }, yaxis: { title: { text: String(label) } }, showlegend: false }} height={250} />)}
      </SimpleGrid>
      <Table striped highlightOnHover withTableBorder>
        <Table.Thead><Table.Tr><Table.Th>Event</Table.Th><Table.Th>Category</Table.Th><Table.Th>Outcome</Table.Th><Table.Th>First detection</Table.Th><Table.Th>Delay</Table.Th></Table.Tr></Table.Thead>
        <Table.Tbody>{eventRows.map((event) => <Table.Tr key={event.event_id} style={{ cursor: 'pointer' }} onClick={() => { const range = { start_timestamp: event.detection_window_start, end_timestamp: event.detection_window_end }; setFocusedRange(range); onFocusRange?.(range); }}><Table.Td>{event.name ?? event.event_id}</Table.Td><Table.Td>{event.category ?? 'Uncategorized'}</Table.Td><Table.Td><Badge color={event.detected ? 'green' : 'red'} variant="light">{event.detected ? 'Detected' : 'Missed'}</Badge></Table.Td><Table.Td>{event.first_detection?.replace('T', ' ') ?? '—'}</Table.Td><Table.Td>{formatEvaluationMetric(event.delay_seconds, 's')}</Table.Td></Table.Tr>)}</Table.Tbody>
      </Table>
    </Stack>
  );
}

export function EvaluationResults({
  evaluation,
  preview,
  events,
  activeQuantile,
  onActiveQuantileChange,
  onFocusRange,
}: {
  evaluation: ModelEvaluation;
  preview: EvaluationScorePreview | null;
  events: EvaluationLabelEvent[];
  activeQuantile: number;
  onActiveQuantileChange: (quantile: number) => void;
  onFocusRange?: (range: EvaluationTimeRange) => void;
}) {
  const [categories, setCategories] = useState<string[]>([]);
  const [validity, setValidity] = useState('all');
  const [detectionFilter, setDetectionFilter] = useState('all');
  const categoryOptions = useMemo(() => [...new Set([
    ...events.filter((event) => event.type === 'target').map((event) => event.category || 'Uncategorized'),
    ...(evaluation.separation_result?.events ?? []).map((event) => event.category ?? 'Uncategorized'),
  ])].sort().map((value) => ({ value, label: value })), [evaluation.separation_result?.events, events]);
  const warningMessages = [
    ...(evaluation.warnings ?? []).map((warning) => typeof warning === 'string' ? warning : typeof warning.warning === 'string' ? warning.warning : warning.warning?.message ?? JSON.stringify(warning)),
    ...(evaluation.separation_result?.warnings ?? []).map((warning) => warning.message),
    ...(evaluation.drift_result?.warnings ?? []).map((warning) => warning.message),
    ...(evaluation.detection_result?.warnings ?? []).map((warning) => warning.message),
  ];
  const warnings = [...new Set(warningMessages)];

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="end">
        <div><Title order={3}>Eight principal metrics</Title><Text size="sm" c="dimmed">C metrics use the active fixed operating point; switching it never recalculates the evaluation.</Text></div>
        <SegmentedControl value={String(activeQuantile)} onChange={(value) => onActiveQuantileChange(Number(value))} data={EVALUATION_QUANTILES.map((quantile) => ({ value: String(quantile), label: `q=${quantile}` }))} />
      </Group>
      <EvaluationMetricCards evaluation={evaluation} quantile={activeQuantile} />
      {warnings.length > 0 && <Alert color="yellow" icon={<AlertTriangle size={17} />} title="Evaluation diagnostics">{warnings.map((warning) => <Text key={warning} size="sm">{warning}</Text>)}</Alert>}
      <Paper withBorder p="md">
        <Stack gap="sm">
          <Group justify="space-between"><Text fw={600}>Detail filters</Text><Group gap="xs"><CheckCircle2 size={16} /><Text size="xs" c="dimmed">Filters affect diagnostics only, never persisted metrics.</Text></Group></Group>
          <SimpleGrid cols={{ base: 1, sm: 3 }}>
            <MultiSelect label="Categories" data={categoryOptions} value={categories} onChange={setCategories} searchable clearable />
            <Select label="Validity" description="Invalid A events block A and are reported in its error instead of being omitted." data={[{ value: 'all', label: 'All calculated events' }, { value: 'valid', label: 'Valid only' }, { value: 'invalid', label: 'Invalid only (available in errors)', disabled: true }]} value={validity} onChange={(value) => setValidity(value ?? 'all')} />
            <Select label="Detection outcome" data={[{ value: 'all', label: 'Detected and missed' }, { value: 'detected', label: 'Detected only' }, { value: 'missed', label: 'Missed only' }]} value={detectionFilter} onChange={(value) => setDetectionFilter(value ?? 'all')} />
          </SimpleGrid>
        </Stack>
      </Paper>
      <Paper withBorder p="md"><Stack><Group><Badge color="violet">A</Badge><Title order={4}>Event separation</Title></Group><SeparationDiagnostics evaluation={evaluation} preview={preview} events={events} categories={categories} validity={validity} onFocusRange={onFocusRange} /></Stack></Paper>
      <Paper withBorder p="md"><Stack><Group><Badge color="teal">B</Badge><Title order={4}>Score stability</Title></Group><DriftDiagnostics evaluation={evaluation} onFocusRange={onFocusRange} /></Stack></Paper>
      <Paper withBorder p="md"><Stack><Group><Badge color="orange">C</Badge><Title order={4}>Detection performance · q={activeQuantile}</Title></Group><DetectionDiagnostics evaluation={evaluation} preview={preview} events={events} quantile={activeQuantile} categories={categories} validity={validity} detectionFilter={detectionFilter} onFocusRange={onFocusRange} /></Stack></Paper>
    </Stack>
  );
}
