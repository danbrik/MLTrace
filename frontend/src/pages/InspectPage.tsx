import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Collapse,
  Divider,
  Group,
  NumberInput,
  MultiSelect,
  Pagination,
  Paper,
  Progress,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { ChevronDown, ChevronRight, Download, Eye, FileVideo, Info, RefreshCw, Save, Square, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent } from 'react';

import {
  abortInspectRun,
  abortHeatmapRange,
  createRoi,
  createInspectRun,
  deleteInspectRun,
  deleteHeatmapRange,
  getInspectCsvData,
  getTemporalDynamicsRunResult,
  heatmapRangeVideoUrl,
  inspectRunCsvUrl,
  inspectRunVideoUrl,
  inspectPreviewVideoUrl,
  listInspectArtifacts,
  listRois,
  listInspectRuns,
  listPreprocessingPipelines,
  listTrainingDatasets,
  previewInspect,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { StepCard } from '../components/StepCard';
import { PlotlyChart } from '../components/PlotlyChart';
import { usePendingIds } from '../hooks/usePendingIds';
import type { InspectArtifactRun, InspectCsvData, InspectPreview, InspectRun, PreprocessingPipeline, RoiDefinition, TemporalDynamicsResult, TrainingDataset } from '../types';

type InspectAnalysisMode = 'preprocessed_video' | 'contrast_enhanced' | 'energy' | 'optical_flow' | 'temporal_dynamics';
type RoiPoint = { x: number; y: number };

const TILE_OPTIONS = [1, 2, 3, 4, 5].map((value) => ({ value: String(value), label: String(value) }));

function toInputDateTime(value: string | null): string {
  return value ? value.slice(0, 19) : '';
}

function formatTimestamp(value: string | null): string {
  return value ? value.replace('T', ' ').slice(0, 19) : 'n/a';
}

function midpointTimestamp(start: string | null, end: string | null): string {
  if (!start) return '';
  if (!end) return toInputDateTime(start);
  // Dataset timestamps are timezone-naive wall times. Append Z only for the
  // arithmetic so toISOString does not apply the browser's local UTC offset.
  const asStableIso = (value: string) => /(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? value : `${value}Z`;
  const startMs = new Date(asStableIso(start)).getTime();
  const endMs = new Date(asStableIso(end)).getTime();
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return toInputDateTime(start);
  return new Date(startMs + (endMs - startMs) / 2).toISOString().slice(0, 19);
}

function parseLagValues(value: string): number[] {
  const parsed = value
    .split(/[\s,;]+/)
    .filter(Boolean)
    .map(Number)
    .filter((lag) => Number.isInteger(lag) && lag > 0);
  return [...new Set(parsed)].sort((left, right) => left - right);
}

function centeredTimeRange(reference: string, windowMinutes: number): { start: string; end: string } {
  const stable = /(?:Z|[+-]\d{2}:\d{2})$/.test(reference) ? reference : `${reference}Z`;
  const center = new Date(stable).getTime();
  const halfWindowMs = Math.max(1000, windowMinutes * 30_000);
  return {
    start: new Date(center - halfWindowMs).toISOString().slice(0, 19),
    end: new Date(center + halfWindowMs).toISOString().slice(0, 19),
  };
}

function formatDurationSeconds(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return 'estimating…';
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${Math.round(seconds % 60)} s`;
}

function estimatedRemainingSeconds(run: { status: string; started_at: string | null; done_count: number; frame_count: number | null }): number | null {
  if (run.status !== 'running' || !run.started_at || !run.frame_count || run.done_count <= 0) return null;
  const stableStarted = /(?:Z|[+-]\d{2}:\d{2})$/.test(run.started_at) ? run.started_at : `${run.started_at}Z`;
  const started = new Date(stableStarted).getTime();
  const elapsed = Math.max(0, (Date.now() - started) / 1000);
  return elapsed * Math.max(0, run.frame_count - run.done_count) / run.done_count;
}

function statusColor(status: string): string {
  if (status === 'finished') return 'green';
  if (status === 'failed') return 'red';
  if (status === 'aborted') return 'yellow';
  if (status === 'running') return 'blue';
  return 'gray';
}

function progressLabel(run: InspectRun): string {
  return `${run.done_count}${run.frame_count ? ` / ${run.frame_count}` : ''}`;
}

function InfoLabel({ label, info }: { label: string; info: string }) {
  return (
    <Group gap={5} wrap="nowrap">
      <Text size="sm">{label}</Text>
      <Tooltip label={info} multiline w={320} withArrow>
        <Info size={14} aria-label={`${label} information`} tabIndex={0} />
      </Tooltip>
    </Group>
  );
}

function roiLabel(roi: RoiDefinition): string {
  return `${roi.name} (${roi.image_width}x${roi.image_height}, ${roi.tile_rows ?? 1}x${roi.tile_cols ?? 1} tiles)`;
}

function roiPoints(roi: RoiDefinition): RoiPoint[] {
  if (roi.points?.length === 4) return roi.points.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
  return [
    { x: roi.x, y: roi.y },
    { x: roi.x + roi.width, y: roi.y },
    { x: roi.x + roi.width, y: roi.y + roi.height },
    { x: roi.x, y: roi.y + roi.height },
  ];
}

function defaultRoiPoints(preview: InspectPreview): RoiPoint[] {
  const width = Math.max(1, Math.round(preview.width * 0.5));
  const height = Math.max(1, Math.round(preview.height * 0.5));
  const x = Math.round((preview.width - width) / 2);
  const y = Math.round((preview.height - height) / 2);
  return [
    { x, y },
    { x: x + width, y },
    { x: x + width, y: y + height },
    { x, y: y + height },
  ];
}

function boundingRect(points: RoiPoint[]) {
  const minX = Math.floor(Math.min(...points.map((point) => point.x)));
  const maxX = Math.ceil(Math.max(...points.map((point) => point.x)));
  const minY = Math.floor(Math.min(...points.map((point) => point.y)));
  const maxY = Math.ceil(Math.max(...points.map((point) => point.y)));
  return { x: minX, y: minY, width: Math.max(1, maxX - minX), height: Math.max(1, maxY - minY) };
}

function interp(points: RoiPoint[], u: number, v: number): RoiPoint {
  const [tl, tr, br, bl] = points;
  const top = { x: tl.x + (tr.x - tl.x) * u, y: tl.y + (tr.y - tl.y) * u };
  const bottom = { x: bl.x + (br.x - bl.x) * u, y: bl.y + (br.y - bl.y) * u };
  return { x: top.x + (bottom.x - top.x) * v, y: top.y + (bottom.y - top.y) * v };
}

function pointFromClient(container: HTMLDivElement, clientX: number, clientY: number, preview: InspectPreview): RoiPoint {
  const rect = container.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * preview.width;
  const y = ((clientY - rect.top) / rect.height) * preview.height;
  return { x: Math.round(Math.max(0, Math.min(preview.width, x))), y: Math.round(Math.max(0, Math.min(preview.height, y))) };
}

function PolygonRoiPicker({
  preview,
  points,
  tileRows,
  tileCols,
  onChange,
}: {
  preview: InspectPreview;
  points: RoiPoint[];
  tileRows: number;
  tileCols: number;
  onChange: (points: RoiPoint[]) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<number | null>(null);
  const polygonPoints = points.map((point) => `${(point.x / preview.width) * 100},${(point.y / preview.height) * 100}`).join(' ');
  function startDrag(index: number, event: PointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    dragRef.current = index;
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }
  function onMove(event: PointerEvent<HTMLDivElement>) {
    const index = dragRef.current;
    const container = containerRef.current;
    if (index === null || !container) return;
    const point = pointFromClient(container, event.clientX, event.clientY, preview);
    onChange(points.map((current, currentIndex) => (currentIndex === index ? point : current)));
  }
  return (
    <Stack gap="xs">
      <div ref={containerRef} className="warp-picker" onPointerMove={onMove} onPointerUp={() => (dragRef.current = null)} onPointerLeave={() => (dragRef.current = null)}>
        <img src={preview.image_data_url} alt="ROI preview" className="warp-picker-image" />
        <svg className="roi-overlay" viewBox="0 0 100 100" preserveAspectRatio="none">
          <polygon points={polygonPoints} className="roi-polygon" />
          {Array.from({ length: Math.max(0, tileCols - 1) }, (_, index) => {
            const u = (index + 1) / tileCols;
            const a = interp(points, u, 0);
            const b = interp(points, u, 1);
            return <line key={`col-${index}`} x1={(a.x / preview.width) * 100} y1={(a.y / preview.height) * 100} x2={(b.x / preview.width) * 100} y2={(b.y / preview.height) * 100} className="roi-grid-line" />;
          })}
          {Array.from({ length: Math.max(0, tileRows - 1) }, (_, index) => {
            const v = (index + 1) / tileRows;
            const a = interp(points, 0, v);
            const b = interp(points, 1, v);
            return <line key={`row-${index}`} x1={(a.x / preview.width) * 100} y1={(a.y / preview.height) * 100} x2={(b.x / preview.width) * 100} y2={(b.y / preview.height) * 100} className="roi-grid-line" />;
          })}
        </svg>
        {points.map((point, index) => (
          <button key={index} type="button" className="warp-point" style={{ left: `${(point.x / preview.width) * 100}%`, top: `${(point.y / preview.height) * 100}%` }} onPointerDown={(event) => startDrag(index, event)}>
            {index + 1}
          </button>
        ))}
      </div>
    </Stack>
  );
}

function selectionSignature(values: {
  trainingDatasetId: number | null;
  preprocessingPipelineId: number | null;
  start: string;
  end: string;
  stride: number;
  analysisMode: InspectAnalysisMode;
  analysisConfig: Record<string, unknown>;
  roiId: number | null;
  generateVideo: boolean;
  contrastEnabled: boolean;
  contrastReferenceFrames: number;
  contrastShift: number;
  contrastVmax: number;
  contrastMaRadius: number;
}): string {
  return JSON.stringify(values);
}

function artifactVideoUrl(artifact: InspectArtifactRun): string {
  return artifact.kind === 'heatmap' ? heatmapRangeVideoUrl(artifact.id) : inspectRunVideoUrl(artifact.id);
}

function artifactFromInspectRun(run: InspectRun): InspectArtifactRun {
  return {
    kind: 'inspect',
    id: run.id,
    mode: run.analysis_mode,
    status: run.status,
    error_message: run.error_message,
    training_dataset_id: run.training_dataset_id,
    training_dataset_name: run.training_dataset_name,
    preprocessing_pipeline_id: run.preprocessing_pipeline_id,
    preprocessing_pipeline_name: run.preprocessing_pipeline_name,
    start_timestamp: run.start_timestamp,
    end_timestamp: run.end_timestamp,
    stride: run.stride,
    fps: run.fps,
    frame_count: run.frame_count,
    done_count: run.done_count,
    started_at: run.started_at,
    duration_seconds: run.duration_seconds,
    has_video: Boolean(run.video_path),
    has_csv: Boolean(run.csv_path),
    has_summary: Boolean(run.summary_json_path),
    created_at: run.created_at,
    updated_at: run.updated_at,
  };
}

function ArtifactViewer({ artifact }: { artifact: InspectArtifactRun | null }) {
  const [view, setView] = useState<'video' | 'csv'>('video');
  const [csvData, setCsvData] = useState<InspectCsvData | null>(null);
  const [csvLoading, setCsvLoading] = useState(false);
  const [xColumn, setXColumn] = useState<string | null>(null);
  const [yColumns, setYColumns] = useState<string[]>([]);

  useEffect(() => {
    setView(artifact?.has_video ? 'video' : 'csv');
    setCsvData(null);
    setXColumn(null);
    setYColumns([]);
  }, [artifact?.kind, artifact?.id]);

  useEffect(() => {
    if (!artifact || view !== 'csv' || !artifact.has_csv || csvData || csvLoading) return;
    setCsvLoading(true);
    getInspectCsvData(artifact.id)
      .then(setCsvData)
      .catch((error) => notifications.show({ color: 'red', title: 'CSV load failed', message: error instanceof Error ? error.message : 'Unknown error' }))
      .finally(() => setCsvLoading(false));
  }, [artifact, view, csvData, csvLoading]);

  if (!artifact) return null;
  const viewOptions = [
    ...(artifact.has_video ? [{ value: 'video', label: 'MP4' }] : []),
    ...(artifact.has_csv ? [{ value: 'csv', label: 'CSV plot' }] : []),
  ];
  const numericColumns = csvData?.columns.filter((column) => column.kind === 'number') ?? [];
  const plotData = yColumns.map((column) => ({
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: column,
    x: csvData?.rows.map((row, index) => xColumn ? row[xColumn] : index) ?? [],
    y: csvData?.rows.map((row) => row[column]) ?? [],
  }));

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="sm">
        <Group justify="space-between" align="flex-end" wrap="wrap">
          <Group gap="xs">
            <Text fw={700}>{artifact.training_dataset_name}</Text>
            <Badge variant="light">{artifact.preprocessing_pipeline_name}</Badge>
            <Badge variant="light" color="cyan">{artifact.mode.replaceAll('_', ' ')}</Badge>
          </Group>
          <Group gap="xs">
            {viewOptions.length > 1 && <Select size="xs" data={viewOptions} value={view} onChange={(value) => setView((value ?? 'video') as 'video' | 'csv')} />}
            {artifact.has_video && <Button size="compact-sm" component="a" href={artifactVideoUrl(artifact)} download leftSection={<Download size={14} />}>MP4</Button>}
            {artifact.has_csv && <Button size="compact-sm" variant="light" component="a" href={inspectRunCsvUrl(artifact.id)} download leftSection={<Download size={14} />}>CSV</Button>}
          </Group>
        </Group>
        {view === 'video' && artifact.has_video && (
          <video src={artifactVideoUrl(artifact)} controls muted playsInline className="inspect-video-player" />
        )}
        {view === 'csv' && artifact.has_csv && (
          <Stack gap="sm">
            <Group grow align="flex-end">
              <Select label="X column" data={(csvData?.columns ?? []).map((column) => ({ value: column.name, label: `${column.name} (${column.kind})` }))} value={xColumn} onChange={setXColumn} searchable disabled={csvLoading} />
              <MultiSelect label="Y columns" data={numericColumns.map((column) => ({ value: column.name, label: column.name }))} value={yColumns} onChange={setYColumns} searchable disabled={csvLoading} />
            </Group>
            {csvLoading && <Progress value={100} animated />}
            {csvData && (!xColumn || yColumns.length === 0) && <Alert color="blue">Choose one X column and at least one numeric Y column.</Alert>}
            {csvData && xColumn && yColumns.length > 0 && <PlotlyChart data={plotData} layout={{ title: { text: `${artifact.mode.replaceAll('_', ' ')} · run ${artifact.id}` }, xaxis: { title: { text: xColumn } }, yaxis: { title: { text: 'Value' } }, hovermode: 'x unified' }} height={480} />}
          </Stack>
        )}
      </Stack>
    </Paper>
  );
}

function TemporalDynamicsResults({ result }: { result: TemporalDynamicsResult }) {
  const lagRows = result.lag_statistics.filter((row) => row.mean != null);
  const lagX = lagRows.map((row) => row.lag_seconds);
  const correlationLabel = result.estimated_correlation_length_seconds == null
    ? 'Not reached or not estimable in the analyzed range'
    : `${result.estimated_correlation_length_seconds} seconds`;
  const number = (value: number | null) => value == null ? 'n/a' : value.toPrecision(5);
  const motionSegments = [...new Set(result.motion_signal.map((row) => row.segment_id))].map((segmentId) => ({
    segmentId,
    rows: result.motion_signal.filter((row) => row.segment_id === segmentId),
  }));

  return (
    <Stack gap="lg">
      <Alert color="cyan" title="Temporal dynamics summary">
        <Stack gap={4}>
          <Text size="sm">Estimated relevant time scale: <strong>{result.estimated_relevant_time_scale_seconds} s</strong></Text>
          <Text size="sm">Estimated temporal correlation length: <strong>{correlationLabel}</strong></Text>
          <Text size="sm">
            Recommended prototype configuration: <strong>{result.recommended_sequence_length} frames</strong>, temporal stride{' '}
            <strong>{result.recommended_temporal_stride}</strong>, covered time window approximately{' '}
            <strong>{result.covered_time_window_seconds} s</strong>.
          </Text>
          <Text size="xs" c="dimmed">
            Heuristic only: the relevant scale conservatively combines the lag-curve plateau and the first motion autocorrelation below {result.autocorrelation_threshold}. This is not hyperparameter optimization.
          </Text>
        </Stack>
      </Alert>

      <Group gap="xs">
        <Badge variant="light">{result.loaded_frame_count} frames</Badge>
        <Badge variant="light">{result.contiguous_segment_count} contiguous segments</Badge>
        <Badge variant="light">Pipeline output {result.image_width}x{result.image_height}</Badge>
        <Badge variant="light">Analysis stride {result.stride}</Badge>
        <Badge variant="light">{result.distance_metric.toUpperCase()}</Badge>
        {result.roi_name && <Badge variant="light" color="grape">ROI: {result.roi_name}</Badge>}
        {result.skipped_frame_count > 0 && <Badge variant="light" color="yellow">{result.skipped_frame_count} unreadable skipped</Badge>}
        {result.cached && <Badge variant="light" color="green">cached result</Badge>}
      </Group>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="sm">
          <Text fw={700}>Image difference over temporal lag</Text>
          <PlotlyChart
            data={[
              { type: 'scatter' as const, mode: 'lines' as const, x: lagX, y: lagRows.map((row) => row.p75), line: { width: 0 }, hoverinfo: 'skip' as const, showlegend: false },
              { type: 'scatter' as const, mode: 'lines' as const, x: lagX, y: lagRows.map((row) => row.p25), fill: 'tonexty' as const, fillcolor: 'rgba(34, 139, 230, 0.18)', line: { width: 0 }, name: '25–75 percentile' },
              { type: 'scatter' as const, mode: 'lines+markers' as const, x: lagX, y: lagRows.map((row) => row.mean), name: 'Mean', line: { color: '#1971c2', width: 3 } },
            ]}
            layout={{ xaxis: { title: { text: 'Temporal Lag [seconds]' }, type: 'log', dtick: 0.30103 }, yaxis: { title: { text: result.distance_label } }, hovermode: 'x unified' }}
            height={380}
          />
          <ScrollArea>
            <Table striped withTableBorder miw={760}>
              <Table.Thead><Table.Tr><Table.Th>Lag [s]</Table.Th><Table.Th>Pairs</Table.Th><Table.Th>Mean</Table.Th><Table.Th>Median</Table.Th><Table.Th>Std</Table.Th><Table.Th>P25</Table.Th><Table.Th>P75</Table.Th></Table.Tr></Table.Thead>
              <Table.Tbody>{result.lag_statistics.map((row) => <Table.Tr key={row.lag_seconds}><Table.Td>{row.lag_seconds}</Table.Td><Table.Td>{row.pair_count}</Table.Td><Table.Td>{number(row.mean)}</Table.Td><Table.Td>{number(row.median)}</Table.Td><Table.Td>{number(row.std)}</Table.Td><Table.Td>{number(row.p25)}</Table.Td><Table.Td>{number(row.p75)}</Table.Td></Table.Tr>)}</Table.Tbody>
            </Table>
          </ScrollArea>
        </Stack>
      </Paper>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="sm">
          <Text fw={700}>Frame-to-frame motion signal</Text>
          <PlotlyChart
            data={motionSegments.map(({ segmentId, rows }, index) => ({ type: 'scatter' as const, mode: 'lines' as const, x: rows.map((row) => row.timestamp), y: rows.map((row) => row.difference), name: index === 0 ? 'Frame difference' : `Segment ${segmentId + 1}`, showlegend: index === 0 }))}
            layout={{ xaxis: { title: { text: 'Time' } }, yaxis: { title: { text: 'Frame-to-frame image difference (MAE)' } }, hovermode: 'x unified' }}
            height={360}
          />
        </Stack>
      </Paper>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="sm">
          <Group justify="space-between" wrap="wrap">
            <Text fw={700}>Motion autocorrelation</Text>
            <Badge variant="light" color={result.estimated_correlation_length_seconds == null ? 'gray' : 'cyan'}>
              Estimated temporal correlation length: {correlationLabel}
            </Badge>
          </Group>
          <PlotlyChart
            data={[{ type: 'scatter' as const, mode: 'lines' as const, x: result.autocorrelation.map((row) => row.lag_seconds), y: result.autocorrelation.map((row) => row.autocorrelation), name: 'Autocorrelation', connectgaps: false }]}
            layout={{
              xaxis: { title: { text: 'Lag [seconds]' } },
              yaxis: { title: { text: 'Autocorrelation' }, range: [-1, 1] },
              shapes: [{ type: 'line', x0: 1, x1: result.autocorrelation.at(-1)?.lag_seconds ?? 128, y0: result.autocorrelation_threshold, y1: result.autocorrelation_threshold, line: { color: '#fa5252', dash: 'dash' } }],
              hovermode: 'x unified',
            }}
            height={360}
          />
          <Text size="xs" c="dimmed">The 0.2 threshold is a pragmatic orientation value, not a hard scientific boundary.</Text>
        </Stack>
      </Paper>

      <div>
        <Text fw={700} mb="sm">Visual pair checks</Text>
        <SimpleGrid cols={{ base: 1, xl: 2 }} spacing="md">
          {result.comparison_examples.map((example) => (
            <Paper key={example.lag_seconds} withBorder p="md" radius="sm">
              <Stack gap="sm">
                <Group justify="space-between"><Text fw={600}>{example.lag_seconds} s lag</Text><Badge variant="light">difference {example.difference.toPrecision(4)}</Badge></Group>
                <Text size="xs" c="dimmed">{formatTimestamp(example.reference_timestamp)} → {formatTimestamp(example.comparison_timestamp)} · actual {example.actual_lag_seconds.toFixed(2)} s</Text>
                <SimpleGrid cols={3} spacing="xs">
                  {[
                    ['Reference I(t)', example.reference_image_data_url],
                    ['Comparison I(t+k)', example.comparison_image_data_url],
                    ['Absolute difference', example.difference_image_data_url],
                  ].map(([label, source]) => <Stack key={label} gap={4}><Text size="xs" ta="center">{label}</Text><img src={source} alt={`${label}, ${example.lag_seconds} second lag`} style={{ width: '100%', borderRadius: 4 }} /></Stack>)}
                </SimpleGrid>
              </Stack>
            </Paper>
          ))}
        </SimpleGrid>
      </div>
    </Stack>
  );
}

export function InspectPage({ active = true }: { active?: boolean }) {
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [preprocessingPipelines, setPreprocessingPipelines] = useState<PreprocessingPipeline[]>([]);
  const [rois, setRois] = useState<RoiDefinition[]>([]);
  const [runs, setRuns] = useState<InspectRun[]>([]);
  const [artifactItems, setArtifactItems] = useState<InspectArtifactRun[]>([]);
  const [artifactTotal, setArtifactTotal] = useState(0);
  const [artifactPages, setArtifactPages] = useState(1);
  const [artifactActiveTotal, setArtifactActiveTotal] = useState(0);
  const [artifactPage, setArtifactPage] = useState(1);
  const [artifactModeFilter, setArtifactModeFilter] = useState<string | null>(null);
  const [artifactDatasetFilter, setArtifactDatasetFilter] = useState<number | null>(null);
  const [artifactPipelineFilter, setArtifactPipelineFilter] = useState<number | null>(null);
  const [artifactStatusFilter, setArtifactStatusFilter] = useState<string | null>(null);
  const [artifactsOpen, setArtifactsOpen] = useState(true);
  const [selectedArtifact, setSelectedArtifact] = useState<InspectArtifactRun | null>(null);
  const [matchingArtifacts, setMatchingArtifacts] = useState<InspectArtifactRun[]>([]);
  const [trainingDatasetId, setTrainingDatasetId] = useState<number | null>(null);
  const [preprocessingPipelineId, setPreprocessingPipelineId] = useState<number | null>(null);
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [stride, setStride] = useState(1);
  const [fps, setFps] = useState(12);
  const [analysisMode, setAnalysisMode] = useState<InspectAnalysisMode>('preprocessed_video');
  const [generateVideo, setGenerateVideo] = useState(true);
  const [contrastEnabled, setContrastEnabled] = useState(false);
  const [contrastReferenceFrames, setContrastReferenceFrames] = useState(100);
  const [contrastShift, setContrastShift] = useState(10000);
  const [contrastVmax, setContrastVmax] = useState(12000);
  const [contrastMaRadius, setContrastMaRadius] = useState(3);
  const [energyVariant, setEnergyVariant] = useState<'pairwise' | 'window'>('pairwise');
  const [energyAggregation, setEnergyAggregation] = useState<'sum' | 'mean' | 'p95'>('sum');
  const [energyWindowSize, setEnergyWindowSize] = useState(5);
  const [energyNormalize, setEnergyNormalize] = useState(false);
  const [flowAggregation, setFlowAggregation] = useState<'mean_magnitude' | 'p95_magnitude' | 'max_magnitude'>('mean_magnitude');
  const [flowPyrScale, setFlowPyrScale] = useState(0.5);
  const [flowLevels, setFlowLevels] = useState(3);
  const [flowWinSize, setFlowWinSize] = useState(15);
  const [flowIterations, setFlowIterations] = useState(3);
  const [flowPolyN, setFlowPolyN] = useState(5);
  const [flowPolySigma, setFlowPolySigma] = useState(1.2);
  const [flowNormalize, setFlowNormalize] = useState(true);
  const [roiEnabled, setRoiEnabled] = useState(false);
  const [selectedRoiId, setSelectedRoiId] = useState<number | null>(null);
  const [roiName, setRoiName] = useState('');
  const [roiPointsDraft, setRoiPointsDraft] = useState<RoiPoint[] | null>(null);
  const [tileRows, setTileRows] = useState(1);
  const [tileCols, setTileCols] = useState(1);
  const [savingRoi, setSavingRoi] = useState(false);
  const [preview, setPreview] = useState<InspectPreview | null>(null);
  const [previewSignature, setPreviewSignature] = useState('');
  const [previewLoading, setPreviewLoading] = useState(false);
  const [runLoading, setRunLoading] = useState(false);
  const [temporalReference, setTemporalReference] = useState('');
  const [temporalWindowMinutes, setTemporalWindowMinutes] = useState(30);
  const [temporalStride, setTemporalStride] = useState(1);
  const [temporalLags, setTemporalLags] = useState('1, 2, 4, 8, 16, 32, 64, 128');
  const [temporalMetric, setTemporalMetric] = useState<'mae' | 'mse' | 'ssim'>('mae');
  const [temporalUseRoi, setTemporalUseRoi] = useState(false);
  const [temporalRunId, setTemporalRunId] = useState<number | null>(null);
  const [temporalResult, setTemporalResult] = useState<TemporalDynamicsResult | null>(null);
  const rowActions = usePendingIds();

  async function refreshArtifacts() {
    const result = await listInspectArtifacts({
      page: artifactPage,
      training_dataset_id: artifactDatasetFilter,
      preprocessing_pipeline_id: artifactPipelineFilter,
      mode: artifactModeFilter,
      status: artifactStatusFilter,
    });
    setArtifactItems(result.items);
    setArtifactTotal(result.total);
    setArtifactPages(result.pages);
    setArtifactActiveTotal(result.active_total);
    if (result.page !== artifactPage) setArtifactPage(result.page);
  }

  async function refresh() {
    const [nextDatasets, nextPipelines, nextRois, nextRuns] = await Promise.all([
      listTrainingDatasets(),
      listPreprocessingPipelines(),
      listRois(),
      listInspectRuns(),
    ]);
    setTrainingDatasets(nextDatasets);
    setPreprocessingPipelines(nextPipelines);
    setRois(nextRois);
    setRuns(nextRuns);
    await refreshArtifacts();
  }

  useEffect(() => {
    if (!active) return;
    refresh().catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Could not load Inspect data',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    });
  }, [active]);

  useEffect(() => {
    if (!active) return;
    refreshArtifacts().catch(() => undefined);
  }, [active, artifactPage, artifactModeFilter, artifactDatasetFilter, artifactPipelineFilter, artifactStatusFilter]);

  useEffect(() => {
    if (!active || trainingDatasetId == null || preprocessingPipelineId == null) {
      setMatchingArtifacts([]);
      return;
    }
    let cancelled = false;
    (async () => {
      const all: InspectArtifactRun[] = [];
      let page = 1;
      let pages = 1;
      do {
        const result = await listInspectArtifacts({ page, training_dataset_id: trainingDatasetId, preprocessing_pipeline_id: preprocessingPipelineId });
        all.push(...result.items);
        pages = result.pages;
        page += 1;
      } while (page <= pages);
      if (!cancelled) setMatchingArtifacts(all);
    })().catch(() => { if (!cancelled) setMatchingArtifacts([]); });
    return () => { cancelled = true; };
  }, [active, trainingDatasetId, preprocessingPipelineId, artifactItems]);

  useEffect(() => {
    if (!active) return;
    const hasActive = artifactActiveTotal > 0 || runs.some((run) => run.status === 'queued' || run.status === 'running');
    if (!hasActive) return;
    const timer = window.setInterval(() => {
      Promise.all([listInspectRuns().then(setRuns), refreshArtifacts()]).catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [active, runs, artifactActiveTotal, artifactPage, artifactModeFilter, artifactDatasetFilter, artifactPipelineFilter, artifactStatusFilter]);

  const selectedDataset = trainingDatasets.find((dataset) => dataset.id === trainingDatasetId) ?? null;
  const selectedRoi = rois.find((roi) => roi.id === selectedRoiId) ?? null;
  const minDate = toInputDateTime(selectedDataset?.start_timestamp ?? null);
  const maxDate = toInputDateTime(selectedDataset?.end_timestamp ?? null);
  const analysisConfig = useMemo<Record<string, unknown>>(() => {
    if (analysisMode === 'temporal_dynamics') {
      const lags = parseLagValues(temporalLags);
      return {
        reference_timestamp: temporalReference,
        analysis_window_seconds: Math.max(2, Math.round(temporalWindowMinutes * 60)),
        lags_seconds: lags,
        distance_metric: temporalMetric,
        autocorrelation_max_lag_seconds: Math.max(128, ...lags),
        autocorrelation_threshold: 0.2,
      };
    }
    if (analysisMode === 'energy') {
      return {
        energy_variant: energyVariant,
        aggregation: energyAggregation,
        window_size: energyWindowSize,
        window_aggregation: 'sum',
        normalize_by_pixels: energyNormalize,
      };
    }
    if (analysisMode === 'optical_flow') {
      return {
        flow_method: 'farneback',
        aggregation: flowAggregation,
        normalize_by_pixels: flowNormalize,
        pyr_scale: flowPyrScale,
        levels: flowLevels,
        winsize: flowWinSize,
        iterations: flowIterations,
        poly_n: flowPolyN,
        poly_sigma: flowPolySigma,
      };
    }
    if (analysisMode === 'contrast_enhanced') {
      return {
        reference_frames: contrastReferenceFrames,
        shift: contrastShift,
        vmax: contrastVmax,
        ma_radius: contrastMaRadius,
      };
    }
    return {};
  }, [
    analysisMode,
    contrastMaRadius,
    contrastReferenceFrames,
    contrastShift,
    contrastVmax,
    energyAggregation,
    energyNormalize,
    energyVariant,
    energyWindowSize,
    flowAggregation,
    flowIterations,
    flowLevels,
    flowNormalize,
    flowPolyN,
    flowPolySigma,
    flowPyrScale,
    flowWinSize,
    temporalLags,
    temporalMetric,
    temporalReference,
    temporalWindowMinutes,
  ]);
  const currentSignature = selectionSignature({
    trainingDatasetId,
    preprocessingPipelineId,
    start,
    end,
    stride,
    analysisMode,
    analysisConfig,
    roiId: roiEnabled ? selectedRoiId : null,
    generateVideo,
    contrastEnabled: analysisMode === 'contrast_enhanced',
    contrastReferenceFrames,
    contrastShift,
    contrastVmax,
    contrastMaRadius,
  });
  const previewFresh = Boolean(preview && previewSignature === currentSignature);
  const invalidRange = Boolean(start && end && end < start);
  const canPreview = analysisMode !== 'temporal_dynamics' && Boolean(trainingDatasetId && preprocessingPipelineId && start && end && !invalidRange);
  const roiReadyForRun = !roiEnabled || selectedRoiId !== null;
  const temporalLagValues = parseLagValues(temporalLags);
  const temporalLagsValid = temporalLagValues.length > 0 && temporalLagValues.length <= 32;
  const canRunTemporal = Boolean(
    trainingDatasetId
    && preprocessingPipelineId
    && temporalReference
    && temporalWindowMinutes > 0
    && temporalLagsValid
    && (!temporalUseRoi || selectedRoiId != null)
    && !runLoading
  );
  const canRun = analysisMode === 'temporal_dynamics'
    ? canRunTemporal
    : Boolean(canPreview && !runLoading && roiReadyForRun && (analysisMode !== 'contrast_enhanced' || contrastVmax > 0));
  const temporalCurrentRun = temporalRunId == null ? null : runs.find((run) => run.id === temporalRunId) ?? null;

  useEffect(() => {
    if (temporalCurrentRun?.status !== 'finished' || temporalResult) return;
    getTemporalDynamicsRunResult(temporalCurrentRun.id)
      .then((result) => {
        setTemporalResult(result);
        notifications.show({ color: 'green', title: 'Temporal analysis complete', message: `${result.loaded_frame_count} frames analyzed` });
      })
      .catch((error) => notifications.show({ color: 'red', title: 'Could not load temporal result', message: error instanceof Error ? error.message : 'Unknown error' }));
  }, [temporalCurrentRun?.status, temporalCurrentRun?.id, temporalResult]);

  function handleDatasetChange(value: string | null) {
    const id = value ? Number(value) : null;
    const dataset = trainingDatasets.find((item) => item.id === id) ?? null;
    setTrainingDatasetId(id);
    setStart(toInputDateTime(dataset?.start_timestamp ?? null));
    setEnd(toInputDateTime(dataset?.end_timestamp ?? null));
    setTemporalReference(midpointTimestamp(dataset?.start_timestamp ?? null, dataset?.end_timestamp ?? null));
    setTemporalResult(null);
    setPreview(null);
    setPreviewSignature('');
  }

  function markPreviewStale() {
    setPreviewSignature('');
  }

  function handleAutoFitContrast() {
    if (
      !preview?.contrast_enabled ||
      preview.contrast_diff_min == null ||
      preview.contrast_diff_max == null
    ) {
      return;
    }
    const diffMin = preview.contrast_diff_min;
    const diffMax = preview.contrast_diff_max;
    setContrastShift(Math.round(-diffMin));
    setContrastVmax(Math.max(1, Math.round(diffMax - diffMin)));
    markPreviewStale();
  }

  async function handleSaveRoi() {
    if (!preview || !roiPointsDraft || !roiName.trim()) return;
    const rect = boundingRect(roiPointsDraft);
    setSavingRoi(true);
    try {
      const roi = await createRoi({
        name: roiName.trim(),
        description: 'Created from Inspect',
        image_width: preview.width,
        image_height: preview.height,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        geometry_type: 'polygon',
        points: roiPointsDraft,
        tile_rows: tileRows,
        tile_cols: tileCols,
      });
      const nextRois = await listRois();
      setRois(nextRois);
      setSelectedRoiId(roi.id);
      notifications.show({ color: 'green', title: 'ROI saved', message: roi.name });
      markPreviewStale();
    } catch (error) {
      notifications.show({ color: 'red', title: 'ROI save failed', message: error instanceof Error ? error.message : 'Unknown error' });
    } finally {
      setSavingRoi(false);
    }
  }

  async function handlePreview() {
    if (!canPreview || trainingDatasetId == null || preprocessingPipelineId == null) return;
    setPreviewLoading(true);
    try {
      const result = await previewInspect({
        training_dataset_id: trainingDatasetId,
        preprocessing_pipeline_id: preprocessingPipelineId,
        start_timestamp: start,
        end_timestamp: end,
        stride,
        analysis_mode: analysisMode,
        analysis_config: analysisConfig,
        roi_id: roiEnabled ? selectedRoiId : null,
        generate_video: generateVideo,
        fps,
        contrast_enabled: analysisMode === 'contrast_enhanced',
        contrast_reference_frames: contrastReferenceFrames,
        contrast_shift: contrastShift,
        contrast_vmax: contrastVmax,
        contrast_ma_radius: contrastMaRadius,
      });
      setPreview(result);
      if (roiEnabled && !selectedRoiId && !roiPointsDraft) {
        setRoiPointsDraft(defaultRoiPoints(result));
      }
      setPreviewSignature(currentSignature);
    } catch (error) {
      setPreview(null);
      setPreviewSignature('');
      notifications.show({
        color: 'red',
        title: 'Inspect preview failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setPreviewLoading(false);
    }
  }

  async function handleRun() {
    if (!canRun || trainingDatasetId == null || preprocessingPipelineId == null) return;
    setRunLoading(true);
    try {
      const temporalRange = analysisMode === 'temporal_dynamics'
        ? centeredTimeRange(temporalReference, temporalWindowMinutes)
        : null;
      const created = await createInspectRun({
        training_dataset_id: trainingDatasetId,
        preprocessing_pipeline_id: preprocessingPipelineId,
        start_timestamp: temporalRange?.start ?? start,
        end_timestamp: temporalRange?.end ?? end,
        stride: analysisMode === 'temporal_dynamics' ? temporalStride : stride,
        fps,
        analysis_mode: analysisMode,
        analysis_config: analysisConfig,
        roi_id: analysisMode === 'temporal_dynamics' ? (temporalUseRoi ? selectedRoiId : null) : (roiEnabled ? selectedRoiId : null),
        generate_video: analysisMode === 'temporal_dynamics' ? false : generateVideo,
        contrast_enabled: analysisMode === 'contrast_enhanced',
        contrast_reference_frames: contrastReferenceFrames,
        contrast_shift: contrastShift,
        contrast_vmax: contrastVmax,
        contrast_ma_radius: contrastMaRadius,
      });
      const queuedArtifact = artifactFromInspectRun(created);
      const matchesRunFilters =
        (!artifactModeFilter || queuedArtifact.mode === artifactModeFilter)
        && (artifactDatasetFilter == null || queuedArtifact.training_dataset_id === artifactDatasetFilter)
        && (artifactPipelineFilter == null || queuedArtifact.preprocessing_pipeline_id === artifactPipelineFilter)
        && (!artifactStatusFilter || queuedArtifact.status === artifactStatusFilter);
      setRuns((current) => [created, ...current.filter((run) => run.id !== created.id)]);
      if (analysisMode === 'temporal_dynamics') {
        setTemporalRunId(created.id);
        setTemporalResult(null);
      }
      if (matchesRunFilters) {
        setArtifactItems((current) => [queuedArtifact, ...current.filter((run) => run.kind !== 'inspect' || run.id !== created.id)].slice(0, 15));
        setArtifactTotal((current) => current + 1);
      }
      setArtifactActiveTotal((current) => current + 1);
      if (created.training_dataset_id === trainingDatasetId && created.preprocessing_pipeline_id === preprocessingPipelineId) {
        setMatchingArtifacts((current) => [queuedArtifact, ...current.filter((run) => run.kind !== 'inspect' || run.id !== created.id)]);
      }
      setRunLoading(false);
      refreshArtifacts().catch(() => undefined);
      notifications.show({ color: 'green', title: analysisMode === 'temporal_dynamics' ? 'Temporal analysis queued' : 'Inspect run queued', message: created.training_dataset_name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Inspect run failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
      setRunLoading(false);
    }
  }

  async function handleArtifactAbort(run: InspectArtifactRun) {
    await rowActions.runPending(`artifact-abort:${run.kind}:${run.id}`, async () => {
      if (run.kind === 'heatmap') await abortHeatmapRange(run.id);
      else await abortInspectRun(run.id);
      await refreshArtifacts();
    }).catch((error) => notifications.show({ color: 'red', title: 'Abort failed', message: error instanceof Error ? error.message : 'Unknown error' }));
  }

  async function handleArtifactLoad(run: InspectArtifactRun) {
    if (run.kind === 'inspect' && run.mode === 'temporal_dynamics' && run.has_summary) {
      try {
        const result = await getTemporalDynamicsRunResult(run.id);
        setAnalysisMode('temporal_dynamics');
        setTrainingDatasetId(result.training_dataset_id);
        setPreprocessingPipelineId(result.preprocessing_pipeline_id);
        setTemporalReference(toInputDateTime(result.reference_timestamp));
        setTemporalStride(result.stride);
        setTemporalRunId(run.id);
        setTemporalResult(result);
        setSelectedArtifact(null);
      } catch (error) {
        notifications.show({ color: 'red', title: 'Could not load temporal result', message: error instanceof Error ? error.message : 'Unknown error' });
      }
      return;
    }
    setSelectedArtifact(run);
  }

  async function handleArtifactDelete(run: InspectArtifactRun) {
    if (!window.confirm(`Delete ${run.kind} run ${run.id}?`)) return;
    await rowActions.runPending(`artifact-delete:${run.kind}:${run.id}`, async () => {
      if (run.kind === 'heatmap') await deleteHeatmapRange(run.id);
      else await deleteInspectRun(run.id);
      if (selectedArtifact?.kind === run.kind && selectedArtifact.id === run.id) setSelectedArtifact(null);
      await refreshArtifacts();
    }).catch((error) => notifications.show({ color: 'red', title: 'Delete failed', message: error instanceof Error ? error.message : 'Unknown error' }));
  }

  return (
    <Stack gap="lg">
      <div>
        <Title order={2}>Inspect Dataset</Title>
        <Text c="dimmed" size="sm">
          Inspect preprocessed image sequences with videos, diagnostics, and training-free temporal analysis.
        </Text>
      </div>

      <StepCard title="Inspect source" subtitle="Choose the source and analysis method; only the required range and output controls are shown." color="blue">
        <Stack gap="md">
          <Paper withBorder p="md" radius="md" bg="var(--mantine-color-blue-0)">
            <Stack gap="sm">
              <div>
                <Text fw={700}>1. Source combination</Text>
                <Text size="xs" c="dimmed">Select the dataset and preprocessing pipeline whose existing and new artifacts you want to inspect.</Text>
              </div>
              <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
            <Select
              label="Train/Test Dataset"
              placeholder="Select dataset rules"
              searchable
              data={trainingDatasets.map((dataset) => ({
                value: String(dataset.id),
                label: `${dataset.name} (${dataset.total_selected_images} images)`,
              }))}
              value={trainingDatasetId == null ? null : String(trainingDatasetId)}
              onChange={handleDatasetChange}
            />
            <Select
              label="Preprocessing Pipeline"
              placeholder="Select preprocessing"
              searchable
              data={preprocessingPipelines.map((pipeline) => ({
                value: String(pipeline.id),
                label: `${pipeline.name} (${pipeline.output_width ?? '?'}x${pipeline.output_height ?? '?'})`,
              }))}
              value={preprocessingPipelineId == null ? null : String(preprocessingPipelineId)}
              onChange={(value) => {
                setPreprocessingPipelineId(value ? Number(value) : null);
                setTemporalResult(null);
                markPreviewStale();
              }}
            />
              </SimpleGrid>
            </Stack>
          </Paper>

          {trainingDatasetId != null && preprocessingPipelineId != null && (
            <Paper withBorder p="md" radius="md">
              <Stack gap="xs">
                <Group justify="space-between">
                  <Text fw={600} size="sm">Available artifacts for this combination</Text>
                  <Badge variant="light">{matchingArtifacts.length}</Badge>
                </Group>
                {matchingArtifacts.length === 0 ? (
                  <Text size="sm" c="dimmed">No Inspect or heatmap artifacts yet.</Text>
                ) : (
                  <Group gap="xs">
                    {matchingArtifacts.map((artifact) => (
                      <Button
                        key={`${artifact.kind}:${artifact.id}`}
                        size="compact-sm"
                        variant="light"
                        disabled={artifact.status !== 'finished' || (!artifact.has_video && !artifact.has_csv && !artifact.has_summary)}
                        onClick={() => void handleArtifactLoad(artifact)}
                      >
                        {artifact.mode.replaceAll('_', ' ')} · #{artifact.id} · {artifact.status}
                      </Button>
                    ))}
                  </Group>
                )}
              </Stack>
            </Paper>
          )}

          <Paper withBorder p="md" radius="md">
            <Stack gap="md">
              <div>
                <Text fw={700}>2. Analysis method</Text>
                <Text size="xs" c="dimmed">Only controls relevant to the selected method are shown below.</Text>
              </div>
              <Divider />
              <Select
                label={<InfoLabel label="Inspect method" info="Choose the analysis generated from the selected Train/Test dataset and preprocessing pipeline. Only method-specific controls are shown." />}
                data={[
                  { value: 'preprocessed_video', label: 'Preprocessed video' },
                  { value: 'contrast_enhanced', label: 'Contrast enhance' },
                  { value: 'energy', label: 'Energy' },
                  { value: 'optical_flow', label: 'Optical flow' },
                  { value: 'temporal_dynamics', label: 'Temporal Dynamics Analysis' },
                ]}
                value={analysisMode}
                onChange={(value) => {
                  const next = (value ?? 'preprocessed_video') as InspectAnalysisMode;
                  setAnalysisMode(next);
                  setContrastEnabled(next === 'contrast_enhanced');
                  setGenerateVideo(next === 'preprocessed_video' || next === 'contrast_enhanced');
                  markPreviewStale();
                }}
              />

              {analysisMode === 'temporal_dynamics' && (
                <Paper withBorder p="md" radius="sm" bg="var(--mantine-color-grape-0)">
                  <Stack gap="md">
                    <Text size="sm" c="dimmed">
                      Uses the final preprocessing-pipeline output directly. No additional resize or downsampling is applied.
                    </Text>
                    <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md">
                      <DateTime24Input
                        label="Reference timestamp t"
                        value={temporalReference}
                        min={minDate}
                        max={maxDate}
                        onChange={(value) => { setTemporalReference(value); setTemporalResult(null); }}
                      />
                      <NumberInput
                        label="Analysis period around t [minutes]"
                        description="Total centered window (30 min = ±15 min)"
                        min={1 / 30}
                        step={1}
                        decimalScale={2}
                        value={temporalWindowMinutes}
                        onChange={(value) => { setTemporalWindowMinutes(Math.max(1 / 30, Number(value) || 30)); setTemporalResult(null); }}
                      />
                      <NumberInput
                        label="Analysis stride"
                        description="Use every nth frame selected by the Train/Test dataset"
                        min={1}
                        value={temporalStride}
                        onChange={(value) => { setTemporalStride(Math.max(1, Number(value) || 1)); setTemporalResult(null); }}
                      />
                      <Select
                        label="Distance metric"
                        data={[
                          { value: 'mae', label: 'MAE' },
                          { value: 'mse', label: 'MSE' },
                          { value: 'ssim', label: 'SSIM distance (1 - SSIM)' },
                        ]}
                        value={temporalMetric}
                        onChange={(value) => { setTemporalMetric((value ?? 'mae') as 'mae' | 'mse' | 'ssim'); setTemporalResult(null); }}
                      />
                    </SimpleGrid>
                    <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
                      <TextInput
                        label="Lag values [seconds]"
                        description="Comma or space separated"
                        value={temporalLags}
                        error={temporalLagsValid ? undefined : 'Enter 1–32 positive whole-second lag values.'}
                        onChange={(event) => { setTemporalLags(event.currentTarget.value); setTemporalResult(null); }}
                      />
                      <Switch
                        label="Use saved ROI"
                        description="Aggregate the exact saved Inspect ROI across its tiles"
                        checked={temporalUseRoi}
                        onChange={(event) => { setTemporalUseRoi(event.currentTarget.checked); setTemporalResult(null); }}
                      />
                      <Select
                        label="Temporal analysis ROI"
                        placeholder="Choose saved ROI"
                        disabled={!temporalUseRoi}
                        searchable
                        data={rois.map((roi) => ({ value: String(roi.id), label: roiLabel(roi) }))}
                        value={selectedRoiId == null ? null : String(selectedRoiId)}
                        onChange={(value) => { setSelectedRoiId(value ? Number(value) : null); setTemporalResult(null); }}
                      />
                    </SimpleGrid>
                    <Text size="xs" c="dimmed">
                      Missing frames and cadence gaps are skipped. Lag pairs never cross source folders or detected gaps.
                    </Text>
                  </Stack>
                </Paper>
              )}

              {(analysisMode === 'energy' || analysisMode === 'optical_flow') && (
                <Switch
                  label={<InfoLabel label="Generate video" info="Off stores CSV, summary JSON and plot preview only. On additionally writes an MP4 overlay/flow video." />}
                  checked={generateVideo}
                  onChange={(event) => {
                    setGenerateVideo(event.currentTarget.checked);
                    markPreviewStale();
                  }}
                />
              )}

              {analysisMode === 'contrast_enhanced' && (
                <>
                  <Group grow align="flex-start">
                    <NumberInput label={<InfoLabel label="Reference frames" info="First N frames averaged into the mean reference that is subtracted from each frame." />} min={1} value={contrastReferenceFrames} onChange={(value) => { setContrastReferenceFrames(Math.max(1, Number(value) || 1)); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Shift" info="Value added after frame minus reference. Useful to move negative differences into visible range." />} value={contrastShift} onChange={(value) => { setContrastShift(Number(value) || 0); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="vmax" info="Upper clip value mapped to white. Must be greater than zero." />} min={1} value={contrastVmax} error={contrastVmax > 0 ? undefined : 'Must be > 0'} onChange={(value) => { setContrastVmax(Math.max(1, Number(value) || 1)); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Moving average radius" info="Centered temporal smoothing radius. 0 disables smoothing." />} min={0} value={contrastMaRadius} onChange={(value) => { setContrastMaRadius(Math.max(0, Number(value) || 0)); markPreviewStale(); }} />
                  </Group>
                  {preview?.contrast_enabled && preview.contrast_diff_min != null && (
                    <Alert color="grape" title="Preview diff range">
                      <Text size="sm">
                        Frame − reference spans {Math.round(preview.contrast_diff_min)} to {Math.round(preview.contrast_diff_max ?? 0)}.
                      </Text>
                      <Button mt="xs" size="compact-sm" variant="light" color="grape" onClick={handleAutoFitContrast}>
                        Auto-fit shift &amp; vmax
                      </Button>
                    </Alert>
                  )}
                </>
              )}

              {analysisMode === 'energy' && (
                <Group grow align="flex-start">
                  <Select label={<InfoLabel label="Energy variant" info="Pairwise compares each consecutive frame pair. Window aggregates pairwise differences over a causal time window." />} data={[{ value: 'pairwise', label: 'Pairwise frame energy' }, { value: 'window', label: 'Window activity energy' }]} value={energyVariant} onChange={(value) => { setEnergyVariant((value ?? 'pairwise') as 'pairwise' | 'window'); markPreviewStale(); }} />
                  <Select label={<InfoLabel label="Aggregation" info="Sum measures total changed intensity, mean normalizes by area, p95 focuses on strong local changes." />} data={[{ value: 'sum', label: 'Sum' }, { value: 'mean', label: 'Mean' }, { value: 'p95', label: 'P95' }]} value={energyAggregation} onChange={(value) => { setEnergyAggregation((value ?? 'sum') as 'sum' | 'mean' | 'p95'); markPreviewStale(); }} />
                  <NumberInput label={<InfoLabel label="Window size" info="Number of consecutive pairwise energies aggregated when variant is Window." />} min={1} disabled={energyVariant !== 'window'} value={energyWindowSize} onChange={(value) => { setEnergyWindowSize(Math.max(1, Number(value) || 1)); markPreviewStale(); }} />
                  <Switch label={<InfoLabel label="Normalize by pixels" info="For sum aggregation, divide by pixel count so tiles of different size are comparable." />} checked={energyNormalize} onChange={(event) => { setEnergyNormalize(event.currentTarget.checked); markPreviewStale(); }} />
                </Group>
              )}

              {analysisMode === 'optical_flow' && (
                <Stack gap="sm">
                  <Group grow align="flex-start">
                    <Select label={<InfoLabel label="Flow aggregation" info="How motion magnitude is reduced to one score per frame or tile." />} data={[{ value: 'mean_magnitude', label: 'Mean magnitude' }, { value: 'p95_magnitude', label: 'P95 magnitude' }, { value: 'max_magnitude', label: 'Max magnitude' }]} value={flowAggregation} onChange={(value) => { setFlowAggregation((value ?? 'mean_magnitude') as 'mean_magnitude' | 'p95_magnitude' | 'max_magnitude'); markPreviewStale(); }} />
                    <Switch label={<InfoLabel label="Normalize by pixels" info="Keeps scores comparable between full image and ROI/tile masks." />} checked={flowNormalize} onChange={(event) => { setFlowNormalize(event.currentTarget.checked); markPreviewStale(); }} />
                  </Group>
                  <Group grow align="flex-start">
                    <NumberInput label={<InfoLabel label="Pyramid scale" info="Farneback scale between pyramid levels. 0.5 is the common OpenCV default." />} min={0.1} max={0.99} step={0.05} value={flowPyrScale} onChange={(value) => { setFlowPyrScale(Number(value) || 0.5); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Levels" info="Number of pyramid levels used by Farneback optical flow." />} min={1} value={flowLevels} onChange={(value) => { setFlowLevels(Math.max(1, Number(value) || 1)); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Window size" info="Averaging window size for Farneback; larger windows smooth motion more." />} min={3} value={flowWinSize} onChange={(value) => { setFlowWinSize(Math.max(3, Number(value) || 15)); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Iterations" info="Solver iterations at each pyramid level." />} min={1} value={flowIterations} onChange={(value) => { setFlowIterations(Math.max(1, Number(value) || 1)); markPreviewStale(); }} />
                  </Group>
                  <Group grow align="flex-start">
                    <NumberInput label={<InfoLabel label="Poly N" info="Pixel neighborhood size for polynomial expansion. Typical values are 5 or 7." />} min={5} value={flowPolyN} onChange={(value) => { setFlowPolyN(Math.max(5, Number(value) || 5)); markPreviewStale(); }} />
                    <NumberInput label={<InfoLabel label="Poly sigma" info="Gaussian sigma for polynomial expansion. 1.1–1.5 is typical." />} min={0.1} step={0.1} value={flowPolySigma} onChange={(value) => { setFlowPolySigma(Number(value) || 1.2); markPreviewStale(); }} />
                  </Group>
                </Stack>
              )}

              {(analysisMode === 'energy' || analysisMode === 'optical_flow') && (
                <Paper withBorder p="sm" radius="sm">
                  <Stack gap="sm">
                    <Switch label={<InfoLabel label="Use ROI / tiles" info="Restrict diagnostics to a reusable quadrilateral ROI. If the ROI has tiles, a separate time series is calculated per tile." />} checked={roiEnabled} onChange={(event) => { setRoiEnabled(event.currentTarget.checked); markPreviewStale(); }} />
                    {roiEnabled && (
                      <>
                        <Group grow align="flex-end">
                          <Select
                            label="Saved ROI"
                            placeholder="Choose saved ROI or define a new one"
                            data={rois.map((roi) => ({ value: String(roi.id), label: roiLabel(roi) }))}
                            value={selectedRoiId == null ? null : String(selectedRoiId)}
                            clearable
                            searchable
                            onChange={(value) => {
                              const roi = value ? rois.find((item) => item.id === Number(value)) ?? null : null;
                              setSelectedRoiId(roi?.id ?? null);
                              if (roi) {
                                setTileRows(roi.tile_rows ?? 1);
                                setTileCols(roi.tile_cols ?? 1);
                                setRoiPointsDraft(roiPoints(roi));
                              }
                              markPreviewStale();
                            }}
                          />
                          <TextInput label="New ROI name" placeholder="Optional new ROI" value={roiName} onChange={(event) => setRoiName(event.currentTarget.value)} />
                          <Button leftSection={<Save size={16} />} variant="light" loading={savingRoi} disabled={!preview || !roiPointsDraft || !roiName.trim()} onClick={handleSaveRoi}>
                            Save ROI
                          </Button>
                        </Group>
                        <Group grow>
                          <Select label="Tile rows" data={TILE_OPTIONS} value={String(tileRows)} onChange={(value) => { setTileRows(Number(value ?? 1)); markPreviewStale(); }} />
                          <Select label="Tile columns" data={TILE_OPTIONS} value={String(tileCols)} onChange={(value) => { setTileCols(Number(value ?? 1)); markPreviewStale(); }} />
                        </Group>
                        {preview && (
                          <PolygonRoiPicker
                            preview={preview}
                            points={roiPointsDraft ?? defaultRoiPoints(preview)}
                            tileRows={tileRows}
                            tileCols={tileCols}
                            onChange={(points) => {
                              setRoiPointsDraft(points);
                              setSelectedRoiId(null);
                              markPreviewStale();
                            }}
                          />
                        )}
                        {selectedRoi && preview && (selectedRoi.image_width !== preview.width || selectedRoi.image_height !== preview.height) && (
                          <Alert color="red">
                            Selected ROI is tuned for {selectedRoi.image_width}x{selectedRoi.image_height}, but preview output is {preview.width}x{preview.height}.
                          </Alert>
                        )}
                        {roiEnabled && roiPointsDraft && selectedRoiId === null && (
                          <Alert color="yellow">
                            Save the drawn ROI before running so Inspect can reuse the global ROI definition.
                          </Alert>
                        )}
                      </>
                    )}
                  </Stack>
                </Paper>
              )}
            </Stack>
          </Paper>

          {analysisMode !== 'temporal_dynamics' && <Paper withBorder p="md" radius="md">
            <Stack gap="sm">
              <div>
                <Text fw={700}>3. Time range &amp; output</Text>
                <Text size="xs" c="dimmed">The selected range is queued immediately; exact frame counting happens in the background worker.</Text>
              </div>
              {selectedDataset && (
                <Alert color="blue" title="Dataset bounds">
                  {selectedDataset.name}: {formatTimestamp(selectedDataset.start_timestamp)} to{' '}
                  {formatTimestamp(selectedDataset.end_timestamp)}
                </Alert>
              )}
              <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md">
                <DateTime24Input
                  label="Start"
                  value={start}
                  min={minDate}
                  max={maxDate}
                  error={invalidRange ? 'Invalid range' : undefined}
                  onChange={(value) => { setStart(value); markPreviewStale(); }}
                />
                <DateTime24Input
                  label="End"
                  value={end}
                  min={minDate}
                  max={maxDate}
                  error={invalidRange ? 'Invalid range' : undefined}
                  onChange={(value) => { setEnd(value); markPreviewStale(); }}
                />
                <NumberInput
                  label="Inspect stride"
                  min={1}
                  value={stride}
                  onChange={(value) => { setStride(Math.max(1, Number(value) || 1)); markPreviewStale(); }}
                />
                {(analysisMode === 'preprocessed_video' || analysisMode === 'contrast_enhanced' || generateVideo) && <NumberInput
                  label="Video fps"
                  min={1}
                  max={60}
                  value={fps}
                  onChange={(value) => setFps(Math.max(1, Math.min(60, Number(value) || 1)))}
                />}
              </SimpleGrid>
            </Stack>
          </Paper>}

          <Paper withBorder p="md" radius="md" bg="var(--mantine-color-gray-0)">
          <Group justify="space-between" align="center" wrap="wrap">
            <div>
              <Text fw={700}>{analysisMode === 'temporal_dynamics' ? '3. Queue temporal analysis' : '4. Preview or queue run'}</Text>
              <Text size="xs" c="dimmed">The queued job appears immediately in Inspect runs with live progress and estimated remaining time.</Text>
            </div>
            <Group gap="sm">
            {analysisMode !== 'temporal_dynamics' && <Button
              leftSection={<Eye size={18} />}
              variant="light"
              loading={previewLoading}
              disabled={!canPreview || previewLoading}
              onClick={handlePreview}
            >
              Load preview
            </Button>}
            <Button
              leftSection={analysisMode === 'temporal_dynamics' ? undefined : <FileVideo size={18} />}
              loading={runLoading}
              disabled={!canRun}
              onClick={handleRun}
            >
              {analysisMode === 'temporal_dynamics' ? 'Run Temporal Analysis' : 'Run'}
            </Button>
            </Group>
          </Group>
          </Paper>

          {preview && analysisMode !== 'temporal_dynamics' && (
            <Stack gap="md" style={{ order: 5 }}>
              <Paper withBorder p="md" radius="sm">
                <Stack gap="sm">
                  <Group gap="xs">
                    <Text fw={700}>Preview MP4</Text>
                    <Badge variant="light" color={previewFresh ? 'green' : 'yellow'}>{previewFresh ? 'Preview current' : 'Preview stale'}</Badge>
                    <Badge variant="light">{preview.selected_images} selected images</Badge>
                    <Badge variant="light" color="gray">{preview.preview_frame_count} preview frames</Badge>
                    <Badge variant="light" color="gray">{preview.width}x{preview.height}, {preview.channels} ch</Badge>
                  </Group>
                  {preview.preview_video_url && <video src={inspectPreviewVideoUrl(preview.preview_video_url)} controls muted playsInline className="inspect-video-player" />}
                </Stack>
              </Paper>
              {preview.plot_image_data_url && (
                <Paper withBorder p="md" radius="sm">
                  <Stack gap="sm">
                    <Group gap="xs">
                      <Text fw={700}>Diagnostic preview</Text>
                      <Badge variant="light">{preview.diagnostic_series.length} samples</Badge>
                    </Group>
                    <img src={preview.plot_image_data_url} alt="Inspect diagnostic plot preview" style={{ maxWidth: '100%', borderRadius: 6 }} />
                  </Stack>
                </Paper>
              )}
            </Stack>
          )}

          {analysisMode === 'temporal_dynamics' && temporalCurrentRun && (
            <Paper withBorder p="md" radius="sm" style={{ order: 5 }}>
              <Stack gap="sm">
                <Group justify="space-between" wrap="wrap">
                  <Group gap="xs">
                    <Text fw={700}>Temporal Dynamics run #{temporalCurrentRun.id}</Text>
                    <Badge color={statusColor(temporalCurrentRun.status)} variant="light">{temporalCurrentRun.status}</Badge>
                  </Group>
                  <Text size="sm" c="dimmed">
                    {temporalCurrentRun.status === 'finished'
                      ? `Completed in ${formatDurationSeconds(temporalCurrentRun.duration_seconds)}`
                      : `Estimated remaining: ${formatDurationSeconds(estimatedRemainingSeconds(temporalCurrentRun))}`}
                  </Text>
                </Group>
                <Progress
                  value={temporalCurrentRun.frame_count ? Math.min(100, temporalCurrentRun.done_count / temporalCurrentRun.frame_count * 100) : 0}
                  animated={temporalCurrentRun.status === 'queued' || temporalCurrentRun.status === 'running'}
                />
                <Text size="xs" c="dimmed">
                  {temporalCurrentRun.done_count}{temporalCurrentRun.frame_count ? ` / ${temporalCurrentRun.frame_count} work units` : ' · preparing frame index'}
                </Text>
                {temporalCurrentRun.error_message && <Alert color="red">{temporalCurrentRun.error_message}</Alert>}
              </Stack>
            </Paper>
          )}

          {analysisMode === 'temporal_dynamics' && temporalResult && (
            <div style={{ order: 6 }}><TemporalDynamicsResults result={temporalResult} /></div>
          )}
        </Stack>
      </StepCard>

      <ArtifactViewer artifact={selectedArtifact} />

      <StepCard
        title={`Inspect runs (${artifactTotal})`}
        color="cyan"
        action={
          <Group gap="xs">
            <Button size="compact-sm" variant="subtle" leftSection={<RefreshCw size={14} />} onClick={() => refreshArtifacts()}>Refresh</Button>
            <ActionIcon variant="subtle" aria-label={artifactsOpen ? 'Collapse Inspect runs' : 'Expand Inspect runs'} onClick={() => setArtifactsOpen((current) => !current)}>
              {artifactsOpen ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
            </ActionIcon>
          </Group>
        }
      >
        <Collapse in={artifactsOpen}>
          <Stack gap="md">
            <Group grow align="flex-end">
              <Select label="Mode" clearable data={[
                { value: 'preprocessed_video', label: 'Preprocessed video' },
                { value: 'contrast_enhanced', label: 'Contrast enhanced' },
                { value: 'energy', label: 'Energy' },
                { value: 'optical_flow', label: 'Optical flow' },
                { value: 'temporal_dynamics', label: 'Temporal dynamics' },
                { value: 'heatmap', label: 'Heatmap' },
              ]} value={artifactModeFilter} onChange={(value) => { setArtifactModeFilter(value); setArtifactPage(1); }} />
              <Select label="Train/Test Dataset" clearable searchable data={trainingDatasets.map((item) => ({ value: String(item.id), label: item.name }))} value={artifactDatasetFilter == null ? null : String(artifactDatasetFilter)} onChange={(value) => { setArtifactDatasetFilter(value ? Number(value) : null); setArtifactPage(1); }} />
              <Select label="Preprocessing" clearable searchable data={preprocessingPipelines.map((item) => ({ value: String(item.id), label: item.name }))} value={artifactPipelineFilter == null ? null : String(artifactPipelineFilter)} onChange={(value) => { setArtifactPipelineFilter(value ? Number(value) : null); setArtifactPage(1); }} />
              <Select label="Status" clearable data={['queued', 'running', 'finished', 'failed', 'aborted']} value={artifactStatusFilter} onChange={(value) => { setArtifactStatusFilter(value); setArtifactPage(1); }} />
            </Group>
            <ScrollArea>
              <Table striped verticalSpacing="sm" miw={1050}>
                <Table.Thead><Table.Tr>
                  <Table.Th>Status</Table.Th><Table.Th>Mode</Table.Th><Table.Th>Dataset</Table.Th><Table.Th>Preprocessing</Table.Th><Table.Th>Range</Table.Th><Table.Th>Progress</Table.Th><Table.Th>Artifacts</Table.Th><Table.Th />
                </Table.Tr></Table.Thead>
                <Table.Tbody>
                  {artifactItems.map((run) => {
                    const busy = run.status === 'queued' || run.status === 'running';
                    return <Table.Tr key={`${run.kind}:${run.id}`}>
                      <Table.Td><Badge color={statusColor(run.status)} variant="light">{run.status}</Badge>{run.error_message && <Text size="xs" c="red">{run.error_message}</Text>}</Table.Td>
                      <Table.Td><Badge variant="light" color={run.kind === 'heatmap' ? 'red' : 'cyan'}>{run.mode.replaceAll('_', ' ')}</Badge></Table.Td>
                      <Table.Td>{run.training_dataset_name}</Table.Td>
                      <Table.Td>{run.preprocessing_pipeline_name}</Table.Td>
                      <Table.Td><Text size="xs">{formatTimestamp(run.start_timestamp)}</Text><Text size="xs">{formatTimestamp(run.end_timestamp)}</Text></Table.Td>
                      <Table.Td>
                        <Text size="xs">{run.done_count}{run.frame_count ? ` / ${run.frame_count}` : ''}</Text>
                        {run.frame_count ? <Progress value={(run.done_count / run.frame_count) * 100} size="xs" animated={busy} /> : null}
                        {run.status === 'running' && <Text size="xs" c="dimmed">ETA {formatDurationSeconds(estimatedRemainingSeconds(run))}</Text>}
                        {run.status === 'finished' && run.duration_seconds != null && <Text size="xs" c="dimmed">{formatDurationSeconds(run.duration_seconds)}</Text>}
                      </Table.Td>
                      <Table.Td><Group gap={4} wrap="nowrap">
                        {run.has_video && <Button size="compact-xs" variant="light" component="a" href={artifactVideoUrl(run)} download>MP4</Button>}
                        {run.has_csv && <Button size="compact-xs" variant="light" component="a" href={inspectRunCsvUrl(run.id)} download>CSV</Button>}
                        {run.mode === 'temporal_dynamics' && run.has_summary && <Badge variant="light" color="grape">Results</Badge>}
                        {run.status === 'finished' && !run.has_video && run.kind === 'heatmap' && <Text size="xs" c="orange">Re-render MP4 in Analysis</Text>}
                      </Group></Table.Td>
                      <Table.Td><Group gap={4} justify="flex-end" wrap="nowrap">
                        {run.status === 'finished' && (run.has_video || run.has_csv || run.has_summary) && <Button size="compact-xs" onClick={() => void handleArtifactLoad(run)}>Load</Button>}
                        {busy && <ActionIcon color="yellow" variant="subtle" loading={rowActions.isPending(`artifact-abort:${run.kind}:${run.id}`)} onClick={() => handleArtifactAbort(run)}><Square size={16} /></ActionIcon>}
                        <ActionIcon color="red" variant="subtle" disabled={busy} loading={rowActions.isPending(`artifact-delete:${run.kind}:${run.id}`)} onClick={() => handleArtifactDelete(run)}><Trash2 size={16} /></ActionIcon>
                      </Group></Table.Td>
                    </Table.Tr>;
                  })}
                </Table.Tbody>
              </Table>
            </ScrollArea>
            {artifactItems.length === 0 && <Alert color="blue">No runs match the combined filters.</Alert>}
            {artifactPages > 1 && <Group justify="center"><Pagination total={artifactPages} value={artifactPage} onChange={setArtifactPage} /></Group>}
          </Stack>
        </Collapse>
      </StepCard>
    </Stack>
  );
}
