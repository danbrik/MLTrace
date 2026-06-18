import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Collapse,
  Group,
  Loader,
  Modal,
  NumberInput,
  Paper,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { ArrowDown, ArrowUp, Check, ChevronDown, ChevronRight, Info, Pause, Pencil, Play, Plus, RotateCcw, Search, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import type React from 'react';

import {
  createHeatmap,
  getTestingRunResults,
  listMethodConfigurations,
  listPreprocessingPipelines,
  listTestingRuns,
  listTrainingDatasets,
  listTrainingPipelines,
  listTrainingRuns,
} from '../api';
import { StepCard } from '../components/StepCard';
import { formatValue } from '../methods/utils';
import { datasetResolutions, formatResolution, orderedGraphNodes, stepDetail } from '../training/graph';
import type {
  HeatmapRun,
  MethodConfiguration,
  PreprocessingPipeline,
  TestingRun,
  TestingRunResult,
  TestingRunResults,
  TrainingDataset,
  TrainingPipeline,
  TrainingRun,
} from '../types';

type PlotType = 'timeseries' | 'heatmap';
type HeatmapMode = 'single' | 'range';

type PlotDraft = {
  plotType: PlotType;
  testingRunId: string | null;
  title: string;
  start: string;
  end: string;
  sampling: number;
  movingAverage: number;
  heatmapMode: HeatmapMode;
  timestamp: string | null;
};

type AnalysisPlot = PlotDraft & {
  id: string;
  sources: PlotSourceConfig[];
};

type DetailModalState = {
  title: string;
  body: React.ReactNode;
} | null;

type PlotSourceConfig = {
  testingRunId: string;
  start: string;
  end: string;
  sampling: number;
  timestamp: string | null;
};

type CombinedResult = TestingRunResult & {
  testingRunId: number;
  testingRunName: string;
  heatmapTimestampOnly?: boolean;
};

function notifyError(title: string, error: unknown) {
  notifications.show({ color: 'red', title, message: error instanceof Error ? error.message : 'Unknown error' });
}

function valueAsNumber(value: string | number, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function scoreValue(result: TestingRunResult): number {
  return result.score ?? result.roi_mse ?? result.full_mse;
}

function resultLabel(result: TestingRunResult): string {
  return new Date(result.timestamp).toLocaleString();
}

function pad(value: number): string {
  return String(value).padStart(2, '0');
}

function toDateTimeLocal(value: string | null | undefined): string {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function sourceBounds(dataset: TrainingDataset | null | undefined): { start: string; end: string } {
  if (dataset?.start_timestamp && dataset?.end_timestamp) {
    return { start: toDateTimeLocal(dataset.start_timestamp), end: toDateTimeLocal(dataset.end_timestamp) };
  }
  const starts = (dataset?.rules ?? []).map((rule) => rule.start_timestamp).filter(Boolean).sort();
  const ends = (dataset?.rules ?? []).map((rule) => rule.end_timestamp).filter(Boolean).sort();
  return {
    start: toDateTimeLocal(starts[0]),
    end: toDateTimeLocal(ends.at(-1)),
  };
}

function filterAndSampleResults(
  results: TestingRunResult[],
  start: string,
  end: string,
  sampling: number,
): TestingRunResult[] {
  const startMs = start ? new Date(start).getTime() : Number.NEGATIVE_INFINITY;
  const endMs = end ? new Date(end).getTime() : Number.POSITIVE_INFINITY;
  const step = Math.max(1, Math.floor(sampling));
  return results
    .filter((result) => {
      const timestamp = new Date(result.timestamp).getTime();
      return timestamp >= startMs && timestamp <= endMs;
    })
    .filter((_, index) => index % step === 0);
}

function movingAverage(values: number[], windowSize: number): number[] {
  const size = Math.max(1, Math.floor(windowSize));
  if (size <= 1) return values;
  return values.map((_, index) => {
    const start = Math.max(0, index - size + 1);
    const slice = values.slice(start, index + 1);
    return slice.reduce((sum, value) => sum + value, 0) / slice.length;
  });
}

function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a';
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(3);
  return value.toFixed(5);
}

function DetailButton({ title, body, onOpen }: { title: string; body: React.ReactNode; onOpen: (detail: DetailModalState) => void }) {
  return (
    <Tooltip label={`Inspect ${title.toLowerCase()}`}>
      <ActionIcon size="sm" variant="subtle" onClick={() => onOpen({ title, body })}>
        <Info size={15} />
      </ActionIcon>
    </Tooltip>
  );
}

function DetailModal({ detail, onClose }: { detail: DetailModalState; onClose: () => void }) {
  return (
    <Modal opened={detail !== null} onClose={onClose} title={detail?.title ?? ''} size="xl">
      <Paper withBorder p="sm" radius="sm">
        <ScrollArea h={460}>
          {detail?.body}
        </ScrollArea>
      </Paper>
    </Modal>
  );
}

const USAGE_LABELS: Record<string, string> = { train: 'Train', test: 'Test', validation: 'Validation', mixed: 'Mixed' };

function usageLabel(value: string | undefined): string {
  const key = value ?? 'train';
  return USAGE_LABELS[key] ?? key;
}

function usageColor(value: string | undefined): string {
  if (value === 'test') return 'orange';
  if (value === 'validation') return 'violet';
  if (value === 'mixed') return 'gray';
  return 'teal';
}

function datasetStrides(dataset: TrainingDataset | null): string {
  if (!dataset) return '—';
  const strides = [...new Set(dataset.rules.map((rule) => rule.stride))].sort((a, b) => a - b);
  return strides.length > 0 ? strides.join(', ') : '—';
}

function renderTrainsetDetails(dataset: TrainingDataset | null) {
  if (!dataset) return <Alert color="yellow">Trainset details are not available.</Alert>;
  return (
    <Stack gap="md">
      <Group justify="space-between">
        <div>
          <Text fw={700}>{dataset.name}</Text>
          <Text size="sm" c="dimmed">
            Label {dataset.usage_label} · {dataset.counts_missing ? 'Counts need refresh' : `${dataset.total_selected_images} selected images`} · Sources {dataset.dataset_names.join(', ')}
          </Text>
        </div>
        <Badge variant="light">{dataset.image_resolutions.join(', ') || 'n/a'}</Badge>
      </Group>
      <Table striped verticalSpacing="xs">
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Dataset</Table.Th>
            <Table.Th>Folder</Table.Th>
            <Table.Th>Start</Table.Th>
            <Table.Th>End</Table.Th>
            <Table.Th>Stride</Table.Th>
            <Table.Th>Images</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {dataset.rules.map((rule) => (
            <Table.Tr key={rule.id}>
              <Table.Td>{rule.dataset_name}</Table.Td>
              <Table.Td>{rule.folder_relative_path}</Table.Td>
              <Table.Td>{new Date(rule.start_timestamp).toLocaleString()}</Table.Td>
              <Table.Td>{new Date(rule.end_timestamp).toLocaleString()}</Table.Td>
              <Table.Td>{rule.stride}</Table.Td>
              <Table.Td>{rule.selected_images == null ? 'Needs refresh' : rule.selected_images}</Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Stack>
  );
}

function renderPreprocessingDetails(pipeline: PreprocessingPipeline | null) {
  if (!pipeline) return <Alert color="yellow">Preprocessing details are not available.</Alert>;
  return (
    <Stack gap="sm">
      <Group>
        <Badge variant="light">Input {pipeline.input_width && pipeline.input_height ? `${pipeline.input_width}x${pipeline.input_height}` : 'n/a'}</Badge>
        <Badge variant="light" color="yellow">
          Output {pipeline.output_width && pipeline.output_height ? `${pipeline.output_width}x${pipeline.output_height}` : 'n/a'}
        </Badge>
      </Group>
      {orderedGraphNodes(pipeline).map((node, index) => (
        <Paper key={node.id} withBorder p="sm" radius="sm">
          <Text fw={700} size="sm">
            {index + 1}. {node.type}
          </Text>
          <Text size="xs" c="dimmed">
            {stepDetail(node)}
          </Text>
          <Group gap={6} mt={6}>
            {Object.entries(node.config ?? {}).map(([key, value]) => (
              <Badge key={key} size="sm" variant="light" color="gray">
                {key}={formatValue(value)}
              </Badge>
            ))}
          </Group>
        </Paper>
      ))}
    </Stack>
  );
}

function renderMethodDetails(configuration: MethodConfiguration | null) {
  if (!configuration) return <Alert color="yellow">Method details are not available.</Alert>;
  return (
    <Stack gap="md">
      <div>
        <Text fw={700}>{configuration.name}</Text>
        {configuration.description && <Text size="sm" c="dimmed">{configuration.description}</Text>}
      </div>
      <Group gap={6}>
        {Object.entries(configuration.method_config ?? {}).map(([key, value]) => (
          <Badge key={key} size="sm" variant="light" color="gray">
            {key}={formatValue(value)}
          </Badge>
        ))}
      </Group>
      {(['encoder', 'decoder'] as const).map((section) => {
        const layers = configuration.method_graph[section] ?? [];
        if (!Array.isArray(layers) || layers.length === 0) return null;
        return (
          <Stack key={section} gap="xs">
            <Text fw={700}>{section}</Text>
            {layers.map((layer, index) => (
              <Paper key={layer.id} withBorder p="sm" radius="sm">
                <Text size="sm" fw={600}>{index + 1}. {layer.type}</Text>
                <Group gap={6} mt={6}>
                  {Object.entries(layer.config ?? {}).map(([key, value]) => (
                    <Badge key={key} size="sm" variant="light" color="gray">
                      {key}={formatValue(value)}
                    </Badge>
                  ))}
                </Group>
              </Paper>
            ))}
          </Stack>
        );
      })}
    </Stack>
  );
}

function renderPipelineDetails(
  pipeline: TrainingPipeline | null,
  trainsets: TrainingDataset[],
  preprocessing: PreprocessingPipeline | null,
  method: MethodConfiguration | null,
) {
  if (!pipeline) return <Alert color="yellow">Training pipeline details are not available.</Alert>;
  return (
    <Stack gap="md">
      <div>
        <Text fw={700}>{pipeline.name}</Text>
        {pipeline.description && <Text size="sm" c="dimmed">{pipeline.description}</Text>}
      </div>
      <Group>
        <Badge variant={pipeline.shuffle ? 'filled' : 'outline'} color="teal">{pipeline.shuffle ? 'shuffled' : 'in order'}</Badge>
        <Badge variant="light">{pipeline.total_selected_images} images</Badge>
      </Group>
      {renderTrainsetPipelineSummary(pipeline, trainsets)}
      <Title order={5}>Preprocessing</Title>
      {renderPreprocessingDetails(preprocessing)}
      <Title order={5}>Method</Title>
      {renderMethodDetails(method)}
    </Stack>
  );
}

function renderTrainsetPipelineSummary(pipeline: TrainingPipeline, datasets: TrainingDataset[]) {
  const byId = new Map(datasets.map((dataset) => [dataset.id, dataset]));
  return (
    <Stack gap="sm">
      <Title order={5}>Trainsets</Title>
      {pipeline.training_datasets.map((entry) => (
        <Paper key={entry.training_dataset_id} withBorder p="sm" radius="sm">
          {renderTrainsetDetails(byId.get(entry.training_dataset_id) ?? null)}
        </Paper>
      ))}
    </Stack>
  );
}

function TimeSeriesPlot({ plot, results }: { plot: AnalysisPlot; results: CombinedResult[] }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [zoomRange, setZoomRange] = useState<[number, number] | null>(null);
  const [dragRange, setDragRange] = useState<{ startX: number; currentX: number } | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const allPoints = useMemo(() => {
    const values = movingAverage(results.map(scoreValue), plot.movingAverage);
    return results.map((result, index) => ({
      result,
      timestamp: new Date(result.timestamp).getTime(),
      value: values[index],
    }));
  }, [plot.movingAverage, results]);
  const points = useMemo(
    () =>
      zoomRange
        ? allPoints.filter((point) => point.timestamp >= zoomRange[0] && point.timestamp <= zoomRange[1])
        : allPoints,
    [allPoints, zoomRange],
  );

  if (allPoints.length === 0) {
    return <Alert color="yellow">No results match this time range.</Alert>;
  }
  if (points.length === 0) {
    return (
      <Alert color="yellow">
        <Group justify="space-between">
          <Text size="sm">No results match the current zoom range.</Text>
          <Button size="compact-xs" variant="light" onClick={() => setZoomRange(null)}>
            Reset zoom
          </Button>
        </Group>
      </Alert>
    );
  }

  const width = 960;
  const height = 320;
  const margin = { top: 24, right: 28, bottom: 46, left: 72 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const minTime = Math.min(...points.map((point) => point.timestamp));
  const maxTime = Math.max(...points.map((point) => point.timestamp));
  const minValue = Math.min(...points.map((point) => point.value));
  const maxValue = Math.max(...points.map((point) => point.value));
  const valuePadding = Math.max((maxValue - minValue) * 0.08, maxValue === minValue ? Math.max(maxValue * 0.1, 1) : 0);
  const yMin = minValue - valuePadding;
  const yMax = maxValue + valuePadding;
  const xScale = (timestamp: number) =>
    margin.left + ((timestamp - minTime) / Math.max(1, maxTime - minTime)) * innerWidth;
  const yScale = (value: number) =>
    margin.top + innerHeight - ((value - yMin) / Math.max(1e-12, yMax - yMin)) * innerHeight;
  const path = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${xScale(point.timestamp).toFixed(2)} ${yScale(point.value).toFixed(2)}`)
    .join(' ');
  const hovered = hoveredIndex !== null ? points[hoveredIndex] : null;

  function svgX(event: MouseEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return null;
    return ((event.clientX - rect.left) / rect.width) * width;
  }

  function timestampAtX(x: number) {
    const ratio = Math.max(0, Math.min(1, (x - margin.left) / innerWidth));
    return minTime + ratio * Math.max(1, maxTime - minTime);
  }

  function onMouseMove(event: MouseEvent<SVGSVGElement>) {
    const x = svgX(event);
    if (x === null) return;
    if (dragRange) {
      setDragRange((current) => (current ? { ...current, currentX: x } : current));
    }
    let bestIndex = 0;
    let bestDistance = Infinity;
    points.forEach((point, index) => {
      const distance = Math.abs(xScale(point.timestamp) - x);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });
    setHoveredIndex(bestIndex);
  }

  return (
    <Stack gap="xs">
      <ScrollArea>
        <svg
          ref={svgRef}
          className="analysis-timeseries"
          viewBox={`0 0 ${width} ${height}`}
          onMouseDown={(event) => {
            const x = svgX(event);
            if (x !== null && x >= margin.left && x <= width - margin.right) {
              setDragRange({ startX: x, currentX: x });
            }
          }}
          onMouseMove={onMouseMove}
          onMouseUp={() => {
            if (!dragRange) return;
            const x1 = Math.min(dragRange.startX, dragRange.currentX);
            const x2 = Math.max(dragRange.startX, dragRange.currentX);
            if (x2 - x1 > 12) {
              setZoomRange([timestampAtX(x1), timestampAtX(x2)]);
            }
            setDragRange(null);
          }}
          onMouseLeave={() => {
            setHoveredIndex(null);
            setDragRange(null);
          }}
        >
          <rect x={0} y={0} width={width} height={height} rx={8} className="analysis-chart-bg" />
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const y = margin.top + tick * innerHeight;
            const value = yMax - tick * (yMax - yMin);
            return (
              <g key={tick}>
                <line x1={margin.left} x2={width - margin.right} y1={y} y2={y} className="analysis-grid-line" />
                <text x={margin.left - 10} y={y + 4} textAnchor="end" className="analysis-axis-label">
                  {formatMetric(value)}
                </text>
              </g>
            );
          })}
          <line x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} className="analysis-axis" />
          <line x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} className="analysis-axis" />
          <path d={path} className="analysis-line" />
          {dragRange && (
            <rect
              x={Math.min(dragRange.startX, dragRange.currentX)}
              y={margin.top}
              width={Math.abs(dragRange.currentX - dragRange.startX)}
              height={innerHeight}
              className="analysis-zoom-selection"
            />
          )}
          {hovered && (
            <g>
              <line
                x1={xScale(hovered.timestamp)}
                x2={xScale(hovered.timestamp)}
                y1={margin.top}
                y2={height - margin.bottom}
                className="analysis-hover-line"
              />
              <circle cx={xScale(hovered.timestamp)} cy={yScale(hovered.value)} r={5} className="analysis-hover-point" />
              <rect
                x={Math.min(width - 282, xScale(hovered.timestamp) + 10)}
                y={Math.max(12, yScale(hovered.value) - 48)}
                width={270}
                height={58}
                rx={6}
                className="analysis-tooltip-box"
              />
              <text
                x={Math.min(width - 268, xScale(hovered.timestamp) + 20)}
                y={Math.max(34, yScale(hovered.value) - 26)}
                className="analysis-tooltip-text"
              >
                {new Date(hovered.result.timestamp).toLocaleString()}
              </text>
              <text
                x={Math.min(width - 268, xScale(hovered.timestamp) + 20)}
                y={Math.max(54, yScale(hovered.value) - 6)}
                className="analysis-tooltip-text"
              >
                score {formatMetric(hovered.value)}
              </text>
            </g>
          )}
          <text x={margin.left} y={height - 16} className="analysis-axis-label">
            {new Date(minTime).toLocaleString()}
          </text>
          <text x={width - margin.right} y={height - 16} textAnchor="end" className="analysis-axis-label">
            {new Date(maxTime).toLocaleString()}
          </text>
        </svg>
      </ScrollArea>
      <Group gap="xs">
        <Badge variant="light">{points.length} points</Badge>
        {zoomRange && (
          <Button size="compact-xs" variant="subtle" onClick={() => setZoomRange(null)}>
            Reset zoom
          </Button>
        )}
        {plot.movingAverage > 1 && <Badge variant="light" color="blue">moving avg {plot.movingAverage}</Badge>}
        {plot.sampling > 1 && <Badge variant="light" color="gray">sample every {plot.sampling}</Badge>}
      </Group>
    </Stack>
  );
}

function HeatmapPlot({
  plot,
  results,
  heatmapCache,
  loadingHeatmaps,
  ensureHeatmap,
}: {
  plot: AnalysisPlot;
  results: CombinedResult[];
  heatmapCache: Record<string, HeatmapRun>;
  loadingHeatmaps: Record<string, boolean>;
  ensureHeatmap: (frame: CombinedResult) => Promise<void>;
}) {
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const frames = useMemo(() => {
    if (plot.heatmapMode === 'single') return results.slice(0, Math.max(1, results.length));
    return results;
  }, [plot.heatmapMode, results]);

  useEffect(() => {
    setFrameIndex(0);
  }, [plot.id, plot.start, plot.end, plot.sampling, plot.timestamp, plot.heatmapMode]);

  useEffect(() => {
    if (frameIndex >= frames.length) setFrameIndex(Math.max(0, frames.length - 1));
  }, [frameIndex, frames.length]);

  const current = frames[frameIndex] ?? null;
  const currentKey = current ? `${current.testingRunId}:${current.heatmapTimestampOnly ? current.timestamp : current.id}` : '';
  useEffect(() => {
    if (!current) return;
    ensureHeatmap(current).catch((error) => notifyError('Could not compute heatmap', error));
  }, [current, ensureHeatmap]);

  useEffect(() => {
    if (!playing || frames.length <= 1) return undefined;
    const timer = window.setInterval(() => {
      setFrameIndex((currentFrame) => (currentFrame + 1) % frames.length);
    }, 800);
    return () => window.clearInterval(timer);
  }, [frames.length, playing]);

  if (!current) {
    return <Alert color="yellow">No result image matches this selection.</Alert>;
  }

  const heatmap = heatmapCache[currentKey];
  const loading = loadingHeatmaps[currentKey] === true;

  return (
    <Stack gap="sm">
      <Group justify="space-between" align="center">
        <Group gap="xs">
          <Badge variant="light">{resultLabel(current)}</Badge>
          <Badge variant="light" color="red">score {formatMetric(scoreValue(current))}</Badge>
          <Badge variant="light" color="gray">{current.testingRunName}</Badge>
        </Group>
        {plot.heatmapMode === 'range' && frames.length > 1 && (
          <Group gap="xs">
            <Button
              size="compact-sm"
              variant="light"
              leftSection={playing ? <Pause size={14} /> : <Play size={14} />}
              onClick={() => setPlaying((currentState) => !currentState)}
            >
              {playing ? 'Pause' : 'Play'}
            </Button>
            <Text size="xs" c="dimmed">
              Frame {frameIndex + 1}/{frames.length}
            </Text>
          </Group>
        )}
      </Group>
      {plot.heatmapMode === 'range' && frames.length > 1 && (
        <input
          type="range"
          min={0}
          max={frames.length - 1}
          value={frameIndex}
          className="analysis-frame-slider"
          onChange={(event) => setFrameIndex(Number(event.currentTarget.value))}
        />
      )}
      <div className="analysis-heatmap-wrap">
        {heatmap ? (
          <>
            <img src={heatmap.source_image_data_url} alt="Testing result source" className="analysis-heatmap-image" />
            <img src={heatmap.heatmap_image_data_url} alt="Pixel reconstruction error heatmap" className="analysis-heatmap-overlay-image" />
          </>
        ) : (
          <div className="analysis-heatmap-loading">
            {loading ? (
              <Stack gap="xs" align="center">
                <Loader size="sm" />
                <Text size="sm">Computing CPU pixel heatmap…</Text>
              </Stack>
            ) : (
              <Text size="sm">Heatmap queued for computation…</Text>
            )}
          </div>
        )}
      </div>
      <Text size="xs" c="dimmed">
        CPU heatmap · max pixel ({heatmap?.max_x ?? '—'}, {heatmap?.max_y ?? '—'}) · max {formatMetric(heatmap?.max_error)} · mean {formatMetric(heatmap?.mean_error)}
      </Text>
    </Stack>
  );
}

function AnalysisPlotCard({
  plot,
  results,
  heatmapCache,
  loadingHeatmaps,
  ensureHeatmap,
  onMove,
  onRemove,
}: {
  plot: AnalysisPlot;
  results: CombinedResult[];
  heatmapCache: Record<string, HeatmapRun>;
  loadingHeatmaps: Record<string, boolean>;
  ensureHeatmap: (frame: CombinedResult) => Promise<void>;
  onMove: (direction: -1 | 1) => void;
  onRemove: () => void;
}) {
  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Group justify="space-between" align="flex-start">
          <div>
            <Group gap="xs">
              <Title order={3}>{plot.title}</Title>
              <Badge variant="light" color={plot.plotType === 'heatmap' ? 'red' : 'blue'}>
                {plot.plotType === 'heatmap' ? 'Heatmap' : 'Time series'}
              </Badge>
            </Group>
            <Text size="sm" c="dimmed">
              {results.length} deduplicated rows · {plot.sources.length} source{plot.sources.length === 1 ? '' : 's'}
            </Text>
          </div>
          <Group gap={4}>
            <Tooltip label="Move up">
              <ActionIcon variant="subtle" onClick={() => onMove(-1)}>
                <ArrowUp size={16} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Move down">
              <ActionIcon variant="subtle" onClick={() => onMove(1)}>
                <ArrowDown size={16} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Remove plot">
              <ActionIcon color="red" variant="subtle" onClick={onRemove}>
                <Trash2 size={16} />
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
        {plot.plotType === 'timeseries' ? (
          <TimeSeriesPlot plot={plot} results={results} />
        ) : (
          <HeatmapPlot
            plot={plot}
            results={results}
            heatmapCache={heatmapCache}
            loadingHeatmaps={loadingHeatmaps}
            ensureHeatmap={ensureHeatmap}
          />
        )}
      </Stack>
    </Paper>
  );
}

function defaultDraft(): PlotDraft {
  return {
    plotType: 'timeseries',
    testingRunId: null,
    title: '',
    start: '',
    end: '',
    sampling: 1,
    movingAverage: 1,
    heatmapMode: 'single',
    timestamp: null,
  };
}

export function AnalysisPage({ active = true }: { active?: boolean }) {
  const [testingRuns, setTestingRuns] = useState<TestingRun[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [trainingPipelines, setTrainingPipelines] = useState<TrainingPipeline[]>([]);
  const [preprocessingPipelines, setPreprocessingPipelines] = useState<PreprocessingPipeline[]>([]);
  const [methodConfigurations, setMethodConfigurations] = useState<MethodConfiguration[]>([]);
  const [resultsByRunId, setResultsByRunId] = useState<Record<number, TestingRunResults>>({});
  const [loadingRunId, setLoadingRunId] = useState<number | null>(null);
  const [draft, setDraft] = useState<PlotDraft>(defaultDraft());
  const [plots, setPlots] = useState<AnalysisPlot[]>([]);
  const [heatmapCache, setHeatmapCache] = useState<Record<string, HeatmapRun>>({});
  const [loadingHeatmaps, setLoadingHeatmaps] = useState<Record<string, boolean>>({});
  const [addPlotOpen, setAddPlotOpen] = useState(true);
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);
  const [pipelineSearch, setPipelineSearch] = useState('');
  const [selectedRoiKey, setSelectedRoiKey] = useState<string | null>(null);
  const [selectedSources, setSelectedSources] = useState<PlotSourceConfig[]>([]);
  const [detailModal, setDetailModal] = useState<DetailModalState>(null);

  async function refresh() {
    const [nextTestingRuns, nextTrainingRuns, nextDatasets, nextPipelines, nextPreprocessing, nextMethods] =
      await Promise.all([
        listTestingRuns(),
        listTrainingRuns(),
        listTrainingDatasets(),
        listTrainingPipelines(),
        listPreprocessingPipelines(),
        listMethodConfigurations(),
      ]);
    setTestingRuns(nextTestingRuns);
    setTrainingRuns(nextTrainingRuns);
    setTrainingDatasets(nextDatasets);
    setTrainingPipelines(nextPipelines);
    setPreprocessingPipelines(nextPreprocessing);
    setMethodConfigurations(nextMethods);
  }

  useEffect(() => {
    if (!active) return;
    refresh().catch((error) => notifyError('Could not load testing runs', error));
  }, [active]);

  const finishedRuns = useMemo(
    () => testingRuns.filter((run) => run.status === 'finished' && (run.image_count ?? 0) > 0),
    [testingRuns],
  );

  const selectedRunId = draft.testingRunId ? Number(draft.testingRunId) : null;
  const selectedRun = selectedRunId !== null ? testingRuns.find((run) => run.id === selectedRunId) ?? null : null;
  const selectedResults = selectedRunId !== null ? resultsByRunId[selectedRunId] : undefined;
  const trainingRunById = useMemo(() => new Map(trainingRuns.map((run) => [run.id, run])), [trainingRuns]);
  const trainingDatasetById = useMemo(() => new Map(trainingDatasets.map((dataset) => [dataset.id, dataset])), [trainingDatasets]);
  const trainingPipelineById = useMemo(() => new Map(trainingPipelines.map((pipeline) => [pipeline.id, pipeline])), [trainingPipelines]);
  const preprocessingById = useMemo(() => new Map(preprocessingPipelines.map((pipeline) => [pipeline.id, pipeline])), [preprocessingPipelines]);
  const methodById = useMemo(() => new Map(methodConfigurations.map((method) => [method.id, method])), [methodConfigurations]);
  const selectedPipeline = selectedPipelineId ? trainingPipelineById.get(Number(selectedPipelineId)) ?? null : null;

  const finishedRunsForPipeline = useMemo(() => {
    if (!selectedPipelineId) return [];
    return finishedRuns.filter((run) => trainingRunById.get(run.training_run_id)?.training_pipeline_id === Number(selectedPipelineId));
  }, [finishedRuns, selectedPipelineId, trainingRunById]);

  const roiOptionsForPipeline = useMemo(() => {
    const seen = new Map<string, { value: string; label: string }>();
    for (const run of finishedRunsForPipeline) {
      const key = run.roi_id === null ? 'none' : String(run.roi_id);
      if (!seen.has(key)) {
        seen.set(key, {
          value: key,
          label: run.roi_id === null ? 'No ROI' : run.roi_name ?? `ROI #${run.roi_id}`,
        });
      }
    }
    return [...seen.values()];
  }, [finishedRunsForPipeline]);

  const candidateRuns = useMemo(() => {
    if (!selectedRoiKey) return [];
    return finishedRunsForPipeline.filter((run) => (run.roi_id === null ? 'none' : String(run.roi_id)) === selectedRoiKey);
  }, [finishedRunsForPipeline, selectedRoiKey]);

  const fetchResults = useCallback(
    async (runId: number) => {
      if (resultsByRunId[runId]) return resultsByRunId[runId];
      setLoadingRunId(runId);
      try {
        const next = await getTestingRunResults(runId);
        setResultsByRunId((current) => ({ ...current, [runId]: next }));
        return next;
      } finally {
        setLoadingRunId(null);
      }
    },
    [resultsByRunId],
  );

  const ensureHeatmap = useCallback(
    async (frame: CombinedResult) => {
      const key = `${frame.testingRunId}:${frame.heatmapTimestampOnly ? frame.timestamp : frame.id}`;
      if (heatmapCache[key] || loadingHeatmaps[key]) return;
      setLoadingHeatmaps((current) => ({ ...current, [key]: true }));
      try {
        const heatmap = await createHeatmap(
          frame.heatmapTimestampOnly
            ? { testing_run_id: frame.testingRunId, timestamp: frame.timestamp }
            : { testing_run_id: frame.testingRunId, testing_result_id: frame.id },
        );
        setHeatmapCache((current) => ({ ...current, [key]: heatmap }));
      } finally {
        setLoadingHeatmaps((current) => ({ ...current, [key]: false }));
      }
    },
    [heatmapCache, loadingHeatmaps],
  );

  useEffect(() => {
    if (selectedRunId === null) return;
    if (draft.plotType === 'heatmap' && draft.heatmapMode === 'single') {
      const run = testingRuns.find((item) => item.id === selectedRunId);
      const bounds = sourceBounds(run ? trainingDatasetById.get(run.training_dataset_id) : null);
      setDraft((current) => {
        if (current.testingRunId !== String(selectedRunId)) return current;
        return {
          ...current,
          title: current.title || `Heatmap · ${run?.name ?? `Testing run #${selectedRunId}`}`,
          start: current.start || bounds.start,
          end: current.end || bounds.end,
          timestamp: current.timestamp ?? bounds.start,
        };
      });
      return;
    }
    fetchResults(selectedRunId)
      .then((data) => {
        if (data.results.length === 0) return;
        const run = testingRuns.find((item) => item.id === selectedRunId);
        const bounds = sourceBounds(run ? trainingDatasetById.get(run.training_dataset_id) : null);
        setDraft((current) => {
          if (current.testingRunId !== String(selectedRunId)) return current;
          const first = data.results[0];
          const last = data.results[data.results.length - 1];
          return {
            ...current,
            title: current.title || `${current.plotType === 'heatmap' ? 'Heatmap' : 'Time series'} · ${data.testing_run.name}`,
            start: current.start || bounds.start || toDateTimeLocal(first.timestamp),
            end: current.end || bounds.end || toDateTimeLocal(last.timestamp),
            timestamp: current.timestamp ?? first.timestamp,
          };
        });
      })
      .catch((error) => notifyError('Could not load testing results', error));
  }, [draft.heatmapMode, draft.plotType, fetchResults, selectedRunId, testingRuns, trainingDatasetById]);

  useEffect(() => {
    if (draft.plotType === 'heatmap' && draft.heatmapMode === 'single') return;
    selectedSources.forEach((source) => {
      fetchResults(Number(source.testingRunId)).catch((error) => notifyError('Could not load testing results', error));
    });
  }, [draft.heatmapMode, draft.plotType, fetchResults, selectedSources]);

  const filteredDraftResults = useMemo(() => {
    if (!selectedResults) return [];
    return filterAndSampleResults(selectedResults.results, draft.start, draft.end, draft.sampling);
  }, [draft.end, draft.sampling, draft.start, selectedResults]);

  function combinedResultsForSources(sources: PlotSourceConfig[], plotType: PlotType, heatmapMode: HeatmapMode): CombinedResult[] {
    const dedup = new Map<string, CombinedResult>();
    for (const source of sources) {
      if (plotType === 'heatmap' && heatmapMode === 'single') {
        if (!source.timestamp) continue;
        const run = testingRuns.find((item) => item.id === Number(source.testingRunId));
        const key = `${source.testingRunId}|${source.timestamp}`;
        if (!dedup.has(key)) {
          dedup.set(key, {
            id: -Number(source.testingRunId),
            position: 0,
            image_path: '',
            timestamp: source.timestamp,
            score: Number.NaN,
            full_mse: Number.NaN,
            roi_mse: null,
            tile_scores: null,
            width: 0,
            height: 0,
            testingRunId: Number(source.testingRunId),
            testingRunName: run?.name ?? `Testing run #${source.testingRunId}`,
            heatmapTimestampOnly: true,
          });
        }
        continue;
      }
      const data = resultsByRunId[Number(source.testingRunId)];
      if (!data) continue;
      const sourceResults =
        plotType === 'heatmap' && heatmapMode === 'single'
          ? data.results.filter((result) => result.timestamp === source.timestamp).slice(0, 1)
          : filterAndSampleResults(data.results, source.start, source.end, source.sampling);
      for (const result of sourceResults) {
        const key = `${result.image_path}|${result.timestamp}`;
        if (!dedup.has(key)) {
          dedup.set(key, {
            ...result,
            testingRunId: Number(source.testingRunId),
            testingRunName: data.testing_run.name,
          });
        }
      }
    }
    return [...dedup.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }

  const combinedDraftResults = useMemo(
    () => combinedResultsForSources(selectedSources, draft.plotType, draft.heatmapMode),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [draft.heatmapMode, draft.plotType, resultsByRunId, selectedSources, testingRuns],
  );

  function addPlot() {
    if (selectedSources.length === 0) {
      notifications.show({ color: 'yellow', title: 'Select inference datasets', message: 'Add at least one inference dataset source before adding a plot.' });
      return;
    }
    const availableResults = combinedDraftResults;
    if (availableResults.length === 0) {
      notifications.show({ color: 'yellow', title: 'No matching results', message: 'Adjust time range or sampling.' });
      return;
    }
    const title =
      draft.title.trim() ||
      `${draft.plotType === 'heatmap' ? 'Heatmap' : 'Time series'} · ${selectedPipeline?.name ?? 'selected sources'}`;
    const nextPlot: AnalysisPlot = {
      ...draft,
      id: crypto.randomUUID(),
      title,
      sources: selectedSources,
      timestamp: draft.timestamp ?? availableResults[0]?.timestamp ?? null,
    };
    setPlots((current) => [...current, nextPlot]);
  }

  function movePlot(plotId: string, direction: -1 | 1) {
    setPlots((current) => {
      const index = current.findIndex((plot) => plot.id === plotId);
      const nextIndex = index + direction;
      if (index < 0 || nextIndex < 0 || nextIndex >= current.length) return current;
      const next = [...current];
      [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
      return next;
    });
  }

  function detailObjectsForRun(run: TestingRun) {
    const trainingRun = trainingRunById.get(run.training_run_id) ?? null;
    const trainingPipeline = trainingRun ? trainingPipelineById.get(trainingRun.training_pipeline_id) ?? null : null;
    const trainset = trainingDatasetById.get(run.training_dataset_id) ?? null;
    const preprocessing = trainingPipeline ? preprocessingById.get(trainingPipeline.preprocessing_pipeline_id) ?? null : null;
    const method = trainingPipeline ? methodById.get(trainingPipeline.method_configuration_id) ?? null : null;
    return { trainingRun, trainingPipeline, trainset, preprocessing, method };
  }

  function addSource(run: TestingRun) {
    if (selectedSources.some((source) => source.testingRunId === String(run.id))) return;
    const bounds = sourceBounds(trainingDatasetById.get(run.training_dataset_id));
    if (draft.plotType === 'heatmap' && draft.heatmapMode === 'single') {
      setSelectedSources((current) => [
        ...current,
        {
          testingRunId: String(run.id),
          start: bounds.start,
          end: bounds.end,
          sampling: 1,
          timestamp: bounds.start,
        },
      ]);
      return;
    }
    fetchResults(run.id)
      .then((data) => {
        const first = data.results[0];
        setSelectedSources((current) => [
          ...current,
          {
            testingRunId: String(run.id),
            start: bounds.start || toDateTimeLocal(first?.timestamp),
            end: bounds.end || toDateTimeLocal(data.results[data.results.length - 1]?.timestamp),
            sampling: 1,
            timestamp: first?.timestamp ?? null,
          },
        ]);
      })
      .catch((error) => notifyError('Could not load testing results', error));
  }

  function updateSource(runId: string, patch: Partial<PlotSourceConfig>) {
    setSelectedSources((current) =>
      current.map((source) => (source.testingRunId === runId ? { ...source, ...patch } : source)),
    );
  }

  // Pipelines that have at least one finished testing run can be analysed.
  const analysablePipelineIds = useMemo(() => {
    const ids = new Set<number>();
    for (const run of testingRuns) {
      if (run.status !== 'finished') continue;
      const trainingRun = trainingRunById.get(run.training_run_id);
      if (trainingRun) ids.add(trainingRun.training_pipeline_id);
    }
    return ids;
  }, [testingRuns, trainingRunById]);
  const analysablePipelines = useMemo(() => {
    const query = pipelineSearch.trim().toLowerCase();
    return trainingPipelines.filter((pipeline) => {
      if (!analysablePipelineIds.has(pipeline.id)) return false;
      if (!query) return true;
      return (
        pipeline.name.toLowerCase().includes(query) ||
        pipeline.method_configuration_name.toLowerCase().includes(query) ||
        pipeline.training_datasets.some((entry) => entry.name.toLowerCase().includes(query))
      );
    });
  }, [trainingPipelines, analysablePipelineIds, pipelineSearch]);

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={2}>Analysis</Title>
          <Text c="dimmed" size="sm">
            Compose interactive plots from saved testing runs and compare them on one board.
          </Text>
        </div>
        <Button
          variant="default"
          leftSection={<RotateCcw size={18} />}
          disabled={plots.length === 0}
          onClick={() => setPlots([])}
        >
          Reset board
        </Button>
      </Group>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Group justify="space-between" align="center">
            <Button
              variant="subtle"
              leftSection={addPlotOpen ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
              onClick={() => setAddPlotOpen((open) => !open)}
            >
              Add plot
            </Button>
            {selectedRunId !== null && loadingRunId === selectedRunId && <Loader size="sm" />}
          </Group>
          <Collapse in={addPlotOpen}>
            <Stack gap="md">
              <StepCard index={1} title="Training pipeline" color="blue">
                {selectedPipeline ? (
                  <Group justify="space-between" align="center" wrap="wrap">
                    <Group gap="xs">
                      <Text fw={600}>{selectedPipeline.name}</Text>
                      <Text size="sm" c="dimmed">
                        {selectedPipeline.method_configuration_name} · input{' '}
                        {formatResolution(selectedPipeline.preprocessing_input_width, selectedPipeline.preprocessing_input_height) ?? 'n/a'}
                      </Text>
                      <Badge variant="light">{finishedRunsForPipeline.length} finished inference runs</Badge>
                      <DetailButton
                        title="Training pipeline details"
                        body={renderPipelineDetails(
                          selectedPipeline,
                          trainingDatasets,
                          preprocessingById.get(selectedPipeline.preprocessing_pipeline_id) ?? null,
                          methodById.get(selectedPipeline.method_configuration_id) ?? null,
                        )}
                        onOpen={setDetailModal}
                      />
                    </Group>
                    <Button
                      variant="subtle"
                      leftSection={<Pencil size={16} />}
                      onClick={() => {
                        setSelectedPipelineId(null);
                        setSelectedRoiKey(null);
                        setSelectedSources([]);
                      }}
                    >
                      Change pipeline
                    </Button>
                  </Group>
                ) : (
                  <>
                    <TextInput
                      placeholder="Search by pipeline, method or trainset"
                      leftSection={<Search size={16} />}
                      value={pipelineSearch}
                      onChange={(event) => setPipelineSearch(event.currentTarget.value)}
                    />
                    <ScrollArea h={240}>
                      <Table striped highlightOnHover verticalSpacing="sm" miw={900}>
                        <Table.Thead>
                          <Table.Tr>
                            <Table.Th>Pipeline</Table.Th>
                            <Table.Th>Trainsets</Table.Th>
                            <Table.Th>Preprocessing</Table.Th>
                            <Table.Th>Method</Table.Th>
                            <Table.Th>Input size</Table.Th>
                            <Table.Th />
                          </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                          {analysablePipelines.map((pipeline) => {
                            const preprocessing = preprocessingById.get(pipeline.preprocessing_pipeline_id) ?? null;
                            const method = methodById.get(pipeline.method_configuration_id) ?? null;
                            return (
                              <Table.Tr key={pipeline.id}>
                                <Table.Td>{pipeline.name}</Table.Td>
                                <Table.Td>
                                  <Group gap={4} wrap="nowrap">
                                    <Text size="sm">{pipeline.training_datasets.map((entry) => entry.name).join(', ')}</Text>
                                    <DetailButton
                                      title="Trainset details"
                                      body={
                                        <Stack gap="md">
                                          {pipeline.training_datasets.map((entry) => (
                                            <div key={entry.training_dataset_id}>
                                              {renderTrainsetDetails(trainingDatasetById.get(entry.training_dataset_id) ?? null)}
                                            </div>
                                          ))}
                                        </Stack>
                                      }
                                      onOpen={setDetailModal}
                                    />
                                  </Group>
                                </Table.Td>
                                <Table.Td>
                                  <Group gap={4} wrap="nowrap">
                                    <Text size="sm">{pipeline.preprocessing_pipeline_name}</Text>
                                    <DetailButton title="Preprocessing details" body={renderPreprocessingDetails(preprocessing)} onOpen={setDetailModal} />
                                  </Group>
                                </Table.Td>
                                <Table.Td>
                                  <Group gap={4} wrap="nowrap">
                                    <Text size="sm">{pipeline.method_configuration_name}</Text>
                                    <DetailButton title="Method details" body={renderMethodDetails(method)} onOpen={setDetailModal} />
                                  </Group>
                                </Table.Td>
                                <Table.Td>
                                  <Badge variant="light" color="blue">
                                    {formatResolution(pipeline.preprocessing_input_width, pipeline.preprocessing_input_height) ?? 'n/a'}
                                  </Badge>
                                </Table.Td>
                                <Table.Td>
                                  <Group justify="flex-end">
                                    <Button
                                      size="compact-sm"
                                      variant="light"
                                      leftSection={<Check size={14} />}
                                      onClick={() => {
                                        setSelectedPipelineId(String(pipeline.id));
                                        setSelectedRoiKey(null);
                                        setSelectedSources([]);
                                      }}
                                    >
                                      Use
                                    </Button>
                                  </Group>
                                </Table.Td>
                              </Table.Tr>
                            );
                          })}
                          {analysablePipelines.length === 0 && (
                            <Table.Tr>
                              <Table.Td colSpan={6}>
                                <Text c="dimmed" ta="center" py="md">
                                  No training pipelines with finished inference runs yet.
                                </Text>
                              </Table.Td>
                            </Table.Tr>
                          )}
                        </Table.Tbody>
                      </Table>
                    </ScrollArea>
                  </>
                )}
              </StepCard>

              {selectedPipeline && (
                <StepCard index={2} title="ROI & plot type" color="violet">
                  <SimpleGrid cols={{ base: 1, md: 2 }}>
                    <Select
                      label="ROI"
                      placeholder="Select ROI variant"
                      data={roiOptionsForPipeline}
                      value={selectedRoiKey}
                      searchable
                      onChange={(value) => {
                        setSelectedRoiKey(value);
                        setSelectedSources([]);
                      }}
                    />
                    <Select
                      label="Plot type"
                      data={[
                        { value: 'timeseries', label: 'Time series' },
                        { value: 'heatmap', label: 'Heatmap' },
                      ]}
                      value={draft.plotType}
                      onChange={(value) =>
                        setDraft((current) => ({
                          ...current,
                          plotType: (value ?? 'timeseries') as PlotType,
                        }))
                      }
                    />
                  </SimpleGrid>
                </StepCard>
              )}

              {selectedPipeline && selectedRoiKey && (
                <StepCard index={3} title="Inference datasets" color="teal">
              <Paper withBorder radius="sm" className="analysis-run-picker">
                <ScrollArea>
                  <Table striped highlightOnHover verticalSpacing="sm" miw={900}>
                    <Table.Thead>
                      <Table.Tr>
                        <Table.Th>Name</Table.Th>
                        <Table.Th>Label</Table.Th>
                        <Table.Th>Datasets</Table.Th>
                        <Table.Th>Image size</Table.Th>
                        <Table.Th>Stride</Table.Th>
                        <Table.Th>Images</Table.Th>
                        <Table.Th />
                        <Table.Th />
                      </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                      {candidateRuns.map((run) => {
                        const selected = draft.testingRunId === String(run.id);
                        const dataset = detailObjectsForRun(run).trainset;
                        const added = selectedSources.some((source) => source.testingRunId === String(run.id));
                        return (
                          <Table.Tr key={run.id} className={selected ? 'analysis-selected-row' : undefined}>
                            <Table.Td>{run.training_dataset_name}</Table.Td>
                            <Table.Td>
                              <Badge size="xs" variant="light" color={usageColor(dataset?.usage_label)}>
                                {usageLabel(dataset?.usage_label)}
                              </Badge>
                            </Table.Td>
                            <Table.Td>
                              <Group gap={4}>
                                {(dataset?.dataset_names ?? []).map((name) => (
                                  <Badge key={name} size="xs" variant="light" color="gray">
                                    {name}
                                  </Badge>
                                ))}
                              </Group>
                            </Table.Td>
                            <Table.Td>
                              <Group gap={4}>
                                {(dataset ? datasetResolutions(dataset) : []).map((res) => (
                                  <Badge key={res} size="xs" variant="light" color="teal">
                                    {res}
                                  </Badge>
                                ))}
                              </Group>
                            </Table.Td>
                            <Table.Td>{datasetStrides(dataset)}</Table.Td>
                            <Table.Td>{run.image_count ?? dataset?.total_selected_images ?? '—'}</Table.Td>
                            <Table.Td>
                              <DetailButton title="Inference dataset details" body={renderTrainsetDetails(dataset)} onOpen={setDetailModal} />
                            </Table.Td>
                            <Table.Td>
                              <Button
                                size="compact-sm"
                                variant={added ? 'filled' : 'light'}
                                color={added ? 'green' : 'blue'}
                                onClick={() => addSource(run)}
                                disabled={added}
                              >
                                {added ? 'Added' : 'Add'}
                              </Button>
                            </Table.Td>
                          </Table.Tr>
                        );
                      })}
                      {candidateRuns.length === 0 && (
                        <Table.Tr>
                          <Table.Td colSpan={8}>
                            <Text c="dimmed" ta="center" py="md">
                              {selectedPipelineId && selectedRoiKey ? 'No inference datasets match this model/ROI.' : 'Select a training pipeline and ROI first.'}
                            </Text>
                          </Table.Td>
                        </Table.Tr>
                      )}
                    </Table.Tbody>
                  </Table>
                </ScrollArea>
              </Paper>
              {selectedSources.length > 0 && (
                <Paper withBorder p="sm" radius="sm">
                  <Stack gap="sm">
                    <Title order={4}>Selected inference datasets</Title>
                    {selectedSources.map((source) => {
                      const run = testingRuns.find((item) => item.id === Number(source.testingRunId));
                      const bounds = sourceBounds(run ? trainingDatasetById.get(run.training_dataset_id) : null);
                      return (
                        <Paper key={source.testingRunId} withBorder p="sm" radius="sm">
                          <Stack gap="xs">
                            <Group justify="space-between">
                              <Text fw={700} size="sm">{run?.training_dataset_name ?? `Testing run #${source.testingRunId}`}</Text>
                              <ActionIcon color="red" variant="subtle" onClick={() => setSelectedSources((current) => current.filter((item) => item.testingRunId !== source.testingRunId))}>
                                <Trash2 size={16} />
                              </ActionIcon>
                            </Group>
                            {draft.plotType === 'heatmap' && draft.heatmapMode === 'single' ? (
                              <TextInput
                                label="Timestamp"
                                type="datetime-local"
                                step={1}
                                min={bounds.start}
                                max={bounds.end}
                                value={toDateTimeLocal(source.timestamp)}
                                description={bounds.start && bounds.end ? `${bounds.start.replace('T', ' ')} to ${bounds.end.replace('T', ' ')}` : undefined}
                                onChange={(event) => {
                                  const value = event.currentTarget.value;
                                  updateSource(source.testingRunId, { timestamp: value });
                                }}
                              />
                            ) : (
                              <SimpleGrid cols={{ base: 1, md: 3 }}>
                                <TextInput label="Start" type="datetime-local" min={bounds.start} max={bounds.end} value={source.start} onChange={(event) => updateSource(source.testingRunId, { start: event.currentTarget.value })} />
                                <TextInput label="End" type="datetime-local" min={bounds.start} max={bounds.end} value={source.end} onChange={(event) => updateSource(source.testingRunId, { end: event.currentTarget.value })} />
                                <NumberInput label="Sampling rate" min={1} value={source.sampling} onChange={(value) => updateSource(source.testingRunId, { sampling: valueAsNumber(value, 1) })} />
                              </SimpleGrid>
                            )}
                          </Stack>
                        </Paper>
                      );
                    })}
                  </Stack>
                </Paper>
              )}
                </StepCard>
              )}

              {selectedPipeline && selectedRoiKey && (
                <StepCard index={4} title="Plot configuration" color="gray">
              <TextInput
                label="Plot title"
                value={draft.title}
                onChange={(event) => {
                  const value = event.currentTarget.value;
                  setDraft((current) => ({ ...current, title: value }));
                }}
              />
              {draft.plotType === 'timeseries' ? (
                <SimpleGrid cols={{ base: 1, md: 3 }}>
                  <NumberInput label="Moving average window" min={1} value={draft.movingAverage} onChange={(value) => setDraft((current) => ({ ...current, movingAverage: valueAsNumber(value, 1) }))} />
                  <TextInput label="Y-axis" value="Reconstruction error" disabled />
                  <TextInput label="X-axis" value="Time" disabled />
                </SimpleGrid>
              ) : (
                <SimpleGrid cols={{ base: 1, md: 3 }}>
                  <Select
                    label="Heatmap mode"
                    data={[
                      { value: 'single', label: 'Single timestamp' },
                      { value: 'range', label: 'Date range video' },
                    ]}
                    value={draft.heatmapMode}
                    onChange={(value) =>
                      setDraft((current) => ({
                        ...current,
                        heatmapMode: (value ?? 'single') as HeatmapMode,
                        timestamp: value === 'range' ? null : current.timestamp,
                      }))
                    }
                  />
                  <TextInput label="Frames" value={`${combinedDraftResults.length} deduplicated frames`} disabled />
                  <TextInput label="Overlay" value="CPU pixel reconstruction error" disabled />
                </SimpleGrid>
              )}
              {finishedRuns.length === 0 && <Alert color="blue">No finished testing runs with images are available yet.</Alert>}
              <Group justify="space-between" align="center">
                <Text size="sm" c="dimmed">
                  {selectedSources.length > 0 ? `${combinedDraftResults.length} deduplicated result rows` : 'Add one or more inference dataset sources.'}
                </Text>
                <Button leftSection={<Plus size={18} />} onClick={addPlot} disabled={selectedSources.length === 0 || combinedDraftResults.length === 0}>
                  Add plot
                </Button>
              </Group>
                </StepCard>
              )}
            </Stack>
          </Collapse>
        </Stack>
      </Paper>

      {plots.length === 0 ? (
        <Alert color="blue">Add time series or heatmap plots to start comparing testing runs.</Alert>
      ) : (
        <Stack gap="md">
          {plots.map((plot) => {
            const hasAllData =
              plot.plotType === 'heatmap' && plot.heatmapMode === 'single'
                ? true
                : plot.sources.every((source) => resultsByRunId[Number(source.testingRunId)]);
            const results = combinedResultsForSources(plot.sources, plot.plotType, plot.heatmapMode);
            return hasAllData ? (
              <AnalysisPlotCard
                key={plot.id}
                plot={plot}
                results={results}
                heatmapCache={heatmapCache}
                loadingHeatmaps={loadingHeatmaps}
                ensureHeatmap={ensureHeatmap}
                onMove={(direction) => movePlot(plot.id, direction)}
                onRemove={() => setPlots((current) => current.filter((item) => item.id !== plot.id))}
              />
            ) : (
              <Paper key={plot.id} withBorder p="md" radius="sm">
                <Group gap="sm">
                  <Loader size="sm" />
                  <Text>Loading plot data…</Text>
                </Group>
              </Paper>
            );
          })}
        </Stack>
      )}
      <DetailModal detail={detailModal} onClose={() => setDetailModal(null)} />
    </Stack>
  );
}
