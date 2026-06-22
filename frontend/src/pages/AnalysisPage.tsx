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
import { ArrowDown, ArrowUp, Check, ChevronDown, ChevronRight, Info, Pause, Pencil, Play, Plus, RotateCcw, Search, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import type React from 'react';

import {
  abortHeatmapRange,
  createHeatmap,
  createHeatmapRange,
  getHeatmapRange,
  getTestingRunResults,
  heatmapRangeFrameUrl,
  listMethodConfigurations,
  listPreprocessingPipelines,
  listTestingRuns,
  listTrainingDatasets,
  listTrainingPipelines,
  listTrainingRuns,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart } from '../components/PlotlyChart';
import { StepCard } from '../components/StepCard';
import type { Data, Layout } from '../lib/plotly';
import { usePendingIds } from '../hooks/usePendingIds';
import { formatValue } from '../methods/utils';
import { datasetResolutions, formatResolution, orderedGraphNodes, stepDetail } from '../training/graph';
import type {
  HeatmapRangeRun,
  HeatmapRun,
  HeatmapVisualizationConfig,
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
  includeReference: boolean;
  heatmapConfig: HeatmapVisualizationConfig;
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

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown error';
}

function valueAsNumber(value: string | number, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function defaultHeatmapConfig(): HeatmapVisualizationConfig {
  return {
    error_mode: 'squared',
    threshold_enabled: false,
    threshold: 0,
    max_clip_enabled: false,
    max_clip: 0.33,
    max_opacity: 0.55,
    fixed_ceiling_enabled: false,
    fixed_ceiling: 1,
    signed_deviations: false,
    positive_weight: 1,
    negative_weight: 1,
  };
}

function heatmapConfigKey(config: HeatmapVisualizationConfig): string {
  return [
    config.error_mode,
    Number(config.threshold_enabled),
    config.threshold,
    Number(config.max_clip_enabled),
    config.max_clip,
    config.max_opacity,
    Number(config.fixed_ceiling_enabled),
    config.fixed_ceiling,
    Number(config.signed_deviations),
    config.positive_weight,
    config.negative_weight,
  ].join(':');
}

function heatmapCacheKey(frame: CombinedResult, config: HeatmapVisualizationConfig): string {
  const source = frame.heatmapTimestampOnly ? frame.timestamp : frame.id;
  return `${frame.testingRunId}:${source}:${heatmapConfigKey(config)}`;
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

const TRACE_COLORS = ['#1c7ed6', '#e8590c', '#2f9e44', '#9c36b5', '#0c8599', '#e03131', '#5f3dc4', '#66a80f'];

function TimeSeriesPlot({ plot, results }: { plot: AnalysisPlot; results: CombinedResult[] }) {
  const traces = useMemo<Data[]>(() => {
    const groups = new Map<number, { name: string; results: CombinedResult[] }>();
    for (const result of results) {
      const group = groups.get(result.testingRunId);
      if (group) group.results.push(result);
      else groups.set(result.testingRunId, { name: result.testingRunName, results: [result] });
    }
    return [...groups.values()].map((group, index) => {
      const smoothed = movingAverage(group.results.map(scoreValue), plot.movingAverage);
      return {
        type: 'scatter',
        mode: 'lines',
        name: group.name,
        x: group.results.map((result) => result.timestamp),
        y: smoothed,
        line: { color: TRACE_COLORS[index % TRACE_COLORS.length], width: 1.5 },
        hovertemplate: '%{x|%Y-%m-%d %H:%M:%S}<br>error %{y:.5g}<extra>%{fullData.name}</extra>',
      } as Data;
    });
  }, [plot.movingAverage, results]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      showlegend: traces.length > 1,
      legend: { orientation: 'h', y: -0.28, x: 0 },
      hovermode: 'x unified',
      xaxis: {
        title: { text: 'Time', font: { size: 12 } },
        type: 'date',
        rangeslider: { thickness: 0.08 },
        showgrid: true,
        gridcolor: 'rgba(128,128,128,0.15)',
      },
      yaxis: {
        title: { text: 'Reconstruction error', font: { size: 12 } },
        showgrid: true,
        gridcolor: 'rgba(128,128,128,0.15)',
        zeroline: false,
      },
      margin: { l: 64, r: 24, t: 12, b: 56 },
    }),
    [traces.length],
  );

  if (results.length === 0) {
    return <Alert color="yellow">No results match this time range.</Alert>;
  }

  return (
    <Stack gap="xs">
      <PlotlyChart data={traces} layout={layout} height={420} />
      <Group gap="xs">
        <Badge variant="light">{results.length} points</Badge>
        {plot.movingAverage > 1 && <Badge variant="light" color="blue">moving avg {plot.movingAverage}</Badge>}
        {plot.sampling > 1 && <Badge variant="light" color="gray">sample every {plot.sampling}</Badge>}
      </Group>
    </Stack>
  );
}

function transparentHeatmapScale(config: HeatmapVisualizationConfig) {
  const maxAlpha = config.max_clip_enabled ? 1 : config.max_opacity;
  if (config.signed_deviations) {
    return [
      [0, `rgba(0,0,255,${maxAlpha})`],
      [0.25, `rgba(0,255,255,${maxAlpha / 2})`],
      [0.5, 'rgba(255,255,255,0)'],
      [0.75, `rgba(255,255,0,${maxAlpha / 2})`],
      [1, `rgba(255,0,0,${maxAlpha})`],
    ];
  }
  return [
    [0, 'rgba(0,0,143,0)'],
    [0.25, `rgba(0,0,255,${maxAlpha * 0.25})`],
    [0.5, `rgba(0,255,255,${maxAlpha * 0.5})`],
    [0.75, `rgba(255,255,0,${maxAlpha * 0.75})`],
    [1, `rgba(255,0,0,${maxAlpha})`],
  ];
}

function HeatmapPlot({
  plot,
  results,
  heatmapCache,
  loadingHeatmaps,
  heatmapErrors,
  ensureHeatmap,
}: {
  plot: AnalysisPlot;
  results: CombinedResult[];
  heatmapCache: Record<string, HeatmapRun>;
  loadingHeatmaps: Record<string, boolean>;
  heatmapErrors: Record<string, string>;
  ensureHeatmap: (
    frame: CombinedResult,
    config: HeatmapVisualizationConfig,
    options?: { force?: boolean },
  ) => Promise<void>;
}) {
  const current = results[0] ?? null;
  const currentKey = current ? heatmapCacheKey(current, plot.heatmapConfig) : '';
  useEffect(() => {
    if (!current) return;
    ensureHeatmap(current, plot.heatmapConfig);
  }, [current, ensureHeatmap, plot.heatmapConfig]);

  const heatmap = currentKey ? heatmapCache[currentKey] : undefined;
  const loading = loadingHeatmaps[currentKey] === true;
  const error = heatmapErrors[currentKey];

  const relativeErrorMatrix = useMemo(() => {
    if (!heatmap?.error_matrix) return null;
    const config = heatmap.visualization_config;
    const clipFactor = config.max_clip_enabled ? config.max_clip : 1;
    const ceiling = config.fixed_ceiling_enabled
      ? config.fixed_ceiling
      : heatmap.max_error > 0
        ? heatmap.max_error * clipFactor
        : 1;
    return heatmap.error_matrix.map((row) =>
      row.map((value) =>
        config.signed_deviations
          ? Math.max(-1, Math.min(1, value / ceiling))
          : Math.max(0, Math.min(1, value / ceiling)),
      ),
    );
  }, [heatmap]);

  const errorTrace = useMemo<Data[]>(() => {
    if (!heatmap?.error_matrix || !relativeErrorMatrix) return [];
    return [
      {
        type: 'heatmap',
        z: relativeErrorMatrix,
        customdata: heatmap.error_matrix,
        colorscale: transparentHeatmapScale(heatmap.visualization_config),
        zmin: heatmap.visualization_config.signed_deviations ? -1 : 0,
        zmax: 1,
        zsmooth: false,
        showscale: false,
        hovertemplate: `x %{x}<br>y %{y}<br>${heatmap.visualization_config.signed_deviations ? 'Signed relative error' : 'Relative error'} %{z:.4f}<br>Configured pixel error %{customdata:.6g}<extra></extra>`,
      } as Data,
    ];
  }, [heatmap, relativeErrorMatrix]);

  const errorLayout = useMemo<Partial<Layout>>(() => {
    if (!heatmap) return {};
    return {
      // Keep the drawable image area identical to the adjacent image frames.
      // Pixel coordinates remain available in hover while zoom/pan stay active.
      margin: { l: 0, r: 0, t: 0, b: 0 },
      xaxis: {
        range: [-0.5, heatmap.width - 0.5],
        scaleanchor: 'y',
        constrain: 'domain',
        showgrid: false,
        zeroline: false,
        showticklabels: false,
      },
      yaxis: {
        range: [heatmap.height - 0.5, -0.5],
        showgrid: false,
        zeroline: false,
        showticklabels: false,
      },
      images: [
        {
          source: heatmap.source_image_data_url,
          xref: 'x',
          yref: 'y',
          x: -0.5,
          y: -0.5,
          sizex: heatmap.width,
          sizey: heatmap.height,
          xanchor: 'left',
          yanchor: 'top',
          sizing: 'stretch',
          layer: 'below',
          opacity: 1,
        },
      ],
    };
  }, [heatmap]);

  const frameStyle = heatmap ? { aspectRatio: `${heatmap.width} / ${heatmap.height}` } : undefined;

  const errorPanel = heatmap ? (
    <div className="analysis-heatmap-panel">
      <Text size="sm" fw={500} c="dimmed" ta="center">
        Reconstruction error
      </Text>
      {heatmap.error_matrix ? (
        <>
          <div className="analysis-heatmap-image-frame analysis-heatmap-plot-frame" style={frameStyle}>
            <PlotlyChart data={errorTrace} layout={errorLayout} height="100%" />
          </div>
          <div className="analysis-relative-colorbar" aria-label="Relative reconstruction error scale from zero to one">
            <Group justify="space-between" gap="xs">
              <Text size="xs" c="dimmed">{heatmap.visualization_config.signed_deviations ? '-1' : '0'}</Text>
              <Text size="xs" c="dimmed">{heatmap.visualization_config.signed_deviations ? 'Signed relative error' : 'Relative reconstruction error'}</Text>
              <Text size="xs" c="dimmed">1</Text>
            </Group>
            <div
              className="analysis-relative-colorbar-gradient"
              style={{
                background: heatmap.visualization_config.signed_deviations ? SIGNED_GRADIENT : JET_GRADIENT,
              }}
            />
          </div>
        </>
      ) : (
        <div className="analysis-heatmap-image-frame" style={frameStyle}>
          <img src={heatmap.source_image_data_url} alt="Original with heatmap overlay" className="analysis-heatmap-image" />
          <img src={heatmap.heatmap_image_data_url} alt="Reconstruction error heatmap overlay" className="analysis-heatmap-overlay-image" />
        </div>
      )}
    </div>
  ) : null;

  if (!current) {
    return <Alert color="yellow">No result image matches this selection.</Alert>;
  }

  return (
    <Stack gap="sm">
      <Group gap="xs">
        <Badge variant="light">{resultLabel(current)}</Badge>
        <Badge variant="light" color="red">score {formatMetric(scoreValue(current))}</Badge>
        <Badge variant="light" color="gray">{current.testingRunName}</Badge>
        <Badge variant="light" color="blue">{plot.heatmapConfig.error_mode} error</Badge>
        {plot.heatmapConfig.threshold_enabled && (
          <Badge variant="light" color="yellow">threshold {formatMetric(plot.heatmapConfig.threshold)}</Badge>
        )}
        {plot.heatmapConfig.signed_deviations && (
          <Badge variant="light" color="grape">
            signed +{formatMetric(plot.heatmapConfig.positive_weight)} / -{formatMetric(plot.heatmapConfig.negative_weight)}
          </Badge>
        )}
        <Badge variant="light" color="gray">
          {plot.heatmapConfig.fixed_ceiling_enabled
            ? `ceiling ${formatMetric(plot.heatmapConfig.fixed_ceiling)}`
            : plot.heatmapConfig.max_clip_enabled
              ? `max clip ${Math.round(plot.heatmapConfig.max_clip * 100)}%`
              : `opacity ${Math.round(plot.heatmapConfig.max_opacity * 100)}%`}
        </Badge>
      </Group>
      <div className="analysis-heatmap-wrap">
        {heatmap ? (
          (plot.includeReference ?? true) ? (
            <SimpleGrid cols={{ base: 1, md: 3 }} spacing="md">
              <div className="analysis-heatmap-panel">
                <Text size="sm" fw={500} c="dimmed" ta="center">
                  Original
                </Text>
                <div className="analysis-heatmap-image-frame" style={frameStyle}>
                  <img src={heatmap.source_image_data_url} alt="Original source" className="analysis-heatmap-image" />
                </div>
              </div>
              <div className="analysis-heatmap-panel">
                <Text size="sm" fw={500} c="dimmed" ta="center">
                  Reconstructed
                </Text>
                <div className="analysis-heatmap-image-frame" style={frameStyle}>
                  <img
                    src={heatmap.reconstruction_image_data_url || heatmap.source_image_data_url}
                    alt="Model reconstruction"
                    className="analysis-heatmap-image"
                  />
                </div>
              </div>
              {errorPanel}
            </SimpleGrid>
          ) : (
            errorPanel
          )
        ) : (
          <div className="analysis-heatmap-loading">
            {loading ? (
              <Stack gap="xs" align="center">
                <Loader size="sm" />
                <Text size="sm">Computing pixel heatmap…</Text>
              </Stack>
            ) : error ? (
              <Stack gap="xs" align="center">
                <Badge color="red" variant="light">
                  Failed
                </Badge>
                <Text size="sm" ta="center">
                  {error}
                </Text>
                <Button size="compact-sm" variant="light" onClick={() => ensureHeatmap(current, plot.heatmapConfig, { force: true })}>
                  Retry heatmap
                </Button>
              </Stack>
            ) : (
              <Text size="sm">Heatmap queued for computation…</Text>
            )}
          </div>
        )}
      </div>
      <Text size="xs" c="dimmed">
        Pixel heatmap · {heatmap?.visualization_config.error_mode ?? plot.heatmapConfig.error_mode} error · max pixel ({heatmap?.max_x ?? '—'}, {heatmap?.max_y ?? '—'}) · max magnitude {formatMetric(heatmap?.max_error)} · mean magnitude {formatMetric(heatmap?.mean_error)}
      </Text>
    </Stack>
  );
}

// Jet-style gradient matching the backend overlay, for the static video legend.
const JET_GRADIENT = 'linear-gradient(to right, #00008f, #0000ff, #00ffff, #ffff00, #ff0000, #800000)';
const SIGNED_GRADIENT = 'linear-gradient(to right, #0000ff, #00ffff, transparent, #ffff00, #ff0000)';

function HeatmapVideo({ plot, results }: { plot: AnalysisPlot; results: CombinedResult[] }) {
  const params = useMemo(() => {
    const source = plot.sources[0];
    if (!source || results.length === 0) return null;
    const startIso = source.start || results[0].timestamp;
    const endIso = source.end || results[results.length - 1].timestamp;
    return {
      testing_run_id: Number(source.testingRunId),
      start_timestamp: startIso,
      end_timestamp: endIso,
      stride: Math.max(1, Math.floor(source.sampling || 1)),
      testingRunName: results[0].testingRunName,
      visualizationConfig: plot.heatmapConfig,
      visualizationConfigKey: heatmapConfigKey(plot.heatmapConfig),
    };
  }, [plot.heatmapConfig, plot.sources, results]);

  const [scaleMode, setScaleMode] = useState<'per_frame' | 'shared'>('per_frame');
  const [job, setJob] = useState<HeatmapRangeRun | null>(null);
  const [starting, setStarting] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(8);
  const fixedCeiling = plot.heatmapConfig.fixed_ceiling_enabled;
  const effectiveScaleMode: 'per_frame' | 'shared' = fixedCeiling ? 'per_frame' : scaleMode;

  // Reset whenever the plot's range/source changes.
  useEffect(() => {
    setJob(null);
    setVideoError(null);
    setFrameIndex(0);
    setPlaying(false);
  }, [plot.id, params?.testing_run_id, params?.start_timestamp, params?.end_timestamp, params?.stride, params?.visualizationConfigKey, effectiveScaleMode]);

  const polling = job != null && (job.status === 'queued' || job.status === 'running');
  useEffect(() => {
    if (!polling || !job) return undefined;
    const timer = window.setInterval(async () => {
      try {
        setJob(await getHeatmapRange(job.id));
      } catch {
        /* transient; keep last state */
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [polling, job?.id]);

  const frameCount = job?.frame_count ?? 0;
  const ready = job?.status === 'finished' && frameCount > 0;

  useEffect(() => {
    if (!ready || !playing || frameCount <= 1) return undefined;
    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % frameCount);
    }, Math.max(33, 1000 / Math.max(1, fps)));
    return () => window.clearInterval(timer);
  }, [ready, playing, frameCount, fps]);

  async function startJob() {
    if (!params) return;
    setStarting(true);
    setVideoError(null);
    try {
      const created = await createHeatmapRange({
        testing_run_id: params.testing_run_id,
        start_timestamp: params.start_timestamp,
        end_timestamp: params.end_timestamp,
        stride: params.stride,
        scale_mode: effectiveScaleMode,
        visualization_config: params.visualizationConfig,
      });
      setJob(created);
      setFrameIndex(0);
    } catch (error) {
      setVideoError(errorMessage(error));
    } finally {
      setStarting(false);
    }
  }

  async function abortJob() {
    if (!job) return;
    try {
      setJob(await abortHeatmapRange(job.id));
    } catch (error) {
      setVideoError(errorMessage(error));
    }
  }

  if (!params) {
    return <Alert color="yellow">Select one inference source with a time range to render a heatmap video.</Alert>;
  }

  return (
    <Stack gap="sm">
      <Group gap="xs" align="flex-end">
        <Select
          label="Color scale"
          size="xs"
          w={170}
          data={fixedCeiling
            ? [{ value: 'fixed', label: 'Fixed ceiling' }]
            : [
                { value: 'per_frame', label: 'Per-frame (auto)' },
                { value: 'shared', label: 'Shared (comparable)' },
              ]}
          value={fixedCeiling ? 'fixed' : scaleMode}
          disabled={fixedCeiling || polling}
          onChange={(value) => value && setScaleMode(value as 'per_frame' | 'shared')}
        />
        <Badge variant="light" color="gray">{params.testingRunName}</Badge>
        <Badge variant="light">stride {params.stride}</Badge>
        <Badge variant="light" color="blue">{plot.heatmapConfig.error_mode} error</Badge>
        {plot.heatmapConfig.signed_deviations && <Badge variant="light" color="grape">signed</Badge>}
        {fixedCeiling && <Badge variant="light" color="gray">ceiling {formatMetric(plot.heatmapConfig.fixed_ceiling)}</Badge>}
        {!ready && (
          <Button size="compact-sm" onClick={startJob} loading={starting || polling} disabled={polling}>
            {polling ? 'Rendering…' : 'Render heatmap video'}
          </Button>
        )}
        {ready && (
          <Button size="compact-sm" variant="light" onClick={startJob} loading={starting}>
            Re-render
          </Button>
        )}
      </Group>

      {videoError && <Alert color="red">{videoError}</Alert>}

      {polling && job && (
        <Stack gap={6}>
          <Group gap="xs">
            <Loader size="sm" />
            <Text size="sm">
              Rendering frame {job.done_count}
              {job.frame_count ? ` / ${job.frame_count}` : ''}…
            </Text>
            <Button size="compact-xs" color="red" variant="light" onClick={abortJob}>
              Abort
            </Button>
          </Group>
          {job.frame_count ? (
            <Progress value={(job.done_count / job.frame_count) * 100} size="sm" />
          ) : null}
        </Stack>
      )}

      {job && (job.status === 'failed' || job.status === 'aborted') && (
        <Alert color={job.status === 'failed' ? 'red' : 'yellow'}>
          Heatmap video {job.status}{job.error_message ? `: ${job.error_message}` : '.'}
        </Alert>
      )}

      {ready && job && (
        <Stack gap="xs">
          <Group justify="space-between" align="center">
            <Group gap="xs">
              <Button
                size="compact-sm"
                variant="light"
                leftSection={playing ? <Pause size={14} /> : <Play size={14} />}
                onClick={() => setPlaying((current) => !current)}
              >
                {playing ? 'Pause' : 'Play'}
              </Button>
              <Text size="xs" c="dimmed">Frame {frameIndex + 1}/{frameCount}</Text>
            </Group>
            <NumberInput
              label="fps"
              size="xs"
              w={90}
              min={1}
              max={60}
              value={fps}
              onChange={(value) => setFps(Math.max(1, Math.min(60, Number(value) || 1)))}
            />
          </Group>
          <div className="analysis-heatmap-image-frame analysis-heatmap-video-frame">
            <img
              src={heatmapRangeFrameUrl(job.id, frameIndex)}
              alt={`Heatmap frame ${frameIndex + 1}`}
              className="analysis-heatmap-image"
            />
          </div>
          <input
            type="range"
            min={0}
            max={frameCount - 1}
            value={frameIndex}
            className="analysis-frame-slider"
            onChange={(event) => setFrameIndex(Number(event.currentTarget.value))}
          />
          <Stack gap={3}>
            <Group gap="xs" align="center">
              <Text size="xs" c="dimmed">{job.visualization_config.signed_deviations ? '-1' : '0'}</Text>
              <div
                style={{
                  flex: 1,
                  height: 10,
                  borderRadius: 3,
                  background: job.visualization_config.signed_deviations ? SIGNED_GRADIENT : JET_GRADIENT,
                }}
              />
              <Text size="xs" c="dimmed">1</Text>
            </Group>
            <Text size="xs" c="dimmed" ta="center">
              Relative reconstruction error · {job.visualization_config.fixed_ceiling_enabled
                ? `fixed ceiling ${formatMetric(job.visualization_config.fixed_ceiling)}`
                : job.scale_mode === 'shared'
                  ? 'shared scale'
                  : 'per-frame scale'} · absolute max{' '}
              {job.scale_mode === 'shared' && !job.visualization_config.fixed_ceiling_enabled
                ? formatMetric(job.global_vmax)
                : formatMetric(job.frame_max_errors?.[frameIndex])}
            </Text>
          </Stack>
        </Stack>
      )}
    </Stack>
  );
}

function AnalysisPlotCard({
  plot,
  results,
  heatmapCache,
  loadingHeatmaps,
  heatmapErrors,
  ensureHeatmap,
  onMove,
  onRemove,
}: {
  plot: AnalysisPlot;
  results: CombinedResult[];
  heatmapCache: Record<string, HeatmapRun>;
  loadingHeatmaps: Record<string, boolean>;
  heatmapErrors: Record<string, string>;
  ensureHeatmap: (
    frame: CombinedResult,
    config: HeatmapVisualizationConfig,
    options?: { force?: boolean },
  ) => Promise<void>;
  onMove: (direction: -1 | 1) => void;
  onRemove: () => void;
}) {
  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Group justify="space-between" align="flex-start">
          <div>
            <Group gap="xs">
              <Title order={4} fw={500}>{plot.title}</Title>
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
        ) : plot.heatmapMode === 'range' ? (
          <HeatmapVideo plot={plot} results={results} />
        ) : (
          <HeatmapPlot
            plot={plot}
            results={results}
            heatmapCache={heatmapCache}
            loadingHeatmaps={loadingHeatmaps}
            heatmapErrors={heatmapErrors}
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
    includeReference: true,
    heatmapConfig: defaultHeatmapConfig(),
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
  const [heatmapErrors, setHeatmapErrors] = useState<Record<string, string>>({});
  const sourceActions = usePendingIds();
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
    async (
      frame: CombinedResult,
      config: HeatmapVisualizationConfig,
      options?: { force?: boolean },
    ) => {
      const key = heatmapCacheKey(frame, config);
      if (!options?.force && (heatmapCache[key] || loadingHeatmaps[key] || heatmapErrors[key])) return;
      if (options?.force) {
        setHeatmapErrors((current) => {
          const next = { ...current };
          delete next[key];
          return next;
        });
      }
      setLoadingHeatmaps((current) => ({ ...current, [key]: true }));
      try {
        const heatmap = await createHeatmap(
          frame.heatmapTimestampOnly
            ? {
                testing_run_id: frame.testingRunId,
                timestamp: frame.timestamp,
                force_recompute: options?.force ?? false,
                visualization_config: config,
              }
            : {
                testing_run_id: frame.testingRunId,
                testing_result_id: frame.id,
                force_recompute: options?.force ?? false,
                visualization_config: config,
              },
        );
        setHeatmapCache((current) => ({ ...current, [key]: heatmap }));
        setHeatmapErrors((current) => {
          if (!current[key]) return current;
          const next = { ...current };
          delete next[key];
          return next;
        });
      } catch (error) {
        const message = errorMessage(error);
        setHeatmapErrors((current) => ({ ...current, [key]: message }));
        notifyError('Could not compute heatmap', error);
      } finally {
        setLoadingHeatmaps((current) => ({ ...current, [key]: false }));
      }
    },
    [heatmapCache, heatmapErrors, loadingHeatmaps],
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

  async function addSource(run: TestingRun) {
    if (selectedSources.some((source) => source.testingRunId === String(run.id))) return;
    await sourceActions.runPending(`add-source:${run.id}`, async () => {
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
      const data = await fetchResults(run.id);
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
    }).catch((error) => notifyError('Could not load testing results', error));
  }

  function updateSource(runId: string, patch: Partial<PlotSourceConfig>) {
    setSelectedSources((current) =>
      current.map((source) => (source.testingRunId === runId ? { ...source, ...patch } : source)),
    );
  }

  function updateHeatmapConfig(patch: Partial<HeatmapVisualizationConfig>) {
    setDraft((current) => ({
      ...current,
      heatmapConfig: { ...current.heatmapConfig, ...patch },
    }));
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
                                loading={sourceActions.isPending(`add-source:${run.id}`)}
                                onClick={() => addSource(run)}
                                disabled={added || sourceActions.isPending(`add-source:${run.id}`)}
                              >
                                {sourceActions.isPending(`add-source:${run.id}`) ? 'Adding…' : added ? 'Added' : 'Add'}
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
                              <DateTime24Input
                                label="Timestamp"
                                min={bounds.start}
                                max={bounds.end}
                                value={toDateTimeLocal(source.timestamp)}
                                description={bounds.start && bounds.end ? `${bounds.start.replace('T', ' ')} to ${bounds.end.replace('T', ' ')}` : undefined}
                                onChange={(value) => updateSource(source.testingRunId, { timestamp: value })}
                              />
                            ) : (
                              <SimpleGrid cols={{ base: 1, md: 3 }}>
                                <DateTime24Input label="Start" min={bounds.start} max={bounds.end} value={source.start} onChange={(value) => updateSource(source.testingRunId, { start: value })} />
                                <DateTime24Input label="End" min={bounds.start} max={bounds.end} value={source.end} onChange={(value) => updateSource(source.testingRunId, { end: value })} />
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
                <Stack gap="md">
                  <SimpleGrid cols={{ base: 1, md: 3 }}>
                    <Select
                      label={<InfoLabel label="Heatmap mode" info="Single timestamp computes one interactive heatmap. Date range renders a sampled heatmap video." />}
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
                    <TextInput
                      label={<InfoLabel label="Frames" info="Number of deduplicated source timestamps before the selected sampling rate is applied." />}
                      value={`${combinedDraftResults.length} deduplicated frames`}
                      disabled
                    />
                    <Switch
                      label={<InfoLabel label="Include reference" info="Shows original and reconstructed images next to the transparent error overlay." />}
                      checked={draft.includeReference}
                      onChange={(event) => {
                        const checked = event.currentTarget.checked;
                        setDraft((current) => ({ ...current, includeReference: checked }));
                      }}
                    />
                  </SimpleGrid>

                  <Text fw={600} size="sm">Error calculation</Text>
                  <SimpleGrid cols={{ base: 1, md: 3 }}>
                    <Select
                      label={<InfoLabel label="Pixel error" info="Absolute uses |input - reconstruction|. Squared uses (input - reconstruction)² and emphasizes large deviations more strongly." />}
                      data={[
                        { value: 'squared', label: 'Squared error' },
                        { value: 'absolute', label: 'Absolute error' },
                      ]}
                      value={draft.heatmapConfig.error_mode}
                      onChange={(value) => updateHeatmapConfig({ error_mode: (value ?? 'squared') as 'squared' | 'absolute' })}
                    />
                    <Switch
                      label={<InfoLabel label="Signed deviations" info="Off treats only deviation magnitude. On preserves direction: input brighter than reconstruction is positive; input darker is negative." />}
                      checked={draft.heatmapConfig.signed_deviations}
                      onChange={(event) => updateHeatmapConfig({ signed_deviations: event.currentTarget.checked })}
                    />
                    <Switch
                      label={<InfoLabel label="Threshold" info="Suppresses pixels whose absolute input/reconstruction difference is below the specified value. The value uses preprocessed pixel units." />}
                      checked={draft.heatmapConfig.threshold_enabled}
                      onChange={(event) => updateHeatmapConfig({ threshold_enabled: event.currentTarget.checked })}
                    />
                  </SimpleGrid>

                  {(draft.heatmapConfig.signed_deviations || draft.heatmapConfig.threshold_enabled) && (
                    <SimpleGrid cols={{ base: 1, md: 3 }}>
                      {draft.heatmapConfig.signed_deviations && (
                        <>
                          <NumberInput
                            label={<InfoLabel label="Positive weight" info="Multiplier for pixels where input is brighter than reconstruction. Zero suppresses positive deviations." />}
                            min={0}
                            step={0.1}
                            decimalScale={3}
                            value={draft.heatmapConfig.positive_weight}
                            onChange={(value) => updateHeatmapConfig({ positive_weight: valueAsNumber(value, 1) })}
                          />
                          <NumberInput
                            label={<InfoLabel label="Negative weight" info="Multiplier for pixels where input is darker than reconstruction. Zero suppresses negative deviations." />}
                            min={0}
                            step={0.1}
                            decimalScale={3}
                            value={draft.heatmapConfig.negative_weight}
                            onChange={(value) => updateHeatmapConfig({ negative_weight: valueAsNumber(value, 1) })}
                          />
                        </>
                      )}
                      {draft.heatmapConfig.threshold_enabled && (
                        <NumberInput
                          label={<InfoLabel label="Threshold value" info="Absolute difference below this value becomes zero before absolute/squared error and sign weighting are applied." />}
                          min={0}
                          step={0.01}
                          decimalScale={6}
                          value={draft.heatmapConfig.threshold}
                          onChange={(value) => updateHeatmapConfig({ threshold: valueAsNumber(value, 0) })}
                        />
                      )}
                    </SimpleGrid>
                  )}

                  <Text fw={600} size="sm">Visibility</Text>
                  <SimpleGrid cols={{ base: 1, md: 3 }}>
                    <Switch
                      label={<InfoLabel label="Fixed ceiling" info="Uses this absolute error value as relative 1.0. Tiny residual errors therefore remain faint instead of being stretched to full visibility. The value uses the selected error metric's units." />}
                      checked={draft.heatmapConfig.fixed_ceiling_enabled}
                      onChange={(event) => {
                        const checked = event.currentTarget.checked;
                        updateHeatmapConfig({
                          fixed_ceiling_enabled: checked,
                          max_clip_enabled: checked ? false : draft.heatmapConfig.max_clip_enabled,
                        });
                      }}
                    />
                    <Switch
                      label={<InfoLabel label="Max clip" info="When enabled, errors at max clip × the strongest current error already reach full opacity. At 0.33, the strongest third saturates like the legacy visualization." />}
                      checked={draft.heatmapConfig.max_clip_enabled}
                      onChange={(event) => {
                        const checked = event.currentTarget.checked;
                        updateHeatmapConfig({
                          max_clip_enabled: checked,
                          fixed_ceiling_enabled: checked ? false : draft.heatmapConfig.fixed_ceiling_enabled,
                        });
                      }}
                    />
                    {!draft.heatmapConfig.max_clip_enabled && (
                      <NumberInput
                        label={<InfoLabel label="Maximum opacity (%)" info="Maximum overlay coverage when max clip is disabled. This remains configurable with fixed ceiling; the previous MLTrace behavior used 55%." />}
                        min={0}
                        max={100}
                        step={5}
                        value={Math.round(draft.heatmapConfig.max_opacity * 100)}
                        onChange={(value) => updateHeatmapConfig({ max_opacity: valueAsNumber(value, 55) / 100 })}
                      />
                    )}
                  </SimpleGrid>
                  {(draft.heatmapConfig.fixed_ceiling_enabled || draft.heatmapConfig.max_clip_enabled) && (
                    <SimpleGrid cols={{ base: 1, md: 3 }}>
                      {draft.heatmapConfig.fixed_ceiling_enabled && (
                        <NumberInput
                          label={<InfoLabel label="Ceiling value" info="Error magnitude mapped to relative 1.0. For absolute error this is a direct pixel difference; for squared error it is a squared pixel difference." />}
                          min={Number.EPSILON}
                          step={0.01}
                          decimalScale={8}
                          value={draft.heatmapConfig.fixed_ceiling}
                          onChange={(value) => updateHeatmapConfig({ fixed_ceiling: valueAsNumber(value, 1) })}
                        />
                      )}
                      {draft.heatmapConfig.max_clip_enabled && (
                        <NumberInput
                          label={<InfoLabel label="Max clip (%)" info="Relative fraction of the strongest remaining error at which the overlay becomes fully opaque." />}
                          min={1}
                          max={100}
                          step={1}
                          value={Math.round(draft.heatmapConfig.max_clip * 100)}
                          onChange={(value) => updateHeatmapConfig({ max_clip: valueAsNumber(value, 33) / 100 })}
                        />
                      )}
                    </SimpleGrid>
                  )}
                </Stack>
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
                heatmapErrors={heatmapErrors}
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
