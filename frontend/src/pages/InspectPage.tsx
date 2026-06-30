import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Group,
  NumberInput,
  Paper,
  Progress,
  ScrollArea,
  Select,
  Stack,
  Switch,
  Table,
  Text,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Download, Eye, FileVideo, ImageDown, Pause, Play, RefreshCw, Square, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

import {
  abortInspectRun,
  createInspectRun,
  deleteInspectRun,
  inspectRunFrameUrl,
  inspectRunVideoUrl,
  listInspectRuns,
  listPreprocessingPipelines,
  listTrainingDatasets,
  previewInspect,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { StepCard } from '../components/StepCard';
import { usePendingIds } from '../hooks/usePendingIds';
import type { InspectPreview, InspectRun, PreprocessingPipeline, TrainingDataset } from '../types';

function toInputDateTime(value: string | null): string {
  return value ? value.slice(0, 19) : '';
}

function formatTimestamp(value: string | null): string {
  return value ? value.replace('T', ' ').slice(0, 19) : 'n/a';
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

function selectionSignature(values: {
  trainingDatasetId: number | null;
  preprocessingPipelineId: number | null;
  start: string;
  end: string;
  stride: number;
  contrastEnabled: boolean;
  contrastReferenceFrames: number;
  contrastShift: number;
  contrastVmax: number;
  contrastMaRadius: number;
}): string {
  return JSON.stringify(values);
}

type InspectFrameSource = {
  frameCount: number;
  fps: number;
  imageUrl: (index: number) => string;
  timestamp?: (index: number) => string | null;
};

function InspectFramePlayer({
  title,
  source,
  meta,
  actions,
}: {
  title: string;
  source: InspectFrameSource;
  meta?: ReactNode;
  actions?: ReactNode;
}) {
  const frameCount = source.frameCount;
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setFrameIndex(0);
    setPlaying(false);
  }, [source]);

  useEffect(() => {
    if (!playing || frameCount <= 1) return;
    const delay = Math.max(40, 1000 / Math.max(1, source.fps));
    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % frameCount);
    }, delay);
    return () => window.clearInterval(timer);
  }, [playing, frameCount, source.fps]);

  if (frameCount <= 0) return null;

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="sm">
        <Group justify="space-between" align="center" wrap="wrap">
          <Group gap="xs">
            <Text fw={700}>{title}</Text>
            {meta}
          </Group>
          <Group gap="xs">{actions}</Group>
        </Group>

        <Group gap="xs" align="center">
          <Button
            size="compact-sm"
            variant="light"
            leftSection={playing ? <Pause size={14} /> : <Play size={14} />}
            onClick={() => setPlaying((current) => !current)}
          >
            {playing ? 'Pause frames' : 'Play frames'}
          </Button>
          <Text size="xs" c="dimmed">
            Frame {frameIndex + 1}/{frameCount}
            {source.timestamp?.(frameIndex) ? ` · ${formatTimestamp(source.timestamp(frameIndex) ?? null)}` : ''}
          </Text>
        </Group>
        <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
          <img
            src={source.imageUrl(frameIndex)}
            alt={`Inspect frame ${frameIndex + 1}`}
            style={{
              display: 'block',
              maxWidth: '100%',
              maxHeight: 'min(70vh, 680px)',
              width: 'auto',
              height: 'auto',
              borderRadius: 6,
            }}
          />
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(0, frameCount - 1)}
          value={frameIndex}
          onChange={(event) => setFrameIndex(Number(event.currentTarget.value))}
        />
      </Stack>
    </Paper>
  );
}

function InspectRunPlayer({ run }: { run: InspectRun }) {
  const frameCount = run.frame_count ?? 0;

  if (run.status !== 'finished' || frameCount <= 0) return null;

  return (
    <InspectFramePlayer
      title={run.training_dataset_name}
      source={{
        frameCount,
        fps: run.fps,
        imageUrl: (index) => inspectRunFrameUrl(run.id, index),
      }}
      meta={
        <>
          <Badge variant="light">{run.preprocessing_pipeline_name}</Badge>
          <Badge variant="light" color="gray">
            {frameCount} frames · {run.fps} fps
          </Badge>
        </>
      }
      actions={
        <>
          <Button
            size="compact-sm"
            component="a"
            href={inspectRunVideoUrl(run.id)}
            download={`inspect-run-${run.id}.mp4`}
            leftSection={<Download size={14} />}
          >
            Download MP4
          </Button>
          <Button
            size="compact-sm"
            variant="light"
            component="a"
            href={inspectRunFrameUrl(run.id, 0)}
            download={`inspect-run-${run.id}-frame.png`}
            leftSection={<ImageDown size={14} />}
          >
            Download PNG
          </Button>
        </>
      }
    />
  );
}

export function InspectPage({ active = true }: { active?: boolean }) {
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [preprocessingPipelines, setPreprocessingPipelines] = useState<PreprocessingPipeline[]>([]);
  const [runs, setRuns] = useState<InspectRun[]>([]);
  const [trainingDatasetId, setTrainingDatasetId] = useState<number | null>(null);
  const [preprocessingPipelineId, setPreprocessingPipelineId] = useState<number | null>(null);
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [stride, setStride] = useState(1);
  const [fps, setFps] = useState(12);
  const [contrastEnabled, setContrastEnabled] = useState(false);
  const [contrastReferenceFrames, setContrastReferenceFrames] = useState(100);
  const [contrastShift, setContrastShift] = useState(10000);
  const [contrastVmax, setContrastVmax] = useState(12000);
  const [contrastMaRadius, setContrastMaRadius] = useState(3);
  const [preview, setPreview] = useState<InspectPreview | null>(null);
  const [previewSignature, setPreviewSignature] = useState('');
  const [previewLoading, setPreviewLoading] = useState(false);
  const [runLoading, setRunLoading] = useState(false);
  const rowActions = usePendingIds();

  async function refresh() {
    const [nextDatasets, nextPipelines, nextRuns] = await Promise.all([
      listTrainingDatasets(),
      listPreprocessingPipelines(),
      listInspectRuns(),
    ]);
    setTrainingDatasets(nextDatasets);
    setPreprocessingPipelines(nextPipelines);
    setRuns(nextRuns);
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
    const hasActive = runs.some((run) => run.status === 'queued' || run.status === 'running');
    if (!hasActive) return;
    const timer = window.setInterval(() => {
      listInspectRuns().then(setRuns).catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [active, runs]);

  const selectedDataset = trainingDatasets.find((dataset) => dataset.id === trainingDatasetId) ?? null;
  const selectedPipeline = preprocessingPipelines.find((pipeline) => pipeline.id === preprocessingPipelineId) ?? null;
  const minDate = toInputDateTime(selectedDataset?.start_timestamp ?? null);
  const maxDate = toInputDateTime(selectedDataset?.end_timestamp ?? null);
  const currentSignature = selectionSignature({
    trainingDatasetId,
    preprocessingPipelineId,
    start,
    end,
    stride,
    contrastEnabled,
    contrastReferenceFrames,
    contrastShift,
    contrastVmax,
    contrastMaRadius,
  });
  const previewFresh = Boolean(preview && previewSignature === currentSignature);
  const invalidRange = Boolean(start && end && end < start);
  const canPreview = Boolean(trainingDatasetId && preprocessingPipelineId && start && end && !invalidRange);
  const canRun = canPreview && !runLoading && (!contrastEnabled || contrastVmax > 0);

  function handleDatasetChange(value: string | null) {
    const id = value ? Number(value) : null;
    const dataset = trainingDatasets.find((item) => item.id === id) ?? null;
    setTrainingDatasetId(id);
    setStart(toInputDateTime(dataset?.start_timestamp ?? null));
    setEnd(toInputDateTime(dataset?.end_timestamp ?? null));
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
        contrast_enabled: contrastEnabled,
        contrast_reference_frames: contrastReferenceFrames,
        contrast_shift: contrastShift,
        contrast_vmax: contrastVmax,
        contrast_ma_radius: contrastMaRadius,
      });
      setPreview(result);
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
      const created = await createInspectRun({
        training_dataset_id: trainingDatasetId,
        preprocessing_pipeline_id: preprocessingPipelineId,
        start_timestamp: start,
        end_timestamp: end,
        stride,
        fps,
        contrast_enabled: contrastEnabled,
        contrast_reference_frames: contrastReferenceFrames,
        contrast_shift: contrastShift,
        contrast_vmax: contrastVmax,
        contrast_ma_radius: contrastMaRadius,
      });
      setRuns((current) => [created, ...current.filter((run) => run.id !== created.id)]);
      notifications.show({ color: 'green', title: 'Inspect run queued', message: created.training_dataset_name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Inspect run failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setRunLoading(false);
    }
  }

  async function handleAbort(run: InspectRun) {
    await rowActions.runPending(`abort:${run.id}`, async () => {
      const updated = await abortInspectRun(run.id);
      setRuns((current) => current.map((item) => (item.id === updated.id ? updated : item)));
    }).catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Abort failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    });
  }

  async function handleDelete(run: InspectRun) {
    if (!window.confirm(`Delete inspect run ${run.id}?`)) return;
    await rowActions.runPending(`delete:${run.id}`, async () => {
      await deleteInspectRun(run.id);
      setRuns((current) => current.filter((item) => item.id !== run.id));
    }).catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Delete failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    });
  }

  const latestFinished = runs.find((run) => run.status === 'finished') ?? null;

  return (
    <Stack gap="lg">
      <div>
        <Title order={2}>Inspect</Title>
        <Text c="dimmed" size="sm">
          Render selected Train/Test dataset images through a preprocessing pipeline into a playable inspection video.
        </Text>
      </div>

      <StepCard title="Inspect source" color="blue">
        <Stack gap="md">
          <Group grow align="flex-end">
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
                markPreviewStale();
              }}
            />
          </Group>

          {selectedDataset && (
            <Alert color="blue" title="Dataset bounds">
              {selectedDataset.name}: {formatTimestamp(selectedDataset.start_timestamp)} to{' '}
              {formatTimestamp(selectedDataset.end_timestamp)}
            </Alert>
          )}

          <Group grow align="flex-start">
            <DateTime24Input
              label="Start"
              value={start}
              min={minDate}
              max={maxDate}
              error={invalidRange ? 'Invalid range' : undefined}
              onChange={(value) => {
                setStart(value);
                markPreviewStale();
              }}
            />
            <DateTime24Input
              label="End"
              value={end}
              min={minDate}
              max={maxDate}
              error={invalidRange ? 'Invalid range' : undefined}
              onChange={(value) => {
                setEnd(value);
                markPreviewStale();
              }}
            />
            <NumberInput
              label="Inspect stride"
              min={1}
              value={stride}
              onChange={(value) => {
                setStride(Math.max(1, Number(value) || 1));
                markPreviewStale();
              }}
            />
            <NumberInput
              label="Video fps"
              min={1}
              max={60}
              value={fps}
              onChange={(value) => setFps(Math.max(1, Math.min(60, Number(value) || 1)))}
            />
          </Group>

          <Paper withBorder p="md" radius="sm">
            <Stack gap="sm">
              <Group justify="space-between" align="center">
                <div>
                  <Text fw={600}>Contrast-enhanced video</Text>
                  <Text size="xs" c="dimmed">
                    Subtract a mean reference (first N frames) from every frame, then shift, clip and rescale.
                  </Text>
                </div>
                <Switch
                  checked={contrastEnabled}
                  onChange={(event) => {
                    setContrastEnabled(event.currentTarget.checked);
                    markPreviewStale();
                  }}
                  label={contrastEnabled ? 'Enabled' : 'Disabled'}
                />
              </Group>

              {contrastEnabled && (
                <>
                  <Group grow align="flex-start">
                    <NumberInput
                      label="Reference frames (mean)"
                      description="First N frames averaged into the reference"
                      min={1}
                      value={contrastReferenceFrames}
                      onChange={(value) => {
                        setContrastReferenceFrames(Math.max(1, Number(value) || 1));
                        markPreviewStale();
                      }}
                    />
                    <NumberInput
                      label="Shift"
                      description="Added to (frame − reference)"
                      value={contrastShift}
                      onChange={(value) => {
                        setContrastShift(Number(value) || 0);
                        markPreviewStale();
                      }}
                    />
                    <NumberInput
                      label="vmax (clip)"
                      description="Upper clip value mapped to white"
                      min={1}
                      value={contrastVmax}
                      error={contrastVmax > 0 ? undefined : 'Must be > 0'}
                      onChange={(value) => {
                        setContrastVmax(Math.max(1, Number(value) || 1));
                        markPreviewStale();
                      }}
                    />
                    <NumberInput
                      label="Moving average ±frames"
                      description="0 disables temporal smoothing"
                      min={0}
                      value={contrastMaRadius}
                      onChange={(value) => {
                        setContrastMaRadius(Math.max(0, Number(value) || 0));
                        markPreviewStale();
                      }}
                    />
                  </Group>
                  <Text size="xs" c="dimmed">
                    Intensities are normalised to a 0–65535 scale, so shift/vmax match the legacy 16-bit workflow
                    (e.g. shift 10000, vmax 12000).
                  </Text>
                  {preview?.contrast_enabled && preview.contrast_diff_min != null && (
                    <Alert color="grape" title="Preview diff range">
                      <Stack gap="xs">
                        <Text size="sm">
                          Frame − reference spans {Math.round(preview.contrast_diff_min)} to{' '}
                          {Math.round(preview.contrast_diff_max ?? 0)} (reference built from{' '}
                          {preview.contrast_reference_frames_used} frames). For full contrast set shift ≈{' '}
                          {Math.round(-preview.contrast_diff_min)} and vmax ≈{' '}
                          {Math.max(1, Math.round((preview.contrast_diff_max ?? 0) - preview.contrast_diff_min))}.
                        </Text>
                        <Group>
                          <Button size="compact-sm" variant="light" color="grape" onClick={handleAutoFitContrast}>
                            Auto-fit shift &amp; vmax
                          </Button>
                        </Group>
                      </Stack>
                    </Alert>
                  )}
                </>
              )}
            </Stack>
          </Paper>

          <Group justify="flex-end">
            <Button
              leftSection={<Eye size={18} />}
              variant="light"
              loading={previewLoading}
              disabled={!canPreview || previewLoading}
              onClick={handlePreview}
            >
              Load preview
            </Button>
            <Button
              leftSection={<FileVideo size={18} />}
              loading={runLoading}
              disabled={!canRun}
              onClick={handleRun}
            >
              Run
            </Button>
          </Group>

          {preview && (
            <InspectFramePlayer
              title="Preview"
              source={{
                frameCount: preview.preview_frames?.length || 1,
                fps,
                imageUrl: (index) => preview.preview_frames?.[index]?.image_data_url ?? preview.image_data_url,
                timestamp: (index) => preview.preview_frames?.[index]?.timestamp ?? preview.first_timestamp,
              }}
              meta={
                <>
                  <Badge variant="light" color={previewFresh ? 'green' : 'yellow'}>
                    {previewFresh ? 'Preview current' : 'Preview stale'}
                  </Badge>
                  <Badge variant="light">{preview.selected_images} selected images</Badge>
                  <Badge variant="light" color="gray">
                    {preview.preview_frame_count ?? 1} preview frames
                  </Badge>
                  <Badge variant="light" color="gray">
                    {preview.width}x{preview.height}, {preview.channels} ch, {preview.dtype}
                  </Badge>
                </>
              }
              actions={
                <Button
                  size="compact-sm"
                  variant="light"
                  component="a"
                  href={preview.image_data_url}
                  download="inspect-preview-frame.png"
                  leftSection={<ImageDown size={14} />}
                >
                  Download PNG
                </Button>
              }
            />
          )}
        </Stack>
      </StepCard>

      {latestFinished && <InspectRunPlayer run={latestFinished} />}

      <StepCard title="Inspect runs" color="cyan">
        <Stack gap="md">
          <Group justify="flex-end">
            <Button variant="light" leftSection={<RefreshCw size={16} />} onClick={refresh}>
              Refresh
            </Button>
          </Group>
          <ScrollArea>
            <Table striped verticalSpacing="sm" miw={1080}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Status</Table.Th>
                  <Table.Th>Train/Test Dataset</Table.Th>
                  <Table.Th>Preprocessing</Table.Th>
                  <Table.Th>Range</Table.Th>
                  <Table.Th>Stride</Table.Th>
                  <Table.Th>FPS</Table.Th>
                  <Table.Th>Progress</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {runs.map((run) => {
                  const busy = run.status === 'queued' || run.status === 'running';
                  return (
                    <Table.Tr key={run.id}>
                      <Table.Td>
                        <Stack gap={4}>
                          <Badge color={statusColor(run.status)} variant="light">
                            {run.status}
                          </Badge>
                          {run.error_message && (
                            <Text size="xs" c="red">
                              {run.error_message}
                            </Text>
                          )}
                        </Stack>
                      </Table.Td>
                      <Table.Td>{run.training_dataset_name}</Table.Td>
                      <Table.Td>
                        <Group gap={6} wrap="nowrap">
                          <Text size="sm">{run.preprocessing_pipeline_name}</Text>
                          {run.contrast_enabled && (
                            <Badge size="xs" color="grape" variant="light">
                              contrast
                            </Badge>
                          )}
                        </Group>
                      </Table.Td>
                      <Table.Td>
                        <Text size="sm">{formatTimestamp(run.start_timestamp)}</Text>
                        <Text size="sm">{formatTimestamp(run.end_timestamp)}</Text>
                      </Table.Td>
                      <Table.Td>{run.stride}</Table.Td>
                      <Table.Td>{run.fps}</Table.Td>
                      <Table.Td>
                        <Stack gap={4}>
                          <Text size="xs">{progressLabel(run)}</Text>
                          {run.frame_count ? (
                            <Progress value={(run.done_count / run.frame_count) * 100} size="sm" />
                          ) : null}
                        </Stack>
                      </Table.Td>
                      <Table.Td>
                        <Group gap="xs" justify="flex-end" wrap="nowrap">
                          {run.status === 'finished' && (
                            <>
                              <Tooltip label="Download MP4">
                                <ActionIcon
                                  variant="subtle"
                                  component="a"
                                  href={inspectRunVideoUrl(run.id)}
                                  download={`inspect-run-${run.id}.mp4`}
                                >
                                  <Download size={18} />
                                </ActionIcon>
                              </Tooltip>
                              <Tooltip label="Open first frame">
                                <ActionIcon
                                  variant="subtle"
                                  component="a"
                                  href={inspectRunFrameUrl(run.id, 0)}
                                  target="_blank"
                                >
                                  <ImageDown size={18} />
                                </ActionIcon>
                              </Tooltip>
                            </>
                          )}
                          {busy && (
                            <Tooltip label="Abort">
                              <ActionIcon
                                color="yellow"
                                variant="subtle"
                                loading={rowActions.isPending(`abort:${run.id}`)}
                                onClick={() => handleAbort(run)}
                              >
                                <Square size={18} />
                              </ActionIcon>
                            </Tooltip>
                          )}
                          <Tooltip label="Delete">
                            <ActionIcon
                              color="red"
                              variant="subtle"
                              loading={rowActions.isPending(`delete:${run.id}`)}
                              disabled={busy}
                              onClick={() => handleDelete(run)}
                            >
                              <Trash2 size={18} />
                            </ActionIcon>
                          </Tooltip>
                        </Group>
                      </Table.Td>
                    </Table.Tr>
                  );
                })}
              </Table.Tbody>
            </Table>
          </ScrollArea>
          {runs.length === 0 && <Alert color="blue">Inspect runs will appear here.</Alert>}
        </Stack>
      </StepCard>
    </Stack>
  );
}
