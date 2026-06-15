import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Group,
  Modal,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Eye, Image as ImageIcon, Play, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent } from 'react';

import {
  createRoi,
  createTestingRun,
  deleteRoi,
  deleteTestingRun,
  getTestingRunResults,
  listRois,
  listTestingRuns,
  listTrainingDatasets,
  listTrainingRuns,
  previewRoi,
} from '../api';
import { formatDuration, runStatusColor } from '../training/runStatus';
import type { RoiDefinition, RoiPreview, TestingRun, TestingRunResults, TrainingDataset, TrainingRun } from '../types';

type Rect = { x: number; y: number; width: number; height: number };
type DragMode = 'move' | 'tl' | 'br';
type DragState = Rect & { mode: DragMode; startX: number; startY: number };

function notifyError(title: string, error: unknown) {
  notifications.show({ color: 'red', title, message: error instanceof Error ? error.message : 'Unknown error' });
}

function formatTimestamp(value: string | null): string {
  if (!value) return 'n/a';
  return new Date(value).toLocaleString();
}

function formatScore(value: number | null): string {
  if (value === null || value === undefined) return '—';
  if (Math.abs(value) >= 1000) return value.toExponential(3);
  return value.toFixed(6);
}

function roiLabel(roi: RoiDefinition): string {
  return `${roi.name} (${roi.image_width}x${roi.image_height}, x${roi.x} y${roi.y} ${roi.width}x${roi.height})`;
}

function pointFromEvent(event: PointerEvent<HTMLDivElement>, preview: RoiPreview) {
  const rect = event.currentTarget.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * preview.width;
  const y = ((event.clientY - rect.top) / rect.height) * preview.height;
  return {
    x: Math.round(Math.max(0, Math.min(preview.width, x))),
    y: Math.round(Math.max(0, Math.min(preview.height, y))),
  };
}

function defaultRect(preview: RoiPreview): Rect {
  const width = Math.max(1, Math.round(preview.width * 0.5));
  const height = Math.max(1, Math.round(preview.height * 0.5));
  return {
    x: Math.round((preview.width - width) / 2),
    y: Math.round((preview.height - height) / 2),
    width,
    height,
  };
}

function RoiPicker({
  preview,
  rect,
  onChange,
}: {
  preview: RoiPreview;
  rect: Rect;
  onChange: (rect: Rect) => void;
}) {
  const dragRef = useRef<DragState | null>(null);
  const style = {
    left: `${(rect.x / preview.width) * 100}%`,
    top: `${(rect.y / preview.height) * 100}%`,
    width: `${(rect.width / preview.width) * 100}%`,
    height: `${(rect.height / preview.height) * 100}%`,
  };

  function startDrag(event: PointerEvent<HTMLElement>, mode: DragMode) {
    event.preventDefault();
    event.stopPropagation();
    const point = pointFromEvent(event as unknown as PointerEvent<HTMLDivElement>, preview);
    dragRef.current = { ...rect, mode, startX: point.x, startY: point.y };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }

  function onMove(event: PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    const point = pointFromEvent(event, preview);
    if (drag.mode === 'move') {
      const x = Math.max(0, Math.min(preview.width - drag.width, drag.x + point.x - drag.startX));
      const y = Math.max(0, Math.min(preview.height - drag.height, drag.y + point.y - drag.startY));
      onChange({ ...rect, x, y });
    } else if (drag.mode === 'tl') {
      const right = drag.x + drag.width;
      const bottom = drag.y + drag.height;
      const x = Math.max(0, Math.min(right - 1, point.x));
      const y = Math.max(0, Math.min(bottom - 1, point.y));
      onChange({ x, y, width: right - x, height: bottom - y });
    } else {
      onChange({
        ...rect,
        width: Math.max(1, Math.min(preview.width - drag.x, point.x - drag.x)),
        height: Math.max(1, Math.min(preview.height - drag.y, point.y - drag.y)),
      });
    }
  }

  return (
    <Stack gap="xs">
      <div className="warp-picker" onPointerMove={onMove} onPointerUp={() => (dragRef.current = null)} onPointerLeave={() => (dragRef.current = null)}>
        <img src={preview.image_data_url} alt="ROI preview" className="warp-picker-image" />
        <div className="crop-rect" style={style} onPointerDown={(event) => startDrag(event, 'move')}>
          <span className="crop-handle crop-handle-tl" onPointerDown={(event) => startDrag(event, 'tl')} />
          <span className="crop-handle crop-handle-br" onPointerDown={(event) => startDrag(event, 'br')} />
        </div>
      </div>
      <Group gap="xs">
        <Badge variant="light">x: {rect.x}</Badge>
        <Badge variant="light">y: {rect.y}</Badge>
        <Badge variant="light">w: {rect.width}</Badge>
        <Badge variant="light">h: {rect.height}</Badge>
      </Group>
    </Stack>
  );
}

export function TestingRunsPage({ active = true }: { active?: boolean }) {
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [rois, setRois] = useState<RoiDefinition[]>([]);
  const [testingRuns, setTestingRuns] = useState<TestingRun[]>([]);
  const [selectedTrainingRunId, setSelectedTrainingRunId] = useState<string | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [selectedRoiId, setSelectedRoiId] = useState<string | null>(null);
  const [runName, setRunName] = useState('');
  const [roiName, setRoiName] = useState('');
  const [preview, setPreview] = useState<RoiPreview | null>(null);
  const [rect, setRect] = useState<Rect | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [savingRoi, setSavingRoi] = useState(false);
  const [running, setRunning] = useState(false);
  const [inspected, setInspected] = useState<TestingRunResults | null>(null);

  async function refresh() {
    const [nextTrainingRuns, nextTrainingDatasets, nextRois, nextTestingRuns] = await Promise.all([
      listTrainingRuns(),
      listTrainingDatasets(),
      listRois(),
      listTestingRuns(),
    ]);
    setTrainingRuns(nextTrainingRuns);
    setTrainingDatasets(nextTrainingDatasets);
    setRois(nextRois);
    setTestingRuns(nextTestingRuns);
  }

  useEffect(() => {
    if (!active) return;
    refresh().catch((error) => notifyError('Could not load testing data', error));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  const finishedTrainingRuns = useMemo(
    () => trainingRuns.filter((run) => run.status === 'finished' && run.artifact_path),
    [trainingRuns],
  );
  const selectedTrainingRun = useMemo(
    () => finishedTrainingRuns.find((run) => String(run.id) === selectedTrainingRunId) ?? null,
    [finishedTrainingRuns, selectedTrainingRunId],
  );
  const selectedDataset = useMemo(
    () => trainingDatasets.find((dataset) => String(dataset.id) === selectedDatasetId) ?? null,
    [trainingDatasets, selectedDatasetId],
  );
  const selectedRoi = useMemo(
    () => rois.find((roi) => String(roi.id) === selectedRoiId) ?? null,
    [rois, selectedRoiId],
  );
  const selectedRoiMismatch =
    preview && selectedRoi && (preview.width !== selectedRoi.image_width || preview.height !== selectedRoi.image_height);

  async function handlePreview() {
    if (!selectedTrainingRunId || !selectedDatasetId) return;
    setLoadingPreview(true);
    try {
      const nextPreview = await previewRoi({
        training_run_id: Number(selectedTrainingRunId),
        training_dataset_id: Number(selectedDatasetId),
      });
      setPreview(nextPreview);
      setRect(defaultRect(nextPreview));
      if (!roiName) setRoiName(`ROI ${nextPreview.width}x${nextPreview.height}`);
    } catch (error) {
      notifyError('Could not load ROI preview', error);
    } finally {
      setLoadingPreview(false);
    }
  }

  async function handleSaveRoi() {
    if (!preview || !rect || !roiName.trim()) return;
    setSavingRoi(true);
    try {
      const roi = await createRoi({
        name: roiName.trim(),
        image_width: preview.width,
        image_height: preview.height,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      });
      setSelectedRoiId(String(roi.id));
      setRoiName('');
      await refresh();
      notifications.show({ color: 'green', title: 'ROI saved', message: roi.name });
    } catch (error) {
      notifyError('Could not save ROI', error);
    } finally {
      setSavingRoi(false);
    }
  }

  async function handleDeleteRoi(roi: RoiDefinition) {
    if (!window.confirm(`Delete ROI "${roi.name}"?`)) return;
    try {
      await deleteRoi(roi.id);
      if (selectedRoiId === String(roi.id)) setSelectedRoiId(null);
      await refresh();
      notifications.show({ color: 'green', title: 'ROI deleted', message: roi.name });
    } catch (error) {
      notifyError('Could not delete ROI', error);
    }
  }

  async function handleRunTesting() {
    if (!selectedTrainingRunId || !selectedDatasetId || selectedRoiMismatch) return;
    setRunning(true);
    try {
      const run = await createTestingRun({
        training_run_id: Number(selectedTrainingRunId),
        training_dataset_id: Number(selectedDatasetId),
        roi_id: selectedRoiId ? Number(selectedRoiId) : null,
        name: runName.trim() || null,
      });
      setRunName('');
      await refresh();
      notifications.show({
        color: run.status === 'finished' ? 'green' : 'red',
        title: run.status === 'finished' ? 'Testing run finished' : 'Testing run failed',
        message: run.error_message ?? run.name,
      });
    } catch (error) {
      notifyError('Could not run testing', error);
    } finally {
      setRunning(false);
    }
  }

  async function handleInspect(run: TestingRun) {
    try {
      setInspected(await getTestingRunResults(run.id));
    } catch (error) {
      notifyError('Could not load testing results', error);
    }
  }

  async function handleDeleteTestingRun(run: TestingRun) {
    if (!window.confirm(`Delete testing run "${run.name}"?`)) return;
    try {
      await deleteTestingRun(run.id);
      await refresh();
      notifications.show({ color: 'green', title: 'Testing run deleted', message: run.name });
    } catch (error) {
      notifyError('Could not delete testing run', error);
    }
  }

  return (
    <Stack gap="lg">
      <div>
        <Title order={2}>Testing Runs</Title>
        <Text c="dimmed" size="sm">
          Run trained reconstruction artifacts on train/test datasets and persist one reconstruction error per image.
        </Text>
      </div>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Title order={3}>Start Testing</Title>
          <Group grow align="flex-end">
            <Select
              label="Training run artifact"
              placeholder="Select finished training run"
              data={finishedTrainingRuns.map((run) => ({
                value: String(run.id),
                label: `${run.training_pipeline_name} · ${run.method_type} · ${run.artifact_kind}`,
              }))}
              value={selectedTrainingRunId}
              onChange={(value) => {
                setSelectedTrainingRunId(value);
                setPreview(null);
                setRect(null);
              }}
              searchable
            />
            <Select
              label="Test dataset"
              placeholder="Train/test dataset"
              data={trainingDatasets.map((dataset) => ({
                value: String(dataset.id),
                label: `${dataset.name} · ${dataset.usage_label ?? 'train'} · ${dataset.total_selected_images} images`,
              }))}
              value={selectedDatasetId}
              onChange={(value) => {
                setSelectedDatasetId(value);
                setPreview(null);
                setRect(null);
              }}
              searchable
            />
            <Select
              label="ROI"
              placeholder="Full image MSE"
              data={rois.map((roi) => ({ value: String(roi.id), label: roiLabel(roi) }))}
              value={selectedRoiId}
              onChange={setSelectedRoiId}
              clearable
              searchable
            />
          </Group>

          <Group align="flex-end">
            <TextInput
              label="Run name"
              placeholder={selectedTrainingRun && selectedDataset ? `${selectedDataset.name} on ${selectedTrainingRun.training_pipeline_name}` : 'Optional'}
              value={runName}
              onChange={(event) => setRunName(event.currentTarget.value)}
              style={{ flex: 1 }}
            />
            <Button
              leftSection={<ImageIcon size={16} />}
              variant="light"
              onClick={handlePreview}
              loading={loadingPreview}
              disabled={!selectedTrainingRunId || !selectedDatasetId}
            >
              Load ROI Preview
            </Button>
            <Button
              leftSection={<Play size={16} />}
              onClick={handleRunTesting}
              loading={running}
              disabled={!selectedTrainingRunId || !selectedDatasetId || Boolean(selectedRoiMismatch)}
            >
              Run Testing
            </Button>
          </Group>

          {selectedTrainingRun && (
            <Alert color="blue">
              Preprocessing is inherited from <b>{selectedTrainingRun.training_pipeline_name}</b>: {selectedTrainingRun.preprocessing_pipeline_name}.
            </Alert>
          )}
          {selectedRoiMismatch && (
            <Alert color="red">
              Selected ROI is tuned for {selectedRoi.image_width}x{selectedRoi.image_height}, but the current preview is {preview.width}x{preview.height}.
            </Alert>
          )}

          {preview && rect && (
            <Stack gap="sm">
              <Group justify="space-between" align="center">
                <Text size="sm" c="dimmed" className="path-text">
                  Preview: {preview.width}x{preview.height}, {preview.dtype}, {preview.source_image_path}
                </Text>
                <Group>
                  <TextInput
                    placeholder="ROI name"
                    value={roiName}
                    onChange={(event) => setRoiName(event.currentTarget.value)}
                  />
                  <Button variant="light" onClick={handleSaveRoi} loading={savingRoi} disabled={!roiName.trim()}>
                    Save ROI
                  </Button>
                </Group>
              </Group>
              <RoiPicker preview={preview} rect={rect} onChange={setRect} />
            </Stack>
          )}
        </Stack>
      </Paper>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Title order={3}>Saved ROIs</Title>
          <ScrollArea>
            <Table striped highlightOnHover verticalSpacing="sm">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>Image size</Table.Th>
                  <Table.Th>Rect</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {rois.map((roi) => (
                  <Table.Tr key={roi.id}>
                    <Table.Td>{roi.name}</Table.Td>
                    <Table.Td>{roi.image_width}x{roi.image_height}</Table.Td>
                    <Table.Td>x {roi.x}, y {roi.y}, {roi.width}x{roi.height}</Table.Td>
                    <Table.Td>
                      <Group justify="flex-end">
                        <Tooltip label="Delete ROI">
                          <ActionIcon aria-label={`Delete ROI ${roi.name}`} color="red" variant="subtle" onClick={() => handleDeleteRoi(roi)}>
                            <Trash2 size={18} />
                          </ActionIcon>
                        </Tooltip>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
          {rois.length === 0 && <Alert color="blue">No ROIs saved yet.</Alert>}
        </Stack>
      </Paper>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Title order={3}>Saved Testing Runs</Title>
          <ScrollArea>
            <Table striped highlightOnHover verticalSpacing="sm" miw={1100}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>Training run</Table.Th>
                  <Table.Th>Dataset</Table.Th>
                  <Table.Th>ROI</Table.Th>
                  <Table.Th>Status</Table.Th>
                  <Table.Th>Images</Table.Th>
                  <Table.Th>Mean score</Table.Th>
                  <Table.Th>Duration</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {testingRuns.map((run) => (
                  <Table.Tr key={run.id}>
                    <Table.Td>{run.name}</Table.Td>
                    <Table.Td>
                      <Stack gap={2}>
                        <Text size="sm">{run.training_pipeline_name}</Text>
                        <Text size="xs" c="dimmed">{run.method_type}</Text>
                      </Stack>
                    </Table.Td>
                    <Table.Td>{run.training_dataset_name}</Table.Td>
                    <Table.Td>{run.roi_name ?? 'Full image'}</Table.Td>
                    <Table.Td><Badge color={runStatusColor(run.status)}>{run.status}</Badge></Table.Td>
                    <Table.Td>{run.image_count ?? '—'}</Table.Td>
                    <Table.Td>{formatScore(run.score_mean)}</Table.Td>
                    <Table.Td>{formatDuration(run.duration_seconds)}</Table.Td>
                    <Table.Td>
                      <Group gap="xs" justify="flex-end" wrap="nowrap">
                        <Tooltip label="Inspect results">
                          <ActionIcon aria-label={`Inspect testing run ${run.name}`} variant="subtle" onClick={() => handleInspect(run)}>
                            <Eye size={18} />
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Delete">
                          <ActionIcon aria-label={`Delete testing run ${run.name}`} color="red" variant="subtle" onClick={() => handleDeleteTestingRun(run)}>
                            <Trash2 size={18} />
                          </ActionIcon>
                        </Tooltip>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
          {testingRuns.length === 0 && <Alert color="blue">No testing runs have been saved yet.</Alert>}
        </Stack>
      </Paper>

      <Modal opened={inspected !== null} onClose={() => setInspected(null)} title={inspected?.testing_run.name ?? 'Testing results'} size="xl">
        {inspected && (
          <Stack gap="md">
            <Group>
              <Badge color={runStatusColor(inspected.testing_run.status)}>{inspected.testing_run.status}</Badge>
              <Badge variant="light">{inspected.testing_run.image_count ?? 0} images</Badge>
              <Badge variant="light">mean {formatScore(inspected.testing_run.score_mean)}</Badge>
              {inspected.testing_run.roi_name && <Badge variant="light">ROI {inspected.testing_run.roi_name}</Badge>}
            </Group>
            <Text size="xs" c="dimmed" className="path-text">
              CSV: {inspected.testing_run.results_path ?? 'n/a'}
            </Text>
            {inspected.testing_run.error_message && <Alert color="red">{inspected.testing_run.error_message}</Alert>}
            <ScrollArea h={420}>
              <Table striped highlightOnHover verticalSpacing="xs" miw={1000}>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>#</Table.Th>
                    <Table.Th>Timestamp</Table.Th>
                    <Table.Th>Score</Table.Th>
                    <Table.Th>Full MSE</Table.Th>
                    <Table.Th>ROI MSE</Table.Th>
                    <Table.Th>Image</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {inspected.results.map((result) => (
                    <Table.Tr key={result.id}>
                      <Table.Td>{result.position + 1}</Table.Td>
                      <Table.Td>{formatTimestamp(result.timestamp)}</Table.Td>
                      <Table.Td>{formatScore(result.score)}</Table.Td>
                      <Table.Td>{formatScore(result.full_mse)}</Table.Td>
                      <Table.Td>{formatScore(result.roi_mse)}</Table.Td>
                      <Table.Td><Text size="xs" className="path-text">{result.image_path}</Text></Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </ScrollArea>
          </Stack>
        )}
      </Modal>
    </Stack>
  );
}
