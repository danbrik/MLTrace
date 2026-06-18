import {
  Alert,
  Badge,
  Button,
  Divider,
  Group,
  Loader,
  Modal,
  ScrollArea,
  Stack,
  Table,
  Text,
  TextInput,
  Textarea,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';

import { StepCard } from '../components/StepCard';
import { Check, FileSearch, Info, RefreshCw, ScanLine, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import {
  confirmTimestampFormat,
  createDataset,
  deleteDataset,
  listDatasets,
  rescanDataset,
  testDatasetConnection,
} from '../api';
import { usePendingIds } from '../hooks/usePendingIds';
import type { Dataset, DatasetConnectionTest } from '../types';

type ActivityLogEntry = {
  id: number;
  level: 'info' | 'success' | 'error';
  time: string;
  message: string;
};

type RunningOperation = {
  kind: 'detect-timestamps' | 'test-path' | 'scan-dataset' | 'rescan';
  label: string;
  startedAt: number;
};

function formatTimestamp(value: string | null): string {
  if (!value) return 'n/a';
  return value.replace('T', ' ');
}

function formatJsonSummary(value: Record<string, number> | null | undefined): string {
  if (!value) return 'n/a';
  return Object.entries(value)
    .map(([key, count]) => `${key}: ${count}`)
    .join(', ');
}

function statusColor(status: string): string {
  if (status === 'ready') return 'green';
  if (status === 'failed') return 'red';
  if (status === 'scanning') return 'blue';
  return 'yellow';
}

function formatImageMetadata(value: Record<string, unknown> | null | undefined): string {
  if (!value) return 'n/a';
  const parts = [
    value.format ? `format ${value.format}` : null,
    value.mode ? `mode ${value.mode}` : null,
    value.dtype ? `dtype ${value.dtype}` : null,
    value.channels ? `${value.channels} ch` : null,
  ].filter(Boolean);
  return parts.length ? parts.join(', ') : 'n/a';
}

function formatMeanSpacing(folder: Dataset['folders'][number] | undefined): string {
  const seconds = folder?.cadence_summary?.mean_seconds ?? folder?.cadence_summary?.median_seconds;
  return seconds == null ? 'n/a' : `${seconds}s`;
}

export function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetName, setDatasetName] = useState('');
  const [rootPath, setRootPath] = useState('');
  const [timestampRegex, setTimestampRegex] = useState('');
  const [timestampFormat, setTimestampFormat] = useState('');
  const [confirmationDataset, setConfirmationDataset] = useState<Dataset | null>(null);
  const [connectionTest, setConnectionTest] = useState<DatasetConnectionTest | null>(null);
  const [loading, setLoading] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [deletingDatasetId, setDeletingDatasetId] = useState<number | null>(null);
  const [scanning, setScanning] = useState(false);
  const [activityLog, setActivityLog] = useState<ActivityLogEntry[]>([]);
  const [runningOperation, setRunningOperation] = useState<RunningOperation | null>(null);
  const rowActions = usePendingIds();
  const [, setElapsedTick] = useState(0);
  const lastWaitLogSecondRef = useRef(0);

  function addActivity(level: ActivityLogEntry['level'], message: string) {
    setActivityLog((current) => [
      {
        id: Date.now() + Math.random(),
        level,
        time: new Date().toLocaleTimeString(),
        message,
      },
      ...current,
    ].slice(0, 30));
  }

  async function refreshDatasets() {
    const nextDatasets = await listDatasets();
    setDatasets(nextDatasets);
  }

  useEffect(() => {
    refreshDatasets().catch((error) => {
      notifications.show({ color: 'red', title: 'Could not load datasets', message: error.message });
    });
  }, []);

  useEffect(() => {
    if (!runningOperation) return undefined;
    const timer = window.setInterval(() => setElapsedTick((current) => current + 1), 1000);
    return () => window.clearInterval(timer);
  }, [runningOperation]);

  const runningElapsedSeconds = runningOperation
    ? Math.floor((Date.now() - runningOperation.startedAt) / 1000)
    : 0;

  useEffect(() => {
    if (!runningOperation) {
      lastWaitLogSecondRef.current = 0;
      return;
    }
    if (runningOperation.kind !== 'test-path') return;
    if (runningElapsedSeconds === 0) return;
    if (runningElapsedSeconds % 10 !== 0) return;
    if (lastWaitLogSecondRef.current === runningElapsedSeconds) return;

    lastWaitLogSecondRef.current = runningElapsedSeconds;
    addActivity(
      'info',
      `Test path still waiting after ${runningElapsedSeconds}s. No backend response yet from /api/datasets/test-connection.`,
    );
  }, [runningElapsedSeconds, runningOperation]);

  async function handleCreateDataset() {
    if (loading) return;
    setLoading(true);
    setElapsedTick(0);
    setRunningOperation({ kind: 'detect-timestamps', label: `Detect timestamps: ${rootPath}`, startedAt: Date.now() });
    addActivity('info', `Detect timestamps started for ${rootPath}`);
    try {
      const dataset = await createDataset({ name: datasetName, root_path: rootPath });
      addActivity(
        dataset.timestamp_regex ? 'success' : 'error',
        dataset.timestamp_regex
          ? `Timestamp parser detected from example ${dataset.timestamp_example ?? 'n/a'}`
          : `No timestamp parser detected for ${dataset.root_path}`,
      );
      setDatasetName('');
      setRootPath('');
      setConnectionTest(null);
      setTimestampRegex(dataset.timestamp_regex ?? '');
      setTimestampFormat(dataset.timestamp_format ?? '');
      setConfirmationDataset(dataset);
      await refreshDatasets();
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Dataset could not be created',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
      addActivity('error', `Detect timestamps failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
      setRunningOperation(null);
    }
  }

  async function handleTestConnection() {
    if (testingConnection) return;
    const startedAt = Date.now();
    setTestingConnection(true);
    setElapsedTick(0);
    setRunningOperation({ kind: 'test-path', label: `Test path: ${rootPath}`, startedAt });
    addActivity('info', `Test path started for ${rootPath}`);
    addActivity('info', 'Dispatching POST /api/datasets/test-connection');
    addActivity('info', 'Timeout budget: 60s');
    addActivity('info', 'Waiting for backend path probe (resolve -> exists -> is_dir -> first direct TIFF).');
    try {
      const result = await testDatasetConnection({ root_path: rootPath });
      const elapsedSeconds = Math.max(1, Math.floor((Date.now() - startedAt) / 1000));
      setConnectionTest(result);
      addActivity('success', `Backend responded after ${elapsedSeconds}s.`);
      addActivity(
        result.supported_file_found ? 'success' : 'error',
        result.sample_file_path ? `Supported file found: ${result.sample_file_path}` : result.message,
      );
      notifications.show({
        color: result.supported_file_found ? 'green' : result.exists && result.is_directory ? 'yellow' : 'red',
        title: result.supported_file_found ? 'Dataset path reachable' : 'No supported image found',
        message: result.sample_file_path ?? result.message,
      });
    } catch (error) {
      const elapsedSeconds = Math.max(1, Math.floor((Date.now() - startedAt) / 1000));
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      notifications.show({
        color: 'red',
        title: 'Connection test failed',
        message: errorMessage,
      });
      if (errorMessage.startsWith('Request timed out after')) {
        addActivity(
          'error',
          `Test path timed out after ${elapsedSeconds}s. No backend response was received from /api/datasets/test-connection before the client timeout.`,
        );
      }
      addActivity('error', `Test path failed after ${elapsedSeconds}s: ${errorMessage}`);
    } finally {
      setTestingConnection(false);
      setRunningOperation(null);
    }
  }

  async function handleConfirmTimestamp() {
    if (!confirmationDataset) return;
    if (scanning) return;
    setScanning(true);
    setElapsedTick(0);
    setRunningOperation({ kind: 'scan-dataset', label: `Scan dataset: ${confirmationDataset.root_path}`, startedAt: Date.now() });
    addActivity('info', `Confirm timestamp parser started for ${confirmationDataset.root_path}`);
    try {
      const scanned = await confirmTimestampFormat(confirmationDataset.id, {
        timestamp_regex: timestampRegex,
        timestamp_format: timestampFormat,
      });
      addActivity(
        scanned.status === 'ready' ? 'success' : 'error',
        scanned.status === 'ready'
          ? `Scan finished: ${scanned.folders.length} folder summaries indexed`
          : `Scan failed: ${scanned.scan_error ?? 'unknown error'}`,
      );
      setConfirmationDataset(null);
      await refreshDatasets();
      notifications.show({
        color: scanned.status === 'ready' ? 'green' : 'red',
        title: scanned.status === 'ready' ? 'Dataset scanned' : 'Scan failed',
        message:
          scanned.status === 'ready'
            ? `${scanned.folders.length} folder summaries are available.`
            : (scanned.scan_error ?? 'The scanner returned an error.'),
      });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Scan failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
      addActivity('error', `Scan failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setScanning(false);
      setRunningOperation(null);
    }
  }

  async function handleOpenConfirmation(dataset: Dataset) {
    setConfirmationDataset(dataset);
    setTimestampRegex(dataset.timestamp_regex ?? '');
    setTimestampFormat(dataset.timestamp_format ?? '');
  }

  async function handleRescan(datasetId: number) {
    if (rowActions.isPending(`rescan:${datasetId}`)) return;
    setScanning(true);
    const datasetForLog = datasets.find((dataset) => dataset.id === datasetId);
    setElapsedTick(0);
    setRunningOperation({ kind: 'rescan', label: `Rescan: ${datasetForLog?.root_path ?? datasetId}`, startedAt: Date.now() });
    addActivity('info', `Rescan started for ${datasetForLog?.root_path ?? datasetId}`);
    await rowActions.runPending(`rescan:${datasetId}`, async () => {
      const dataset = await rescanDataset(datasetId);
      addActivity('success', `Rescan finished: ${dataset.folders.length} folder summaries indexed`);
      await refreshDatasets();
      notifications.show({ color: 'green', title: 'Dataset rescanned', message: dataset.root_path });
    }).catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Rescan failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
      addActivity('error', `Rescan failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }).finally(() => {
      setScanning(false);
      setRunningOperation(null);
    });
  }

  async function handleDeleteDataset(dataset: Dataset) {
    if (deletingDatasetId === dataset.id) return;
    const confirmed = window.confirm(`Delete dataset "${dataset.name}"? Indexed metadata will be removed.`);
    if (!confirmed) return;
    setDeletingDatasetId(dataset.id);
    try {
      await deleteDataset(dataset.id);
      const nextDatasets = await listDatasets();
      setDatasets(nextDatasets);
      notifications.show({ color: 'green', title: 'Dataset deleted', message: dataset.name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Dataset could not be deleted',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setDeletingDatasetId(null);
    }
  }

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={2}>Datasets</Title>
          <Text c="dimmed" size="sm">
            Register source folders, confirm filename timestamps, and index TIFF metadata.
          </Text>
        </div>
        {scanning && (
          <Group gap="xs">
            <Loader size="sm" />
            <Text size="sm" c="dimmed">
              Scanner running
            </Text>
          </Group>
        )}
      </Group>

      <StepCard title="Add dataset" color="blue">
          <Group grow align="flex-end">
            <TextInput
              label="Name"
              placeholder="February normal state"
              value={datasetName}
              onChange={(event) => setDatasetName(event.currentTarget.value)}
            />
            <TextInput
              label="Root path"
              placeholder="/data/images"
              value={rootPath}
              onChange={(event) => {
                setRootPath(event.currentTarget.value);
                setConnectionTest(null);
              }}
            />
            <Button
              variant="light"
              leftSection={<FileSearch size={18} />}
              onClick={handleTestConnection}
              loading={testingConnection}
              disabled={!rootPath.trim()}
            >
              Test path
            </Button>
            <Button
              leftSection={<ScanLine size={18} />}
              onClick={handleCreateDataset}
              loading={loading}
              disabled={!datasetName.trim() || !rootPath.trim()}
            >
              Detect timestamps
            </Button>
          </Group>
          {connectionTest && (
            <Alert
              color={connectionTest.supported_file_found ? 'green' : connectionTest.exists && connectionTest.is_directory ? 'yellow' : 'red'}
              title={connectionTest.supported_file_found ? 'Supported image found' : 'Path check result'}
            >
              <Stack gap={4}>
                <Text size="sm">{connectionTest.message}</Text>
                <Text size="sm" className="mono">
                  Root: {connectionTest.root_path}
                </Text>
                {connectionTest.sample_file_path && (
                  <Text size="sm" className="mono">
                    Sample: {connectionTest.sample_file_path}
                  </Text>
                )}
              </Stack>
            </Alert>
          )}
          <Divider />
          <Stack gap="xs">
            <Group justify="space-between" align="center">
              <Text fw={600} size="sm">
                Activity log
              </Text>
              {runningOperation ? (
                <Badge color="blue" variant="light">
                  {runningOperation.label} · {runningElapsedSeconds}s
                </Badge>
              ) : (
                <Badge color="gray" variant="light">
                  idle
                </Badge>
              )}
            </Group>
            <ScrollArea h={120} mt="xs">
              <Stack gap={4}>
                {runningOperation && (
                  <Text size="xs" c="blue">
                    Waiting for response since {runningElapsedSeconds}s.
                  </Text>
                )}
                {activityLog.length === 0 ? (
                  <Text size="xs" c="dimmed">
                    No dataset actions yet.
                  </Text>
                ) : (
                  activityLog.map((entry) => (
                    <Text
                      key={entry.id}
                      size="xs"
                      c={entry.level === 'error' ? 'red' : entry.level === 'success' ? 'green' : 'dimmed'}
                    >
                      <span className="mono">{entry.time}</span> {entry.message}
                    </Text>
                  ))
                )}
              </Stack>
            </ScrollArea>
          </Stack>
      </StepCard>

      <StepCard title="Registered datasets" color="cyan">
          <ScrollArea>
            <Table highlightOnHover verticalSpacing="sm">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>Status</Table.Th>
                  <Table.Th>Root path</Table.Th>
                  <Table.Th>Timestamp example</Table.Th>
                  <Table.Th>Images</Table.Th>
                  <Table.Th>Start</Table.Th>
                  <Table.Th>End</Table.Th>
                  <Table.Th>Resolution</Table.Th>
                  <Table.Th>Image metadata</Table.Th>
                  <Table.Th>
                    <Group gap={4}>
                      Mean spacing
                      <Tooltip label="Mean spacing is estimated from sampled filename timestamps, rounded to the nearest second. Exact Train/Test counts are computed by parsing filenames in the selected time range.">
                        <Info size={14} />
                      </Tooltip>
                    </Group>
                  </Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {datasets.map((dataset) => {
                  const folder = dataset.folders[0];
                  return (
                    <Table.Tr key={dataset.id}>
                      <Table.Td>{dataset.name}</Table.Td>
                      <Table.Td>
                        <Badge color={statusColor(dataset.status)}>{dataset.status}</Badge>
                      </Table.Td>
                      <Table.Td className="mono">{dataset.root_path}</Table.Td>
                      <Table.Td className="mono">{dataset.timestamp_example ?? 'n/a'}</Table.Td>
                      <Table.Td>{folder?.image_count ?? 'n/a'}</Table.Td>
                      <Table.Td>{formatTimestamp(folder?.first_timestamp ?? null)}</Table.Td>
                      <Table.Td>{formatTimestamp(folder?.last_timestamp ?? null)}</Table.Td>
                      <Table.Td>{formatJsonSummary(folder?.resolution_summary)}</Table.Td>
                      <Table.Td>{formatImageMetadata(folder?.image_metadata)}</Table.Td>
                      <Table.Td>{formatMeanSpacing(folder)}</Table.Td>
                      <Table.Td>
                        <Group gap="xs" justify="flex-end">
                          <Button
                            size="compact-sm"
                            variant="light"
                            leftSection={<Check size={14} />}
                            onClick={() => handleOpenConfirmation(dataset)}
                            disabled={dataset.is_update_locked}
                          >
                            Timestamp
                          </Button>
                          <Button
                            size="compact-sm"
                            variant="light"
                            leftSection={<RefreshCw size={14} />}
                            loading={rowActions.isPending(`rescan:${dataset.id}`)}
                            disabled={
                              !dataset.timestamp_regex ||
                              !dataset.timestamp_format ||
                              dataset.is_update_locked ||
                              deletingDatasetId === dataset.id
                            }
                            onClick={() => handleRescan(dataset.id)}
                          >
                            {rowActions.isPending(`rescan:${dataset.id}`) ? 'Rescanning…' : 'Rescan'}
                          </Button>
                          <Button
                            size="compact-sm"
                            variant="light"
                            color="red"
                            leftSection={<Trash2 size={14} />}
                            loading={deletingDatasetId === dataset.id}
                            disabled={rowActions.isPending(`rescan:${dataset.id}`)}
                            onClick={() => handleDeleteDataset(dataset)}
                          >
                            Delete
                          </Button>
                        </Group>
                        {dataset.update_lock_reasons.length > 0 && (
                          <Text size="xs" c="dimmed" ta="right" mt={4}>
                            {dataset.update_lock_reasons[0]}
                          </Text>
                        )}
                      </Table.Td>
                    </Table.Tr>
                  );
                })}
              </Table.Tbody>
            </Table>
          </ScrollArea>
      </StepCard>

      <Modal
        opened={confirmationDataset !== null}
        onClose={() => setConfirmationDataset(null)}
        title="Confirm timestamp parser"
        size="lg"
      >
        <Stack gap="md">
          <Text size="sm" c="dimmed">
            Confirm or edit the filename parser before MLTrace indexes the dataset.
          </Text>
          <Textarea
            label="Timestamp regex"
            description="Use a named group `timestamp` or the first capture group."
            minRows={3}
            value={timestampRegex}
            onChange={(event) => setTimestampRegex(event.currentTarget.value)}
          />
          <TextInput
            label="Python datetime format"
            placeholder="%Y%m%d_%H%M%S"
            value={timestampFormat}
            onChange={(event) => setTimestampFormat(event.currentTarget.value)}
          />
          <Text size="sm">
            Example: <span className="mono">{confirmationDataset?.timestamp_example ?? 'n/a'}</span>
          </Text>
          <Group justify="flex-end">
            <Button variant="subtle" onClick={() => setConfirmationDataset(null)}>
              Cancel
            </Button>
            <Button
              leftSection={<ScanLine size={18} />}
              onClick={handleConfirmTimestamp}
              loading={scanning}
              disabled={!timestampRegex.trim() || !timestampFormat.trim()}
            >
              Confirm and scan
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
}
