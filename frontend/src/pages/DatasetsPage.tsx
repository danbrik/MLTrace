import {
  Alert,
  Badge,
  Button,
  Divider,
  Group,
  Loader,
  Modal,
  Paper,
  ScrollArea,
  Stack,
  Table,
  Text,
  TextInput,
  Textarea,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Check, FileSearch, RefreshCw, ScanLine, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import {
  confirmTimestampFormat,
  createDataset,
  deleteDataset,
  getDataset,
  listDatasets,
  rescanDataset,
  testDatasetConnection,
} from '../api';
import type { Dataset, DatasetConnectionTest } from '../types';

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

export function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
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

  async function refreshDatasets(selectId?: number) {
    const nextDatasets = await listDatasets();
    setDatasets(nextDatasets);
    if (selectId) {
      setSelectedDatasetId(selectId);
    } else if (!selectedDatasetId && nextDatasets.length > 0) {
      setSelectedDatasetId(nextDatasets[0].id);
    }
  }

  useEffect(() => {
    refreshDatasets().catch((error) => {
      notifications.show({ color: 'red', title: 'Could not load datasets', message: error.message });
    });
  }, []);

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId],
  );

  async function handleCreateDataset() {
    setLoading(true);
    try {
      const dataset = await createDataset({ name: datasetName, root_path: rootPath });
      setDatasetName('');
      setRootPath('');
      setConnectionTest(null);
      setTimestampRegex(dataset.timestamp_regex ?? '');
      setTimestampFormat(dataset.timestamp_format ?? '');
      setConfirmationDataset(dataset);
      await refreshDatasets(dataset.id);
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Dataset could not be created',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleTestConnection() {
    setTestingConnection(true);
    try {
      const result = await testDatasetConnection({ root_path: rootPath });
      setConnectionTest(result);
      notifications.show({
        color: result.supported_file_found ? 'green' : result.exists && result.is_directory ? 'yellow' : 'red',
        title: result.supported_file_found ? 'Dataset path reachable' : 'No supported image found',
        message: result.sample_file_path ?? result.message,
      });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Connection test failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setTestingConnection(false);
    }
  }

  async function handleConfirmTimestamp() {
    if (!confirmationDataset) return;
    setScanning(true);
    try {
      const scanned = await confirmTimestampFormat(confirmationDataset.id, {
        timestamp_regex: timestampRegex,
        timestamp_format: timestampFormat,
      });
      setConfirmationDataset(null);
      await refreshDatasets(scanned.id);
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
    } finally {
      setScanning(false);
    }
  }

  async function handleOpenConfirmation(dataset: Dataset) {
    setConfirmationDataset(dataset);
    setTimestampRegex(dataset.timestamp_regex ?? '');
    setTimestampFormat(dataset.timestamp_format ?? '');
  }

  async function handleRescan(datasetId: number) {
    setScanning(true);
    try {
      const dataset = await rescanDataset(datasetId);
      await refreshDatasets(dataset.id);
      notifications.show({ color: 'green', title: 'Dataset rescanned', message: dataset.root_path });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Rescan failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setScanning(false);
    }
  }

  async function handleDeleteDataset(dataset: Dataset) {
    const confirmed = window.confirm(`Delete dataset "${dataset.name}"? Indexed metadata will be removed.`);
    if (!confirmed) return;
    setDeletingDatasetId(dataset.id);
    try {
      await deleteDataset(dataset.id);
      const nextDatasets = await listDatasets();
      setDatasets(nextDatasets);
      setSelectedDatasetId((current) => {
        if (current !== dataset.id) return current;
        return nextDatasets[0]?.id ?? null;
      });
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

  async function handleSelectDataset(datasetId: number) {
    setSelectedDatasetId(datasetId);
    try {
      const dataset = await getDataset(datasetId);
      setDatasets((current) =>
        current.map((item) => (item.id === dataset.id ? dataset : item)),
      );
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Could not load dataset',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
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

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Title order={3}>Add dataset</Title>
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
        </Stack>
      </Paper>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Title order={3}>Registered datasets</Title>
          <ScrollArea>
            <Table highlightOnHover verticalSpacing="sm">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>Status</Table.Th>
                  <Table.Th>Root path</Table.Th>
                  <Table.Th>Timestamp example</Table.Th>
                  <Table.Th>Folders</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {datasets.map((dataset) => (
                  <Table.Tr key={dataset.id}>
                    <Table.Td>
                      <Button
                        variant="subtle"
                        size="compact-sm"
                        onClick={() => handleSelectDataset(dataset.id)}
                      >
                        {dataset.name}
                      </Button>
                    </Table.Td>
                    <Table.Td>
                      <Badge color={statusColor(dataset.status)}>{dataset.status}</Badge>
                    </Table.Td>
                    <Table.Td className="mono">{dataset.root_path}</Table.Td>
                    <Table.Td className="mono">{dataset.timestamp_example ?? 'n/a'}</Table.Td>
                    <Table.Td>{dataset.folders.length}</Table.Td>
                    <Table.Td>
                      <Group gap="xs" justify="flex-end">
                        <Button
                          size="compact-sm"
                          variant="light"
                          leftSection={<Check size={14} />}
                          onClick={() => handleOpenConfirmation(dataset)}
                        >
                          Timestamp
                        </Button>
                        <Button
                          size="compact-sm"
                          variant="light"
                          leftSection={<RefreshCw size={14} />}
                          disabled={!dataset.timestamp_regex || !dataset.timestamp_format}
                          onClick={() => handleRescan(dataset.id)}
                        >
                          Rescan
                        </Button>
                        <Button
                          size="compact-sm"
                          variant="light"
                          color="red"
                          leftSection={<Trash2 size={14} />}
                          loading={deletingDatasetId === dataset.id}
                          onClick={() => handleDeleteDataset(dataset)}
                        >
                          Delete
                        </Button>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
        </Stack>
      </Paper>

      {selectedDataset && (
        <Paper withBorder p="md" radius="sm">
          <Stack gap="md">
            <Group justify="space-between" align="flex-start">
              <div>
                <Title order={3}>{selectedDataset.name}</Title>
                <Text size="sm" c="dimmed" className="mono">
                  {selectedDataset.root_path}
                </Text>
              </div>
              <Badge color={statusColor(selectedDataset.status)}>{selectedDataset.status}</Badge>
            </Group>

            {selectedDataset.scan_error && (
              <Alert color="red" title="Last scan error">
                {selectedDataset.scan_error}
              </Alert>
            )}

            <Divider />
            <Title order={4}>Folder summaries</Title>
            <ScrollArea>
              <Table verticalSpacing="sm" striped>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Folder</Table.Th>
                    <Table.Th>Images</Table.Th>
                    <Table.Th>Start</Table.Th>
                    <Table.Th>End</Table.Th>
                    <Table.Th>Resolution</Table.Th>
                    <Table.Th>Image metadata</Table.Th>
                    <Table.Th>Median spacing</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {selectedDataset.folders.map((folder) => (
                    <Table.Tr key={folder.id}>
                      <Table.Td className="mono">{folder.relative_path}</Table.Td>
                      <Table.Td>{folder.image_count}</Table.Td>
                      <Table.Td>{formatTimestamp(folder.first_timestamp)}</Table.Td>
                      <Table.Td>{formatTimestamp(folder.last_timestamp)}</Table.Td>
                      <Table.Td>{formatJsonSummary(folder.resolution_summary)}</Table.Td>
                      <Table.Td>{formatImageMetadata(folder.image_metadata)}</Table.Td>
                      <Table.Td>
                        {folder.cadence_summary?.median_seconds == null
                          ? 'n/a'
                          : `${folder.cadence_summary.median_seconds}s`}
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </ScrollArea>
          </Stack>
        </Paper>
      )}

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
