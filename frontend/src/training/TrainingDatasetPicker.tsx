import {
  ActionIcon,
  Alert,
  Badge,
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
import { ArrowDown, ArrowUp, Info, Plus, Search, Trash2 } from 'lucide-react';
import { useMemo, useState } from 'react';

import { datasetResolutions, datasetSizeSignature } from './graph';
import type { TrainingDataset } from '../types';

const USAGE_OPTIONS = [
  { value: 'train', label: 'Train' },
  { value: 'test', label: 'Test' },
  { value: 'validation', label: 'Validation' },
  { value: 'mixed', label: 'Mixed' },
];

function arraysEqual(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function ImageSizeCell({ resolutions }: { resolutions: string[] }) {
  if (resolutions.length === 0) {
    return (
      <Badge size="xs" variant="outline" color="gray">
        n/a
      </Badge>
    );
  }
  return (
    <Group gap={4}>
      {resolutions.map((resolution) => (
        <Badge key={resolution} size="xs" variant="light" color="teal">
          {resolution}
        </Badge>
      ))}
    </Group>
  );
}

function usageLabel(value: string): string {
  return USAGE_OPTIONS.find((option) => option.value === value)?.label ?? value;
}

function usageColor(value: string): string {
  if (value === 'test') return 'orange';
  if (value === 'validation') return 'violet';
  if (value === 'mixed') return 'gray';
  return 'teal';
}

function strideSummary(dataset: TrainingDataset): string {
  const values = [...new Set(dataset.rules.map((rule) => rule.stride))].sort((a, b) => a - b);
  if (values.length === 0) return 'n/a';
  return values.length === 1 ? String(values[0]) : values.join(', ');
}

function countLabel(value: number | null): string {
  return value == null ? 'Needs refresh' : String(value);
}

function renderDatasetDetails(dataset: TrainingDataset) {
  return (
    <Stack gap="md">
      <div>
        <Text fw={700}>{dataset.name}</Text>
        <Text size="sm" c="dimmed">
          {dataset.counts_missing ? 'Counts need refresh' : `${dataset.total_selected_images} selected images`} · {usageLabel(dataset.usage_label ?? 'train')} · sizes{' '}
          {datasetResolutions(dataset).join(', ') || 'n/a'}
        </Text>
      </div>
      <Table striped verticalSpacing="xs">
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Dataset</Table.Th>
            <Table.Th>Folder</Table.Th>
            <Table.Th>Start</Table.Th>
            <Table.Th>End</Table.Th>
            <Table.Th>Stride</Table.Th>
            <Table.Th>Matching</Table.Th>
            <Table.Th>Selected</Table.Th>
            <Table.Th>Image data</Table.Th>
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
              <Table.Td>{countLabel(rule.matching_images)}</Table.Td>
              <Table.Td>{countLabel(rule.selected_images)}</Table.Td>
              <Table.Td>
                <Text size="xs" c="dimmed">
                  {rule.folder_image_signature ?? datasetResolutions(dataset).join(', ') ?? 'n/a'}
                </Text>
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Stack>
  );
}

export function TrainingDatasetPicker({
  trainingDatasets,
  selectedIds,
  onChange,
  disabled = false,
  embedded = false,
}: {
  trainingDatasets: TrainingDataset[];
  selectedIds: number[];
  onChange: (ids: number[]) => void;
  disabled?: boolean;
  embedded?: boolean;
}) {
  const [search, setSearch] = useState('');
  const [sizeFilter, setSizeFilter] = useState<string | null>(null);
  const [usageFilter, setUsageFilter] = useState<string | null>(null);
  const [detailDataset, setDetailDataset] = useState<TrainingDataset | null>(null);

  const datasetById = useMemo(
    () => new Map(trainingDatasets.map((dataset) => [dataset.id, dataset])),
    [trainingDatasets],
  );

  const sizeOptions = useMemo(() => {
    const values = new Set<string>();
    trainingDatasets.forEach((dataset) => datasetResolutions(dataset).forEach((resolution) => values.add(resolution)));
    return [...values].sort();
  }, [trainingDatasets]);

  // Image-size signature of the already-selected datasets. Once set, only
  // datasets of the same size (or unknown size) stay addable.
  const selectedSignature = useMemo(
    () => datasetSizeSignature(selectedIds, datasetById),
    [selectedIds, datasetById],
  );

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return trainingDatasets.filter((dataset) => {
      const resolutions = datasetResolutions(dataset);
      if (query) {
        const matches =
          dataset.name.toLowerCase().includes(query) ||
          dataset.dataset_names.some((name) => name.toLowerCase().includes(query));
        if (!matches) return false;
      }
      if (sizeFilter && !resolutions.includes(sizeFilter)) return false;
      if (usageFilter && (dataset.usage_label ?? 'train') !== usageFilter) return false;
      // Selection-driven size constraint: known sizes must match the signature;
      // unknown-size datasets stay visible (compatibility can't be disproven).
      if (selectedSignature && resolutions.length > 0 && !arraysEqual(resolutions, selectedSignature)) {
        return false;
      }
      return true;
    });
  }, [trainingDatasets, search, sizeFilter, usageFilter, selectedSignature]);

  const selected = selectedIds
    .map((id) => datasetById.get(id))
    .filter((dataset): dataset is TrainingDataset => dataset !== undefined);

  function move(index: number, offset: number) {
    if (disabled) return;
    const target = index + offset;
    if (target < 0 || target >= selectedIds.length) return;
    const next = [...selectedIds];
    [next[index], next[target]] = [next[target], next[index]];
    onChange(next);
  }

  return (
    <Paper withBorder={!embedded} p={embedded ? 0 : 'md'} radius="sm">
      <Stack gap="md">
        {!embedded && (
          <Group justify="space-between" align="center">
            <Title order={3}>1. Trainsets</Title>
            <Group gap="xs">
              <Badge variant="light" color={selectedIds.length > 0 ? 'green' : 'gray'}>
                {selectedIds.length} selected
              </Badge>
            </Group>
          </Group>
        )}
        <Group grow>
          <TextInput
            placeholder="Search by name or source dataset"
            leftSection={<Search size={16} />}
            value={search}
            onChange={(event) => setSearch(event.currentTarget.value)}
          />
          <Select placeholder="Image size" data={sizeOptions} value={sizeFilter} onChange={setSizeFilter} clearable />
          <Select placeholder="Label" data={USAGE_OPTIONS} value={usageFilter} onChange={setUsageFilter} clearable />
        </Group>
        {selectedSignature && (
          <Text size="xs" c="dimmed">
            Locked to image size {selectedSignature.join(', ')} by the current selection. Unknown-size sets stay
            listed.
          </Text>
        )}
        <ScrollArea h={selectedIds.length > 0 ? 180 : 240}>
          <Table striped highlightOnHover verticalSpacing="xs">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Label</Table.Th>
                <Table.Th>Datasets</Table.Th>
                <Table.Th>Image size</Table.Th>
                <Table.Th>Stride</Table.Th>
                <Table.Th>Images</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filtered.map((dataset) => {
                const alreadySelected = selectedIds.includes(dataset.id);
                return (
                  <Table.Tr key={dataset.id} className={alreadySelected ? 'pipeline-step selected' : 'pipeline-step'}>
                    <Table.Td>{dataset.name}</Table.Td>
                    <Table.Td>
                      <Badge size="xs" variant="light" color={usageColor(dataset.usage_label ?? 'train')}>
                        {usageLabel(dataset.usage_label ?? 'train')}
                      </Badge>
                    </Table.Td>
                    <Table.Td>
                      <Group gap={4}>
                        {dataset.dataset_names.map((name) => (
                          <Badge key={name} size="xs" variant="light" color="gray">
                            {name}
                          </Badge>
                        ))}
                      </Group>
                    </Table.Td>
                    <Table.Td>
                      <ImageSizeCell resolutions={datasetResolutions(dataset)} />
                    </Table.Td>
                    <Table.Td>{strideSummary(dataset)}</Table.Td>
                    <Table.Td>{dataset.counts_missing ? 'Needs refresh' : dataset.total_selected_images}</Table.Td>
                    <Table.Td>
                      <Group gap={4} justify="flex-end" wrap="nowrap">
                        <Tooltip label="Inspect trainset rules">
                          <ActionIcon variant="subtle" onClick={() => setDetailDataset(dataset)}>
                            <Info size={16} />
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label={alreadySelected ? 'Already added' : 'Add to pipeline'}>
                          <ActionIcon
                            variant="subtle"
                            disabled={disabled || alreadySelected}
                            onClick={() => onChange([...selectedIds, dataset.id])}
                          >
                            <Plus size={18} />
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
        {trainingDatasets.length === 0 && (
          <Alert color="blue">No train/test datasets available yet. Create one on the Train/Test Datasets page.</Alert>
        )}

        {selected.length > 0 && (
          <Stack gap="xs">
            <Text fw={700} size="sm">
              Selected (combined in this order)
            </Text>
            {selected.map((dataset, index) => (
              <Paper key={dataset.id} withBorder p="xs" radius="sm" className="pipeline-step selected">
                <Group justify="space-between" wrap="nowrap">
                  <Group gap="xs" wrap="nowrap">
                    <Badge variant="filled" color="teal" size="sm">
                      {index + 1}
                    </Badge>
                    <div>
                      <Text size="sm" fw={600}>
                        {dataset.name}
                      </Text>
                      <Text size="xs" c="dimmed">
                        {dataset.counts_missing ? 'Counts need refresh' : `${dataset.total_selected_images} images`} ·{' '}
                        {usageLabel(dataset.usage_label ?? 'train')} · stride {strideSummary(dataset)} ·{' '}
                        {datasetResolutions(dataset).join(', ') || 'size n/a'} · {dataset.dataset_names.join(', ')}
                      </Text>
                    </div>
                  </Group>
                  <Group gap={4} wrap="nowrap">
                    <ActionIcon variant="subtle" onClick={() => setDetailDataset(dataset)}>
                      <Info size={16} />
                    </ActionIcon>
                    <ActionIcon variant="subtle" disabled={disabled || index === 0} onClick={() => move(index, -1)}>
                      <ArrowUp size={16} />
                    </ActionIcon>
                    <ActionIcon
                      variant="subtle"
                      disabled={disabled || index === selected.length - 1}
                      onClick={() => move(index, 1)}
                    >
                      <ArrowDown size={16} />
                    </ActionIcon>
                    <ActionIcon
                      color="red"
                      variant="subtle"
                      disabled={disabled}
                      onClick={() => onChange(selectedIds.filter((id) => id !== dataset.id))}
                    >
                      <Trash2 size={16} />
                    </ActionIcon>
                  </Group>
                </Group>
              </Paper>
            ))}
          </Stack>
        )}
        <Modal
          opened={detailDataset !== null}
          onClose={() => setDetailDataset(null)}
          title={detailDataset ? `Trainset: ${detailDataset.name}` : 'Trainset'}
          size="xl"
          scrollAreaComponent={ScrollArea.Autosize}
        >
          {detailDataset ? renderDatasetDetails(detailDataset) : null}
        </Modal>
      </Stack>
    </Paper>
  );
}
