import {
  ActionIcon,
  Alert,
  Badge,
  Group,
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
import { ArrowDown, ArrowUp, Plus, Search, Trash2 } from 'lucide-react';
import { useMemo, useState } from 'react';

import { datasetResolutions, datasetSizeSignature } from './graph';
import type { TrainingDataset } from '../types';

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

export function TrainingDatasetPicker({
  trainingDatasets,
  selectedIds,
  onChange,
}: {
  trainingDatasets: TrainingDataset[];
  selectedIds: number[];
  onChange: (ids: number[]) => void;
}) {
  const [search, setSearch] = useState('');
  const [sizeFilter, setSizeFilter] = useState<string | null>(null);

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
      // Selection-driven size constraint: known sizes must match the signature;
      // unknown-size datasets stay visible (compatibility can't be disproven).
      if (selectedSignature && resolutions.length > 0 && !arraysEqual(resolutions, selectedSignature)) {
        return false;
      }
      return true;
    });
  }, [trainingDatasets, search, sizeFilter, selectedSignature]);

  const selected = selectedIds
    .map((id) => datasetById.get(id))
    .filter((dataset): dataset is TrainingDataset => dataset !== undefined);

  function move(index: number, offset: number) {
    const target = index + offset;
    if (target < 0 || target >= selectedIds.length) return;
    const next = [...selectedIds];
    [next[index], next[target]] = [next[target], next[index]];
    onChange(next);
  }

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Group justify="space-between" align="center">
          <Title order={3}>1. Training Sets</Title>
          <Badge variant="light">{selectedIds.length} selected</Badge>
        </Group>
        <Group grow>
          <TextInput
            placeholder="Search by name or source dataset"
            leftSection={<Search size={16} />}
            value={search}
            onChange={(event) => setSearch(event.currentTarget.value)}
          />
          <Select placeholder="Image size" data={sizeOptions} value={sizeFilter} onChange={setSizeFilter} clearable />
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
                <Table.Th>Datasets</Table.Th>
                <Table.Th>Image size</Table.Th>
                <Table.Th>Images</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filtered.map((dataset) => {
                const alreadySelected = selectedIds.includes(dataset.id);
                return (
                  <Table.Tr key={dataset.id}>
                    <Table.Td>{dataset.name}</Table.Td>
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
                    <Table.Td>{dataset.total_selected_images}</Table.Td>
                    <Table.Td>
                      <Tooltip label={alreadySelected ? 'Already added' : 'Add to pipeline'}>
                        <ActionIcon
                          variant="subtle"
                          disabled={alreadySelected}
                          onClick={() => onChange([...selectedIds, dataset.id])}
                        >
                          <Plus size={18} />
                        </ActionIcon>
                      </Tooltip>
                    </Table.Td>
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
        </ScrollArea>
        {trainingDatasets.length === 0 && (
          <Alert color="blue">No training datasets available yet. Create one on the Training Datasets page.</Alert>
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
                        {dataset.total_selected_images} images ·{' '}
                        {datasetResolutions(dataset).join(', ') || 'size n/a'} · {dataset.dataset_names.join(', ')}
                      </Text>
                    </div>
                  </Group>
                  <Group gap={4} wrap="nowrap">
                    <ActionIcon variant="subtle" disabled={index === 0} onClick={() => move(index, -1)}>
                      <ArrowUp size={16} />
                    </ActionIcon>
                    <ActionIcon
                      variant="subtle"
                      disabled={index === selected.length - 1}
                      onClick={() => move(index, 1)}
                    >
                      <ArrowDown size={16} />
                    </ActionIcon>
                    <ActionIcon
                      color="red"
                      variant="subtle"
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
      </Stack>
    </Paper>
  );
}
