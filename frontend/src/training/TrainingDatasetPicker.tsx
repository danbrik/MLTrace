import {
  ActionIcon,
  Alert,
  Badge,
  Group,
  Paper,
  ScrollArea,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { ArrowDown, ArrowUp, Plus, Search, Trash2 } from 'lucide-react';
import { useMemo, useState } from 'react';

import type { TrainingDataset } from '../types';

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

  const datasetById = useMemo(
    () => new Map(trainingDatasets.map((dataset) => [dataset.id, dataset])),
    [trainingDatasets],
  );

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return trainingDatasets;
    return trainingDatasets.filter(
      (dataset) =>
        dataset.name.toLowerCase().includes(query) ||
        dataset.dataset_names.some((name) => name.toLowerCase().includes(query)),
    );
  }, [trainingDatasets, search]);

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
        <TextInput
          placeholder="Search by name or source dataset"
          leftSection={<Search size={16} />}
          value={search}
          onChange={(event) => setSearch(event.currentTarget.value)}
        />
        <ScrollArea h={selectedIds.length > 0 ? 180 : 240}>
          <Table striped highlightOnHover verticalSpacing="xs">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Datasets</Table.Th>
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
                          <Badge key={name} size="xs" variant="light" color="teal">
                            {name}
                          </Badge>
                        ))}
                      </Group>
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
                        {dataset.total_selected_images} images · {dataset.dataset_names.join(', ')}
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
