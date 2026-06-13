import { Alert, Badge, Group, Paper, ScrollArea, Stack, Table, TextInput, Title } from '@mantine/core';
import { Search } from 'lucide-react';
import { useMemo, useState } from 'react';

import type { PreprocessingPipeline } from '../types';

function resolutionLabel(width: number | null, height: number | null): string {
  return width && height ? `${width}x${height}` : 'n/a';
}

export function PreprocessingPipelinePicker({
  pipelines,
  selectedId,
  onChange,
}: {
  pipelines: PreprocessingPipeline[];
  selectedId: number | null;
  onChange: (id: number | null) => void;
}) {
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return pipelines;
    return pipelines.filter(
      (pipeline) =>
        pipeline.name.toLowerCase().includes(query) ||
        (pipeline.description ?? '').toLowerCase().includes(query),
    );
  }, [pipelines, search]);

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Group justify="space-between" align="center">
          <Title order={3}>2. Preprocessing Pipeline</Title>
          {selectedId != null && <Badge variant="light">selected</Badge>}
        </Group>
        <TextInput
          placeholder="Search by name or description"
          leftSection={<Search size={16} />}
          value={search}
          onChange={(event) => setSearch(event.currentTarget.value)}
        />
        <ScrollArea h={240}>
          <Table striped highlightOnHover verticalSpacing="xs">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Steps</Table.Th>
                <Table.Th>Output</Table.Th>
                <Table.Th>Description</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filtered.map((pipeline) => (
                <Table.Tr
                  key={pipeline.id}
                  className={pipeline.id === selectedId ? 'pipeline-step selected' : 'pipeline-step'}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onChange(pipeline.id === selectedId ? null : pipeline.id)}
                >
                  <Table.Td>{pipeline.name}</Table.Td>
                  <Table.Td>{pipeline.graph.nodes.length}</Table.Td>
                  <Table.Td>
                    <Badge size="xs" variant="light" color="yellow">
                      {resolutionLabel(pipeline.output_width, pipeline.output_height)}
                    </Badge>
                  </Table.Td>
                  <Table.Td>{pipeline.description ?? ''}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea>
        {pipelines.length === 0 && (
          <Alert color="blue">No preprocessing pipelines available yet. Create one on the Preprocessing page.</Alert>
        )}
      </Stack>
    </Paper>
  );
}
