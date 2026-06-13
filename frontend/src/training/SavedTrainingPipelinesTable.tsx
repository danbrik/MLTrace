import { ActionIcon, Alert, Badge, Group, Paper, ScrollArea, Stack, Table, Text, Title, Tooltip } from '@mantine/core';
import { Trash2, Upload } from 'lucide-react';

import { formatValue, trainingModeColor, trainingModeLabel } from '../methods/utils';
import type { TrainingPipeline } from '../types';

function keyTrainingParameters(pipeline: TrainingPipeline): string {
  const parameters = pipeline.training_parameters ?? {};
  const interesting = ['epochs', 'batch_size', 'learning_rate', 'loss', 'reconstruction_loss'];
  const parts = interesting
    .filter((key) => parameters[key] !== undefined)
    .map((key) => `${key} ${formatValue(parameters[key])}`);
  return parts.length > 0 ? parts.join(', ') : 'no training parameters';
}

export function SavedTrainingPipelinesTable({
  pipelines,
  onLoad,
  onDelete,
}: {
  pipelines: TrainingPipeline[];
  onLoad: (pipelineId: number) => void;
  onDelete: (pipeline: TrainingPipeline) => void;
}) {
  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Title order={3}>Saved Training Pipelines</Title>
        <ScrollArea>
          <Table striped verticalSpacing="sm" miw={1080}>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Training sets</Table.Th>
                <Table.Th>Preprocessing</Table.Th>
                <Table.Th>Method</Table.Th>
                <Table.Th>Shuffle</Table.Th>
                <Table.Th>Key parameters</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {pipelines.map((pipeline) => (
                <Table.Tr key={pipeline.id}>
                  <Table.Td>{pipeline.name}</Table.Td>
                  <Table.Td>
                    <Stack gap={2}>
                      <Text size="sm">
                        {pipeline.training_datasets.map((entry) => entry.name).join(', ')}
                      </Text>
                      <Text size="xs" c="dimmed">
                        {pipeline.total_selected_images} images total
                      </Text>
                    </Stack>
                  </Table.Td>
                  <Table.Td>{pipeline.preprocessing_pipeline_name}</Table.Td>
                  <Table.Td>
                    <Group gap="xs" wrap="nowrap">
                      <Text size="sm">{pipeline.method_configuration_name}</Text>
                      <Badge color={trainingModeColor(pipeline.training_mode)} variant="light">
                        {trainingModeLabel(pipeline.training_mode)}
                      </Badge>
                    </Group>
                  </Table.Td>
                  <Table.Td>
                    <Badge variant={pipeline.shuffle ? 'filled' : 'outline'} color="teal" size="sm">
                      {pipeline.shuffle ? 'shuffled' : 'in order'}
                    </Badge>
                  </Table.Td>
                  <Table.Td>{keyTrainingParameters(pipeline)}</Table.Td>
                  <Table.Td>
                    <Group gap="xs" justify="flex-end" wrap="nowrap">
                      <Tooltip label="Load">
                        <ActionIcon variant="subtle" onClick={() => onLoad(pipeline.id)}>
                          <Upload size={18} />
                        </ActionIcon>
                      </Tooltip>
                      <Tooltip label="Delete">
                        <ActionIcon color="red" variant="subtle" onClick={() => onDelete(pipeline)}>
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
        {pipelines.length === 0 && <Alert color="blue">Saved training pipelines will appear here.</Alert>}
      </Stack>
    </Paper>
  );
}
