import { ActionIcon, Alert, Badge, Group, Paper, Progress, ScrollArea, Stack, Table, Text, Title, Tooltip } from '@mantine/core';
import { FileText, RotateCcw, StopCircle, Trash2 } from 'lucide-react';

import { trainingModeColor, trainingModeLabel } from '../methods/utils';
import { formatDuration, runStatusColor } from './runStatus';
import type { TrainingRun } from '../types';

const TERMINAL = new Set(['finished', 'failed', 'aborted']);

function ProgressCell({ run }: { run: TrainingRun }) {
  if (run.builder_kind === 'form') {
    return <Text size="sm">{run.image_count != null ? `${run.image_count} imgs` : '—'}</Text>;
  }
  const total = run.epochs_total ?? run.epochs ?? 0;
  const value = total > 0 ? (run.epochs_completed / total) * 100 : 0;
  return (
    <Stack gap={2}>
      <Text size="xs">
        {run.epochs_completed}/{total || '?'} epochs
      </Text>
      {total > 0 && <Progress value={value} size="sm" radius="sm" color={runStatusColor(run.status)} />}
    </Stack>
  );
}

export function TrainingRunsTable({
  runs,
  onAbort,
  onRestart,
  onDelete,
  onShowLog,
}: {
  runs: TrainingRun[];
  onAbort: (run: TrainingRun) => void;
  onRestart: (run: TrainingRun) => void;
  onDelete: (run: TrainingRun) => void;
  onShowLog: (run: TrainingRun) => void;
}) {
  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Title order={3}>Runs</Title>
        <ScrollArea>
          <Table striped highlightOnHover verticalSpacing="sm" miw={1100}>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Pipeline</Table.Th>
                <Table.Th>Method</Table.Th>
                <Table.Th>Datasets</Table.Th>
                <Table.Th>Status</Table.Th>
                <Table.Th>Device</Table.Th>
                <Table.Th>Progress</Table.Th>
                <Table.Th>Duration</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {runs.map((run) => {
                const terminal = TERMINAL.has(run.status);
                const abortable = run.status === 'running' || run.status === 'queued';
                return (
                  <Table.Tr key={run.id}>
                    <Table.Td>{run.training_pipeline_name}</Table.Td>
                    <Table.Td>
                      <Group gap="xs" wrap="nowrap">
                        <Text size="sm">{run.method_type}</Text>
                        <Badge color={trainingModeColor(run.training_mode)} variant="light" size="sm">
                          {trainingModeLabel(run.training_mode)}
                        </Badge>
                      </Group>
                    </Table.Td>
                    <Table.Td>
                      <Text size="xs" c="dimmed">
                        {run.dataset_names.join(', ')}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Badge color={runStatusColor(run.status)}>{run.status}</Badge>
                    </Table.Td>
                    <Table.Td>
                      {run.device ? (
                        <Badge variant="light" color={run.device === 'CPU' ? 'gray' : 'grape'}>
                          {run.device}
                        </Badge>
                      ) : (
                        '—'
                      )}
                    </Table.Td>
                    <Table.Td>
                      <ProgressCell run={run} />
                    </Table.Td>
                    <Table.Td>{formatDuration(run.duration_seconds)}</Table.Td>
                    <Table.Td>
                      <Group gap="xs" justify="flex-end" wrap="nowrap">
                        <Tooltip label="Show log & metrics">
                          <ActionIcon variant="subtle" onClick={() => onShowLog(run)}>
                            <FileText size={18} />
                          </ActionIcon>
                        </Tooltip>
                        {abortable && (
                          <Tooltip label="Abort">
                            <ActionIcon color="orange" variant="subtle" onClick={() => onAbort(run)}>
                              <StopCircle size={18} />
                            </ActionIcon>
                          </Tooltip>
                        )}
                        {terminal && (
                          <Tooltip label="Restart">
                            <ActionIcon variant="subtle" onClick={() => onRestart(run)}>
                              <RotateCcw size={18} />
                            </ActionIcon>
                          </Tooltip>
                        )}
                        {run.status !== 'running' && (
                          <Tooltip label="Remove">
                            <ActionIcon color="red" variant="subtle" onClick={() => onDelete(run)}>
                              <Trash2 size={18} />
                            </ActionIcon>
                          </Tooltip>
                        )}
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                );
              })}
            </Table.Tbody>
          </Table>
        </ScrollArea>
        {runs.length === 0 && <Alert color="blue">No training runs match the current filters.</Alert>}
      </Stack>
    </Paper>
  );
}
