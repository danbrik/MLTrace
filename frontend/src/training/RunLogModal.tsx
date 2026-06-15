import { Badge, Group, Modal, Paper, ScrollArea, Stack, Table, Text } from '@mantine/core';
import { useEffect, useState } from 'react';

import { getTrainingRunLog } from '../api';
import { formatLoss, runStatusColor } from './runStatus';
import type { TrainingRun } from '../types';

/** Detail view for a run: live worker log + per-epoch loss table. */
export function RunLogModal({ run, onClose }: { run: TrainingRun | null; onClose: () => void }) {
  const [log, setLog] = useState('');

  useEffect(() => {
    if (!run) return undefined;
    let cancelled = false;
    const load = () =>
      getTrainingRunLog(run.id)
        .then((result) => {
          if (!cancelled) setLog(result.log);
        })
        .catch(() => undefined);
    load();
    // Keep the log fresh while the run is active.
    const interval = run.status === 'running' || run.status === 'queued' ? window.setInterval(load, 2000) : null;
    return () => {
      cancelled = true;
      if (interval) window.clearInterval(interval);
    };
  }, [run]);

  return (
    <Modal opened={run !== null} onClose={onClose} title={run?.training_pipeline_name ?? ''} size="xl">
      {run && (
        <Stack gap="md">
          <Group gap="xs">
            <Badge color={runStatusColor(run.status)}>{run.status}</Badge>
            <Text size="sm" c="dimmed">
              {run.method_type} · {run.builder_kind}
              {run.gpu_index !== null ? ` · GPU ${run.gpu_index}` : ''}
            </Text>
          </Group>

          {run.error_message && (
            <Paper withBorder p="xs" radius="sm" bg="var(--mantine-color-red-0)">
              <Text size="sm" c="red">
                {run.error_message}
              </Text>
            </Paper>
          )}

          {run.metrics.length > 0 && (
            <Stack gap={4}>
              <Text fw={700} size="sm">
                Loss per epoch
              </Text>
              <ScrollArea h={180}>
                <Table striped stickyHeader>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Epoch</Table.Th>
                      <Table.Th>Train loss</Table.Th>
                      <Table.Th>Val loss</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {run.metrics.map((metric) => (
                      <Table.Tr key={metric.epoch}>
                        <Table.Td>{metric.epoch}</Table.Td>
                        <Table.Td>{formatLoss(metric.train_loss)}</Table.Td>
                        <Table.Td>{formatLoss(metric.val_loss)}</Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              </ScrollArea>
            </Stack>
          )}

          <Stack gap={4}>
            <Text fw={700} size="sm">
              Worker log
            </Text>
            <Paper withBorder p="xs" radius="sm">
              <ScrollArea h={220}>
                <Text size="xs" ff="monospace" style={{ whiteSpace: 'pre-wrap' }}>
                  {log || 'No log output yet.'}
                </Text>
              </ScrollArea>
            </Paper>
          </Stack>
        </Stack>
      )}
    </Modal>
  );
}
