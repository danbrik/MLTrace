import { Alert, Badge, Button, Group, Paper, Progress, Stack, Text } from '@mantine/core';
import { Play } from 'lucide-react';

import type { MethodTorchCheckResponse, MethodValidationResponse } from '../../types';

export function TorchCheckPanel({
  validation,
  torchCheck,
  loading,
  logs,
  onRun,
}: {
  validation: MethodValidationResponse | null;
  torchCheck: MethodTorchCheckResponse | null;
  loading: boolean;
  logs: string[];
  onRun: () => void;
}) {
  const torchStatus = loading ? 'running' : (torchCheck?.status ?? 'not run');
  const torchColor =
    torchStatus === 'available'
      ? 'green'
      : torchStatus === 'failed'
        ? 'red'
        : torchStatus === 'running'
          ? 'blue'
          : 'gray';
  const visibleLogs = loading || logs.length > 0 ? logs : (torchCheck?.logs ?? []);

  return (
    <Paper withBorder p="sm" radius="sm">
      <Stack gap="sm">
        <Group justify="space-between" align="center">
          <div>
            <Text fw={700} size="sm">
              Torch Dummy Forward
            </Text>
            <Text size="xs" c="dimmed">
              Manual runtime check. Save and automatic validation use static shapes only.
            </Text>
          </div>
          <Group gap="xs">
            <Badge color={torchColor} variant="light">
              {torchStatus === 'available' ? 'Passed' : torchStatus}
            </Badge>
            <Button
              size="xs"
              variant="light"
              leftSection={<Play size={14} />}
              loading={loading}
              disabled={validation?.valid !== true}
              onClick={onRun}
            >
              Run Torch Check
            </Button>
          </Group>
        </Group>
        {(loading || torchCheck) && <Progress value={loading ? 65 : 100} color={torchColor} size="sm" radius="sm" />}
        {visibleLogs.length > 0 && (
          <Paper withBorder p="xs" radius="sm" className="torch-check-log">
            <Stack gap={3}>
              {visibleLogs.map((log, index) => (
                <Text key={`${log}-${index}`} size="xs" ff="monospace">
                  {log}
                </Text>
              ))}
            </Stack>
          </Paper>
        )}
        {torchCheck?.errors.length ? (
          <Alert color="red" title="Torch check errors">
            <Stack gap={4}>
              {torchCheck.errors.map((error) => (
                <Text key={error} size="sm">
                  {error}
                </Text>
              ))}
            </Stack>
          </Alert>
        ) : null}
        {torchCheck?.warnings.length ? (
          <Alert color="yellow" title="Torch check warnings">
            <Stack gap={4}>
              {torchCheck.warnings.map((warning) => (
                <Text key={warning} size="sm">
                  {warning}
                </Text>
              ))}
            </Stack>
          </Alert>
        ) : null}
      </Stack>
    </Paper>
  );
}
