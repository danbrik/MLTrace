import { ActionIcon, Alert, Group, Paper, ScrollArea, Stack, Table, Title, Tooltip } from '@mantine/core';
import { Trash2, Upload } from 'lucide-react';

import type { MethodConfiguration, MethodDefinition } from '../../types';
import { keyParameters, methodLabel } from '../utils';

export function SavedMethodsTable({
  methods,
  methodByType,
  onLoad,
  onDelete,
  isLoading,
  isDeleting,
}: {
  methods: MethodConfiguration[];
  methodByType: Map<string, MethodDefinition>;
  onLoad: (methodId: number) => void;
  onDelete: (method: MethodConfiguration) => void;
  isLoading?: (methodId: number) => boolean;
  isDeleting?: (methodId: number) => boolean;
}) {
  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Title order={3}>Saved Methods</Title>
        <ScrollArea>
          <Table striped verticalSpacing="sm" miw={980}>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Method</Table.Th>
                <Table.Th>Key parameters</Table.Th>
                <Table.Th>Description</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {methods.map((method) => {
                const methodDefinition = methodByType.get(method.method_type);
                const loading = isLoading?.(method.id) ?? false;
                const deleting = isDeleting?.(method.id) ?? false;
                return (
                  <Table.Tr key={method.id}>
                    <Table.Td>{method.name}</Table.Td>
                    <Table.Td>{methodLabel(methodDefinition, method.method_type)}</Table.Td>
                    <Table.Td>{keyParameters(method)}</Table.Td>
                    <Table.Td>{method.description ?? ''}</Table.Td>
                    <Table.Td>
                      <Group gap="xs" justify="flex-end" wrap="nowrap">
                        <Tooltip label="Load">
                          <ActionIcon variant="subtle" loading={loading} disabled={deleting} onClick={() => onLoad(method.id)}>
                            <Upload size={18} />
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Delete">
                          <ActionIcon color="red" variant="subtle" loading={deleting} disabled={loading} onClick={() => onDelete(method)}>
                            <Trash2 size={18} />
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
        {methods.length === 0 && <Alert color="blue">Saved method configurations will appear here.</Alert>}
      </Stack>
    </Paper>
  );
}
