import {
  Alert,
  Badge,
  Group,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Table,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { Search } from 'lucide-react';
import { useMemo, useState } from 'react';

import { keyParameters, methodLabel, trainingModeColor, trainingModeLabel } from '../methods/utils';
import type { MethodConfiguration, MethodDefinition } from '../types';

export function MethodConfigurationPicker({
  configurations,
  methodByType,
  selectedId,
  onChange,
}: {
  configurations: MethodConfiguration[];
  methodByType: Map<string, MethodDefinition>;
  selectedId: number | null;
  onChange: (id: number | null) => void;
}) {
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string | null>(null);

  const typeOptions = useMemo(() => {
    const types = [...new Set(configurations.map((configuration) => configuration.method_type))];
    return types.map((type) => ({ value: type, label: methodLabel(methodByType.get(type), type) }));
  }, [configurations, methodByType]);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return configurations.filter((configuration) => {
      if (typeFilter && configuration.method_type !== typeFilter) return false;
      if (!query) return true;
      return (
        configuration.name.toLowerCase().includes(query) ||
        (configuration.description ?? '').toLowerCase().includes(query)
      );
    });
  }, [configurations, search, typeFilter]);

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Group justify="space-between" align="center">
          <Title order={3}>3. Method / Architecture</Title>
          {selectedId != null && <Badge variant="light">selected</Badge>}
        </Group>
        <Group grow>
          <TextInput
            placeholder="Search by name or description"
            leftSection={<Search size={16} />}
            value={search}
            onChange={(event) => setSearch(event.currentTarget.value)}
          />
          <Select
            placeholder="All method types"
            data={typeOptions}
            value={typeFilter}
            onChange={setTypeFilter}
            clearable
          />
        </Group>
        <ScrollArea h={240}>
          <Table striped highlightOnHover verticalSpacing="xs">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Method</Table.Th>
                <Table.Th>Training</Table.Th>
                <Table.Th>Key parameters</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filtered.map((configuration) => {
                const supported = configuration.supports_training_pipeline;
                const row = (
                  <Table.Tr
                    key={configuration.id}
                    className={configuration.id === selectedId ? 'pipeline-step selected' : 'pipeline-step'}
                    style={{ cursor: supported ? 'pointer' : 'not-allowed', opacity: supported ? 1 : 0.45 }}
                    onClick={() => {
                      if (!supported) return;
                      onChange(configuration.id === selectedId ? null : configuration.id);
                    }}
                  >
                    <Table.Td>{configuration.name}</Table.Td>
                    <Table.Td>{methodLabel(methodByType.get(configuration.method_type), configuration.method_type)}</Table.Td>
                    <Table.Td>
                      <Badge color={trainingModeColor(configuration.training_mode)} variant="light">
                        {trainingModeLabel(configuration.training_mode)}
                      </Badge>
                    </Table.Td>
                    <Table.Td>{keyParameters(configuration)}</Table.Td>
                  </Table.Tr>
                );
                if (supported) return row;
                return (
                  <Tooltip key={configuration.id} label="This method cannot be used in training pipelines">
                    {row}
                  </Tooltip>
                );
              })}
            </Table.Tbody>
          </Table>
        </ScrollArea>
        {configurations.length === 0 && (
          <Alert color="blue">No saved methods available yet. Create one on the Methods page.</Alert>
        )}
      </Stack>
    </Paper>
  );
}
