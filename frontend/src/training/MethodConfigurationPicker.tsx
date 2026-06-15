import {
  ActionIcon,
  Alert,
  Badge,
  Collapse,
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
import { ChevronDown, ChevronRight, Search } from 'lucide-react';
import { Fragment, useMemo, useState } from 'react';

import { methodInputResolution } from './graph';
import { formatValue, keyParameters, methodLabel, trainingModeColor, trainingModeLabel } from '../methods/utils';
import type { MethodConfiguration, MethodDefinition } from '../types';

const DETAIL_CONFIG_KEYS = [
  'input_channels',
  'input_width',
  'input_height',
  'latent_dim',
  'output_activation',
  'kl_weight',
  'aggregation',
  'accumulator_dtype',
  'output_dtype_policy',
];

export function MethodConfigurationPicker({
  configurations,
  methodByType,
  selectedId,
  onChange,
  requiredInputResolution = null,
}: {
  configurations: MethodConfiguration[];
  methodByType: Map<string, MethodDefinition>;
  selectedId: number | null;
  onChange: (id: number | null) => void;
  // Input size the method must accept (= selected pipeline output). When set,
  // only methods of that input size (plus unknown-size ones) remain visible.
  requiredInputResolution?: string | null;
}) {
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [trainingModeFilter, setTrainingModeFilter] = useState<string | null>(null);
  const [resolutionFilter, setResolutionFilter] = useState<string | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  const typeOptions = useMemo(() => {
    const types = [...new Set(configurations.map((configuration) => configuration.method_type))];
    return types.map((type) => ({ value: type, label: methodLabel(methodByType.get(type), type) }));
  }, [configurations, methodByType]);

  const trainingModeOptions = useMemo(() => {
    const modes = [...new Set(configurations.map((configuration) => configuration.training_mode))];
    return modes.map((mode) => ({ value: mode, label: trainingModeLabel(mode) }));
  }, [configurations]);

  const resolutionOptions = useMemo(() => {
    const values = new Set<string>();
    configurations.forEach((configuration) => {
      const resolution = methodInputResolution(configuration);
      if (resolution) values.add(resolution);
    });
    return [...values].sort();
  }, [configurations]);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return configurations.filter((configuration) => {
      // The currently selected method is always shown so the selection is never lost.
      if (configuration.id === selectedId) return true;
      // Cross-filter to the required input size (= selected pipeline output).
      // Unknown size stays visible since compatibility cannot be disproven.
      if (requiredInputResolution) {
        const resolution = methodInputResolution(configuration);
        if (resolution !== null && resolution !== requiredInputResolution) return false;
      }
      if (typeFilter && configuration.method_type !== typeFilter) return false;
      if (trainingModeFilter && configuration.training_mode !== trainingModeFilter) return false;
      if (resolutionFilter && methodInputResolution(configuration) !== resolutionFilter) return false;
      if (!query) return true;
      return (
        configuration.name.toLowerCase().includes(query) ||
        (configuration.description ?? '').toLowerCase().includes(query)
      );
    });
  }, [configurations, search, typeFilter, trainingModeFilter, resolutionFilter, requiredInputResolution, selectedId]);

  function toggleExpanded(id: number) {
    setExpandedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Group justify="space-between" align="center">
          <Title order={3}>3. Method / Architecture</Title>
          {selectedId != null && <Badge variant="light">selected</Badge>}
        </Group>
        <TextInput
          placeholder="Search by name or description"
          leftSection={<Search size={16} />}
          value={search}
          onChange={(event) => setSearch(event.currentTarget.value)}
        />
        <Group grow>
          <Select placeholder="Method type" data={typeOptions} value={typeFilter} onChange={setTypeFilter} clearable />
          <Select
            placeholder="Training mode"
            data={trainingModeOptions}
            value={trainingModeFilter}
            onChange={setTrainingModeFilter}
            clearable
          />
          <Select
            placeholder="Input resolution"
            data={resolutionOptions}
            value={resolutionFilter}
            onChange={setResolutionFilter}
            clearable
          />
        </Group>
        {requiredInputResolution && (
          <Text size="xs" c="dimmed">
            Filtered to methods accepting {requiredInputResolution} (selected pipeline output). Unknown-size methods stay listed.
          </Text>
        )}
        <ScrollArea h={280}>
          <Table striped highlightOnHover verticalSpacing="xs">
            <Table.Thead>
              <Table.Tr>
                <Table.Th w={32} />
                <Table.Th>Name</Table.Th>
                <Table.Th>Method</Table.Th>
                <Table.Th>Input size</Table.Th>
                <Table.Th>Training</Table.Th>
                <Table.Th>Key parameters</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filtered.map((configuration) => {
                const supported = configuration.supports_training_pipeline;
                const expanded = expandedIds.has(configuration.id);
                const mainRow = (
                  <Table.Tr
                    className={configuration.id === selectedId ? 'pipeline-step selected' : 'pipeline-step'}
                    style={{ cursor: supported ? 'pointer' : 'not-allowed', opacity: supported ? 1 : 0.45 }}
                    onClick={() => {
                      if (!supported) return;
                      onChange(configuration.id === selectedId ? null : configuration.id);
                    }}
                  >
                    <Table.Td>
                      <ActionIcon
                        variant="subtle"
                        size="sm"
                        onClick={(event) => {
                          event.stopPropagation();
                          toggleExpanded(configuration.id);
                        }}
                      >
                        {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                      </ActionIcon>
                    </Table.Td>
                    <Table.Td>{configuration.name}</Table.Td>
                    <Table.Td>{methodLabel(methodByType.get(configuration.method_type), configuration.method_type)}</Table.Td>
                    <Table.Td>
                      {methodInputResolution(configuration) ? (
                        <Badge size="xs" variant="light" color="grape">
                          {methodInputResolution(configuration)}
                        </Badge>
                      ) : (
                        <Badge size="xs" variant="outline" color="gray">
                          n/a
                        </Badge>
                      )}
                    </Table.Td>
                    <Table.Td>
                      <Badge color={trainingModeColor(configuration.training_mode)} variant="light">
                        {trainingModeLabel(configuration.training_mode)}
                      </Badge>
                    </Table.Td>
                    <Table.Td>{keyParameters(configuration)}</Table.Td>
                  </Table.Tr>
                );
                return (
                  <Fragment key={configuration.id}>
                    {supported ? (
                      mainRow
                    ) : (
                      <Tooltip label="This method cannot be used in training pipelines">{mainRow}</Tooltip>
                    )}
                    <Table.Tr>
                      <Table.Td colSpan={6} p={0} style={{ borderBottom: expanded ? undefined : 'none' }}>
                        <Collapse in={expanded}>
                          <Stack gap={6} p="sm">
                            {configuration.description && (
                              <Text size="xs" c="dimmed">
                                {configuration.description}
                              </Text>
                            )}
                            <Group gap={6}>
                              {DETAIL_CONFIG_KEYS.filter((key) => configuration.method_config?.[key] !== undefined).map(
                                (key) => (
                                  <Badge key={key} size="sm" variant="light" color="gray">
                                    {key}={formatValue(configuration.method_config[key])}
                                  </Badge>
                                ),
                              )}
                            </Group>
                            {(configuration.diagram?.nodes?.length ?? 0) > 0 && (
                              <Group gap={6}>
                                {configuration.diagram.nodes.map((node, index) => (
                                  <Badge
                                    key={node.id}
                                    size="sm"
                                    variant="light"
                                    className={`model-section-${node.section}`}
                                  >
                                    {index + 1}. {node.label}
                                  </Badge>
                                ))}
                              </Group>
                            )}
                          </Stack>
                        </Collapse>
                      </Table.Td>
                    </Table.Tr>
                  </Fragment>
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
