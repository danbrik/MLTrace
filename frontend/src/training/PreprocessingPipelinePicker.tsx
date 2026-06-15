import {
  ActionIcon,
  Alert,
  Badge,
  Collapse,
  Group,
  MultiSelect,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { ChevronDown, ChevronRight, Search } from 'lucide-react';
import { Fragment, useMemo, useState } from 'react';

import { orderedGraphNodes, pipelineInputResolution, pipelineOutputResolution, stepDetail } from './graph';
import type { PreprocessingPipeline } from '../types';

function resolutionLabel(width: number | null, height: number | null): string {
  return width && height ? `${width}x${height}` : 'n/a';
}

export function PreprocessingPipelinePicker({
  pipelines,
  selectedId,
  onChange,
  requiredInputResolutions = null,
}: {
  pipelines: PreprocessingPipeline[];
  selectedId: number | null;
  onChange: (id: number | null) => void;
  // Input sizes accepted by the selected datasets (= dataset image sizes).
  // When set, only pipelines whose input is in this set (plus unknown-input
  // ones) remain visible.
  requiredInputResolutions?: string[] | null;
}) {
  const [search, setSearch] = useState('');
  const [inputFilter, setInputFilter] = useState<string | null>(null);
  const [outputFilter, setOutputFilter] = useState<string | null>(null);
  const [stepFilter, setStepFilter] = useState<string[]>([]);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  const inputResolutionOptions = useMemo(() => {
    const values = new Set<string>();
    pipelines.forEach((pipeline) => {
      const resolution = pipelineInputResolution(pipeline);
      if (resolution) values.add(resolution);
    });
    return [...values].sort();
  }, [pipelines]);

  const outputResolutionOptions = useMemo(() => {
    const values = new Set<string>();
    pipelines.forEach((pipeline) => {
      const resolution = pipelineOutputResolution(pipeline);
      if (resolution) values.add(resolution);
    });
    return [...values].sort();
  }, [pipelines]);

  const stepOptions = useMemo(() => {
    const types = new Set<string>();
    pipelines.forEach((pipeline) => pipeline.graph.nodes.forEach((node) => types.add(node.type)));
    return [...types].sort();
  }, [pipelines]);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    const requiredSet =
      requiredInputResolutions && requiredInputResolutions.length > 0 ? new Set(requiredInputResolutions) : null;
    return pipelines.filter((pipeline) => {
      // The currently selected pipeline is always shown so selection is never lost.
      if (pipeline.id === selectedId) return true;
      // Cross-filter to the dataset image size. Unknown input stays visible
      // since compatibility cannot be disproven.
      if (requiredSet) {
        const resolution = pipelineInputResolution(pipeline);
        if (resolution !== null && !requiredSet.has(resolution)) return false;
      }
      if (query) {
        const matches =
          pipeline.name.toLowerCase().includes(query) ||
          (pipeline.description ?? '').toLowerCase().includes(query);
        if (!matches) return false;
      }
      if (inputFilter && pipelineInputResolution(pipeline) !== inputFilter) return false;
      if (outputFilter && pipelineOutputResolution(pipeline) !== outputFilter) return false;
      if (stepFilter.length > 0) {
        const stepTypes = new Set(pipeline.graph.nodes.map((node) => node.type));
        if (!stepFilter.every((type) => stepTypes.has(type))) return false;
      }
      return true;
    });
  }, [pipelines, search, inputFilter, outputFilter, stepFilter, requiredInputResolutions, selectedId]);

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
          <Title order={3}>2. Preprocessing Pipeline</Title>
          {selectedId != null && <Badge variant="light">selected</Badge>}
        </Group>
        <TextInput
          placeholder="Search by name or description"
          leftSection={<Search size={16} />}
          value={search}
          onChange={(event) => setSearch(event.currentTarget.value)}
        />
        <Group grow>
          <Select
            placeholder="Input resolution"
            data={inputResolutionOptions}
            value={inputFilter}
            onChange={setInputFilter}
            clearable
          />
          <Select
            placeholder="Output resolution"
            data={outputResolutionOptions}
            value={outputFilter}
            onChange={setOutputFilter}
            clearable
          />
          <MultiSelect
            placeholder="Contains step"
            data={stepOptions}
            value={stepFilter}
            onChange={setStepFilter}
            clearable
            searchable
          />
        </Group>
        {requiredInputResolutions && requiredInputResolutions.length > 0 && (
          <Text size="xs" c="dimmed">
            Filtered to pipelines accepting {requiredInputResolutions.join(', ')} (selected dataset image size).
            Unknown-input pipelines stay listed.
          </Text>
        )}
        <ScrollArea h={280}>
          <Table striped highlightOnHover verticalSpacing="xs">
            <Table.Thead>
              <Table.Tr>
                <Table.Th w={32} />
                <Table.Th>Name</Table.Th>
                <Table.Th>Steps</Table.Th>
                <Table.Th>Input</Table.Th>
                <Table.Th>Output</Table.Th>
                <Table.Th>Description</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filtered.map((pipeline) => {
                const expanded = expandedIds.has(pipeline.id);
                return (
                  <Fragment key={pipeline.id}>
                    <Table.Tr
                      className={pipeline.id === selectedId ? 'pipeline-step selected' : 'pipeline-step'}
                      style={{ cursor: 'pointer' }}
                      onClick={() => onChange(pipeline.id === selectedId ? null : pipeline.id)}
                    >
                      <Table.Td>
                        <ActionIcon
                          variant="subtle"
                          size="sm"
                          onClick={(event) => {
                            event.stopPropagation();
                            toggleExpanded(pipeline.id);
                          }}
                        >
                          {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                        </ActionIcon>
                      </Table.Td>
                      <Table.Td>{pipeline.name}</Table.Td>
                      <Table.Td>{pipeline.graph.nodes.length}</Table.Td>
                      <Table.Td>
                        <Badge
                          size="xs"
                          variant={pipeline.input_width ? 'light' : 'outline'}
                          color={pipeline.input_width ? 'blue' : 'gray'}
                        >
                          {resolutionLabel(pipeline.input_width, pipeline.input_height)}
                        </Badge>
                      </Table.Td>
                      <Table.Td>
                        <Badge size="xs" variant="light" color="yellow">
                          {resolutionLabel(pipeline.output_width, pipeline.output_height)}
                        </Badge>
                      </Table.Td>
                      <Table.Td>{pipeline.description ?? ''}</Table.Td>
                    </Table.Tr>
                    <Table.Tr>
                      <Table.Td colSpan={6} p={0} style={{ borderBottom: expanded ? undefined : 'none' }}>
                        <Collapse in={expanded}>
                          <Stack gap={6} p="sm">
                            <Text size="xs">
                              Input {resolutionLabel(pipeline.input_width, pipeline.input_height)} → Output{' '}
                              {resolutionLabel(pipeline.output_width, pipeline.output_height)}
                            </Text>
                            <Group gap={6}>
                              {orderedGraphNodes(pipeline).map((node, index) => (
                                <Badge key={node.id} size="sm" variant="light" color="gray">
                                  {index + 1}. {node.type} ({stepDetail(node)})
                                </Badge>
                              ))}
                            </Group>
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
        {pipelines.length === 0 && (
          <Alert color="blue">No preprocessing pipelines available yet. Create one on the Preprocessing page.</Alert>
        )}
      </Stack>
    </Paper>
  );
}
