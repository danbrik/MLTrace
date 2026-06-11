import { ActionIcon, Alert, Badge, Collapse, Group, Paper, ScrollArea, Stack, Table, Text } from '@mantine/core';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';

import type { MethodValidationResponse } from '../../types';
import { formatValue } from '../utils';

function inputShapeLabel(modelConfig: Record<string, unknown>) {
  return `N,${formatValue(modelConfig.input_channels)},${formatValue(modelConfig.input_height)},${formatValue(modelConfig.input_width)}`;
}

function validationShapeRows(validation: MethodValidationResponse | null, modelConfig: Record<string, unknown>) {
  const specs = validation?.layer_specs ?? [];
  if (specs.length === 0) return [];
  const firstInput = specs[0]?.input_label ?? inputShapeLabel(modelConfig);
  const lastOutput = specs[specs.length - 1]?.output_label ?? firstInput;
  return [
    { key: 'input', layer: 'Input', input: 'source image', output: firstInput },
    ...specs.map((spec) => ({
      key: `${spec.section}-${spec.index}-${spec.layer_id ?? spec.layer_type}`,
      layer: `${spec.section} ${spec.index}: ${spec.layer_type}`,
      input: spec.input_label,
      output: spec.output_label,
    })),
    { key: 'output', layer: 'Output', input: lastOutput, output: 'reconstruction' },
  ];
}

export function ArchitectureCheckPanel({
  validation,
  modelConfig,
}: {
  validation: MethodValidationResponse | null;
  modelConfig: Record<string, unknown>;
}) {
  const [shapesExpanded, setShapesExpanded] = useState(false);

  if (!validation) {
    return <Alert color="blue">Architecture check will run after the method definition is complete.</Alert>;
  }

  const shapeRows = validationShapeRows(validation, modelConfig);
  return (
    <Stack gap="sm">
      <Group justify="space-between" align="center">
        <Text fw={700}>Architecture Check</Text>
        <Badge color={validation.valid ? 'green' : 'red'} variant="light">
          {validation.valid ? 'VALIDATION PASSED' : 'VALIDATION FAILED'}
        </Badge>
      </Group>
      {validation.errors.length > 0 && (
        <Alert color="red" title="Hard errors block saving">
          <Stack gap={4}>
            {validation.errors.map((error) => (
              <Text key={error} size="sm">
                {error}
              </Text>
            ))}
          </Stack>
        </Alert>
      )}
      {validation.warnings.length > 0 && (
        <Alert color="yellow" title="Warnings">
          <Stack gap={4}>
            {validation.warnings.map((warning) => (
              <Text key={warning} size="sm">
                {warning}
              </Text>
            ))}
          </Stack>
        </Alert>
      )}
      {shapeRows.length > 0 && (
        <Paper withBorder p="sm" radius="sm">
          <Stack gap="sm">
            <Group justify="space-between">
              <Group gap="xs">
                <ActionIcon
                  variant="subtle"
                  onClick={() => setShapesExpanded((current) => !current)}
                  aria-label={shapesExpanded ? 'Collapse layer shape table' : 'Expand layer shape table'}
                >
                  {shapesExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                </ActionIcon>
                <div>
                  <Text fw={700} size="sm">
                    Layer shape flow
                  </Text>
                  <Text size="xs" c="dimmed">
                    Includes explicit input and output rows
                  </Text>
                </div>
              </Group>
              <Badge variant="light">{shapeRows.length} rows</Badge>
            </Group>
            <Collapse in={shapesExpanded}>
              <ScrollArea>
                <Table striped verticalSpacing={4} miw={420}>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Layer</Table.Th>
                      <Table.Th>Input</Table.Th>
                      <Table.Th>Output</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {shapeRows.map((row) => (
                      <Table.Tr key={row.key}>
                        <Table.Td>
                          <Text size="xs">{row.layer}</Text>
                        </Table.Td>
                        <Table.Td>
                          <Text size="xs">{row.input}</Text>
                        </Table.Td>
                        <Table.Td>
                          <Text size="xs">{row.output}</Text>
                        </Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              </ScrollArea>
            </Collapse>
          </Stack>
        </Paper>
      )}
    </Stack>
  );
}
