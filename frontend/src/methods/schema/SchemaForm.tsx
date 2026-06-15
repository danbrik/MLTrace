import { Group, Paper, Select, SimpleGrid, Stack, Switch, Text, TextInput, Tooltip } from '@mantine/core';
import { Info } from 'lucide-react';
import type { ReactNode } from 'react';

import type { SchemaProperty } from '../../types';
import type { SchemaFormProps } from '../types';
import { BufferedNumberInput } from './BufferedNumberInput';

function fieldLabel(key: string, property: SchemaProperty): ReactNode {
  const label = property.label ?? key;
  const description = property.description ?? property.help_text;
  if (!description) return label;
  return (
    <Group component="span" gap={5} wrap="nowrap" className="schema-field-label">
      <span>{label}</span>
      <Tooltip label={description} multiline w={280} withArrow>
        <span className="schema-info-button" aria-label={`${label} info`} role="img">
          <Info size={12} />
        </span>
      </Tooltip>
    </Group>
  );
}

function SchemaField({
  config,
  fieldKey,
  property,
  onChange,
  disabled,
  fieldPrefix,
  onNumberDraftChange,
}: {
  config: Record<string, unknown>;
  fieldKey: string;
  property: SchemaProperty;
  onChange: (key: string, value: unknown) => void;
  disabled?: boolean;
  fieldPrefix?: string;
  onNumberDraftChange?: SchemaFormProps['onNumberDraftChange'];
}) {
  const value = config[fieldKey] ?? property.default ?? '';
  const label = fieldLabel(fieldKey, property);

  if (property.enum) {
    return (
      <Select
        key={fieldKey}
        label={label}
        data={property.enum}
        value={String(value)}
        disabled={disabled}
        onChange={(next) => onChange(fieldKey, next ?? property.default)}
      />
    );
  }

  if (property.type === 'boolean') {
    return (
      <Switch
        key={fieldKey}
        label={label}
        checked={value === true}
        disabled={disabled}
        onChange={(event) => onChange(fieldKey, event.currentTarget.checked)}
      />
    );
  }

  if (property.type === 'integer' || property.type === 'number') {
    return (
      <BufferedNumberInput
        key={fieldKey}
        label={label}
        min={property.minimum}
        max={property.maximum}
        integerOnly={property.type === 'integer'}
        value={typeof value === 'number' || typeof value === 'string' ? value : ''}
        disabled={disabled}
        onCommit={(next) => onChange(fieldKey, next)}
        onDraftStateChange={(state) => onNumberDraftChange?.(`${fieldPrefix ?? 'schema'}.${fieldKey}`, state)}
      />
    );
  }

  return (
    <TextInput
      key={fieldKey}
      label={label}
      value={String(value)}
      disabled={disabled}
      onChange={(event) => onChange(fieldKey, event.currentTarget.value)}
    />
  );
}

export function SchemaForm({ title, schema, config, keys, onChange, disabled, fieldPrefix, onNumberDraftChange }: SchemaFormProps) {
  const entries = Object.entries(schema?.properties ?? {}).filter(([key]) => !keys || keys.includes(key));
  const fields = entries.map(([key, property]) => (
    <SchemaField
      key={key}
      config={config}
      fieldKey={key}
      property={property}
      onChange={onChange}
      disabled={disabled}
      fieldPrefix={fieldPrefix}
      onNumberDraftChange={onNumberDraftChange}
    />
  ));

  if (fields.length === 0) return null;
  const grid = <SimpleGrid cols={{ base: 1, sm: 2, xl: 3 }}>{fields}</SimpleGrid>;

  if (!title) return grid;
  return (
    <Paper withBorder p="sm" radius="sm">
      <Stack gap="sm">
        <Text fw={700}>{title}</Text>
        {grid}
      </Stack>
    </Paper>
  );
}
