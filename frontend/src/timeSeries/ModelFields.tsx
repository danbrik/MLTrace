import { useEffect, useState } from 'react';
import { Badge, Checkbox, Group, NumberInput, Paper, SimpleGrid, Stack, Text, TextInput } from '@mantine/core';
import type { FieldDefinition, Parameters } from './types';

export function fieldDefaults(fields: Record<string, FieldDefinition>): Parameters {
  return Object.fromEntries(Object.entries(fields).map(([key, field]) => [key, field.default]));
}

function DilationsInput({ value, onChange, label }: { value: number[]; onChange: (value: number[]) => void; label: string }) {
  const [text, setText] = useState(value.join(', '));
  useEffect(() => { setText(value.join(', ')); }, [JSON.stringify(value)]);
  return <TextInput label={label} description="Kommagetrennte positive ganze Zahlen" value={text} onChange={e => setText(e.currentTarget.value)} onBlur={() => onChange(text.split(',').map(Number))} />;
}

export function ModelFields({ fields, values, onChange }: { fields: Record<string, FieldDefinition>; values: Parameters; onChange: (values: Parameters) => void }) {
  return <SimpleGrid cols={{ base: 1, md: 2, xl: 3 }}>{Object.entries(fields).map(([key, field]) => {
    const value = values[key] ?? field.default;
    const changed = JSON.stringify(value) !== JSON.stringify(field.default);
    return <Paper key={key} withBorder p="sm"><Stack gap={5}>
      {typeof field.default === 'boolean' ? <Checkbox label={field.label} checked={Boolean(value)} onChange={e => onChange({ ...values, [key]: e.currentTarget.checked })} />
        : Array.isArray(field.default) ? <DilationsInput label={field.label} value={value as number[]} onChange={v => onChange({ ...values, [key]: v })} />
          : <NumberInput label={field.label} value={value as number} min={field.minimum ?? undefined} max={field.maximum ?? undefined} allowDecimal={!Number.isInteger(field.default) || key === 'noise_std' || key === 'kl_weight' || key === 'dropout'} onChange={v => onChange({ ...values, [key]: typeof v === 'number' ? v : Number(v) })} />}
      <Group gap={5}><Badge size="xs" variant="light">{field.origin}</Badge>{changed && <Badge size="xs" color="orange">Überschrieben</Badge>}<Text size="xs" c="dimmed">Default: {String(field.default)}</Text></Group>
      <Text size="xs" c="dimmed">{field.reference}</Text>{field.adaptation && <Text size="xs" c="orange">{field.adaptation}</Text>}
    </Stack></Paper>;
  })}</SimpleGrid>;
}
export function Flow({ steps }: { steps: string[] }) {
  return <Group gap="xs" align="stretch" wrap="wrap">{steps.map((step, i) => <Group gap="xs" key={`${i}-${step}`}><Paper withBorder radius="md" p="sm" style={{ maxWidth: 220 }}><Text size="xs" c="dimmed">{String(i + 1).padStart(2, '0')}</Text><Text size="sm" fw={600}>{step}</Text></Paper>{i < steps.length - 1 && <Text c="dimmed">→</Text>}</Group>)}</Group>;
}
