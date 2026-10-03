import { Alert, Button, Group, Loader, Paper, ScrollArea, Stack, Table, Text, TextInput } from '@mantine/core';
import { useState } from 'react';
import { formatPresetTime, formatRangeDuration } from './helpers';
import { TimeRangePresetActions, type PresetAction } from './TimeRangePresetActions';
import { timeRangePresetStore, useTimeRangePresets } from './useTimeRangePresets';

type Props = { projectId: string; active?: boolean; onBusyChange?: (busy: boolean) => void };
export function TimeRangePresetManager(props: Props) { return <Manager key={props.projectId} {...props} />; }

function Manager({ projectId, active = true, onBusyChange }: Props) {
  const { items, loading, error } = useTimeRangePresets(projectId, active);
  const [search, setSearch] = useState('');
  const [action, setAction] = useState<PresetAction | null>(null);
  const [busy, setBusy] = useState(false);
  const visible = items.filter(item => item.name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()))
    .sort((a, b) => a.name.localeCompare(b.name));
  return <Stack>
    <Text size="sm" c="dimmed">Gemeinsame Vorlagen für dieses Projekt. Das Übernehmen kopiert Beginn und Ende; Änderungen an Vorlagen verändern keine übernommenen Zeiten oder gespeicherten Analyse-Läufe.</Text>
    {error && <Alert color="red">{error}<Button variant="subtle" onClick={() => void timeRangePresetStore.load(projectId)}>Erneut laden</Button></Alert>}
    {action ? <Paper withBorder p="md">
      <TimeRangePresetActions key={`${action.mode}-${'id' in action.value ? action.value.id : 'new'}`} projectId={projectId} action={action}
        onClose={() => { setAction(null); setBusy(false); onBusyChange?.(false); }}
        onBusyChange={value => { setBusy(value); onBusyChange?.(value); }} />
    </Paper> : <Group justify="space-between">
      <TextInput label="Zeiträume nach Namen suchen" value={search} onChange={event => setSearch(event.currentTarget.value)} />
      <Button onClick={() => setAction({ mode: 'create', value: { name: '', start: '', end: '' } })}>Neuer Zeitraum</Button>
    </Group>}
    {loading && <Loader size="sm" />}
    <ScrollArea><Table striped highlightOnHover miw={1000}>
      <Table.Thead><Table.Tr>{['Name', 'Beginn (einschließlich)', 'Ende (einschließlich)', 'Dauer', 'Erstellt (UTC)', 'Geändert (UTC)', 'Aktionen'].map(label => <Table.Th key={label}>{label}</Table.Th>)}</Table.Tr></Table.Thead>
      <Table.Tbody>{visible.map(item => <Table.Tr key={item.id}>
        <Table.Td>{item.name}</Table.Td><Table.Td>{formatPresetTime(item.start)}</Table.Td><Table.Td>{formatPresetTime(item.end)}</Table.Td>
        <Table.Td>{formatRangeDuration(item)}</Table.Td><Table.Td>{formatPresetTime(item.created_at)}</Table.Td><Table.Td>{formatPresetTime(item.updated_at)}</Table.Td>
        <Table.Td><Group gap="xs" wrap="nowrap">
          <Button size="xs" variant="subtle" disabled={busy} onClick={() => setAction({ mode: 'details', value: item })}>Details</Button>
          <Button size="xs" variant="light" disabled={busy} onClick={() => setAction({ mode: 'edit', value: item })}>Bearbeiten</Button>
          <Button size="xs" color="red" variant="subtle" disabled={busy} onClick={() => setAction({ mode: 'delete', value: item })}>Löschen</Button>
        </Group></Table.Td>
      </Table.Tr>)}</Table.Tbody>
    </Table></ScrollArea>
    {!loading && !error && visible.length === 0 && <Text c="dimmed">{items.length ? 'Keine passenden Zeiträume gefunden.' : 'Noch keine Zeiträume gespeichert.'}</Text>}
  </Stack>;
}
