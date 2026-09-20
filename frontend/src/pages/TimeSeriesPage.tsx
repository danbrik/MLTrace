import {
  Alert, Badge, Button, Checkbox, FileInput, Group, Loader, Modal, MultiSelect, Paper,
  Select, SimpleGrid, Stack, Table, TagsInput, Text, TextInput, Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Database, Eye, Pencil, Plus, Save, Trash2, Upload } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { timeSeriesApi as api } from '../api';
import { StepCard } from '../components/StepCard';
import { displayTime, inputTime, intervalError, sortIntervals, utcTime } from '../timeSeries/intervals';
import type { CsvPreview, LabelSplitPreview, Subset, TimeInterval, TimeSeriesDataset, TimeSeriesSplit } from '../timeSeries/types';

const SUBSETS: { value: Subset; label: string; color: string }[] = [
  { value: 'train', label: 'Train', color: 'blue' },
  { value: 'test', label: 'Test', color: 'orange' },
  { value: 'validation', label: 'Validation', color: 'teal' },
];
const FORMATS = [
  { value: 'ISO8601', label: 'ISO-Datum (2026-09-20 14:30:00)' },
  { value: '%d.%m.%Y %H:%M:%S', label: 'TT.MM.JJJJ HH:MM:SS' },
  { value: '%d.%m.%Y %H:%M:%S.%f', label: 'TT.MM.JJJJ HH:MM:SS.Bruchteile' },
  { value: 'unix_s', label: 'Unix-Zeit in Sekunden' },
  { value: 'unix_ms', label: 'Unix-Zeit in Millisekunden' },
  { value: 'unix_us', label: 'Unix-Zeit in Mikrosekunden' },
  { value: 'unix_ns', label: 'Unix-Zeit in Nanosekunden' },
  { value: 'custom', label: 'Eigenes Zeitformat' },
];

function DataPreview({ columns, rows }: { columns: string[]; rows: string[][] }) {
  return <Stack gap="xs">
    <Text size="sm" fw={600}>Datenvorschau · erste {rows.length} Zeilen</Text>
    <Table.ScrollContainer minWidth={Math.max(500, columns.length * 160)} maxHeight={320}>
      <Table striped withTableBorder stickyHeader>
        <Table.Thead><Table.Tr>{columns.map((column) => <Table.Th key={column}>{column}</Table.Th>)}</Table.Tr></Table.Thead>
        <Table.Tbody>{rows.map((row, index) => <Table.Tr key={index}>{columns.map((column, i) => <Table.Td key={column}><Text size="xs" lineClamp={2}>{row[i] || '—'}</Text></Table.Td>)}</Table.Tr>)}</Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  </Stack>;
}

function IntervalBoxes({ intervals, onEdit, onDelete, disabled = false }: {
  intervals: TimeInterval[]; onEdit?: (item: TimeInterval) => void; onDelete?: (id: string) => void; disabled?: boolean;
}) {
  return <SimpleGrid cols={{ base: 1, lg: 3 }} spacing="md">
    {SUBSETS.map((subset) => {
      const items = sortIntervals(intervals.filter((item) => item.subset === subset.value));
      return <Paper key={subset.value} withBorder p="md" radius="md" style={{ borderTop: `4px solid var(--mantine-color-${subset.color}-5)` }}>
        <Stack gap="sm">
          <Group justify="space-between"><Text fw={700}>{subset.label}</Text><Badge color={subset.color}>{items.length} {items.length === 1 ? 'Zeitraum' : 'Zeiträume'}</Badge></Group>
          {!items.length && <Text size="sm" c="dimmed">Noch keine Zeiträume zugeordnet.</Text>}
          {items.map((item) => <Paper key={item.id} withBorder p="sm" radius="sm">
            <Stack gap={6}>
              <Text size="xs" ff="monospace">{displayTime(item.start)}<br />bis {displayTime(item.end)} UTC</Text>
              <Group gap={4}>{item.tags.length ? item.tags.map((tag) => <Badge key={tag} color={subset.color} variant="light" style={{ textTransform: 'none' }}>{tag}</Badge>) : <Text size="xs" c="dimmed">Ohne Tags</Text>}</Group>
              {item.row_count !== undefined && <Text size="xs" c="dimmed">{item.row_count.toLocaleString('de-DE')} Datenzeilen</Text>}
              {onEdit && onDelete && <Group gap="xs"><Button size="compact-xs" variant="subtle" disabled={disabled} onClick={() => onEdit(item)}>Bearbeiten</Button><Button size="compact-xs" variant="subtle" color="red" disabled={disabled} onClick={() => onDelete(item.id)}>Entfernen</Button></Group>}
            </Stack>
          </Paper>)}
        </Stack>
      </Paper>;
    })}
  </SimpleGrid>;
}

export function TimeSeriesPage({ active, section }: { active: boolean; section: 'datasets' | 'splits' }) {
  const [datasets, setDatasets] = useState<TimeSeriesDataset[]>([]);
  const [splits, setSplits] = useState<TimeSeriesSplit[]>([]);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasetModal, setDatasetModal] = useState(false);
  const [datasetRecord, setDatasetRecord] = useState<TimeSeriesDataset | null>(null);
  const [readOnly, setReadOnly] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [name, setName] = useState('');
  const [columns, setColumns] = useState<string[]>([]);
  const [timeColumn, setTimeColumn] = useState<string | null>(null);
  const [format, setFormat] = useState('ISO8601');
  const [customFormat, setCustomFormat] = useState('%Y-%m-%d %H:%M:%S');
  const [timePreview, setTimePreview] = useState<{ start: string; end: string } | null>(null);
  const [labelColumn, setLabelColumn] = useState<string | null>(null);
  const [autoSplit, setAutoSplit] = useState(false);
  const [importSplitName, setImportSplitName] = useState<string | null>(null);
  const [labelPreview, setLabelPreview] = useState<LabelSplitPreview | null>(null);
  const [splitModal, setSplitModal] = useState(false);
  const [splitRecord, setSplitRecord] = useState<TimeSeriesSplit | null>(null);
  const [splitName, setSplitName] = useState('');
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [tags, setTags] = useState<string[]>([]);
  const [intervals, setIntervals] = useState<TimeInterval[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [subset, setSubset] = useState<Subset>('train');
  const [intervalTags, setIntervalTags] = useState<string[]>([]);
  const [deleteTarget, setDeleteTarget] = useState<{ kind: 'dataset' | 'split'; id: number; name: string; splitCount?: number } | null>(null);
  const fileGeneration = useRef(0);
  const selectedDataset = datasets.find((item) => String(item.id) === datasetId);
  const effectiveFormat = format === 'custom' ? customFormat : format;
  const effectiveSplitName = importSplitName ?? `${name.trim().slice(0, 241)} – Label-Split`;
  function invalidatePreview() { setTimePreview(null); setLabelPreview(null); }
  const candidate: TimeInterval = { id: editingId ?? 'new', start: utcTime(start), end: utcTime(end), subset, tags: intervalTags };
  const rangeError = selectedDataset ? intervalError(candidate, intervals, selectedDataset) : 'Bitte zuerst eine Datenbasis auswählen.';

  async function reload() {
    const [nextDatasets, nextSplits] = await Promise.all([api.datasets(), api.splits()]);
    setDatasets(nextDatasets); setSplits(nextSplits);
  }
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    setLoading(true); setError(null);
    Promise.all([api.datasets(), api.splits()]).then(([nextDatasets, nextSplits]) => {
      if (!cancelled) { setDatasets(nextDatasets); setSplits(nextSplits); }
    }).catch((err: Error) => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [active, section]);

  async function perform(action: () => Promise<void>) {
    setBusy(true); setError(null);
    try { await action(); } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setBusy(false); }
  }
  function resetInterval() { setEditingId(null); setStart(''); setEnd(''); setIntervalTags([]); }
  function openImport() {
    setError(null); setDatasetRecord(null); setReadOnly(false); setFile(null); setPreview(null);
    setName(''); setColumns([]); setTimeColumn(null); setFormat('ISO8601'); setTimePreview(null); setDatasetModal(true);
    setLabelColumn(null); setAutoSplit(false); setImportSplitName(null); setLabelPreview(null);
  }
  async function chooseFile(next: File | null) {
    const generation = ++fileGeneration.current;
    setFile(next); setPreview(null); setTimePreview(null); setTimeColumn(null); setColumns([]); setError(null);
    setLabelColumn(null); setAutoSplit(false); setImportSplitName(null); setLabelPreview(null);
    if (!next) return;
    setName(next.name.replace(/\.csv$/i, ''));
    if (next.size > 50 * 1024 * 1024) { setError('Die CSV darf höchstens 50 MB groß sein.'); return; }
    setBusy(true);
    try {
      const result = await api.preview(next);
      if (generation !== fileGeneration.current) return;
      const detectedLabel = result.detected_label_column ?? null;
      setPreview(result); setColumns(result.columns.filter((column) => column !== detectedLabel));
      setLabelColumn(detectedLabel); setAutoSplit(!!detectedLabel);
      setTimeColumn(result.columns.find((column) => /^(timestamp|time|datetime|date|zeit|zeitstempel)$/i.test(column.trim())) ?? result.columns.find((column) => column !== detectedLabel) ?? null);
    } catch (err) { if (generation === fileGeneration.current) setError(err instanceof Error ? err.message : String(err)); }
    finally { if (generation === fileGeneration.current) setBusy(false); }
  }
  function openDataset(record: TimeSeriesDataset, view: boolean) {
    void perform(async () => {
      const detail = await api.dataset(record.id);
      setDatasetRecord(detail); setReadOnly(view); setName(detail.name); setColumns(detail.selected_columns);
      setTimeColumn(detail.timestamp_column); setPreview(null); setFile(null); setDatasetModal(true);
      setLabelColumn(detail.label_column ?? null); setAutoSplit(false); setLabelPreview(null);
    });
  }
  function openSplit(record: TimeSeriesSplit | null, view = false) {
    setError(null); setSplitRecord(record); setReadOnly(view); setSplitName(record?.name ?? '');
    setDatasetId(record ? String(record.dataset_id) : null); setTags(record?.tags ?? []);
    setIntervals(record?.intervals ?? []); resetInterval(); setSplitModal(true);
  }
  const sourceColumns = datasetRecord?.columns ?? preview?.columns ?? [];
  const shownColumns = sourceColumns.filter((column) => columns.includes(column));
  const shownRows = (datasetRecord?.source_rows ?? preview?.rows ?? []).map((row) => sourceColumns.flatMap((column, index) => columns.includes(column) ? [row[index]] : []));
  const errorNotice = error && <Alert color="red" title="Aktion konnte nicht abgeschlossen werden">{error}</Alert>;

  return <Stack gap="lg">
    <Group justify="space-between">
      <div><Title order={2}>{section === 'datasets' ? 'Zeitreihen · Datenbasis' : 'Zeitreihen · Splits'}</Title>
        <Text c="dimmed" size="sm">{section === 'datasets' ? 'CSV-Daten importieren, benennen und relevante Spalten auswählen.' : 'Datenbasis auswählen und Zeiträume mit Tags auf Train, Test und Validation verteilen.'}</Text>
      </div>
      <Button leftSection={section === 'datasets' ? <Upload size={16} /> : <Plus size={16} />} disabled={busy || loading || (section === 'splits' && !datasets.length)} onClick={() => section === 'datasets' ? openImport() : openSplit(null)}>
        {section === 'datasets' ? 'CSV importieren' : 'Neuer Split'}
      </Button>
    </Group>
    {!datasetModal && !splitModal && !deleteTarget && errorNotice}
    {loading && <Loader size="sm" />}
    {section === 'datasets' ? <>
      {!loading && !datasets.length && <Paper withBorder p="xl" radius="md"><Stack align="center"><Database size={32} /><Text fw={600}>Noch keine Zeitreihen-Datenbasis</Text><Text c="dimmed" size="sm">Importiere eine CSV mit Zeitspalte und mindestens einer Datenspalte.</Text><Button variant="light" onClick={openImport}>Erste CSV importieren</Button></Stack></Paper>}
      <SimpleGrid cols={{ base: 1, lg: 2 }}>{datasets.map((record) => <Paper key={record.id} withBorder p="lg" radius="md"><Stack gap="sm">
        <Group justify="space-between"><Text fw={700}>{record.name}</Text><Badge variant="light">CSV</Badge></Group>
        <Text size="xs" c="dimmed">{record.filename}</Text>
        <Group gap="xs"><Badge variant="light">{record.row_count.toLocaleString('de-DE')} Zeilen</Badge><Badge variant="light">{record.selected_columns.length} Spalten</Badge><Badge variant="light">{record.split_count} Splits</Badge></Group>
        <Text size="sm">{displayTime(record.start)} → {displayTime(record.end)} UTC</Text>
        <Text size="xs" c="dimmed">Zeitspalte: {record.timestamp_column}</Text>
        <Group gap="xs"><Button size="xs" variant="light" leftSection={<Eye size={14} />} disabled={busy} onClick={() => openDataset(record, true)}>Anzeigen</Button><Button size="xs" variant="subtle" leftSection={<Pencil size={14} />} disabled={busy} onClick={() => openDataset(record, false)}>Bearbeiten</Button><Button size="xs" variant="subtle" color="red" leftSection={<Trash2 size={14} />} onClick={() => { setError(null); setDeleteTarget({ kind: 'dataset', id: record.id, name: record.name, splitCount: record.split_count }); }}>Löschen</Button></Group>
      </Stack></Paper>)}</SimpleGrid>
    </> : <>
      {!loading && !splits.length && <Paper withBorder p="xl" radius="md"><Text fw={600}>Noch keine Splits gespeichert</Text><Text size="sm" c="dimmed">{datasets.length ? 'Erstelle einen Split und füge beliebig viele getrennte Zeiträume hinzu.' : 'Importiere zuerst unter „Datenbasis“ eine CSV.'}</Text></Paper>}
      {splits.map((record) => <Paper key={record.id} withBorder p="lg" radius="md"><Stack gap="sm">
        <Group justify="space-between"><div><Text fw={700}>{record.name}</Text><Text size="sm" c="dimmed">Datenbasis: {record.dataset_name}</Text></div>
          <Group gap="xs"><Button size="xs" variant="light" leftSection={<Eye size={14} />} onClick={() => openSplit(record, true)}>Anzeigen</Button><Button size="xs" variant="subtle" leftSection={<Pencil size={14} />} onClick={() => openSplit(record)}>Bearbeiten</Button><Button size="xs" variant="subtle" color="red" leftSection={<Trash2 size={14} />} onClick={() => { setError(null); setDeleteTarget({ kind: 'split', id: record.id, name: record.name }); }}>Löschen</Button></Group>
        </Group>
        <Group gap="xs">{SUBSETS.map((group) => <Badge key={group.value} color={group.color} variant="light">{group.label}: {record.intervals.filter((item) => item.subset === group.value).length}</Badge>)}{record.tags.map((tag) => <Badge key={tag} color="gray" variant="outline" style={{ textTransform: 'none' }}>{tag}</Badge>)}</Group>
        <IntervalBoxes intervals={record.intervals} />
      </Stack></Paper>)}
    </>}

    <Modal opened={datasetModal} onClose={() => { if (!busy) setDatasetModal(false); }} title={datasetRecord ? `${readOnly ? 'Datenbasis' : 'Datenbasis bearbeiten'} · ${datasetRecord.name}` : 'CSV-Datenbasis importieren'} size="xl" closeOnClickOutside={false}>
      <Stack>
        {errorNotice}
        {!datasetRecord && <FileInput label="CSV-Datei" description="UTF-8 · Komma, Semikolon, Tab oder | · bis 50 MB" accept=".csv,text/csv,text/plain" value={file} disabled={busy} onChange={(next) => void chooseFile(next)} leftSection={<Upload size={16} />} />}
        {busy && !sourceColumns.length && <Loader size="sm" />}
        {!!sourceColumns.length && <>
          <TextInput label="Name der Datenbasis" required maxLength={255} value={name} readOnly={readOnly} disabled={busy} onChange={(event) => setName(event.currentTarget.value)} />
          {!datasetRecord && <StepCard index={1} title="Zeitspalte festlegen">
            <SimpleGrid cols={{ base: 1, sm: 2 }}>
              <Select label="Zeitspalte" required data={sourceColumns.filter((column) => column !== labelColumn)} value={timeColumn} disabled={busy} searchable onChange={(value) => { setTimeColumn(value); invalidatePreview(); if (value) setColumns((current) => [...new Set([...current, value])]); }} />
              <Select label="Zeitformat" data={FORMATS} value={format} disabled={busy} onChange={(value) => { setFormat(value ?? 'ISO8601'); invalidatePreview(); }} />
            </SimpleGrid>
            {format === 'custom' && <TextInput label="Zeitformat" description="Beispiel: %d/%m/%Y %H:%M:%S" value={customFormat} disabled={busy} onChange={(event) => { setCustomFormat(event.currentTarget.value); invalidatePreview(); }} />}
            <Text size="xs" c="dimmed">Zeitangaben ohne Zeitzone werden als UTC übernommen. Angaben mit Zeitzone werden nach UTC umgerechnet.</Text>
            <Button variant="light" w="fit-content" loading={busy} disabled={!timeColumn || !effectiveFormat} onClick={() => void perform(async () => {
              invalidatePreview();
              const result = await api.preview(file!, timeColumn!, effectiveFormat, autoSplit ? labelColumn! : undefined);
              setTimePreview({ start: result.start!, end: result.end! });
              setLabelPreview(result.label_split ?? null);
            })}>{autoSplit ? 'Zeitspalte und Label-Split prüfen' : 'Zeitspalte prüfen'}</Button>
            {timePreview && <Alert color="teal" title="Zeitspalte geprüft">{preview?.row_count.toLocaleString('de-DE')} Zeilen · {displayTime(timePreview.start)} bis {displayTime(timePreview.end)} UTC</Alert>}
          </StepCard>}
          {datasetRecord && <Text size="sm" c="dimmed">Zeitspalte: {datasetRecord.timestamp_column} · {datasetRecord.row_count.toLocaleString('de-DE')} Zeilen<br />{displayTime(datasetRecord.start)} bis {displayTime(datasetRecord.end)} UTC</Text>}
          {labelColumn && <StepCard title="Labelspalte" subtitle={`„${labelColumn}“ wird als Annotation gespeichert und nicht als Sensor verwendet.`}>
            {!datasetRecord && <>
              <Checkbox label="Split aus Labelspalte erstellen" checked={autoSplit} disabled={busy} onChange={(event) => { setAutoSplit(event.currentTarget.checked); invalidatePreview(); setError(null); }} />
              {autoSplit && <>
                <Text size="sm">normal → Train (ohne Tag) · before_anomaly, anomaly und cooldown → Test mit dem jeweiligen Tag. Validation bleibt leer. Jeder Labelwechsel beginnt einen neuen Zeitraum.</Text>
                <TextInput label="Name des automatisch erstellten Splits" required maxLength={255} value={effectiveSplitName} disabled={busy} onChange={(event) => setImportSplitName(event.currentTarget.value)} />
                {!labelPreview && <Text size="sm" c="dimmed">Bitte oben „Zeitspalte und Label-Split prüfen“ ausführen, um die Zeiträume vor dem Speichern anzuzeigen.</Text>}
                {labelPreview && <>
                  <Group>{SUBSETS.map((group) => <Badge key={group.value} color={group.color} variant="light">{group.label}: {labelPreview.counts[group.value].rows.toLocaleString('de-DE')} Zeilen</Badge>)}</Group>
                  <div style={{ maxHeight: 420, overflowY: 'auto' }}><IntervalBoxes intervals={labelPreview.intervals} /></div>
                </>}
              </>}
            </>}
            <DataPreview columns={[labelColumn]} rows={(datasetRecord?.source_rows ?? preview?.rows ?? []).map((row) => [row[sourceColumns.indexOf(labelColumn)]])} />
          </StepCard>}
          {!readOnly && <StepCard index={datasetRecord ? undefined : 2} title="Spalten auswählen" subtitle="Die Zeitspalte bleibt ausgewählt. Mindestens eine weitere Spalte ist erforderlich.">
            <Group><Button size="compact-xs" variant="subtle" disabled={busy} onClick={() => setColumns(sourceColumns.filter((column) => column !== labelColumn))}>Alle auswählen</Button><Button size="compact-xs" variant="subtle" disabled={busy} onClick={() => setColumns(timeColumn ? [timeColumn] : [])}>Nur Zeitspalte</Button></Group>
            <SimpleGrid cols={{ base: 1, sm: 3 }}>{sourceColumns.map((column) => <Checkbox key={column} label={column === labelColumn ? `${column} (Annotation)` : column} checked={columns.includes(column)} disabled={busy || column === timeColumn || column === labelColumn} onChange={(event) => setColumns(event.currentTarget.checked ? [...columns, column] : columns.filter((item) => item !== column))} />)}</SimpleGrid>
            {datasetRecord && <Text size="xs" c="dimmed">Die Spaltenauswahl gilt auch für vorhandene Splits. Die gespeicherte CSV bleibt erhalten, damit sich Spalten wieder aktivieren lassen.</Text>}
          </StepCard>}
          <DataPreview columns={shownColumns} rows={shownRows} />
          <Group justify="flex-end">
            <Button variant="default" disabled={busy} onClick={() => setDatasetModal(false)}>{readOnly ? 'Schließen' : 'Abbrechen'}</Button>
            {readOnly ? <Button onClick={() => setReadOnly(false)}>Bearbeiten</Button> : <Button leftSection={<Save size={16} />} loading={busy} disabled={!name.trim() || columns.length < 2 || !timeColumn || (!datasetRecord && (!timePreview || (autoSplit && (!labelPreview || !effectiveSplitName.trim()))))} onClick={() => void perform(async () => {
              if (datasetRecord) await api.updateDataset(datasetRecord.id, { name, selected_columns: columns });
              else await api.createDataset(file!, { name, selected_columns: columns, timestamp_column: timeColumn!, timestamp_format: effectiveFormat, label_column: labelColumn, auto_split: autoSplit, ...(autoSplit ? { split_name: effectiveSplitName } : {}) });
              setDatasetModal(false); await reload(); notifications.show({ color: 'teal', message: !datasetRecord && autoSplit ? 'Datenbasis und Label-Split gespeichert. Der Split ist unter „Splits“ verfügbar.' : 'Datenbasis gespeichert.' });
            })}>{!datasetRecord && autoSplit ? 'Datenbasis und Split speichern' : 'Datenbasis speichern'}</Button>}
          </Group>
        </>}
      </Stack>
    </Modal>

    <Modal opened={splitModal} onClose={() => { if (!busy) setSplitModal(false); }} title={splitRecord ? `${readOnly ? 'Split' : 'Split bearbeiten'} · ${splitRecord.name}` : 'Neuer Zeitreihen-Split'} size="min(1400px, 96vw)" closeOnClickOutside={false}>
      <Stack>
        {errorNotice}
        <SimpleGrid cols={{ base: 1, sm: 2 }}>
          <Select label="Datenbasis" required searchable data={datasets.map((item) => ({ value: String(item.id), label: item.name }))} value={datasetId} disabled={busy || readOnly || !!splitRecord || intervals.length > 0} onChange={(value) => { setDatasetId(value); resetInterval(); }} />
          <TextInput label="Name des Splits" required maxLength={255} value={splitName} readOnly={readOnly} disabled={busy} onChange={(event) => setSplitName(event.currentTarget.value)} />
        </SimpleGrid>
        {selectedDataset && <Text size="sm" c="dimmed">Verfügbar: {displayTime(selectedDataset.start)} bis {displayTime(selectedDataset.end)} UTC · {selectedDataset.row_count.toLocaleString('de-DE')} Zeilen</Text>}
        {!readOnly && <>
          <TagsInput label="Tags für diesen Split" description="Tag eingeben und mit Enter hinzufügen, z. B. Anomalie, Puffer oder Normalzustand. Mehrere Tags pro Zeitraum sind möglich. Entfernte Tags werden aus allen Zeiträumen entfernt." value={tags} disabled={busy} onChange={(values) => {
            setTags(values); setIntervals((current) => current.map((item) => ({ ...item, tags: item.tags.filter((tag) => values.includes(tag)) }))); setIntervalTags((current) => current.filter((tag) => values.includes(tag)));
          }} />
          <StepCard title={editingId ? 'Zeitraum bearbeiten' : 'Zeitraum hinzufügen'} color="violet" subtitle="Alle Zeiten in UTC. Start und Ende inklusive. Belegte Zeiträume sind in diesem Split für alle drei Gruppen gesperrt.">
            <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
              <TextInput label="Beginn (UTC)" placeholder="2026-09-20T08:00:00" description="JJJJ-MM-TTTHH:MM:SS" value={start} disabled={!selectedDataset || busy} onChange={(event) => setStart(event.currentTarget.value)} />
              <TextInput label="Ende (UTC)" placeholder="2026-09-20T09:00:00" description="Bruchteile von Sekunden sind möglich" value={end} disabled={!selectedDataset || busy} onChange={(event) => setEnd(event.currentTarget.value)} />
              <Select label="Zuordnung" data={SUBSETS} value={subset} disabled={busy} onChange={(value) => setSubset(value as Subset)} />
              <MultiSelect label="Tags des Zeitraums" placeholder={tags.length ? 'Tags auswählen' : 'Zuerst oben Tags anlegen'} data={tags} value={intervalTags} searchable disabled={busy} onChange={setIntervalTags} />
            </SimpleGrid>
            {selectedDataset && <Group gap="xs"><Button variant="subtle" size="compact-xs" disabled={busy} onClick={() => setStart(inputTime(selectedDataset.start))}>Ersten Zeitstempel übernehmen</Button><Button variant="subtle" size="compact-xs" disabled={busy} onClick={() => setEnd(inputTime(selectedDataset.end))}>Letzten Zeitstempel übernehmen</Button></Group>}
            {(start || end) && rangeError && <Alert color="orange">{rangeError}</Alert>}
            <Group><Button variant="light" leftSection={<Plus size={16} />} disabled={busy || !!rangeError} onClick={() => {
              setIntervals(sortIntervals([...intervals.filter((item) => item.id !== editingId), { ...candidate, id: editingId ?? crypto.randomUUID() }])); resetInterval(); setError(null);
            }}>{editingId ? 'Änderung übernehmen' : 'Zeitraum hinzufügen'}</Button>{(editingId || start || end) && <Button variant="subtle" disabled={busy} onClick={resetInterval}>Eingabe verwerfen</Button>}</Group>
          </StepCard>
        </>}
        <Text size="xs" c="dimmed">Zeiträume sind je Gruppe chronologisch sortiert. Nicht zugeordnete Daten bleiben ungenutzt.</Text>
        <IntervalBoxes intervals={intervals} disabled={busy} onEdit={readOnly ? undefined : (item) => { setEditingId(item.id); setStart(inputTime(item.start)); setEnd(inputTime(item.end)); setSubset(item.subset); setIntervalTags(item.tags); }} onDelete={readOnly ? undefined : (id) => { setIntervals(intervals.filter((item) => item.id !== id)); if (editingId === id) resetInterval(); }} />
        {!readOnly && (start || end || editingId) && <Text size="sm" c="orange">Bitte den eingegebenen Zeitraum zuerst übernehmen oder die Eingabe verwerfen.</Text>}
        <Group justify="flex-end"><Button variant="default" disabled={busy} onClick={() => setSplitModal(false)}>{readOnly ? 'Schließen' : 'Abbrechen'}</Button>
          {readOnly ? <Button onClick={() => setReadOnly(false)}>Bearbeiten</Button> : <Button leftSection={<Save size={16} />} loading={busy} disabled={!splitName.trim() || !datasetId || !intervals.length || !!start || !!end || !!editingId} onClick={() => void perform(async () => {
            await api.saveSplit({ name: splitName, dataset_id: Number(datasetId), tags, intervals }, splitRecord?.id);
            setSplitModal(false); await reload(); notifications.show({ color: 'teal', message: 'Split gespeichert.' });
          })}>Split speichern</Button>}
        </Group>
      </Stack>
    </Modal>

    <Modal opened={!!deleteTarget} onClose={() => { if (!busy) setDeleteTarget(null); }} title="Eintrag löschen" centered>
      <Stack>{errorNotice}<Text>„{deleteTarget?.name}“ dauerhaft löschen?</Text>
        {deleteTarget?.kind === 'dataset' && <Alert color="orange">Die gespeicherte CSV und alle {deleteTarget.splitCount} zugehörigen Splits werden gelöscht.</Alert>}
        <Group justify="flex-end"><Button variant="default" disabled={busy} onClick={() => setDeleteTarget(null)}>Abbrechen</Button><Button color="red" loading={busy} onClick={() => void perform(async () => {
          if (!deleteTarget) return;
          if (deleteTarget.kind === 'dataset') await api.deleteDataset(deleteTarget.id); else await api.deleteSplit(deleteTarget.id);
          setDeleteTarget(null); await reload(); notifications.show({ color: 'teal', message: 'Eintrag gelöscht.' });
        })}>Dauerhaft löschen</Button></Group>
      </Stack>
    </Modal>
  </Stack>;
}
