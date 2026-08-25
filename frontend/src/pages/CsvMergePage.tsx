import {
  Alert,
  Badge,
  Button,
  Checkbox,
  FileInput,
  Group,
  Modal,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Download, FileSpreadsheet, Plus, RotateCcw, Trash2, Upload } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { StepCard, STEP_COLORS } from '../components/StepCard';
import { DEFAULT_TABLE_PAGE_SIZE, TablePagination } from '../components/TablePagination';
import {
  csvMergeResultTable,
  defaultMergedFileName,
  mergeCsvDocuments,
  parseCsvFile,
  timestampNormalizationSuggested,
  timestampNormalizationPreview,
  validateCsvMerge,
  type CsvDocument,
  type CsvAggregationMethod,
  type CsvDuplicatePolicy,
  type CsvJoinType,
  type CsvKeyPair,
  type CsvMergeResult,
  type CsvMergeValidation,
  type CsvSecondaryColumn,
} from '../csvMerge/csvMerge';
import {
  normalizedTableDownloadName,
  tabularTableToCsv,
  tabularTableToParquet,
} from '../lib/tabularExport';

const PREVIEW_ROWS = DEFAULT_TABLE_PAGE_SIZE;

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function delimiterLabel(delimiter: string): string {
  return delimiter === '\t' ? 'Tab' : delimiter === ',' ? 'Comma' : delimiter === ';' ? 'Semicolon' : JSON.stringify(delimiter);
}

function DataPreview({ headers, rows, label }: { headers: string[]; rows: Array<Array<string | null>>; label: string }) {
  const [page, setPage] = useState(1);
  useEffect(() => setPage(1), [headers, rows]);
  const start = (page - 1) * PREVIEW_ROWS;
  const visible = rows.slice(start, start + PREVIEW_ROWS);
  return (
    <Stack gap="xs">
      <Text size="sm" fw={600}>{label}</Text>
      <Table.ScrollContainer minWidth={Math.max(600, headers.length * 150)} maxHeight={420}>
        <Table striped highlightOnHover withTableBorder withColumnBorders stickyHeader>
          <Table.Thead><Table.Tr><Table.Th>#</Table.Th>{headers.map((header) => <Table.Th key={header}>{header}</Table.Th>)}</Table.Tr></Table.Thead>
          <Table.Tbody>
            {visible.map((row, rowOffset) => (
              <Table.Tr key={start + rowOffset}>
                <Table.Td><Text size="xs" c="dimmed">{start + rowOffset + 1}</Text></Table.Td>
                {headers.map((header, columnIndex) => (
                  <Table.Td key={header}><Text size="xs" lineClamp={3}>{row[columnIndex] ?? <Text span c="dimmed">—</Text>}</Text></Table.Td>
                ))}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      {rows.length === 0 && <Text size="sm" c="dimmed">The file contains headers but no data rows.</Text>}
      <TablePagination totalItems={rows.length} page={page} onChange={setPage} pageSize={PREVIEW_ROWS} />
    </Stack>
  );
}

function FileSummary({ document, file }: { document: CsvDocument; file: File | null }) {
  return (
    <Group gap="xs">
      <Badge variant="light">{document.rows.length.toLocaleString()} rows</Badge>
      <Badge variant="light">{document.headers.length.toLocaleString()} columns</Badge>
      <Badge variant="light">{delimiterLabel(document.delimiter)} delimiter</Badge>
      {file && <Badge variant="light">{humanSize(file.size)}</Badge>}
    </Group>
  );
}

function ErrorList({ errors }: { errors: string[] }) {
  if (errors.length === 0) return null;
  return <Alert color="red" title="CSV could not be loaded"><Stack gap={3}>{errors.map((error) => <Text size="sm" key={error}>{error}</Text>)}</Stack></Alert>;
}

function DuplicateSummary({ validation, policy, onKeepFirst }: {
  validation: CsvMergeValidation;
  policy: CsvDuplicatePolicy;
  onKeepFirst: () => void;
}) {
  const duplicateKeys = validation.primaryDuplicateKeyCount + validation.secondaryDuplicateKeyCount;
  if (duplicateKeys === 0) return null;
  const resolved = policy === 'keep_first';
  return (
    <Alert color={resolved ? 'green' : 'yellow'} title={resolved ? 'Duplicate keys resolved with keep first' : 'Duplicate keys found'}>
      <Stack gap="xs">
        <Text size="sm">Primary: <b>{validation.primaryDuplicateKeyCount.toLocaleString()}</b> duplicate keys · <b>{validation.primaryDiscardedDuplicateRows.toLocaleString()}</b> later rows {resolved ? 'ignored' : 'to ignore'}</Text>
        <Text size="sm">Secondary: <b>{validation.secondaryDuplicateKeyCount.toLocaleString()}</b> duplicate keys · <b>{validation.secondaryDiscardedDuplicateRows.toLocaleString()}</b> later rows {resolved ? 'ignored' : 'to ignore'}</Text>
        {!resolved && <Button variant="light" color="yellow" w="fit-content" onClick={onKeepFirst}>Keep first entries</Button>}
      </Stack>
    </Alert>
  );
}

function TimestampNormalizationPreview({ primary, secondary, pair }: {
  primary: CsvDocument;
  secondary: CsvDocument;
  pair: CsvKeyPair;
}) {
  const rows = timestampNormalizationPreview(primary, secondary, pair);
  const renderValue = (value: string | null, comparisonKey?: string | null) => (
    <Stack gap={2}>
      <Text size="xs" ff="monospace">{value ?? <Text span c="dimmed">—</Text>}</Text>
      {value !== null && comparisonKey === null && <Text size="xs" c="red">Invalid timestamp · remains unmatched</Text>}
    </Stack>
  );
  return (
    <Paper withBorder p="sm">
      <Stack gap="xs">
        <div>
          <Text fw={600} size="sm">Timestamp normalization preview · {pair.primary} ↔ {pair.secondary}</Text>
          <Text size="xs" c="dimmed">
            Primary values remain unchanged. Matching uses the canonical comparison key shown below. Valid secondary output values adopt the first valid primary timestamp style; required precision is retained.
          </Text>
        </div>
        <Table.ScrollContainer minWidth={1050}>
          <Table striped withTableBorder withColumnBorders>
            <Table.Thead><Table.Tr>
              <Table.Th w={55}>Head row</Table.Th>
              <Table.Th>Primary · unchanged</Table.Th>
              <Table.Th>Primary · comparison key</Table.Th>
              <Table.Th>Secondary · original</Table.Th>
              <Table.Th>Secondary · output preview</Table.Th>
              <Table.Th>Secondary · comparison key</Table.Th>
            </Table.Tr></Table.Thead>
            <Table.Tbody>{rows.map((row) => <Table.Tr key={row.rowNumber}>
              <Table.Td><Text size="xs">{row.rowNumber}</Text></Table.Td>
              <Table.Td>{renderValue(row.primaryOriginal, row.primaryComparisonKey)}</Table.Td>
              <Table.Td><Text size="xs" ff="monospace" c={row.primaryComparisonKey ? undefined : 'dimmed'}>{row.primaryComparisonKey ?? '—'}</Text></Table.Td>
              <Table.Td>{renderValue(row.secondaryOriginal, row.secondaryComparisonKey)}</Table.Td>
              <Table.Td>{renderValue(row.secondaryOutput, row.secondaryComparisonKey)}</Table.Td>
              <Table.Td><Text size="xs" ff="monospace" c={row.secondaryComparisonKey ? undefined : 'dimmed'}>{row.secondaryComparisonKey ?? '—'}</Text></Table.Td>
            </Table.Tr>)}</Table.Tbody>
          </Table>
        </Table.ScrollContainer>
        <Text size="xs" c="dimmed">Rows are shown by their independent source-file positions; the preview does not imply that equal row numbers match.</Text>
      </Stack>
    </Paper>
  );
}

function MatchedPairsPreview({ rows, pairs }: {
  rows: CsvMergeValidation['matchedPairPreview'];
  pairs: CsvKeyPair[];
}) {
  const renderKeys = (values: Array<string | null>, side: 'primary' | 'secondary') => (
    <Stack gap={2}>{values.map((value, index) => (
      <Text size="xs" ff="monospace" key={`${side}:${pairs[index]?.[side]}`}>
        <Text span c="dimmed">{pairs[index]?.[side]}: </Text>{value ?? '—'}
      </Text>
    ))}</Stack>
  );
  return (
    <Stack gap={5}>
      <Text size="sm" fw={600}>First {Math.min(5, rows.length)} matched row pairs</Text>
      {rows.length === 0
        ? <Text size="sm" c="dimmed">No matching row pairs are available for the current key configuration.</Text>
        : <Table.ScrollContainer minWidth={720}>
          <Table striped withTableBorder withColumnBorders>
            <Table.Thead><Table.Tr><Table.Th>Primary row</Table.Th><Table.Th>Primary original key</Table.Th><Table.Th>Secondary row</Table.Th><Table.Th>Secondary original key</Table.Th></Table.Tr></Table.Thead>
            <Table.Tbody>{rows.map((row) => <Table.Tr key={`${row.primaryRowNumber}:${row.secondaryRowNumber}`}>
              <Table.Td>{row.primaryRowNumber}</Table.Td>
              <Table.Td>{renderKeys(row.primaryKeyValues, 'primary')}</Table.Td>
              <Table.Td>{row.secondaryRowNumber}</Table.Td>
              <Table.Td>{renderKeys(row.secondaryKeyValues, 'secondary')}</Table.Td>
            </Table.Tr>)}</Table.Tbody>
          </Table>
        </Table.ScrollContainer>}
      <Text size="xs" c="dimmed">These are the original source values. A pair appears here only when all configured key components match using their selected comparison modes.</Text>
    </Stack>
  );
}

function TimestampAggregationPreview({ validation, method }: {
  validation: CsvMergeValidation;
  method: CsvAggregationMethod;
}) {
  const rows = validation.aggregationPreview;
  return (
    <Paper withBorder p="sm">
      <Stack gap="xs">
        <div>
          <Text fw={600} size="sm">First {Math.min(5, rows.length)} matched minute groups</Text>
          <Text size="xs" c="dimmed">
            Primary timestamps stay unchanged. All secondary samples within [primary timestamp, primary timestamp + 60 seconds) are used, regardless of cadence or second offset; the last column shows the {method} result.
          </Text>
        </div>
        {rows.length === 0
          ? <Text size="sm" c="dimmed">No minute groups match the current key configuration.</Text>
          : <Table.ScrollContainer minWidth={950}>
            <Table striped withTableBorder withColumnBorders>
              <Table.Thead><Table.Tr>
                <Table.Th>Primary row / timestamp</Table.Th>
                <Table.Th>Secondary rows / timestamps</Table.Th>
                <Table.Th>Raw secondary values</Table.Th>
                <Table.Th>Aggregated values</Table.Th>
              </Table.Tr></Table.Thead>
              <Table.Tbody>{rows.map((row) => <Table.Tr key={row.primaryRowNumber}>
                <Table.Td><Text size="xs">Row {row.primaryRowNumber}</Text><Text size="xs" ff="monospace">{row.primaryTimestamp}</Text></Table.Td>
                <Table.Td><Stack gap={2}>{row.secondaryTimestamps.map((timestamp, index) => (
                  <Text size="xs" ff="monospace" key={`${row.secondaryRowNumbers[index]}:${timestamp}`}>
                    Row {row.secondaryRowNumbers[index]} · {timestamp}
                  </Text>
                ))}</Stack></Table.Td>
                <Table.Td><Stack gap={4}>{row.rawValues.map((item) => (
                  <Text size="xs" ff="monospace" key={item.column}><Text span c="dimmed">{item.column}: </Text>{item.values.map((value) => value ?? '—').join(' · ')}</Text>
                ))}</Stack></Table.Td>
                <Table.Td><Stack gap={4}>{row.aggregatedValues.map((item) => (
                  <Text size="xs" ff="monospace" key={item.column}><Text span c="dimmed">{item.column}: </Text>{item.value ?? '—'}</Text>
                ))}</Stack></Table.Td>
              </Table.Tr>)}</Table.Tbody>
            </Table>
          </Table.ScrollContainer>}
      </Stack>
    </Paper>
  );
}

function DownloadDialog({ result, primaryName, secondaryName, opened, onClose }: {
  result: CsvMergeResult;
  primaryName: string;
  secondaryName: string;
  opened: boolean;
  onClose: () => void;
}) {
  const [format, setFormat] = useState<'csv' | 'parquet'>('csv');
  const [name, setName] = useState(defaultMergedFileName(primaryName, secondaryName));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (opened) {
      setName(defaultMergedFileName(primaryName, secondaryName));
      setError(null);
    }
  }, [opened, primaryName, secondaryName]);

  const download = async () => {
    setLoading(true);
    setError(null);
    try {
      const table = csvMergeResultTable(result);
      const filename = normalizedTableDownloadName(name, format, 'merged-data');
      const blob = format === 'csv'
        ? new Blob([tabularTableToCsv(table)], { type: 'text/csv;charset=utf-8' })
        : new Blob([await tabularTableToParquet(table, 'MLTrace CSV merge', true)], { type: 'application/vnd.apache.parquet' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      anchor.style.display = 'none';
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      onClose();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The merged file could not be exported.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal opened={opened} onClose={() => !loading && onClose()} title="Download merged data" centered>
      <Stack gap="md">
        <Text size="sm" c="dimmed">All {result.rows.length.toLocaleString()} merged rows will be exported. Values remain text and empty fields remain null.</Text>
        {error && <Alert color="red" title="Export failed">{error}</Alert>}
        <TextInput label="File name" value={name} onChange={(event) => setName(event.currentTarget.value)} />
        <Select label="File type" value={format} allowDeselect={false} onChange={(value) => setFormat(value === 'parquet' ? 'parquet' : 'csv')} data={[{ value: 'csv', label: 'CSV' }, { value: 'parquet', label: 'Parquet' }]} />
        <Group justify="flex-end">
          <Button variant="default" disabled={loading} onClick={onClose}>Cancel</Button>
          <Button loading={loading} leftSection={<Download size={16} />} onClick={() => void download()}>Download</Button>
        </Group>
      </Stack>
    </Modal>
  );
}

export function CsvMergePage({ active }: { active: boolean }) {
  const [primaryFile, setPrimaryFile] = useState<File | null>(null);
  const [secondaryFile, setSecondaryFile] = useState<File | null>(null);
  const [primary, setPrimary] = useState<CsvDocument | null>(null);
  const [secondary, setSecondary] = useState<CsvDocument | null>(null);
  const [primaryErrors, setPrimaryErrors] = useState<string[]>([]);
  const [secondaryErrors, setSecondaryErrors] = useState<string[]>([]);
  const [primaryLoading, setPrimaryLoading] = useState(false);
  const [secondaryLoading, setSecondaryLoading] = useState(false);
  const [primaryColumns, setPrimaryColumns] = useState<string[]>([]);
  const [secondaryColumns, setSecondaryColumns] = useState<CsvSecondaryColumn[]>([]);
  const [keyPairs, setKeyPairs] = useState<CsvKeyPair[]>([]);
  const [joinType, setJoinType] = useState<CsvJoinType>('left');
  const [duplicatePolicy, setDuplicatePolicy] = useState<CsvDuplicatePolicy>('block');
  const [aggregationMethod, setAggregationMethod] = useState<CsvAggregationMethod>('mean');
  const [result, setResult] = useState<CsvMergeResult | null>(null);
  const [downloadOpened, setDownloadOpened] = useState(false);
  const primaryLoadId = useRef(0);
  const secondaryLoadId = useRef(0);

  const clearSecondary = () => {
    secondaryLoadId.current += 1;
    setSecondaryFile(null);
    setSecondary(null);
    setSecondaryErrors([]);
    setSecondaryColumns([]);
    setKeyPairs([]);
    setDuplicatePolicy('block');
    setAggregationMethod('mean');
    setResult(null);
    setSecondaryLoading(false);
  };
  const reset = () => {
    primaryLoadId.current += 1;
    setPrimaryFile(null);
    setPrimary(null);
    setPrimaryErrors([]);
    setPrimaryColumns([]);
    clearSecondary();
    setJoinType('left');
    setPrimaryLoading(false);
  };

  const loadPrimary = async (file: File | null) => {
    const loadId = ++primaryLoadId.current;
    setPrimaryFile(file);
    setPrimary(null);
    setPrimaryColumns([]);
    setPrimaryErrors([]);
    clearSecondary();
    setPrimaryLoading(false);
    if (!file) return;
    setPrimaryLoading(true);
    const outcome = await parseCsvFile(file);
    if (loadId !== primaryLoadId.current) return;
    setPrimaryLoading(false);
    setPrimaryErrors(outcome.errors);
    setPrimary(outcome.document);
    setPrimaryColumns(outcome.document?.headers ?? []);
  };

  const loadSecondary = async (file: File | null) => {
    const loadId = ++secondaryLoadId.current;
    setSecondaryFile(file);
    setSecondary(null);
    setSecondaryErrors([]);
    setSecondaryColumns([]);
    setKeyPairs([]);
    setDuplicatePolicy('block');
    setAggregationMethod('mean');
    setResult(null);
    setSecondaryLoading(false);
    if (!file || !primary) return;
    setSecondaryLoading(true);
    const outcome = await parseCsvFile(file);
    if (loadId !== secondaryLoadId.current) return;
    setSecondaryLoading(false);
    setSecondaryErrors(outcome.errors);
    setSecondary(outcome.document);
    if (outcome.document) {
      const common = primary.headers.find((header) => outcome.document?.headers.includes(header));
      setKeyPairs(primary.headers.length && outcome.document.headers.length ? [{
        primary: common ?? primary.headers[0],
        secondary: common ?? outcome.document.headers[0],
        comparison: 'exact',
      }] : []);
    }
  };

  const updatePrimaryColumns = (columns: string[]) => { setPrimaryColumns(columns); setResult(null); };
  const updateSecondaryColumns = (columns: CsvSecondaryColumn[]) => { setSecondaryColumns(columns); setResult(null); };
  const updateKeyPairs = (pairs: CsvKeyPair[]) => { setKeyPairs(pairs); setDuplicatePolicy('block'); setResult(null); };
  const updateJoinType = (value: CsvJoinType) => { setJoinType(value); setResult(null); };
  const updateAggregationMethod = (value: CsvAggregationMethod) => { setAggregationMethod(value); setDuplicatePolicy('block'); setResult(null); };
  const config = useMemo(() => ({ primaryColumns, secondaryColumns, keyPairs, joinType, duplicatePolicy, aggregationMethod }), [aggregationMethod, duplicatePolicy, joinType, keyPairs, primaryColumns, secondaryColumns]);
  const validation = useMemo(() => primary && secondary ? validateCsvMerge(primary, secondary, config) : null, [config, primary, secondary]);
  const selectedSecondary = new Set(secondaryColumns.map((column) => column.source));
  const outputNames = [...primaryColumns, ...secondaryColumns.map((column) => column.output)];
  const outputNameCount = (name: string) => outputNames.filter((item) => item === name).length;

  const toggleSecondaryColumn = (source: string, checked: boolean) => {
    if (!secondary) return;
    const next = checked
      ? [...secondaryColumns, { source, output: source }].sort((left, right) => secondary.headers.indexOf(left.source) - secondary.headers.indexOf(right.source))
      : secondaryColumns.filter((column) => column.source !== source);
    updateSecondaryColumns(next);
  };

  const addKeyPair = () => {
    if (!primary || !secondary) return;
    const usedPrimary = new Set(keyPairs.map((pair) => pair.primary));
    const usedSecondary = new Set(keyPairs.map((pair) => pair.secondary));
    const primaryColumn = primary.headers.find((header) => !usedPrimary.has(header));
    const secondaryColumn = secondary.headers.find((header) => !usedSecondary.has(header));
    if (primaryColumn && secondaryColumn) updateKeyPairs([...keyPairs, {
      primary: primaryColumn,
      secondary: secondaryColumn,
      comparison: 'exact',
    }]);
  };

  const calculate = () => {
    if (!primary || !secondary || !validation || validation.errors.length > 0) return;
    try {
      const merged = mergeCsvDocuments(primary, secondary, config);
      setResult(merged);
      notifications.show({ color: 'green', title: 'Merge complete', message: `${merged.rows.length.toLocaleString()} rows are ready to download.` });
    } catch (error) {
      notifications.show({ color: 'red', title: 'Merge failed', message: error instanceof Error ? error.message : 'Unknown error' });
    }
  };

  if (!active) return <div />;
  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-start">
        <div><Title order={2}>CSV Merge</Title><Text c="dimmed">Join two local CSV files using exact keys, normalized timestamps, or minute aggregation. Files never leave your browser.</Text></div>
        <Button variant="default" leftSection={<RotateCcw size={16} />} onClick={reset} disabled={!primaryFile && !secondaryFile}>Reset</Button>
      </Group>

      <Alert color="blue" icon={<FileSpreadsheet size={18} />} title="Local workspace">
        Uploads and merged results are not sent to MLTrace or stored in the project. Reloading the page discards this workspace.
      </Alert>

      <StepCard index={1} color={STEP_COLORS[0]} title="Upload the primary CSV" subtitle="Choose which primary columns should appear in the result. Hidden columns remain available as join keys." complete={Boolean(primary)}>
        <FileInput accept=".csv,text/csv,text/plain" clearable label="Primary CSV" placeholder="Choose a CSV file" leftSection={<Upload size={16} />} value={primaryFile} onChange={(file) => void loadPrimary(file)} disabled={primaryLoading} />
        {primaryLoading && <Text size="sm" c="dimmed">Parsing the primary CSV…</Text>}
        <ErrorList errors={primaryErrors} />
        {primary && <Stack gap="md"><FileSummary document={primary} file={primaryFile} /><Paper withBorder p="sm"><Group justify="space-between" mb="xs"><Text fw={600} size="sm">Primary output columns</Text><Group gap="xs"><Button size="compact-xs" variant="subtle" onClick={() => updatePrimaryColumns(primary.headers)}>All</Button><Button size="compact-xs" variant="subtle" onClick={() => updatePrimaryColumns([])}>None</Button></Group></Group><SimpleGrid cols={{ base: 1, sm: 2, md: 3 }}>{primary.headers.map((header) => <Checkbox key={header} label={header} checked={primaryColumns.includes(header)} onChange={(event) => updatePrimaryColumns(event.currentTarget.checked ? [...primaryColumns, header].sort((a, b) => primary.headers.indexOf(a) - primary.headers.indexOf(b)) : primaryColumns.filter((item) => item !== header))} />)}</SimpleGrid></Paper><DataPreview headers={primary.headers} rows={primary.rows} label="Primary data preview" /></Stack>}
      </StepCard>

      <StepCard index={2} color={STEP_COLORS[1]} title="Upload the secondary CSV" subtitle="Select only the columns that should be added to the primary data." complete={Boolean(secondary)}>
        {!primary && <Alert color="gray">Load a valid primary CSV first.</Alert>}
        <FileInput accept=".csv,text/csv,text/plain" clearable label="Secondary CSV" placeholder="Choose a CSV file" leftSection={<Upload size={16} />} value={secondaryFile} onChange={(file) => void loadSecondary(file)} disabled={!primary || secondaryLoading} />
        {secondaryLoading && <Text size="sm" c="dimmed">Parsing the secondary CSV…</Text>}
        <ErrorList errors={secondaryErrors} />
        {secondary && <Stack gap="md"><FileSummary document={secondary} file={secondaryFile} /><Paper withBorder p="sm"><Text fw={600} size="sm" mb="xs">Columns to add</Text><Table.ScrollContainer minWidth={620}><Table withTableBorder withColumnBorders><Table.Thead><Table.Tr><Table.Th>Include</Table.Th><Table.Th>Source column</Table.Th><Table.Th>Output column name</Table.Th></Table.Tr></Table.Thead><Table.Tbody>{secondary.headers.map((header) => { const selection = secondaryColumns.find((column) => column.source === header); const conflict = selection && (selection.output.trim() === '' || outputNameCount(selection.output) > 1); return <Table.Tr key={header}><Table.Td><Checkbox aria-label={`Include ${header}`} checked={selectedSecondary.has(header)} onChange={(event) => toggleSecondaryColumn(header, event.currentTarget.checked)} /></Table.Td><Table.Td>{header}</Table.Td><Table.Td>{selection ? <TextInput size="xs" value={selection.output} error={conflict ? 'Enter a unique output name.' : undefined} onChange={(event) => updateSecondaryColumns(secondaryColumns.map((column) => column.source === header ? { ...column, output: event.currentTarget.value } : column))} /> : <Text size="xs" c="dimmed">Not selected</Text>}</Table.Td></Table.Tr>; })}</Table.Tbody></Table></Table.ScrollContainer></Paper><DataPreview headers={secondary.headers} rows={secondary.rows} label="Secondary data preview" /></Stack>}
      </StepCard>

      <StepCard index={3} color={STEP_COLORS[2]} title="Configure the keyed join" subtitle="Compare exact values, normalize timestamp formats, or aggregate all samples within each primary [timestamp, timestamp + 60s) window." complete={Boolean(validation && validation.errors.length === 0)}>
        {(!primary || !secondary) && <Alert color="gray">Load both CSV files to configure the merge.</Alert>}
        {primary && secondary && <Stack gap="md">
          <Select label="Join type" value={joinType} allowDeselect={false} onChange={(value) => updateJoinType((value as CsvJoinType | null) ?? 'left')} data={[{ value: 'left', label: 'Left join · keep every primary row' }, { value: 'inner', label: 'Inner join · keep matching rows only' }, { value: 'full', label: 'Full outer join · keep every row from both files' }]} />
          <Table.ScrollContainer minWidth={850}>
            <Table withTableBorder withColumnBorders>
              <Table.Thead><Table.Tr><Table.Th>Primary key column</Table.Th><Table.Th>Secondary key column</Table.Th><Table.Th>Comparison</Table.Th><Table.Th w={55} /></Table.Tr></Table.Thead>
              <Table.Tbody>{keyPairs.map((pair, index) => {
                const suggestTimestamp = timestampNormalizationSuggested(primary, secondary, pair);
                return <Table.Tr key={`${index}:${pair.primary}:${pair.secondary}:${pair.comparison}`}>
                  <Table.Td><Select searchable value={pair.primary} onChange={(value) => value && updateKeyPairs(keyPairs.map((item, itemIndex) => itemIndex === index ? { ...item, primary: value } : item))} data={primary.headers.map((header) => ({ value: header, label: header, disabled: keyPairs.some((item, itemIndex) => itemIndex !== index && item.primary === header) }))} /></Table.Td>
                  <Table.Td><Select searchable value={pair.secondary} onChange={(value) => value && updateKeyPairs(keyPairs.map((item, itemIndex) => itemIndex === index ? { ...item, secondary: value } : item))} data={secondary.headers.map((header) => ({ value: header, label: header, disabled: keyPairs.some((item, itemIndex) => itemIndex !== index && item.secondary === header) }))} /></Table.Td>
                  <Table.Td>
                    <Stack gap={4}>
                      <Select aria-label={`Comparison for ${pair.primary} and ${pair.secondary}`} value={pair.comparison} allowDeselect={false} onChange={(value) => updateKeyPairs(keyPairs.map((item, itemIndex) => itemIndex === index ? {
                        ...item,
                        comparison: value === 'timestamp' || value === 'timestamp_aggregation' ? value : 'exact',
                      } : item))} data={[{ value: 'exact', label: 'Exact text' }, { value: 'timestamp', label: 'Timestamp normalization' }, { value: 'timestamp_aggregation', label: 'Timestamp aggregation · [primary, +60s)' }]} />
                      {suggestTimestamp && <Stack gap={3}>
                        <Text size="xs" c="blue">Both columns look temporal, but their formats differ.</Text>
                        <Button size="compact-xs" variant="light" color="blue" onClick={() => updateKeyPairs(keyPairs.map((item, itemIndex) => itemIndex === index ? { ...item, comparison: 'timestamp' } : item))}>Normalize timestamps</Button>
                      </Stack>}
                    </Stack>
                  </Table.Td>
                  <Table.Td><Button variant="subtle" color="red" size="compact-sm" aria-label="Remove key mapping" onClick={() => updateKeyPairs(keyPairs.filter((_, itemIndex) => itemIndex !== index))}><Trash2 size={15} /></Button></Table.Td>
                </Table.Tr>;
              })}</Table.Tbody>
            </Table>
          </Table.ScrollContainer>
          {keyPairs.filter((pair) => pair.comparison === 'timestamp').map((pair) => (
            <TimestampNormalizationPreview
              key={`${pair.primary}:${pair.secondary}`}
              primary={primary}
              secondary={secondary}
              pair={pair}
            />
          ))}
          {keyPairs.some((pair) => pair.comparison === 'timestamp_aggregation') && <Select
            label="Aggregation for all selected secondary columns"
            description="One method is applied to every included secondary output column. Mean, min, and max require numeric non-empty values."
            value={aggregationMethod}
            allowDeselect={false}
            onChange={(value) => updateAggregationMethod((value as CsvAggregationMethod | null) ?? 'mean')}
            data={[{ value: 'mean', label: 'Mean' }, { value: 'min', label: 'Min' }, { value: 'max', label: 'Max' }, { value: 'first', label: 'First' }, { value: 'last', label: 'Last' }]}
          />}
          <Button variant="light" leftSection={<Plus size={15} />} onClick={addKeyPair} disabled={keyPairs.length >= Math.min(primary.headers.length, secondary.headers.length)}>Add key column</Button>
          {validation && (() => {
            const effectivePrimaryRows = validation.matchedRows + validation.unmatchedPrimaryRows;
            const matchPercent = effectivePrimaryRows === 0 ? 0 : (validation.matchedRows / effectivePrimaryRows) * 100;
            const matchColor = validation.matchedRows === 0 ? 'red' : matchPercent < 50 ? 'yellow' : 'green';
            return <Alert color={matchColor} title="Live key match">
              <Stack gap="xs">
                <Group justify="space-between"><Text fw={600}>Matched primary rows: {validation.matchedRows.toLocaleString()} / {effectivePrimaryRows.toLocaleString()}</Text><Badge color={matchColor}>{matchPercent.toFixed(1)}%</Badge></Group>
                <SimpleGrid cols={{ base: 1, sm: 3 }}>
                  <Text size="sm">Unmatched primary: <b>{validation.unmatchedPrimaryRows.toLocaleString()}</b></Text>
                  <Text size="sm">Unmatched secondary: <b>{validation.unmatchedSecondaryRows.toLocaleString()}</b></Text>
                  <Text size="sm">Expected result rows: <b>{validation.expectedRows.toLocaleString()}</b></Text>
                </SimpleGrid>
                {keyPairs.some((pair) => pair.comparison === 'timestamp_aggregation') ? <>
                  <SimpleGrid cols={{ base: 1, sm: 4 }}>
                    <Text size="sm">Used secondary rows: <b>{validation.aggregationUsedSecondaryRows.toLocaleString()}</b></Text>
                    <Text size="sm">Samples / minute · min: <b>{validation.aggregationMinSamples?.toLocaleString() ?? 'N/A'}</b></Text>
                    <Text size="sm">max: <b>{validation.aggregationMaxSamples?.toLocaleString() ?? 'N/A'}</b></Text>
                    <Text size="sm">average: <b>{validation.aggregationAverageSamples?.toFixed(2) ?? 'N/A'}</b></Text>
                  </SimpleGrid>
                  <TimestampAggregationPreview validation={validation} method={aggregationMethod} />
                </> : <MatchedPairsPreview rows={validation.matchedPairPreview} pairs={keyPairs} />}
              </Stack>
            </Alert>;
          })()}
        </Stack>}
      </StepCard>

      <StepCard index={4} color={STEP_COLORS[3]} title="Validate and merge" subtitle="The preview is paginated; validation and merge always use all rows." complete={Boolean(result)}>
        {!validation && <Alert color="gray">Complete the previous steps to validate the merge.</Alert>}
        {validation && <Stack gap="md">
          {(validation.primaryMissingKeys > 0 || validation.secondaryMissingKeys > 0) && <Alert color="yellow" title="Empty keys remain unmatched">Primary: {validation.primaryMissingKeys.toLocaleString()} · Secondary: {validation.secondaryMissingKeys.toLocaleString()}</Alert>}
          {(validation.primaryInvalidTimestampRows > 0 || validation.secondaryInvalidTimestampRows > 0) && <Alert color="yellow" title="Invalid timestamps remain unmatched">Primary: {validation.primaryInvalidTimestampRows.toLocaleString()} · Secondary: {validation.secondaryInvalidTimestampRows.toLocaleString()}. Original text values are preserved in the output.</Alert>}
          <DuplicateSummary validation={validation} policy={duplicatePolicy} onKeepFirst={() => { setDuplicatePolicy('keep_first'); setResult(null); }} />
          {validation.errors.filter((error) => !error.includes('not unique')).map((error) => <Alert color="red" key={error}>{error}</Alert>)}
          <Button onClick={calculate} disabled={validation.errors.length > 0} leftSection={<FileSpreadsheet size={16} />}>Merge all rows</Button>
        </Stack>}
      </StepCard>

      {result && primary && secondary && <StepCard index={5} color={STEP_COLORS[4]} title="Merged result" subtitle="Review the merged rows, then download the full result as CSV or Parquet." complete><Group justify="space-between"><Group><Badge color="green">{result.rows.length.toLocaleString()} rows</Badge><Badge color="green">{result.headers.length.toLocaleString()} columns</Badge>{result.validation.primaryDiscardedDuplicateRows > 0 && <Badge color="yellow">Primary duplicate rows ignored: {result.validation.primaryDiscardedDuplicateRows.toLocaleString()}</Badge>}{result.validation.secondaryDiscardedDuplicateRows > 0 && <Badge color="yellow">Secondary duplicate rows ignored: {result.validation.secondaryDiscardedDuplicateRows.toLocaleString()}</Badge>}</Group><Button leftSection={<Download size={16} />} onClick={() => setDownloadOpened(true)}>Download</Button></Group><DataPreview headers={result.headers} rows={result.rows} label="Merged data preview" /><DownloadDialog result={result} primaryName={primary.fileName} secondaryName={secondary.fileName} opened={downloadOpened} onClose={() => setDownloadOpened(false)} /></StepCard>}
    </Stack>
  );
}

export default CsvMergePage;
