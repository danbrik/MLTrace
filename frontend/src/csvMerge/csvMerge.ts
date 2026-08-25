import Papa, { type ParseError, type ParseResult } from 'papaparse';
import type { TabularTable } from '../lib/tabularExport';

export type CsvCell = string | null;

export type CsvDocument = {
  fileName: string;
  delimiter: string;
  headers: string[];
  rows: CsvCell[][];
};

export type CsvParseOutcome = {
  document: CsvDocument | null;
  errors: string[];
};

export type CsvJoinType = 'left' | 'inner' | 'full';
export type CsvDuplicatePolicy = 'block' | 'keep_first';
export type CsvKeyComparison = 'exact' | 'timestamp' | 'timestamp_aggregation';
export type CsvAggregationMethod = 'mean' | 'min' | 'max' | 'first' | 'last';

export type CsvKeyPair = {
  primary: string;
  secondary: string;
  comparison: CsvKeyComparison;
};

export type CsvSecondaryColumn = {
  source: string;
  output: string;
};

export type CsvMergeConfig = {
  primaryColumns: string[];
  secondaryColumns: CsvSecondaryColumn[];
  keyPairs: CsvKeyPair[];
  joinType: CsvJoinType;
  duplicatePolicy: CsvDuplicatePolicy;
  aggregationMethod?: CsvAggregationMethod;
};

export type CsvDuplicateKey = {
  key: string;
  rows: number[];
};

export type CsvMergeValidation = {
  errors: string[];
  primaryDuplicateKeys: CsvDuplicateKey[];
  secondaryDuplicateKeys: CsvDuplicateKey[];
  primaryDuplicateKeyCount: number;
  secondaryDuplicateKeyCount: number;
  primaryDiscardedDuplicateRows: number;
  secondaryDiscardedDuplicateRows: number;
  primaryMissingKeys: number;
  secondaryMissingKeys: number;
  primaryInvalidTimestampRows: number;
  secondaryInvalidTimestampRows: number;
  matchedRows: number;
  unmatchedPrimaryRows: number;
  unmatchedSecondaryRows: number;
  expectedRows: number;
  matchedPairPreview: CsvMatchedPairPreviewRow[];
  aggregationUsedSecondaryRows: number;
  aggregationMinSamples: number | null;
  aggregationMaxSamples: number | null;
  aggregationAverageSamples: number | null;
  aggregationNumericIssues: CsvAggregationNumericIssue[];
  aggregationPreview: CsvAggregationPreviewRow[];
};

export type CsvMergeResult = {
  headers: string[];
  rows: CsvCell[][];
  validation: CsvMergeValidation;
};

export type CsvTimestampPreviewRow = {
  rowNumber: number;
  primaryOriginal: CsvCell;
  primaryComparisonKey: string | null;
  secondaryOriginal: CsvCell;
  secondaryOutput: CsvCell;
  secondaryComparisonKey: string | null;
};

export type CsvMatchedPairPreviewRow = {
  primaryRowNumber: number;
  primaryKeyValues: CsvCell[];
  secondaryRowNumber: number;
  secondaryKeyValues: CsvCell[];
};

export type CsvAggregationNumericIssue = {
  column: string;
  invalidValues: number;
};

export type CsvAggregationPreviewRow = {
  primaryRowNumber: number;
  primaryTimestamp: string;
  secondaryRowNumbers: number[];
  secondaryTimestamps: string[];
  rawValues: Array<{ column: string; values: CsvCell[] }>;
  aggregatedValues: Array<{ column: string; value: CsvCell }>;
};

function parseErrorMessage(error: ParseError): string {
  const row = typeof error.row === 'number' ? ` at data row ${error.row + 1}` : '';
  return `${error.message}${row}`;
}

function finalizeParse(result: ParseResult<string[]>, fileName: string): CsvParseOutcome {
  const structuralErrors = result.errors.filter((error) => error.code !== 'UndetectableDelimiter');
  const errors = structuralErrors.map(parseErrorMessage);
  const matrix = result.data.map((row) => row.map((value) => String(value)));
  if (matrix.length === 0) return { document: null, errors: ['The CSV is empty.'] };
  const headers = matrix[0].map((value, index) => index === 0 ? value.replace(/^\uFEFF/, '') : value);
  if (headers.length === 0 || headers.every((header) => header.trim() === '')) {
    errors.push('The CSV has no header row.');
  }
  const emptyHeaders = headers.flatMap((header, index) => header.trim() === '' ? [index + 1] : []);
  if (emptyHeaders.length > 0) errors.push(`Empty column names at positions: ${emptyHeaders.join(', ')}.`);
  const seen = new Set<string>();
  const duplicates = new Set<string>();
  headers.forEach((header) => {
    if (seen.has(header)) duplicates.add(header);
    seen.add(header);
  });
  if (duplicates.size > 0) errors.push(`Duplicate column names: ${[...duplicates].join(', ')}.`);
  if (matrix.some((row) => row.some((value) => value.includes('\uFFFD')))) {
    errors.push('The file contains invalid UTF-8 characters.');
  }
  const dataRows = matrix.slice(1);
  dataRows.forEach((row, index) => {
    if (row.length !== headers.length) {
      errors.push(`Data row ${index + 1} has ${row.length} fields; expected ${headers.length}.`);
    }
  });
  if (errors.length > 0) return { document: null, errors: [...new Set(errors)] };
  return {
    document: {
      fileName,
      delimiter: result.meta.delimiter || ',',
      headers,
      rows: dataRows.map((row) => row.map((value) => value === '' ? null : value)),
    },
    errors: [],
  };
}

const PARSE_OPTIONS = {
  delimiter: '',
  dynamicTyping: false,
  skipEmptyLines: 'greedy' as const,
};

export function parseCsvText(text: string, fileName = 'upload.csv'): CsvParseOutcome {
  const result = Papa.parse<string[]>(text, PARSE_OPTIONS);
  return finalizeParse(result, fileName);
}

export function parseCsvFile(file: File): Promise<CsvParseOutcome> {
  return new Promise((resolve) => {
    Papa.parse<string[]>(file, {
      ...PARSE_OPTIONS,
      worker: true,
      complete: (result) => resolve(finalizeParse(result, file.name)),
      error: (error) => resolve({ document: null, errors: [error.message] }),
    });
  });
}

function columnIndex(document: CsvDocument, name: string): number {
  return document.headers.indexOf(name);
}

type TimestampStyle = {
  dateSeparator: '-' | '.';
  dateTimeSeparator: ' ' | 'T';
  includeTime: boolean;
  includeSeconds: boolean;
  fractionDigits: number;
};

export type ParsedCsvTimestamp = {
  canonical: string;
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  fraction: string;
  style: TimestampStyle;
};

const TIMESTAMP_PATTERN = /^(\d{4})([-.])(\d{2})\2(\d{2})(?:([ T])(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?)?$/;

function pad(value: number, width = 2): string {
  return String(value).padStart(width, '0');
}

function daysInMonth(year: number, month: number): number {
  if (month === 2) return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 29 : 28;
  return [4, 6, 9, 11].includes(month) ? 30 : 31;
}

export function parseCsvTimestamp(value: string): ParsedCsvTimestamp | null {
  const match = TIMESTAMP_PATTERN.exec(value);
  if (!match) return null;
  const [, yearText, dateSeparator, monthText, dayText, dateTimeSeparator, hourText, minuteText, secondText, fractionText] = match;
  const year = Number(yearText);
  const month = Number(monthText);
  const day = Number(dayText);
  const hour = hourText === undefined ? 0 : Number(hourText);
  const minute = minuteText === undefined ? 0 : Number(minuteText);
  const second = secondText === undefined ? 0 : Number(secondText);
  if (month < 1 || month > 12 || day < 1 || day > daysInMonth(year, month)
    || hour > 23 || minute > 59 || second > 59) return null;
  const fraction = (fractionText ?? '').replace(/0+$/, '');
  const canonical = `${pad(year, 4)}-${pad(month)}-${pad(day)}T${pad(hour)}:${pad(minute)}:${pad(second)}${fraction ? `.${fraction}` : ''}`;
  return {
    canonical,
    year,
    month,
    day,
    hour,
    minute,
    second,
    fraction,
    style: {
      dateSeparator: dateSeparator as '-' | '.',
      dateTimeSeparator: (dateTimeSeparator ?? ' ') as ' ' | 'T',
      includeTime: hourText !== undefined,
      includeSeconds: secondText !== undefined,
      fractionDigits: fractionText?.length ?? 0,
    },
  };
}

function timestampStyleSignature(value: ParsedCsvTimestamp): string {
  return `${value.style.dateSeparator}|${value.style.dateTimeSeparator}|${value.style.includeTime}|${value.style.includeSeconds}|${value.style.fractionDigits}`;
}

function sampledTimestampProfile(document: CsvDocument, column: string): { ratio: number; styles: Set<string> } {
  const index = columnIndex(document, column);
  const values: string[] = [];
  for (const row of document.rows) {
    const value = row[index];
    if (value !== null && value !== '') values.push(value);
    if (values.length === 100) break;
  }
  const parsed = values.map(parseCsvTimestamp).filter((value): value is ParsedCsvTimestamp => value !== null);
  return {
    ratio: values.length === 0 ? 0 : parsed.length / values.length,
    styles: new Set(parsed.map(timestampStyleSignature)),
  };
}

export function timestampNormalizationSuggested(
  primary: CsvDocument,
  secondary: CsvDocument,
  pair: CsvKeyPair,
): boolean {
  if (pair.comparison !== 'exact') return false;
  const primaryProfile = sampledTimestampProfile(primary, pair.primary);
  const secondaryProfile = sampledTimestampProfile(secondary, pair.secondary);
  if (primaryProfile.ratio < 0.8 || secondaryProfile.ratio < 0.8) return false;
  return [...primaryProfile.styles].some((style) => !secondaryProfile.styles.has(style))
    || [...secondaryProfile.styles].some((style) => !primaryProfile.styles.has(style));
}

type KeySide = 'primary' | 'secondary';
type CompositeKey = { key: string | null; reason: 'valid' | 'missing' | 'invalid_timestamp' };

function compositeKey(
  row: CsvCell[],
  document: CsvDocument,
  pairs: CsvKeyPair[],
  side: KeySide,
): CompositeKey {
  const values = pairs.map((pair) => row[columnIndex(document, pair[side])] ?? null);
  if (values.some((value) => value === null || value === '')) return { key: null, reason: 'missing' };
  const normalized: string[] = [];
  for (let index = 0; index < pairs.length; index += 1) {
    const value = values[index] as string;
    if (pairs[index].comparison === 'exact') {
      normalized.push(value);
      continue;
    }
    const parsed = parseCsvTimestamp(value);
    if (!parsed) return { key: null, reason: 'invalid_timestamp' };
    normalized.push(parsed.canonical);
  }
  return { key: JSON.stringify(normalized), reason: 'valid' };
}

function keyIndex(
  document: CsvDocument,
  pairs: CsvKeyPair[],
  side: KeySide,
): {
  rowsByKey: Map<string, number>;
  duplicates: CsvDuplicateKey[];
  discardedDuplicateRows: number;
  retainedRowIndexes: number[];
  missing: number;
  invalidTimestampRows: number;
} {
  const allRows = new Map<string, number[]>();
  const retainedRowIndexes: number[] = [];
  let missing = 0;
  let invalidTimestampRows = 0;
  document.rows.forEach((row, index) => {
    const value = compositeKey(row, document, pairs, side);
    if (value.key === null) {
      if (value.reason === 'missing') missing += 1;
      else invalidTimestampRows += 1;
      retainedRowIndexes.push(index);
      return;
    }
    const rows = allRows.get(value.key) ?? [];
    if (rows.length === 0) retainedRowIndexes.push(index);
    rows.push(index);
    allRows.set(value.key, rows);
  });
  const duplicates = [...allRows.entries()].filter(([, rows]) => rows.length > 1).map(([key, rows]) => ({
    key: (JSON.parse(key) as string[]).join(' | '),
    rows: rows.map((row) => row + 1),
  }));
  return {
    rowsByKey: new Map([...allRows.entries()].map(([key, rows]) => [key, rows[0]])),
    duplicates,
    discardedDuplicateRows: duplicates.reduce((total, item) => total + item.rows.length - 1, 0),
    retainedRowIndexes,
    missing,
    invalidTimestampRows,
  };
}

function firstPrimaryTimestampStyles(primary: CsvDocument, pairs: CsvKeyPair[]): Map<string, TimestampStyle> {
  const styles = new Map<string, TimestampStyle>();
  pairs.filter((pair) => pair.comparison !== 'exact').forEach((pair) => {
    const index = columnIndex(primary, pair.primary);
    for (const row of primary.rows) {
      const value = row[index];
      if (value === null) continue;
      const parsed = parseCsvTimestamp(value);
      if (parsed) {
        styles.set(pair.secondary, parsed.style);
        break;
      }
    }
  });
  return styles;
}

function formatTimestamp(value: string, style: TimestampStyle): string {
  const parsed = parseCsvTimestamp(value);
  if (!parsed) return value;
  const fractionDigits = Math.max(style.fractionDigits, parsed.fraction.length);
  const includeSeconds = style.includeSeconds || parsed.second !== 0 || fractionDigits > 0;
  const date = `${pad(parsed.year, 4)}${style.dateSeparator}${pad(parsed.month)}${style.dateSeparator}${pad(parsed.day)}`;
  const includeTime = style.includeTime || parsed.hour !== 0 || parsed.minute !== 0 || includeSeconds;
  if (!includeTime) return date;
  let time = `${pad(parsed.hour)}:${pad(parsed.minute)}`;
  if (includeSeconds) time += `:${pad(parsed.second)}`;
  if (fractionDigits > 0) time += `.${parsed.fraction.padEnd(fractionDigits, '0')}`;
  return `${date}${style.dateTimeSeparator}${time}`;
}

function secondaryOutputValue(
  row: CsvCell[],
  document: CsvDocument,
  source: string,
  primaryStyles: Map<string, TimestampStyle>,
): CsvCell {
  const value = row[columnIndex(document, source)] ?? null;
  const style = primaryStyles.get(source);
  return value !== null && style ? formatTimestamp(value, style) : value;
}

export function timestampNormalizationPreview(
  primary: CsvDocument,
  secondary: CsvDocument,
  pair: CsvKeyPair,
  limit = 5,
): CsvTimestampPreviewRow[] {
  const primaryIndex = columnIndex(primary, pair.primary);
  const secondaryIndex = columnIndex(secondary, pair.secondary);
  if (primaryIndex < 0 || secondaryIndex < 0 || limit <= 0) return [];
  const primaryStyles = firstPrimaryTimestampStyles(primary, [{ ...pair, comparison: 'timestamp' }]);
  const style = primaryStyles.get(pair.secondary);
  const rowCount = Math.min(Math.max(primary.rows.length, secondary.rows.length), Math.floor(limit));
  return Array.from({ length: rowCount }, (_, index) => {
    const primaryOriginal = primary.rows[index]?.[primaryIndex] ?? null;
    const secondaryOriginal = secondary.rows[index]?.[secondaryIndex] ?? null;
    const primaryParsed = primaryOriginal === null ? null : parseCsvTimestamp(primaryOriginal);
    const secondaryParsed = secondaryOriginal === null ? null : parseCsvTimestamp(secondaryOriginal);
    return {
      rowNumber: index + 1,
      primaryOriginal,
      primaryComparisonKey: primaryParsed?.canonical ?? null,
      secondaryOriginal,
      secondaryOutput: secondaryOriginal !== null && style ? formatTimestamp(secondaryOriginal, style) : secondaryOriginal,
      secondaryComparisonKey: secondaryParsed?.canonical ?? null,
    };
  });
}

function unique(values: string[]): boolean {
  return new Set(values).size === values.length;
}

const AGGREGATION_WINDOW_MS = 60_000;

type AggregationMatchPlan = {
  secondaryRowsByPrimary: Map<number, number[]>;
  matchedSecondary: Set<number>;
  overlappingPrimaryWindows: boolean;
};

function aggregationPair(config: CsvMergeConfig): CsvKeyPair | null {
  return config.keyPairs.find((pair) => pair.comparison === 'timestamp_aggregation') ?? null;
}

function timestampMilliseconds(value: CsvCell): number | null {
  if (value === null) return null;
  const parsed = parseCsvTimestamp(value);
  if (!parsed) return null;
  const fraction = parsed.fraction ? Number(`0.${parsed.fraction}`) * 1000 : 0;
  return Date.UTC(parsed.year, parsed.month - 1, parsed.day, parsed.hour, parsed.minute, parsed.second) + fraction;
}

function groupingKey(
  row: CsvCell[],
  document: CsvDocument,
  pairs: CsvKeyPair[],
  side: KeySide,
): string | null {
  const groupingPairs = pairs.filter((pair) => pair.comparison !== 'timestamp_aggregation');
  if (groupingPairs.length === 0) return JSON.stringify([]);
  return compositeKey(row, document, groupingPairs, side).key;
}

function buildAggregationMatchPlan(
  primary: CsvDocument,
  secondary: CsvDocument,
  config: CsvMergeConfig,
  primaryRowIndexes: number[],
  secondaryRowIndexes: number[],
): AggregationMatchPlan {
  const pair = aggregationPair(config);
  if (!pair) return { secondaryRowsByPrimary: new Map(), matchedSecondary: new Set(), overlappingPrimaryWindows: false };
  const primaryTimeIndex = columnIndex(primary, pair.primary);
  const secondaryTimeIndex = columnIndex(secondary, pair.secondary);
  const anchorsByGroup = new Map<string, Array<{ rowIndex: number; start: number }>>();
  primaryRowIndexes.forEach((rowIndex) => {
    const row = primary.rows[rowIndex];
    const group = groupingKey(row, primary, config.keyPairs, 'primary');
    const start = timestampMilliseconds(row[primaryTimeIndex] ?? null);
    if (group === null || start === null) return;
    const anchors = anchorsByGroup.get(group) ?? [];
    anchors.push({ rowIndex, start });
    anchorsByGroup.set(group, anchors);
  });
  let overlappingPrimaryWindows = false;
  anchorsByGroup.forEach((anchors) => {
    anchors.sort((left, right) => left.start - right.start || left.rowIndex - right.rowIndex);
    if (anchors.some((anchor, index) => index > 0 && anchor.start < anchors[index - 1].start + AGGREGATION_WINDOW_MS)) {
      overlappingPrimaryWindows = true;
    }
  });
  const secondaryRowsByPrimary = new Map<number, number[]>();
  const matchedSecondary = new Set<number>();
  secondaryRowIndexes.forEach((rowIndex) => {
    const row = secondary.rows[rowIndex];
    const group = groupingKey(row, secondary, config.keyPairs, 'secondary');
    const timestamp = timestampMilliseconds(row[secondaryTimeIndex] ?? null);
    if (group === null || timestamp === null) return;
    const anchors = anchorsByGroup.get(group);
    if (!anchors) return;
    let low = 0;
    let high = anchors.length - 1;
    let candidate = -1;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      if (anchors[middle].start <= timestamp) {
        candidate = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    if (candidate < 0 || timestamp >= anchors[candidate].start + AGGREGATION_WINDOW_MS) return;
    const primaryRowIndex = anchors[candidate].rowIndex;
    const matches = secondaryRowsByPrimary.get(primaryRowIndex) ?? [];
    matches.push(rowIndex);
    secondaryRowsByPrimary.set(primaryRowIndex, matches);
    matchedSecondary.add(rowIndex);
  });
  secondaryRowsByPrimary.forEach((rows) => rows.sort((left, right) => {
    const leftTime = timestampMilliseconds(secondary.rows[left][secondaryTimeIndex] ?? null) ?? 0;
    const rightTime = timestampMilliseconds(secondary.rows[right][secondaryTimeIndex] ?? null) ?? 0;
    return leftTime - rightTime || left - right;
  }));
  return { secondaryRowsByPrimary, matchedSecondary, overlappingPrimaryWindows };
}

const NUMERIC_VALUE = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

function numericValue(value: CsvCell): number | null {
  if (value === null) return null;
  const normalized = value.trim();
  if (!NUMERIC_VALUE.test(normalized)) return Number.NaN;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function numericAggregationIssues(
  secondary: CsvDocument,
  config: CsvMergeConfig,
): CsvAggregationNumericIssue[] {
  if (!aggregationPair(config) || !['mean', 'min', 'max'].includes(config.aggregationMethod ?? 'mean')) return [];
  return config.secondaryColumns.flatMap((column) => {
    const index = columnIndex(secondary, column.source);
    const invalidValues = secondary.rows.reduce((count, row) => {
      const value = numericValue(row[index] ?? null);
      return count + (value !== null && Number.isNaN(value) ? 1 : 0);
    }, 0);
    return invalidValues > 0 ? [{ column: column.source, invalidValues }] : [];
  });
}

function aggregateSecondaryValue(
  rowIndexes: number[],
  secondary: CsvDocument,
  source: string,
  method: CsvAggregationMethod,
  primaryStyles: Map<string, TimestampStyle>,
): CsvCell {
  if (rowIndexes.length === 0) return null;
  if (method === 'first' || method === 'last') {
    const rowIndex = method === 'first' ? rowIndexes[0] : rowIndexes[rowIndexes.length - 1];
    return secondaryOutputValue(secondary.rows[rowIndex], secondary, source, primaryStyles);
  }
  const column = columnIndex(secondary, source);
  const values = rowIndexes
    .map((rowIndex) => numericValue(secondary.rows[rowIndex][column] ?? null))
    .filter((value): value is number => value !== null && !Number.isNaN(value));
  if (values.length === 0) return null;
  const result = method === 'mean'
    ? values.reduce((total, value) => total + value, 0) / values.length
    : method === 'min' ? Math.min(...values) : Math.max(...values);
  return Number.isFinite(result) ? String(result) : null;
}

export function validateCsvMerge(
  primary: CsvDocument,
  secondary: CsvDocument,
  config: CsvMergeConfig,
): CsvMergeValidation {
  const errors: string[] = [];
  const aggregationPairs = config.keyPairs.filter((pair) => pair.comparison === 'timestamp_aggregation');
  if (config.keyPairs.length === 0) errors.push('Add at least one key-column mapping.');
  if (aggregationPairs.length > 1) errors.push('Configure only one Timestamp aggregation key.');
  if (!unique(config.keyPairs.map((pair) => pair.primary))) errors.push('Each primary key column may only be used once.');
  if (!unique(config.keyPairs.map((pair) => pair.secondary))) errors.push('Each secondary key column may only be used once.');
  if (config.primaryColumns.some((name) => !primary.headers.includes(name))) errors.push('A selected primary output column no longer exists.');
  if (config.secondaryColumns.some((column) => !secondary.headers.includes(column.source))) errors.push('A selected secondary output column no longer exists.');
  if (config.secondaryColumns.length === 0) errors.push('Select at least one secondary column to merge.');
  if (config.keyPairs.some((pair) => !primary.headers.includes(pair.primary) || !secondary.headers.includes(pair.secondary))) {
    errors.push('Every key mapping must reference an existing column in both files.');
  }
  const outputHeaders = [...config.primaryColumns, ...config.secondaryColumns.map((column) => column.output)];
  if (outputHeaders.length === 0) errors.push('Select at least one output column.');
  if (outputHeaders.some((header) => header.trim() === '')) errors.push('Output column names cannot be empty.');
  if (!unique(outputHeaders)) errors.push('Every output column name must be unique. Rename conflicting secondary columns.');

  const primaryIndexes = config.keyPairs.map((pair) => columnIndex(primary, pair.primary));
  const secondaryIndexes = config.keyPairs.map((pair) => columnIndex(secondary, pair.secondary));
  if (primaryIndexes.some((index) => index < 0) || secondaryIndexes.some((index) => index < 0) || config.keyPairs.length === 0) {
    return {
      errors: [...new Set(errors)], primaryDuplicateKeys: [], secondaryDuplicateKeys: [], primaryMissingKeys: 0,
      primaryDuplicateKeyCount: 0, secondaryDuplicateKeyCount: 0,
      primaryDiscardedDuplicateRows: 0, secondaryDiscardedDuplicateRows: 0,
      primaryInvalidTimestampRows: 0, secondaryInvalidTimestampRows: 0,
      secondaryMissingKeys: 0, matchedRows: 0, unmatchedPrimaryRows: primary.rows.length,
      unmatchedSecondaryRows: secondary.rows.length, expectedRows: 0, matchedPairPreview: [],
      aggregationUsedSecondaryRows: 0, aggregationMinSamples: null, aggregationMaxSamples: null,
      aggregationAverageSamples: null, aggregationNumericIssues: [], aggregationPreview: [],
    };
  }
  const primaryKeys = keyIndex(primary, config.keyPairs, 'primary');
  const secondaryKeys = keyIndex(secondary, config.keyPairs, 'secondary');
  if (config.duplicatePolicy === 'block' && primaryKeys.duplicates.length > 0) errors.push('The selected key is not unique in the primary CSV.');
  if (config.duplicatePolicy === 'block' && secondaryKeys.duplicates.length > 0) errors.push('The selected key is not unique in the secondary CSV.');
  const primaryRowIndexes = config.duplicatePolicy === 'keep_first'
    ? primaryKeys.retainedRowIndexes
    : primary.rows.map((_, index) => index);
  const secondaryRowIndexes = config.duplicatePolicy === 'keep_first'
    ? secondaryKeys.retainedRowIndexes
    : secondary.rows.map((_, index) => index);
  const activeAggregationPair = aggregationPair(config);
  const matchPlan = activeAggregationPair
    ? buildAggregationMatchPlan(primary, secondary, config, primaryRowIndexes, secondaryRowIndexes)
    : null;
  if (matchPlan?.overlappingPrimaryWindows) {
    errors.push('Primary Timestamp aggregation windows overlap for the same compound key.');
  }
  const aggregationNumericIssues = numericAggregationIssues(secondary, config);
  aggregationNumericIssues.forEach((issue) => {
    const method = config.aggregationMethod ?? 'mean';
    errors.push(`Column "${issue.column}" contains ${issue.invalidValues} non-numeric value${issue.invalidValues === 1 ? '' : 's'} and cannot use ${method} aggregation.`);
  });
  let matchedRows = 0;
  const matchedSecondary = new Set<number>();
  const matchedPairRows: CsvMatchedPairPreviewRow[] = [];
  const aggregationPreview: CsvAggregationPreviewRow[] = [];
  const primaryStyles = firstPrimaryTimestampStyles(primary, config.keyPairs);
  primaryRowIndexes.forEach((rowIndex) => {
    const row = primary.rows[rowIndex];
    const secondaryRows = matchPlan
      ? matchPlan.secondaryRowsByPrimary.get(rowIndex) ?? []
      : (() => {
        const value = compositeKey(row, primary, config.keyPairs, 'primary');
        const secondaryRow = value.key === null ? undefined : secondaryKeys.rowsByKey.get(value.key);
        return secondaryRow === undefined ? [] : [secondaryRow];
      })();
    if (secondaryRows.length > 0) {
      matchedRows += 1;
      secondaryRows.forEach((secondaryRow) => matchedSecondary.add(secondaryRow));
      if (matchedPairRows.length < 5) {
        const secondaryRow = secondaryRows[0];
        matchedPairRows.push({
          primaryRowNumber: rowIndex + 1,
          primaryKeyValues: config.keyPairs.map((pair) => row[columnIndex(primary, pair.primary)] ?? null),
          secondaryRowNumber: secondaryRow + 1,
          secondaryKeyValues: config.keyPairs.map((pair) => secondary.rows[secondaryRow][columnIndex(secondary, pair.secondary)] ?? null),
        });
      }
      if (activeAggregationPair && aggregationPreview.length < 5) {
        const primaryTimestamp = row[columnIndex(primary, activeAggregationPair.primary)] as string;
        aggregationPreview.push({
          primaryRowNumber: rowIndex + 1,
          primaryTimestamp,
          secondaryRowNumbers: secondaryRows.map((secondaryRow) => secondaryRow + 1),
          secondaryTimestamps: secondaryRows.map((secondaryRow) => (
            secondary.rows[secondaryRow][columnIndex(secondary, activeAggregationPair.secondary)] as string
          )),
          rawValues: config.secondaryColumns.map((column) => ({
            column: column.source,
            values: secondaryRows.map((secondaryRow) => secondary.rows[secondaryRow][columnIndex(secondary, column.source)] ?? null),
          })),
          aggregatedValues: config.secondaryColumns.map((column) => ({
            column: column.source,
            value: aggregateSecondaryValue(
              secondaryRows,
              secondary,
              column.source,
              config.aggregationMethod ?? 'mean',
              primaryStyles,
            ),
          })),
        });
      }
    }
  });
  const unmatchedPrimaryRows = primaryRowIndexes.length - matchedRows;
  const unmatchedSecondaryRows = secondaryRowIndexes.length - matchedSecondary.size;
  const expectedRows = config.joinType === 'inner'
    ? matchedRows
    : config.joinType === 'full'
      ? primaryRowIndexes.length + unmatchedSecondaryRows
      : primaryRowIndexes.length;
  const sampleCounts = matchPlan ? [...matchPlan.secondaryRowsByPrimary.values()].map((rows) => rows.length) : [];
  return {
    errors: [...new Set(errors)],
    primaryDuplicateKeys: primaryKeys.duplicates,
    secondaryDuplicateKeys: secondaryKeys.duplicates,
    primaryDuplicateKeyCount: primaryKeys.duplicates.length,
    secondaryDuplicateKeyCount: secondaryKeys.duplicates.length,
    primaryDiscardedDuplicateRows: primaryKeys.discardedDuplicateRows,
    secondaryDiscardedDuplicateRows: secondaryKeys.discardedDuplicateRows,
    primaryMissingKeys: primaryKeys.missing,
    secondaryMissingKeys: secondaryKeys.missing,
    primaryInvalidTimestampRows: primaryKeys.invalidTimestampRows,
    secondaryInvalidTimestampRows: secondaryKeys.invalidTimestampRows,
    matchedRows,
    unmatchedPrimaryRows,
    unmatchedSecondaryRows,
    expectedRows,
    matchedPairPreview: matchedPairRows,
    aggregationUsedSecondaryRows: matchPlan?.matchedSecondary.size ?? 0,
    aggregationMinSamples: sampleCounts.length > 0 ? Math.min(...sampleCounts) : null,
    aggregationMaxSamples: sampleCounts.length > 0 ? Math.max(...sampleCounts) : null,
    aggregationAverageSamples: sampleCounts.length > 0
      ? sampleCounts.reduce((total, count) => total + count, 0) / sampleCounts.length
      : null,
    aggregationNumericIssues,
    aggregationPreview,
  };
}

export function mergeCsvDocuments(
  primary: CsvDocument,
  secondary: CsvDocument,
  config: CsvMergeConfig,
): CsvMergeResult {
  const validation = validateCsvMerge(primary, secondary, config);
  if (validation.errors.length > 0) throw new Error(validation.errors.join(' '));
  const primaryOutputIndexes = config.primaryColumns.map((name) => columnIndex(primary, name));
  const primaryKeys = keyIndex(primary, config.keyPairs, 'primary');
  const secondaryKeyDetails = keyIndex(secondary, config.keyPairs, 'secondary');
  const secondaryKeys = secondaryKeyDetails.rowsByKey;
  const primaryTimestampStyles = firstPrimaryTimestampStyles(primary, config.keyPairs);
  const primaryRowIndexes = config.duplicatePolicy === 'keep_first'
    ? primaryKeys.retainedRowIndexes
    : primary.rows.map((_, index) => index);
  const secondaryRowIndexes = config.duplicatePolicy === 'keep_first'
    ? secondaryKeyDetails.retainedRowIndexes
    : secondary.rows.map((_, index) => index);
  const matchPlan = aggregationPair(config)
    ? buildAggregationMatchPlan(primary, secondary, config, primaryRowIndexes, secondaryRowIndexes)
    : null;
  const matchedSecondary = new Set<number>();
  const rows: CsvCell[][] = [];

  primaryRowIndexes.forEach((primaryIndex) => {
    const primaryRow = primary.rows[primaryIndex];
    const secondaryRowIndexesForPrimary = matchPlan
      ? matchPlan.secondaryRowsByPrimary.get(primaryIndex) ?? []
      : (() => {
        const value = compositeKey(primaryRow, primary, config.keyPairs, 'primary');
        const secondaryIndex = value.key === null ? undefined : secondaryKeys.get(value.key);
        return secondaryIndex === undefined ? [] : [secondaryIndex];
      })();
    if (config.joinType === 'inner' && secondaryRowIndexesForPrimary.length === 0) return;
    secondaryRowIndexesForPrimary.forEach((secondaryIndex) => matchedSecondary.add(secondaryIndex));
    rows.push([
      ...primaryOutputIndexes.map((index) => primaryRow[index] ?? null),
      ...config.secondaryColumns.map((column) => matchPlan
        ? aggregateSecondaryValue(
          secondaryRowIndexesForPrimary,
          secondary,
          column.source,
          config.aggregationMethod ?? 'mean',
          primaryTimestampStyles,
        )
        : secondaryRowIndexesForPrimary.length > 0
          ? secondaryOutputValue(secondary.rows[secondaryRowIndexesForPrimary[0]], secondary, column.source, primaryTimestampStyles)
          : null),
    ]);
  });

  if (config.joinType === 'full') {
    secondaryRowIndexes.forEach((secondaryIndex) => {
      if (matchedSecondary.has(secondaryIndex)) return;
      const secondaryRow = secondary.rows[secondaryIndex];
      const primaryValues = primaryOutputIndexes.map((index) => {
        const primaryHeader = primary.headers[index];
        const mappingIndex = config.keyPairs.findIndex((pair) => pair.primary === primaryHeader);
        return mappingIndex >= 0
          ? secondaryOutputValue(secondaryRow, secondary, config.keyPairs[mappingIndex].secondary, primaryTimestampStyles)
          : null;
      });
      rows.push([
        ...primaryValues,
        ...config.secondaryColumns.map((column) => secondaryOutputValue(
          secondaryRow,
          secondary,
          column.source,
          primaryTimestampStyles,
        )),
      ]);
    });
  }
  return {
    headers: [...config.primaryColumns, ...config.secondaryColumns.map((column) => column.output)],
    rows,
    validation,
  };
}

export function csvMergeResultTable(result: CsvMergeResult): TabularTable {
  return {
    rowCount: result.rows.length,
    columns: result.headers.map((header, index) => ({
      name: header,
      values: result.rows.map((row) => row[index] ?? null),
    })),
  };
}

export function defaultMergedFileName(primary: string, secondary: string): string {
  const stem = (name: string) => name.replace(/\.[^.]+$/, '') || 'csv';
  return `${stem(primary)}-${stem(secondary)}-merged`;
}
