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

export type CsvKeyPair = {
  primary: string;
  secondary: string;
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
  matchedRows: number;
  unmatchedPrimaryRows: number;
  unmatchedSecondaryRows: number;
  expectedRows: number;
};

export type CsvMergeResult = {
  headers: string[];
  rows: CsvCell[][];
  validation: CsvMergeValidation;
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

function compositeKey(row: CsvCell[], indexes: number[]): { key: string | null; label: string } {
  const values = indexes.map((index) => row[index] ?? null);
  if (values.some((value) => value === null || value === '')) return { key: null, label: values.map((value) => value ?? '').join(' | ') };
  return { key: JSON.stringify(values), label: values.join(' | ') };
}

function keyIndex(
  document: CsvDocument,
  indexes: number[],
): {
  rowsByKey: Map<string, number>;
  duplicates: CsvDuplicateKey[];
  discardedDuplicateRows: number;
  retainedRowIndexes: number[];
  missing: number;
} {
  const allRows = new Map<string, number[]>();
  const retainedRowIndexes: number[] = [];
  let missing = 0;
  document.rows.forEach((row, index) => {
    const value = compositeKey(row, indexes);
    if (value.key === null) {
      missing += 1;
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
  };
}

function unique(values: string[]): boolean {
  return new Set(values).size === values.length;
}

export function validateCsvMerge(
  primary: CsvDocument,
  secondary: CsvDocument,
  config: CsvMergeConfig,
): CsvMergeValidation {
  const errors: string[] = [];
  if (config.keyPairs.length === 0) errors.push('Add at least one key-column mapping.');
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
      secondaryMissingKeys: 0, matchedRows: 0, unmatchedPrimaryRows: primary.rows.length,
      unmatchedSecondaryRows: secondary.rows.length, expectedRows: 0,
    };
  }
  const primaryKeys = keyIndex(primary, primaryIndexes);
  const secondaryKeys = keyIndex(secondary, secondaryIndexes);
  if (config.duplicatePolicy === 'block' && primaryKeys.duplicates.length > 0) errors.push('The selected key is not unique in the primary CSV.');
  if (config.duplicatePolicy === 'block' && secondaryKeys.duplicates.length > 0) errors.push('The selected key is not unique in the secondary CSV.');
  const primaryRowIndexes = config.duplicatePolicy === 'keep_first'
    ? primaryKeys.retainedRowIndexes
    : primary.rows.map((_, index) => index);
  const secondaryRowIndexes = config.duplicatePolicy === 'keep_first'
    ? secondaryKeys.retainedRowIndexes
    : secondary.rows.map((_, index) => index);
  let matchedRows = 0;
  const matchedSecondary = new Set<number>();
  primaryRowIndexes.forEach((rowIndex) => {
    const row = primary.rows[rowIndex];
    const value = compositeKey(row, primaryIndexes);
    if (value.key === null) return;
    const secondaryRow = secondaryKeys.rowsByKey.get(value.key);
    if (secondaryRow !== undefined) {
      matchedRows += 1;
      matchedSecondary.add(secondaryRow);
    }
  });
  const unmatchedPrimaryRows = primaryRowIndexes.length - matchedRows;
  const unmatchedSecondaryRows = secondaryRowIndexes.length - matchedSecondary.size;
  const expectedRows = config.joinType === 'inner'
    ? matchedRows
    : config.joinType === 'full'
      ? primaryRowIndexes.length + unmatchedSecondaryRows
      : primaryRowIndexes.length;
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
    matchedRows,
    unmatchedPrimaryRows,
    unmatchedSecondaryRows,
    expectedRows,
  };
}

export function mergeCsvDocuments(
  primary: CsvDocument,
  secondary: CsvDocument,
  config: CsvMergeConfig,
): CsvMergeResult {
  const validation = validateCsvMerge(primary, secondary, config);
  if (validation.errors.length > 0) throw new Error(validation.errors.join(' '));
  const primaryKeyIndexes = config.keyPairs.map((pair) => columnIndex(primary, pair.primary));
  const secondaryKeyIndexes = config.keyPairs.map((pair) => columnIndex(secondary, pair.secondary));
  const primaryOutputIndexes = config.primaryColumns.map((name) => columnIndex(primary, name));
  const secondaryOutputIndexes = config.secondaryColumns.map((column) => columnIndex(secondary, column.source));
  const primaryKeys = keyIndex(primary, primaryKeyIndexes);
  const secondaryKeyDetails = keyIndex(secondary, secondaryKeyIndexes);
  const secondaryKeys = secondaryKeyDetails.rowsByKey;
  const primaryRowIndexes = config.duplicatePolicy === 'keep_first'
    ? primaryKeys.retainedRowIndexes
    : primary.rows.map((_, index) => index);
  const secondaryRowIndexes = config.duplicatePolicy === 'keep_first'
    ? secondaryKeyDetails.retainedRowIndexes
    : secondary.rows.map((_, index) => index);
  const matchedSecondary = new Set<number>();
  const rows: CsvCell[][] = [];

  primaryRowIndexes.forEach((primaryIndex) => {
    const primaryRow = primary.rows[primaryIndex];
    const value = compositeKey(primaryRow, primaryKeyIndexes);
    const secondaryIndex = value.key === null ? undefined : secondaryKeys.get(value.key);
    if (config.joinType === 'inner' && secondaryIndex === undefined) return;
    if (secondaryIndex !== undefined) matchedSecondary.add(secondaryIndex);
    const secondaryRow = secondaryIndex === undefined ? null : secondary.rows[secondaryIndex];
    rows.push([
      ...primaryOutputIndexes.map((index) => primaryRow[index] ?? null),
      ...secondaryOutputIndexes.map((index) => secondaryRow?.[index] ?? null),
    ]);
  });

  if (config.joinType === 'full') {
    secondaryRowIndexes.forEach((secondaryIndex) => {
      if (matchedSecondary.has(secondaryIndex)) return;
      const secondaryRow = secondary.rows[secondaryIndex];
      const primaryValues = primaryOutputIndexes.map((index) => {
        const primaryHeader = primary.headers[index];
        const mappingIndex = config.keyPairs.findIndex((pair) => pair.primary === primaryHeader);
        return mappingIndex >= 0 ? secondaryRow[secondaryKeyIndexes[mappingIndex]] ?? null : null;
      });
      rows.push([
        ...primaryValues,
        ...secondaryOutputIndexes.map((index) => secondaryRow[index] ?? null),
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
