import { describe, expect, it } from 'vitest';
import { parquetReadObjects } from 'hyparquet';
import { tabularTableToCsv, tabularTableToParquet } from '../lib/tabularExport';
import {
  csvMergeResultTable,
  mergeCsvDocuments,
  parseCsvTimestamp,
  parseCsvText,
  timestampNormalizationSuggested,
  timestampNormalizationPreview,
  validateCsvMerge,
  type CsvDocument,
  type CsvMergeConfig,
} from './csvMerge';

function document(headers: string[], rows: Array<Array<string | null>>, fileName = 'test.csv'): CsvDocument {
  return { fileName, delimiter: ',', headers, rows };
}

const primary = document(['id', 'site', 'keep', 'drop'], [
  ['001', 'A', 'first', 'x'],
  ['002', 'A', 'second', 'y'],
  [null, 'A', 'missing', 'z'],
], 'primary.csv');
const secondary = document(['key', 'site_key', 'value'], [
  ['001', 'A', 'one'],
  ['003', 'A', 'three'],
  [null, 'A', 'empty'],
], 'secondary.csv');
const config: CsvMergeConfig = {
  primaryColumns: ['id', 'keep'],
  secondaryColumns: [{ source: 'value', output: 'joined_value' }],
  keyPairs: [{ primary: 'id', secondary: 'key', comparison: 'exact' }, { primary: 'site', secondary: 'site_key', comparison: 'exact' }],
  joinType: 'left',
  duplicatePolicy: 'block',
};

describe('CSV parsing', () => {
  it('auto-detects delimiters and preserves quoted values, whitespace and leading zeros', () => {
    const parsed = parseCsvText('\uFEFFid;note;value\r\n001;"hello; world";  exact  \r\n002;"two\nlines";', 'input.csv');
    expect(parsed.errors).toEqual([]);
    expect(parsed.document).toMatchObject({ headers: ['id', 'note', 'value'], delimiter: ';' });
    expect(parsed.document?.rows).toEqual([
      ['001', 'hello; world', '  exact  '],
      ['002', 'two\nlines', null],
    ]);
  });

  it('accepts tab-separated data', () => {
    const parsed = parseCsvText('id\tvalue\n1\ta\n2\tb', 'input.tsv');
    expect(parsed.errors).toEqual([]);
    expect(parsed.document?.delimiter).toBe('\t');
  });

  it('rejects empty, duplicate and structurally inconsistent headers/rows', () => {
    expect(parseCsvText('id,,id\n1,2,3').errors.join(' ')).toContain('Empty column names');
    expect(parseCsvText('id,id\n1,2').errors.join(' ')).toContain('Duplicate column names');
    expect(parseCsvText('id,value\n1,2,3').errors.join(' ')).toContain('expected 2');
  });
});

describe('CSV keyed merge', () => {
  it('performs a stable left join using excluded columns as composite keys', () => {
    const result = mergeCsvDocuments(primary, secondary, config);
    expect(result.headers).toEqual(['id', 'keep', 'joined_value']);
    expect(result.rows).toEqual([
      ['001', 'first', 'one'],
      ['002', 'second', null],
      [null, 'missing', null],
    ]);
    expect(result.validation).toMatchObject({ matchedRows: 1, unmatchedPrimaryRows: 2, unmatchedSecondaryRows: 2 });
  });

  it('supports inner and full joins with deterministic ordering and coalesced output keys', () => {
    const inner = mergeCsvDocuments(primary, secondary, { ...config, joinType: 'inner' });
    expect(inner.rows).toEqual([['001', 'first', 'one']]);

    const full = mergeCsvDocuments(primary, secondary, { ...config, joinType: 'full' });
    expect(full.rows).toEqual([
      ['001', 'first', 'one'],
      ['002', 'second', null],
      [null, 'missing', null],
      ['003', null, 'three'],
      [null, null, 'empty'],
    ]);
  });

  it('uses exact key matching without trimming or changing case', () => {
    const left = document(['key'], [['A'], [' A'], ['a']]);
    const right = document(['key', 'value'], [['A', 'match']]);
    const result = mergeCsvDocuments(left, right, {
      primaryColumns: ['key'], secondaryColumns: [{ source: 'value', output: 'value' }],
      keyPairs: [{ primary: 'key', secondary: 'key', comparison: 'exact' }], joinType: 'left', duplicatePolicy: 'block',
    });
    expect(result.rows).toEqual([['A', 'match'], [' A', null], ['a', null]]);
  });

  it('blocks duplicate keys, reused key mappings and output-name conflicts', () => {
    const duplicate = document(['key', 'value'], [['x', '1'], ['x', '2']]);
    const validation = validateCsvMerge(duplicate, secondary, {
      primaryColumns: ['key'],
      secondaryColumns: [{ source: 'value', output: 'key' }],
      keyPairs: [{ primary: 'key', secondary: 'key', comparison: 'exact' }, { primary: 'key', secondary: 'site_key', comparison: 'exact' }],
      joinType: 'left',
      duplicatePolicy: 'block',
    });
    expect(validation.errors).toEqual(expect.arrayContaining([
      'Each primary key column may only be used once.',
      'Every output column name must be unique. Rename conflicting secondary columns.',
      'The selected key is not unique in the primary CSV.',
    ]));
    expect(validation.primaryDuplicateKeys[0]).toEqual({ key: 'x | x', rows: [1, 2] });
    expect(validation).toMatchObject({ primaryDuplicateKeyCount: 1, primaryDiscardedDuplicateRows: 1 });
  });

  it('keeps only the first primary and secondary row for duplicate keys', () => {
    const duplicatePrimary = document(['key', 'group', 'primary_value'], [
      ['x', 'A', 'primary-first'],
      ['x', 'A', 'primary-second'],
      ['x', 'A', 'primary-third'],
      ['x', 'B', 'different-composite-key'],
      [null, 'A', 'empty-first'],
      [null, 'A', 'empty-second'],
    ]);
    const duplicateSecondary = document(['key', 'group', 'secondary_value'], [
      ['x', 'A', 'secondary-first'],
      ['x', 'A', 'secondary-second'],
      ['z', 'A', 'secondary-unmatched'],
    ]);
    const duplicateConfig: CsvMergeConfig = {
      primaryColumns: ['key', 'group', 'primary_value'],
      secondaryColumns: [{ source: 'secondary_value', output: 'secondary_value' }],
      keyPairs: [{ primary: 'key', secondary: 'key', comparison: 'exact' }, { primary: 'group', secondary: 'group', comparison: 'exact' }],
      joinType: 'left',
      duplicatePolicy: 'keep_first',
    };

    const validation = validateCsvMerge(duplicatePrimary, duplicateSecondary, duplicateConfig);
    expect(validation.errors).toEqual([]);
    expect(validation).toMatchObject({
      primaryDuplicateKeyCount: 1,
      primaryDiscardedDuplicateRows: 2,
      secondaryDuplicateKeyCount: 1,
      secondaryDiscardedDuplicateRows: 1,
      matchedRows: 1,
      unmatchedPrimaryRows: 3,
      unmatchedSecondaryRows: 1,
      expectedRows: 4,
    });
    expect(mergeCsvDocuments(duplicatePrimary, duplicateSecondary, duplicateConfig).rows).toEqual([
      ['x', 'A', 'primary-first', 'secondary-first'],
      ['x', 'B', 'different-composite-key', null],
      [null, 'A', 'empty-first', null],
      [null, 'A', 'empty-second', null],
    ]);

    expect(mergeCsvDocuments(duplicatePrimary, duplicateSecondary, { ...duplicateConfig, joinType: 'inner' }).rows)
      .toEqual([['x', 'A', 'primary-first', 'secondary-first']]);
    expect(mergeCsvDocuments(duplicatePrimary, duplicateSecondary, { ...duplicateConfig, joinType: 'full' }).rows)
      .toEqual([
        ['x', 'A', 'primary-first', 'secondary-first'],
        ['x', 'B', 'different-composite-key', null],
        [null, 'A', 'empty-first', null],
        [null, 'A', 'empty-second', null],
        ['z', 'A', null, 'secondary-unmatched'],
      ]);
  });

  it('reports missing keys and expected output sizes for every join type', () => {
    expect(validateCsvMerge(primary, secondary, config)).toMatchObject({ primaryMissingKeys: 1, secondaryMissingKeys: 1, expectedRows: 3 });
    expect(validateCsvMerge(primary, secondary, { ...config, joinType: 'inner' }).expectedRows).toBe(1);
    expect(validateCsvMerge(primary, secondary, { ...config, joinType: 'full' }).expectedRows).toBe(5);
  });

  it('requires an explicitly selected secondary output column', () => {
    const validation = validateCsvMerge(primary, secondary, { ...config, secondaryColumns: [] });
    expect(validation.errors).toContain('Select at least one secondary column to merge.');
  });

  it('normalizes supported dataset-local timestamps while exact comparison stays strict', () => {
    const left = document(['time'], [
      ['2025-09-16 00:09'],
      ['2024-02-29T01:02:03.100'],
      ['2025-10-01'],
    ]);
    const right = document(['time', 'value'], [
      ['2025.09.16T00:09:00.000', 'first'],
      ['2024.02.29 01:02:03.1', 'leap'],
      ['2025.10.01', 'date only'],
    ]);
    const timestampConfig: CsvMergeConfig = {
      primaryColumns: ['time'],
      secondaryColumns: [{ source: 'value', output: 'value' }],
      keyPairs: [{ primary: 'time', secondary: 'time', comparison: 'timestamp' }],
      joinType: 'left',
      duplicatePolicy: 'block',
    };
    expect(mergeCsvDocuments(left, right, timestampConfig).rows).toEqual([
      ['2025-09-16 00:09', 'first'],
      ['2024-02-29T01:02:03.100', 'leap'],
      ['2025-10-01', 'date only'],
    ]);
    expect(validateCsvMerge(left, right, {
      ...timestampConfig,
      keyPairs: [{ primary: 'time', secondary: 'time', comparison: 'exact' }],
    }).matchedRows).toBe(0);
  });

  it('rejects invalid calendar values and timezone suffixes without treating them as empty', () => {
    expect(parseCsvTimestamp('2024-02-29 23:59:59')).not.toBeNull();
    expect(parseCsvTimestamp('2024.02.29')).not.toBeNull();
    expect(parseCsvTimestamp('2025-02-29 00:00:00')).toBeNull();
    expect(parseCsvTimestamp('2025-13-01 00:00')).toBeNull();
    expect(parseCsvTimestamp('2025-01-01 24:00')).toBeNull();
    expect(parseCsvTimestamp('2025-01-01T00:00:00Z')).toBeNull();
    expect(parseCsvTimestamp('2025-01-01T00:00:00+01:00')).toBeNull();

    const left = document(['time'], [['2025-02-29 00:00'], [null], ['2025-01-01 00:00']]);
    const right = document(['time', 'value'], [['2025.01.01 00:00:00', 'valid'], ['bad', 'invalid']]);
    const validation = validateCsvMerge(left, right, {
      primaryColumns: ['time'], secondaryColumns: [{ source: 'value', output: 'value' }],
      keyPairs: [{ primary: 'time', secondary: 'time', comparison: 'timestamp' }],
      joinType: 'left', duplicatePolicy: 'block',
    });
    expect(validation).toMatchObject({
      matchedRows: 1,
      primaryMissingKeys: 1,
      primaryInvalidTimestampRows: 1,
      secondaryMissingKeys: 0,
      secondaryInvalidTimestampRows: 1,
    });
    expect(validation.errors).toEqual([]);
  });

  it('supports mixed exact and timestamp components and detects normalized duplicates', () => {
    const left = document(['site', 'time', 'value'], [
      ['A', '2025-09-16 00:09:00', 'first'],
      ['A', '2025.09.16T00:09', 'later duplicate'],
      ['B', '2025-09-16 00:09:00', 'other site'],
    ]);
    const right = document(['site', 'time', 'joined'], [
      ['A', '2025.09.16 00:09:00.000', 'match A'],
      ['B', '2025.09.16 00:09:00', 'match B'],
    ]);
    const mixedConfig: CsvMergeConfig = {
      primaryColumns: ['site', 'time', 'value'],
      secondaryColumns: [{ source: 'joined', output: 'joined' }],
      keyPairs: [
        { primary: 'site', secondary: 'site', comparison: 'exact' },
        { primary: 'time', secondary: 'time', comparison: 'timestamp' },
      ],
      joinType: 'left', duplicatePolicy: 'block',
    };
    const blocked = validateCsvMerge(left, right, mixedConfig);
    expect(blocked).toMatchObject({ primaryDuplicateKeyCount: 1, primaryDiscardedDuplicateRows: 1 });
    expect(blocked.errors).toContain('The selected key is not unique in the primary CSV.');
    const kept = mergeCsvDocuments(left, right, { ...mixedConfig, duplicatePolicy: 'keep_first' });
    expect(kept.rows).toEqual([
      ['A', '2025-09-16 00:09:00', 'first', 'match A'],
      ['B', '2025-09-16 00:09:00', 'other site', 'match B'],
    ]);
    expect(kept.validation.matchedRows).toBe(2);
    expect(kept.validation.matchedPairPreview).toEqual([
      {
        primaryRowNumber: 1,
        primaryKeyValues: ['A', '2025-09-16 00:09:00'],
        secondaryRowNumber: 1,
        secondaryKeyValues: ['A', '2025.09.16 00:09:00.000'],
      },
      {
        primaryRowNumber: 3,
        primaryKeyValues: ['B', '2025-09-16 00:09:00'],
        secondaryRowNumber: 2,
        secondaryKeyValues: ['B', '2025.09.16 00:09:00'],
      },
    ]);
    expect(blocked.matchedPairPreview).toHaveLength(3);
  });

  it('formats selected secondary timestamp keys like the primary without losing precision', () => {
    const left = document(['time'], [['2025-09-16 00:09']]);
    const right = document(['time', 'value'], [
      ['2025.09.16T00:09:00.000', 'match'],
      ['2025.09.16T00:10:02.1234', 'unmatched'],
      ['invalid timestamp', 'invalid'],
    ]);
    const result = mergeCsvDocuments(left, right, {
      primaryColumns: ['time'],
      secondaryColumns: [{ source: 'time', output: 'secondary_time' }, { source: 'value', output: 'value' }],
      keyPairs: [{ primary: 'time', secondary: 'time', comparison: 'timestamp' }],
      joinType: 'full', duplicatePolicy: 'block',
    });
    expect(result.rows).toEqual([
      ['2025-09-16 00:09', '2025-09-16 00:09', 'match'],
      ['2025-09-16 00:10:02.1234', '2025-09-16 00:10:02.1234', 'unmatched'],
      ['invalid timestamp', 'invalid timestamp', 'invalid'],
    ]);

    expect(timestampNormalizationPreview(left, right, {
      primary: 'time', secondary: 'time', comparison: 'timestamp',
    }, 3)).toEqual([
      {
        rowNumber: 1,
        primaryOriginal: '2025-09-16 00:09',
        primaryComparisonKey: '2025-09-16T00:09:00',
        secondaryOriginal: '2025.09.16T00:09:00.000',
        secondaryOutput: '2025-09-16 00:09',
        secondaryComparisonKey: '2025-09-16T00:09:00',
      },
      {
        rowNumber: 2,
        primaryOriginal: null,
        primaryComparisonKey: null,
        secondaryOriginal: '2025.09.16T00:10:02.1234',
        secondaryOutput: '2025-09-16 00:10:02.1234',
        secondaryComparisonKey: '2025-09-16T00:10:02.1234',
      },
      {
        rowNumber: 3,
        primaryOriginal: null,
        primaryComparisonKey: null,
        secondaryOriginal: 'invalid timestamp',
        secondaryOutput: 'invalid timestamp',
        secondaryComparisonKey: null,
      },
    ]);
  });

  it('suggests timestamp normalization only for predominantly temporal columns with different styles', () => {
    const left = document(['time'], [['2025-09-16 00:09:00'], ['2025-09-16 00:10:00']]);
    const right = document(['time'], [['2025.09.16 00:09:00'], ['2025.09.16 00:10:00']]);
    const exactPair = { primary: 'time', secondary: 'time', comparison: 'exact' as const };
    expect(timestampNormalizationSuggested(left, right, exactPair)).toBe(true);
    expect(timestampNormalizationSuggested(left, right, { ...exactPair, comparison: 'timestamp' })).toBe(false);
    expect(timestampNormalizationSuggested(document(['key'], [['a'], ['b']]), document(['key'], [['a'], ['b']]), {
      primary: 'key', secondary: 'key', comparison: 'exact',
    })).toBe(false);
  });

  it('aggregates every secondary sample in the primary half-open minute window', () => {
    const left = document(['site', 'time'], [
      ['A', '2025-09-16 00:09:00'],
      ['A', '2025-09-16 00:10:00'],
      ['B', '2025-09-16 00:09:00'],
    ]);
    const right = document(['site', 'time', 'value'], [
      ['A', '2025.09.16 00:09:45', '16'],
      ['A', '2025.09.16 00:09:00', '10'],
      ['A', '2025.09.16 00:09:30', '14'],
      ['A', '2025.09.16 00:09:15', '12'],
      ['A', '2025.09.16 00:10:00', '20'],
      ['A', '2025.09.16 00:10:15', '22'],
      ['B', '2025.09.16 00:09:15', '100'],
      ['A', '2025.09.16 00:08:45', '8'],
    ]);
    const aggregateConfig: CsvMergeConfig = {
      primaryColumns: ['site', 'time'],
      secondaryColumns: [{ source: 'value', output: 'value_mean' }],
      keyPairs: [
        { primary: 'site', secondary: 'site', comparison: 'exact' },
        { primary: 'time', secondary: 'time', comparison: 'timestamp_aggregation' },
      ],
      aggregationMethod: 'mean', joinType: 'left', duplicatePolicy: 'block',
    };
    const result = mergeCsvDocuments(left, right, aggregateConfig);
    expect(result.rows).toEqual([
      ['A', '2025-09-16 00:09:00', '13'],
      ['A', '2025-09-16 00:10:00', '21'],
      ['B', '2025-09-16 00:09:00', '100'],
    ]);
    expect(result.validation).toMatchObject({
      matchedRows: 3,
      unmatchedPrimaryRows: 0,
      unmatchedSecondaryRows: 1,
      aggregationUsedSecondaryRows: 7,
      aggregationMinSamples: 1,
      aggregationMaxSamples: 4,
      aggregationAverageSamples: 7 / 3,
    });
    expect(result.validation.aggregationPreview[0]).toEqual({
      primaryRowNumber: 1,
      primaryTimestamp: '2025-09-16 00:09:00',
      secondaryRowNumbers: [2, 4, 3, 1],
      secondaryTimestamps: [
        '2025.09.16 00:09:00',
        '2025.09.16 00:09:15',
        '2025.09.16 00:09:30',
        '2025.09.16 00:09:45',
      ],
      rawValues: [{ column: 'value', values: ['10', '12', '14', '16'] }],
      aggregatedValues: [{ column: 'value', value: '13' }],
    });
  });

  it('accepts an offset 15-second cadence without requiring fixed second positions', () => {
    const left = document(['time'], [['2025-09-16 00:09:00']]);
    const right = document(['time', 'value'], [
      ['2025-09-16 00:09:03', '1'],
      ['2025-09-16 00:09:18', '2'],
      ['2025-09-16 00:09:33', '3'],
      ['2025-09-16 00:09:48', '4'],
    ]);
    const result = mergeCsvDocuments(left, right, {
      primaryColumns: ['time'], secondaryColumns: [{ source: 'value', output: 'value_mean' }],
      keyPairs: [{ primary: 'time', secondary: 'time', comparison: 'timestamp_aggregation' }],
      joinType: 'left', duplicatePolicy: 'block', aggregationMethod: 'mean',
    });
    expect(result.rows).toEqual([['2025-09-16 00:09:00', '2.5']]);
    expect(result.validation).toMatchObject({
      aggregationUsedSecondaryRows: 4,
      aggregationMinSamples: 4,
      aggregationMaxSamples: 4,
      aggregationAverageSamples: 4,
    });
  });

  it('aggregates a 5-second cadence and reports all twelve samples in the minute', () => {
    const left = document(['time'], [['2025-09-16 00:09:00']]);
    const right = document(['time', 'value'], Array.from({ length: 12 }, (_, index) => [
      `2025-09-16 00:09:${String(index * 5).padStart(2, '0')}`,
      String(index + 1),
    ]));
    const result = mergeCsvDocuments(left, right, {
      primaryColumns: ['time'], secondaryColumns: [{ source: 'value', output: 'value_mean' }],
      keyPairs: [{ primary: 'time', secondary: 'time', comparison: 'timestamp_aggregation' }],
      joinType: 'left', duplicatePolicy: 'block', aggregationMethod: 'mean',
    });
    expect(result.rows).toEqual([['2025-09-16 00:09:00', '6.5']]);
    expect(result.validation).toMatchObject({
      aggregationUsedSecondaryRows: 12,
      aggregationMinSamples: 12,
      aggregationMaxSamples: 12,
      aggregationAverageSamples: 12,
    });
  });

  it('uses irregular available samples and assigns the exclusive boundary only to the next window', () => {
    const left = document(['time'], [
      ['2025-09-16 00:09:00'],
      ['2025-09-16 00:10:00'],
      ['2025-09-16 00:11:00'],
    ]);
    const right = document(['time', 'value'], [
      ['2025-09-16 00:09:07', '2'],
      ['2025-09-16 00:09:41', '4'],
      ['2025-09-16 00:10:00', '10'],
      ['2025-09-16 00:10:52', '20'],
    ]);
    const result = mergeCsvDocuments(left, right, {
      primaryColumns: ['time'], secondaryColumns: [{ source: 'value', output: 'value_mean' }],
      keyPairs: [{ primary: 'time', secondary: 'time', comparison: 'timestamp_aggregation' }],
      joinType: 'left', duplicatePolicy: 'block', aggregationMethod: 'mean',
    });
    expect(result.rows).toEqual([
      ['2025-09-16 00:09:00', '3'],
      ['2025-09-16 00:10:00', '15'],
      ['2025-09-16 00:11:00', null],
    ]);
    expect(result.validation).toMatchObject({
      matchedRows: 2,
      unmatchedPrimaryRows: 1,
      aggregationUsedSecondaryRows: 4,
      aggregationMinSamples: 2,
      aggregationMaxSamples: 2,
      aggregationAverageSamples: 2,
    });
    expect(result.validation.aggregationPreview[0].secondaryTimestamps).not.toContain('2025-09-16 00:10:00');
    expect(result.validation.aggregationPreview[1].secondaryTimestamps).toContain('2025-09-16 00:10:00');
  });

  it('supports one global mean, min, max, first, or last aggregation method', () => {
    const left = document(['time'], [['2025-09-16 00:09:00']]);
    const right = document(['time', 'value'], [
      ['2025-09-16 00:09:45', '16'],
      ['2025-09-16 00:09:00', '10'],
      ['2025-09-16 00:09:30', '14'],
      ['2025-09-16 00:09:15', '12'],
    ]);
    const aggregateConfig: CsvMergeConfig = {
      primaryColumns: ['time'], secondaryColumns: [{ source: 'value', output: 'value' }],
      keyPairs: [{ primary: 'time', secondary: 'time', comparison: 'timestamp_aggregation' }],
      joinType: 'left', duplicatePolicy: 'block', aggregationMethod: 'mean',
    };
    expect(mergeCsvDocuments(left, right, aggregateConfig).rows[0][1]).toBe('13');
    expect(mergeCsvDocuments(left, right, { ...aggregateConfig, aggregationMethod: 'min' }).rows[0][1]).toBe('10');
    expect(mergeCsvDocuments(left, right, { ...aggregateConfig, aggregationMethod: 'max' }).rows[0][1]).toBe('16');
    expect(mergeCsvDocuments(left, right, { ...aggregateConfig, aggregationMethod: 'first' }).rows[0][1]).toBe('10');
    expect(mergeCsvDocuments(left, right, { ...aggregateConfig, aggregationMethod: 'last' }).rows[0][1]).toBe('16');
  });

  it('ignores empty numeric cells but blocks non-numeric selected columns with counts', () => {
    const left = document(['time'], [['2025-09-16 00:09:00'], ['2025-09-16 00:10:00']]);
    const right = document(['time', 'value'], [
      ['2025-09-16 00:09:00', null],
      ['2025-09-16 00:09:15', '2'],
      ['2025-09-16 00:10:00', null],
    ]);
    const aggregateConfig: CsvMergeConfig = {
      primaryColumns: ['time'], secondaryColumns: [{ source: 'value', output: 'value' }],
      keyPairs: [{ primary: 'time', secondary: 'time', comparison: 'timestamp_aggregation' }],
      joinType: 'left', duplicatePolicy: 'block', aggregationMethod: 'mean',
    };
    expect(mergeCsvDocuments(left, right, aggregateConfig).rows).toEqual([
      ['2025-09-16 00:09:00', '2'],
      ['2025-09-16 00:10:00', null],
    ]);
    const invalid = document(['time', 'value'], [
      ['2025-09-16 00:09:00', '2'],
      ['2025-09-16 00:09:15', 'not numeric'],
      ['2025-09-16 00:09:30', 'also invalid'],
    ]);
    const validation = validateCsvMerge(left, invalid, aggregateConfig);
    expect(validation.aggregationNumericIssues).toEqual([{ column: 'value', invalidValues: 2 }]);
    expect(validation.errors).toContain('Column "value" contains 2 non-numeric values and cannot use mean aggregation.');
  });

  it('treats distinct sub-minute samples as a group but identical timestamps as duplicates', () => {
    const left = document(['time'], [['2025-09-16 00:09:00']]);
    const right = document(['time', 'value'], [
      ['2025-09-16 00:09:00', '1'],
      ['2025-09-16 00:09:00', '100'],
      ['2025-09-16 00:09:15', '3'],
    ]);
    const aggregateConfig: CsvMergeConfig = {
      primaryColumns: ['time'], secondaryColumns: [{ source: 'value', output: 'value' }],
      keyPairs: [{ primary: 'time', secondary: 'time', comparison: 'timestamp_aggregation' }],
      joinType: 'left', duplicatePolicy: 'block', aggregationMethod: 'mean',
    };
    const blocked = validateCsvMerge(left, right, aggregateConfig);
    expect(blocked.secondaryDuplicateKeyCount).toBe(1);
    expect(blocked.secondaryDiscardedDuplicateRows).toBe(1);
    expect(blocked.errors).toContain('The selected key is not unique in the secondary CSV.');
    const kept = mergeCsvDocuments(left, right, { ...aggregateConfig, duplicatePolicy: 'keep_first' });
    expect(kept.rows).toEqual([['2025-09-16 00:09:00', '2']]);
    expect(kept.validation.aggregationUsedSecondaryRows).toBe(2);
  });

  it('rejects overlapping primary aggregation windows for the same compound key', () => {
    const left = document(['site', 'time'], [
      ['A', '2025-09-16 00:09:00'],
      ['A', '2025-09-16 00:09:30'],
    ]);
    const right = document(['site', 'time', 'value'], [['A', '2025-09-16 00:09:45', '1']]);
    const validation = validateCsvMerge(left, right, {
      primaryColumns: ['site', 'time'], secondaryColumns: [{ source: 'value', output: 'value' }],
      keyPairs: [
        { primary: 'site', secondary: 'site', comparison: 'exact' },
        { primary: 'time', secondary: 'time', comparison: 'timestamp_aggregation' },
      ],
      joinType: 'left', duplicatePolicy: 'block', aggregationMethod: 'mean',
    });
    expect(validation.errors).toContain('Primary Timestamp aggregation windows overlap for the same compound key.');
  });

  it('exports merged data losslessly as CSV and string-typed Parquet', async () => {
    const result = mergeCsvDocuments(primary, secondary, config);
    const table = csvMergeResultTable(result);
    expect(tabularTableToCsv(table)).toContain('\r\n001,first,one\r\n');
    const buffer = await tabularTableToParquet(table, 'MLTrace CSV merge', true);
    const rows = await parquetReadObjects({ file: buffer });
    expect(rows[0]).toEqual({ id: '001', keep: 'first', joined_value: 'one' });
    expect(rows[1]).toEqual({ id: '002', keep: 'second', joined_value: null });
  });
});
