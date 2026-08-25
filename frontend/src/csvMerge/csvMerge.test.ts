import { describe, expect, it } from 'vitest';
import { parquetReadObjects } from 'hyparquet';
import { tabularTableToCsv, tabularTableToParquet } from '../lib/tabularExport';
import {
  csvMergeResultTable,
  mergeCsvDocuments,
  parseCsvText,
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
  keyPairs: [{ primary: 'id', secondary: 'key' }, { primary: 'site', secondary: 'site_key' }],
  joinType: 'left',
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
      keyPairs: [{ primary: 'key', secondary: 'key' }], joinType: 'left',
    });
    expect(result.rows).toEqual([['A', 'match'], [' A', null], ['a', null]]);
  });

  it('blocks duplicate keys, reused key mappings and output-name conflicts', () => {
    const duplicate = document(['key', 'value'], [['x', '1'], ['x', '2']]);
    const validation = validateCsvMerge(duplicate, secondary, {
      primaryColumns: ['key'],
      secondaryColumns: [{ source: 'value', output: 'key' }],
      keyPairs: [{ primary: 'key', secondary: 'key' }, { primary: 'key', secondary: 'site_key' }],
      joinType: 'left',
    });
    expect(validation.errors).toEqual(expect.arrayContaining([
      'Each primary key column may only be used once.',
      'Every output column name must be unique. Rename conflicting secondary columns.',
      'The selected key is not unique in the primary CSV.',
    ]));
    expect(validation.primaryDuplicateKeys[0]).toEqual({ key: 'x | x', rows: [1, 2] });
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
