export type TabularValue = string | number | boolean | null;

export type TabularColumn = {
  name: string;
  values: TabularValue[];
};

export type TabularTable = {
  columns: TabularColumn[];
  rowCount: number;
};

function csvCell(value: TabularValue): string {
  if (value === null) return '';
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function tabularTableToCsv(table: TabularTable): string {
  if (table.columns.length === 0) return '\uFEFF';
  const lines = [table.columns.map((column) => csvCell(column.name)).join(',')];
  for (let row = 0; row < table.rowCount; row += 1) {
    lines.push(table.columns.map((column) => csvCell(column.values[row] ?? null)).join(','));
  }
  return `\uFEFF${lines.join('\r\n')}`;
}

export async function tabularTableToParquet(
  table: TabularTable,
  source = 'MLTrace table export',
  preserveText = false,
): Promise<ArrayBuffer> {
  const { parquetWriteBuffer } = await import('hyparquet-writer');
  const columnData = table.columns.map((column) => {
    const present = column.values.filter((value) => value !== null);
    if (!preserveText && present.every((value) => typeof value === 'number')) {
      return { name: column.name, data: column.values, type: 'DOUBLE' as const, nullable: present.length !== column.values.length };
    }
    if (!preserveText && present.length > 0 && present.every((value) => typeof value === 'boolean')) {
      return { name: column.name, data: column.values, type: 'BOOLEAN' as const, nullable: present.length !== column.values.length };
    }
    return {
      name: column.name,
      data: column.values.map((value) => value === null ? null : String(value)),
      type: 'STRING' as const,
      nullable: present.length !== column.values.length,
    };
  });
  return parquetWriteBuffer({
    columnData,
    kvMetadata: [{ key: 'source', value: source }],
  });
}

export function normalizedTableDownloadName(
  input: string,
  extension: 'csv' | 'parquet',
  fallback = 'table-data',
): string {
  const withoutExtension = input.trim().replace(/\.(?:csv|parquet)$/i, '');
  const safe = withoutExtension.replace(/[\\/:*?"<>|\u0000-\u001F]/g, '_').replace(/\s+/g, ' ').trim() || fallback;
  return `${safe}.${extension}`;
}
