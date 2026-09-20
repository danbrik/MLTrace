export type Subset = 'train' | 'test' | 'validation';
export type CsvPreview = { columns: string[]; row_count: number; rows: string[][]; start?: string; end?: string };
export type TimeSeriesDataset = {
  id: number; name: string; filename: string; columns: string[]; selected_columns: string[];
  timestamp_column: string; timestamp_format: string; row_count: number; start: string; end: string;
  split_count: number; created_at: string; updated_at: string; rows?: string[][]; source_rows?: string[][];
};
export type TimeInterval = { id: string; start: string; end: string; subset: Subset; tags: string[]; row_count?: number };
export type SplitPayload = { name: string; dataset_id: number; tags: string[]; intervals: TimeInterval[] };
export type TimeSeriesSplit = SplitPayload & { id: number; dataset_name: string; created_at: string; updated_at: string };
export type DatasetPayload = { name: string; selected_columns: string[]; timestamp_column: string; timestamp_format: string };
