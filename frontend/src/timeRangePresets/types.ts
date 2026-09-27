export type TimeRangeValue = { start: string; end: string };
export type TimeRangePresetInput = TimeRangeValue & { name: string };
export type TimeRangePreset = TimeRangePresetInput & {
  id: number;
  created_at: string;
  updated_at: string;
};
