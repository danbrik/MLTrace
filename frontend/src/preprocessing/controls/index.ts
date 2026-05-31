import { cropBoxControl } from './CropBoxControl';
import { pointPickerControl } from './PointPickerControl';
import type { StepControl } from './types';

// Maps a step's config_schema.ui_control value to its interactive control. A new step that
// reuses one of these controls only needs to set ui_control in its backend config_schema —
// no change to the pipeline page is required. A brand-new control type is added here.
export const CONTROL_REGISTRY: Record<string, StepControl> = {
  point_picker: pointPickerControl,
  crop_box: cropBoxControl,
};

export type { StepControl, StepControlProps } from './types';
