import type { ComponentType } from 'react';

import type { PreprocessingPreviewImage } from '../../types';

// Standard contract every interactive step control implements. A control draws on top of
// the step's INPUT image (the previous step's output) and writes back the config keys it owns.
export type StepControlProps = {
  inputImage: PreprocessingPreviewImage;
  config: Record<string, unknown>;
  disabled?: boolean;
  onChange: (partial: Record<string, unknown>) => void;
};

export type StepControl = {
  component: ComponentType<StepControlProps>;
  // Config keys this control manages; the generic config form hides these raw fields.
  ownedKeys: string[];
};
