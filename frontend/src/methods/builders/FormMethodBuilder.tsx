import { Stack } from '@mantine/core';

import { SchemaForm } from '../schema/SchemaForm';
import type { MethodBuilderProps } from '../types';

export function FormMethodBuilder({ method, modelConfig, onConfigChange, onNumberDraftChange }: MethodBuilderProps) {
  return (
    <Stack gap="md">
      <SchemaForm
        title={`${method.label} configuration`}
        schema={method.method_schema}
        config={modelConfig}
        fieldPrefix="method.form"
        onChange={onConfigChange}
        onNumberDraftChange={onNumberDraftChange}
      />
    </Stack>
  );
}
