import { FormMethodBuilder } from './builders/FormMethodBuilder';
import { FastAnoganBuilder } from './builders/FastAnoganBuilder';
import { sequentialAutoencoderBuilder } from './builders/SequentialAutoencoderBuilder';
import { sequentialVaeBuilder } from './builders/SequentialVaeBuilder';
import type { MethodBuilderDefinition } from './types';
import { createDefaultFastAnoganGraph } from './utils';

// Frontend builders are selected by backend-provided builder_kind. Adding a new
// complex method UI should add one registry entry here; schema-only methods can
// reuse the "form" builder without touching MethodsPage.
export const methodBuilderRegistry: Record<string, MethodBuilderDefinition> = {
  form: {
    label: 'Schema form',
    component: FormMethodBuilder,
    createDefaultGraph: () => ({}),
  },
  sequential_autoencoder: sequentialAutoencoderBuilder,
  sequential_spatial_autoencoder: sequentialAutoencoderBuilder,
  sequential_variational_autoencoder: sequentialVaeBuilder,
  spatiotemporal_autoencoder: sequentialAutoencoderBuilder,
  fast_anogan: {
    label: 'fastAnoGAN',
    component: FastAnoganBuilder,
    createDefaultGraph: () => createDefaultFastAnoganGraph(),
  },
};

export function getMethodBuilder(builderKind: string | undefined): MethodBuilderDefinition | undefined {
  return builderKind ? methodBuilderRegistry[builderKind] : undefined;
}
