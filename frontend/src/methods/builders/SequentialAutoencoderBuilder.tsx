import { SequentialMethodBuilder } from './SequentialMethodBuilder';
import { createDefaultSequentialGraph } from '../utils';
import type { MethodBuilderDefinition } from '../types';

export const sequentialAutoencoderBuilder: MethodBuilderDefinition = {
  label: 'Sequential autoencoder',
  component: SequentialMethodBuilder,
  createDefaultGraph: createDefaultSequentialGraph,
};
