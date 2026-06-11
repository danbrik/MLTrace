import { SequentialMethodBuilder } from './SequentialMethodBuilder';
import { createDefaultSequentialGraph } from '../utils';
import type { MethodBuilderDefinition } from '../types';

export const sequentialVaeBuilder: MethodBuilderDefinition = {
  label: 'Sequential variational autoencoder',
  component: SequentialMethodBuilder,
  createDefaultGraph: createDefaultSequentialGraph,
};
