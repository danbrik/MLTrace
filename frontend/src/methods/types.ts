import type { ComponentType } from 'react';

import type {
  ConfigSchema,
  MethodDefinition,
  MethodValidationResponse,
  ModelGraph,
  ModelLayerDefinition,
  ModelLayerInstance,
} from '../types';

export type ModelConfig = Record<string, unknown>;

export type GraphSection = 'encoder' | 'decoder';

export type MethodBuilderProps = {
  method: MethodDefinition;
  modelConfig: ModelConfig;
  modelGraph: ModelGraph;
  layers: ModelLayerDefinition[];
  validation: MethodValidationResponse | null;
  onConfigChange: (key: string, value: unknown) => void;
  onGraphChange: (graph: ModelGraph | ((current: ModelGraph) => ModelGraph)) => void;
  onNumberDraftChange?: (fieldId: string, state: NumericDraftState | null) => void;
};

export type MethodBuilderDefinition = {
  label: string;
  component: ComponentType<MethodBuilderProps>;
  createDefaultGraph?: (
    builderKind: string,
    layerByType: Map<string, ModelLayerDefinition>,
    modelConfig: ModelConfig,
  ) => ModelGraph;
};

export type SchemaFormProps = {
  title?: string;
  schema: ConfigSchema | undefined;
  config: ModelConfig;
  keys?: string[];
  onChange: (key: string, value: unknown) => void;
  fieldPrefix?: string;
  onNumberDraftChange?: (fieldId: string, state: NumericDraftState | null) => void;
};

export type NumericDraftState = {
  dirty: boolean;
  valid: boolean;
  message?: string;
};

export type LayerMutationHandlers = {
  updateLayer: (section: GraphSection, layerId: string, partial: Partial<ModelLayerInstance>) => void;
  updateLayerConfig: (section: GraphSection, layerId: string, key: string, value: unknown) => void;
  addLayer: (section: GraphSection, layerType: string | null) => void;
  removeLayer: (section: GraphSection, layerId: string) => void;
  moveLayer: (section: GraphSection, layerId: string, direction: -1 | 1) => void;
};
