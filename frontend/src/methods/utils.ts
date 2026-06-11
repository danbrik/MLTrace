import type {
  MethodConfiguration,
  MethodConfigurationPayload,
  MethodDefinition,
  ModelGraph,
  ModelLayerDefinition,
  ModelLayerInstance,
} from '../types';
import type { ModelConfig } from './types';

export function newLayerId(layerType: string): string {
  return `${layerType}-${crypto.randomUUID().slice(0, 8)}`;
}

export function nextAvailableMethodName(methods: MethodConfiguration[]): string {
  const used = new Set(methods.map((method) => method.name.trim().toLowerCase()));
  const base = 'Untitled method';
  if (!used.has(base.toLowerCase())) return base;
  for (let index = 2; index < 10000; index += 1) {
    const candidate = `${base} ${index}`;
    if (!used.has(candidate.toLowerCase())) return candidate;
  }
  return `${base} ${Date.now()}`;
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'n/a';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
}

export function methodLabel(method: MethodDefinition | undefined, type: string): string {
  return method?.label ?? type;
}

export function trainingModeLabel(trainingMode: string | undefined): string {
  if (trainingMode === 'gradient') return 'Gradient training';
  if (trainingMode === 'fit') return 'Training';
  return 'No Training';
}

export function trainingModeColor(trainingMode: string | undefined): string {
  if (trainingMode === 'gradient') return 'green';
  if (trainingMode === 'fit') return 'blue';
  return 'gray';
}

export function makeLayer(layerType: string, layerByType: Map<string, ModelLayerDefinition>): ModelLayerInstance {
  const definition = layerByType.get(layerType);
  return {
    id: newLayerId(layerType),
    type: layerType,
    config: { ...(definition?.default_config ?? {}) },
  };
}

export function numericConfig(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function conv2dOutput(size: number, kernelSize: number, stride: number, padding: number): number {
  return Math.floor((size + 2 * padding - (kernelSize - 1) - 1) / stride + 1);
}

export function schemaDefaults(schema: MethodDefinition['method_schema'] | undefined): ModelConfig {
  const defaults: ModelConfig = {};
  for (const [key, property] of Object.entries(schema?.properties ?? {})) {
    if (property.default !== undefined) defaults[key] = property.default;
  }
  return defaults;
}

export function keyParameters(method: MethodConfiguration): string {
  const config = method.method_config ?? method.model_config;
  if (method.method_type === 'mean_image') {
    return [
      `aggregation ${formatValue(config.aggregation)}`,
      `acc ${formatValue(config.accumulator_dtype)}`,
      `out ${formatValue(config.output_dtype_policy)}`,
    ].join(', ');
  }
  const size =
    config.input_width && config.input_height
      ? `${formatValue(config.input_width)}x${formatValue(config.input_height)}`
      : 'size n/a';
  const parts = [`input ${size}`, `latent ${formatValue(config.latent_dim)}`];
  if (config.kl_weight !== undefined) parts.push(`KL ${formatValue(config.kl_weight)}`);
  return parts.join(', ');
}

// Method payload creation is centralized so save, diagram validation, and torch checks
// serialize the graph in the same way. Diagram payloads intentionally omit training
// and inference configs because static shape validation only depends on architecture.
export function buildMethodPayload(
  method: MethodDefinition | undefined,
  modelGraph: ModelGraph,
  modelConfig: ModelConfig,
  trainingConfig: ModelConfig,
  inferenceConfig: ModelConfig,
  options: { diagramOnly?: boolean } = {},
): MethodConfigurationPayload | null {
  if (!method) return null;
  return {
    method_type: method.type,
    method_graph:
      method.builder_kind === 'form'
        ? {}
        : {
            ...modelGraph,
            builder_kind: method.builder_kind,
            latent: {
              ...(modelGraph.latent ?? {}),
              latent_dim: modelConfig.latent_dim,
              kl_weight: modelConfig.kl_weight,
              reparameterization: method.builder_kind === 'sequential_variational_autoencoder',
            },
          },
    method_config: modelConfig,
    training_config: options.diagramOnly ? {} : trainingConfig,
    inference_config: options.diagramOnly ? {} : inferenceConfig,
  };
}

export function createDefaultSequentialGraph(
  builderKind: string,
  layerByType: Map<string, ModelLayerDefinition>,
  modelConfig: ModelConfig,
): ModelGraph {
  const inputChannels = numericConfig(modelConfig.input_channels, 1);
  const inputWidth = numericConfig(modelConfig.input_width, 160);
  const inputHeight = numericConfig(modelConfig.input_height, 120);
  const latentDim = numericConfig(modelConfig.latent_dim, 64);
  const encoderChannels = 8;
  const encodedWidth = conv2dOutput(inputWidth, 3, 2, 1);
  const encodedHeight = conv2dOutput(inputHeight, 3, 2, 1);
  const encodedFeatures = encoderChannels * encodedHeight * encodedWidth;
  const outputPadding = inputWidth % 2 === 0 && inputHeight % 2 === 0 ? 1 : 0;

  const encoder = ['Conv2d', 'ReLU', 'Flatten', 'Linear']
    .filter((type) => layerByType.has(type))
    .map((type) => makeLayer(type, layerByType));
  const encoderConv = encoder.find((layer) => layer.type === 'Conv2d');
  if (encoderConv) {
    encoderConv.config = { ...encoderConv.config, out_channels: encoderChannels, kernel_size: 3, stride: 2, padding: 1 };
  }
  const encoderLinear = encoder.find((layer) => layer.type === 'Linear');
  if (encoderLinear) {
    encoderLinear.config = { ...encoderLinear.config, out_features: latentDim };
  }

  const decoder = ['Linear', 'Unflatten', 'ConvTranspose2d']
    .filter((type) => layerByType.has(type))
    .map((type) => makeLayer(type, layerByType));
  const decoderLinear = decoder.find((layer) => layer.type === 'Linear');
  if (decoderLinear) {
    decoderLinear.config = { ...decoderLinear.config, out_features: encodedFeatures };
  }
  const decoderUnflatten = decoder.find((layer) => layer.type === 'Unflatten');
  if (decoderUnflatten) {
    decoderUnflatten.config = { ...decoderUnflatten.config, channels: encoderChannels, height: encodedHeight, width: encodedWidth };
  }
  const decoderConv = decoder.find((layer) => layer.type === 'ConvTranspose2d');
  if (decoderConv) {
    decoderConv.config = {
      ...decoderConv.config,
      out_channels: inputChannels,
      kernel_size: 3,
      stride: 2,
      padding: 1,
      output_padding: outputPadding,
    };
  }

  return {
    builder_kind: builderKind,
    encoder,
    latent: {
      latent_dim: latentDim,
      reparameterization: builderKind === 'sequential_variational_autoencoder',
    },
    decoder,
  };
}
