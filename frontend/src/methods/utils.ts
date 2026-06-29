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
  const parts = [`input ${size}`];
  if (method.method_type === 'spatiotemporal_autoencoder') {
    parts.push(`clip ${formatValue(config.clip_length)}`);
    if (config.prediction_branch !== false) parts.push(`future ${formatValue(config.future_length)}`);
    return parts.join(', ');
  }
  if (method.method_type === 'fast_anogan') {
    parts.push(`latent ${formatValue(config.latent_dim)}`);
    parts.push(`kappa ${formatValue(config.kappa)}`);
    return parts.join(', ');
  }
  if (config.bottleneck_channels !== undefined) {
    parts.push(`bottleneck ${formatValue(config.bottleneck_channels)} ch`);
  } else {
    parts.push(`latent ${formatValue(config.latent_dim)}`);
  }
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
  const graph =
    method.builder_kind === 'form'
      ? {}
      : method.builder_kind === 'fast_anogan'
        ? { ...modelGraph, builder_kind: method.builder_kind }
        : {
            ...modelGraph,
            builder_kind: method.builder_kind,
            latent: {
              ...(modelGraph.latent ?? {}),
              latent_dim: modelConfig.latent_dim,
              bottleneck_channels: modelConfig.bottleneck_channels,
              kl_weight: modelConfig.kl_weight,
              reparameterization: method.builder_kind === 'sequential_variational_autoencoder',
            },
          };
  return {
    method_type: method.type,
    method_graph: graph,
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
  if (builderKind === 'spatiotemporal_autoencoder') {
    return createDefaultSpatioTemporalGraph(builderKind, layerByType);
  }
  if (builderKind === 'fast_anogan') {
    return createDefaultFastAnoganGraph(builderKind);
  }
  if (builderKind === 'sequential_spatial_autoencoder') {
    return createDefaultSpatialAutoencoderGraph(builderKind, layerByType, modelConfig);
  }
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

export function createDefaultFastAnoganGraph(builderKind = 'fast_anogan'): ModelGraph {
  const block = (prefix: string, index: number, outChannels: number, direction: 'up' | 'down', normalization: 'none' | 'layer_norm') => ({
    id: `${prefix}-${index}`,
    block_type: 'residual',
    direction,
    out_channels: outChannels,
    normalization,
  });
  return {
    builder_kind: builderKind,
    generator_blocks: [512, 256, 128, 64].map((channels, index) => block('gan-gen-up', index + 1, channels, 'up', 'none')),
    critic_blocks: [128, 256, 512, 512].map((channels, index) =>
      block('gan-critic-down', index + 1, channels, 'down', 'layer_norm'),
    ),
    encoder_blocks: [128, 256, 512, 512].map((channels, index) => block('gan-enc-down', index + 1, channels, 'down', 'none')),
    feature_layer: 'critic_blocks',
  };
}

function makeConfiguredLayer(
  layerType: string,
  layerByType: Map<string, ModelLayerDefinition>,
  id: string,
  config: Record<string, unknown>,
): ModelLayerInstance {
  const layer = makeLayer(layerType, layerByType);
  layer.id = id;
  layer.config = { ...layer.config, ...config };
  return layer;
}

function createDefaultSpatioTemporalGraph(
  builderKind: string,
  layerByType: Map<string, ModelLayerDefinition>,
): ModelGraph {
  const encoder: ModelLayerInstance[] = [];
  for (const [index, outChannels] of [32, 64, 128].entries()) {
    encoder.push(
      makeConfiguredLayer('Conv3d', layerByType, `stae-enc-conv-${index + 1}`, {
        out_channels: outChannels,
        kernel_size: 3,
        stride: 1,
        padding: 1,
      }),
      makeConfiguredLayer('BatchNorm3d', layerByType, `stae-enc-bn-${index + 1}`, {
        num_features: outChannels,
      }),
      makeConfiguredLayer('LeakyReLU', layerByType, `stae-enc-act-${index + 1}`, {
        negative_slope: 0.01,
        inplace: false,
      }),
      makeConfiguredLayer('MaxPool3d', layerByType, `stae-enc-pool-${index + 1}`, {
        kernel_size: 2,
        stride: 2,
        padding: 0,
      }),
    );
  }

  const decoder: ModelLayerInstance[] = [];
  const predictionDecoder: ModelLayerInstance[] = [];
  for (const [index, outChannels] of [64, 32, 1].entries()) {
    decoder.push(
      makeConfiguredLayer('ConvTranspose3d', layerByType, `stae-rec-deconv-${index + 1}`, {
        out_channels: outChannels,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        output_padding: 0,
        stride_t: 2,
        stride_xy: 2,
        output_padding_t: 1,
        output_padding_xy: 1,
      }),
    );
    predictionDecoder.push(
      makeConfiguredLayer('ConvTranspose3d', layerByType, `stae-pred-deconv-${index + 1}`, {
        out_channels: outChannels,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        output_padding: 0,
        stride_t: 1,
        stride_xy: 2,
        output_padding_t: 0,
        output_padding_xy: 1,
      }),
    );
    if (index < 2) {
      decoder.push(makeConfiguredLayer('LeakyReLU', layerByType, `stae-rec-act-${index + 1}`, { negative_slope: 0.01, inplace: false }));
      predictionDecoder.push(makeConfiguredLayer('LeakyReLU', layerByType, `stae-pred-act-${index + 1}`, { negative_slope: 0.01, inplace: false }));
    }
  }
  return {
    builder_kind: builderKind,
    encoder,
    latent: { bottleneck_kind: 'spatiotemporal', shape: '128x1x32x32' },
    decoder,
    prediction_decoder: predictionDecoder,
  };
}

function createDefaultSpatialAutoencoderGraph(
  builderKind: string,
  layerByType: Map<string, ModelLayerDefinition>,
  modelConfig: ModelConfig,
): ModelGraph {
  const inputChannels = numericConfig(modelConfig.input_channels, 1);
  const bottleneckChannels = numericConfig(modelConfig.bottleneck_channels, 16);
  const channels = [32, 64, 128, 256, 256];
  const encoder = channels.flatMap((outChannels, index) => {
    const conv = makeLayer('Conv2d', layerByType);
    conv.id = `Conv2d-spatial-enc-${index + 1}`;
    conv.config = { ...conv.config, out_channels: outChannels, kernel_size: 3, stride: 2, padding: 1 };
    const relu = makeLayer('ReLU', layerByType);
    relu.id = `ReLU-spatial-enc-${index + 1}`;
    return [conv, relu];
  });
  const bottleneck = makeLayer('Conv2d', layerByType);
  bottleneck.id = 'Conv2d-spatial-bottleneck';
  bottleneck.config = { ...bottleneck.config, out_channels: bottleneckChannels, kernel_size: 1, stride: 1, padding: 0 };
  encoder.push(bottleneck);

  const seed = makeLayer('Conv2d', layerByType);
  seed.id = 'Conv2d-spatial-seed';
  seed.config = { ...seed.config, out_channels: 256, kernel_size: 1, stride: 1, padding: 0 };
  const decoder = [seed];
  for (const [index, outChannels] of [256, 128, 64, 32, inputChannels].entries()) {
    const deconv = makeLayer('ConvTranspose2d', layerByType);
    deconv.id = `ConvTranspose2d-spatial-dec-${index + 1}`;
    deconv.config = { ...deconv.config, out_channels: outChannels, kernel_size: 3, stride: 2, padding: 1, output_padding: 1 };
    decoder.push(deconv);
    if (index < 4) {
      const relu = makeLayer('ReLU', layerByType);
      relu.id = `ReLU-spatial-dec-${index + 1}`;
      decoder.push(relu);
    }
  }

  return {
    builder_kind: builderKind,
    encoder,
    latent: {
      bottleneck_kind: 'spatial',
      bottleneck_channels: bottleneckChannels,
      height: 8,
      width: 8,
    },
    decoder,
  };
}
