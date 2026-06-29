import { Alert, Badge, Group, Paper, Select, SimpleGrid, Stack, Text, Tooltip } from '@mantine/core';
import { Info } from 'lucide-react';

import { BufferedNumberInput } from '../schema/BufferedNumberInput';
import { SchemaForm } from '../schema/SchemaForm';
import type { MethodBuilderProps, ModelConfig } from '../types';

type FastBlock = {
  id: string;
  block_type: string;
  direction: 'up' | 'down';
  out_channels: number;
  normalization: 'none' | 'layer_norm' | 'batch_norm';
};

const SECTIONS = [
  {
    key: 'generator_blocks',
    title: 'Generator',
    color: 'blue',
    subtitle: 'ResNet upsampling blocks, z -> image',
  },
  {
    key: 'critic_blocks',
    title: 'Critic',
    color: 'orange',
    subtitle: 'ResNet downsampling blocks, image -> critic/features. LayerNorm only; BatchNorm is invalid for WGAN-GP.',
  },
  {
    key: 'encoder_blocks',
    title: 'Encoder',
    color: 'green',
    subtitle: 'ResNet downsampling blocks, image -> latent z',
  },
] as const;

function asBlocks(value: unknown): FastBlock[] {
  return Array.isArray(value) ? (value as FastBlock[]) : [];
}

export function FastAnoganBuilder({
  method,
  modelConfig,
  modelGraph,
  disabled = false,
  onConfigChange,
  onGraphChange,
  onNumberDraftChange,
}: MethodBuilderProps) {
  function updateBlock(section: string, blockId: string, partial: Partial<FastBlock>) {
    if (disabled) return;
    onGraphChange((current) => {
      const blocks = asBlocks(current[section]);
      return {
        ...current,
        [section]: blocks.map((block) => (block.id === blockId ? { ...block, ...partial } : block)),
      };
    });
  }

  return (
    <Stack gap="md">
      <Paper withBorder p="sm" radius="sm">
        <Stack gap="sm">
          <Group gap={6}>
            <Text fw={700}>Input and latent</Text>
            <Tooltip
              label="fastAnoGAN encodes an input image to z, reconstructs it through the generator, and scores pixel plus critic-feature residuals."
              multiline
              w={320}
              withArrow
            >
              <Info size={14} />
            </Tooltip>
          </Group>
          <SchemaForm
            schema={method.method_schema}
            config={modelConfig as ModelConfig}
            keys={[
              'input_channels',
              'input_width',
              'input_height',
              'latent_dim',
              'latent_distribution',
              'encoder_output_activation',
              'generator_seed_size',
              'output_activation',
              'kappa',
            ]}
            disabled={disabled}
            fieldPrefix="fast_anogan.method"
            onChange={onConfigChange}
            onNumberDraftChange={onNumberDraftChange}
          />
        </Stack>
      </Paper>

      <Alert color="orange" title="WGAN-GP critic normalization">
        Critic blocks must use LayerNorm or no normalization. BatchNorm couples samples in a batch and is rejected for
        fastAnoGAN because it breaks the WGAN-GP per-sample gradient penalty assumption.
      </Alert>

      {SECTIONS.map((section) => {
        const blocks = asBlocks(modelGraph[section.key]);
        return (
          <Paper key={section.key} withBorder p="sm" radius="sm" className={`model-section-${section.key}`}>
            <Stack gap="sm">
              <Group justify="space-between">
                <div>
                  <Text fw={700}>{section.title}</Text>
                  <Text size="xs" c="dimmed">
                    {section.subtitle}
                  </Text>
                </div>
                <Badge color={section.color} variant="light">
                  {blocks.length} residual block(s)
                </Badge>
              </Group>
              <Stack gap="xs">
                {blocks.map((block, index) => (
                  <Paper key={block.id} withBorder p="xs" radius="sm">
                    <SimpleGrid cols={{ base: 1, sm: 4 }}>
                      <Text size="sm" fw={600}>
                        {index + 1}. {block.direction === 'up' ? 'Upsample' : 'Downsample'}
                      </Text>
                      <BufferedNumberInput
                        label="Out channels"
                        value={block.out_channels}
                        min={1}
                        integerOnly
                        disabled={disabled}
                        onCommit={(value) => updateBlock(section.key, block.id, { out_channels: value })}
                        onDraftStateChange={(state) => onNumberDraftChange?.(`fast_anogan.${section.key}.${block.id}.out_channels`, state)}
                      />
                      <Select
                        label="Direction"
                        data={['up', 'down']}
                        value={block.direction}
                        disabled
                      />
                      <Select
                        label="Normalization"
                        data={section.key === 'critic_blocks' ? ['layer_norm', 'none'] : ['none', 'layer_norm']}
                        value={block.normalization}
                        disabled={disabled}
                        onChange={(value) => updateBlock(section.key, block.id, { normalization: (value ?? 'none') as FastBlock['normalization'] })}
                        allowDeselect={false}
                      />
                    </SimpleGrid>
                  </Paper>
                ))}
              </Stack>
            </Stack>
          </Paper>
        );
      })}
    </Stack>
  );
}
