import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Collapse,
  Group,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  TextInput,
} from '@mantine/core';
import { ArrowDown, ArrowUp, ChevronDown, ChevronRight, Plus, Trash2 } from 'lucide-react';
import type { ReactNode } from 'react';
import { useMemo, useState } from 'react';

import type { SchemaProperty } from '../../types';
import { makeLayer } from '../utils';
import { BufferedNumberInput } from '../schema/BufferedNumberInput';
import { SchemaForm } from '../schema/SchemaForm';
import type { GraphSection, MethodBuilderProps, ModelConfig } from '../types';

function CollapsibleMethodSection({
  blockId,
  title,
  subtitle,
  rightSection,
  sectionClass,
  expandedBlocks,
  toggleBlock,
  children,
}: {
  blockId: string;
  title: string;
  subtitle: string | null;
  rightSection?: ReactNode;
  sectionClass?: string;
  expandedBlocks: Record<string, boolean>;
  toggleBlock: (blockId: string) => void;
  children: ReactNode;
}) {
  const expanded = expandedBlocks[blockId] === true;
  return (
    <Paper withBorder p="sm" radius="sm" className={`method-builder-section ${sectionClass ?? `model-section-${blockId}`}`}>
      <Stack gap="sm">
        <Group justify="space-between" align="center">
          <Group gap="xs">
            <ActionIcon variant="subtle" onClick={() => toggleBlock(blockId)} aria-label={expanded ? `Collapse ${title}` : `Expand ${title}`}>
              {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </ActionIcon>
            <div>
              <Text fw={700}>{title}</Text>
              {subtitle && (
                <Text size="xs" c="dimmed">
                  {subtitle}
                </Text>
              )}
            </div>
          </Group>
          {rightSection}
        </Group>
        <Collapse in={expanded}>{children}</Collapse>
      </Stack>
    </Paper>
  );
}

function layerConfigField({
  value,
  fieldKey,
  property,
  onChange,
  fieldId,
  onNumberDraftChange,
}: {
  value: unknown;
  fieldKey: string;
  property: SchemaProperty;
  onChange: (key: string, value: unknown) => void;
  fieldId: string;
  onNumberDraftChange?: MethodBuilderProps['onNumberDraftChange'];
}) {
  const resolvedValue = value ?? property.default ?? '';
  if (property.enum) {
    return (
      <Select
        key={fieldKey}
        label={property.label ?? fieldKey}
        data={property.enum}
        value={String(resolvedValue)}
        onChange={(next) => onChange(fieldKey, next ?? property.default)}
      />
    );
  }
  if (property.type === 'boolean') {
    return (
      <Switch
        key={fieldKey}
        label={property.label ?? fieldKey}
        checked={resolvedValue === true}
        onChange={(event) => onChange(fieldKey, event.currentTarget.checked)}
      />
    );
  }
  if (property.type === 'integer' || property.type === 'number') {
    return (
      <BufferedNumberInput
        key={fieldKey}
        label={property.label ?? fieldKey}
        min={property.minimum}
        max={property.maximum}
        integerOnly={property.type === 'integer'}
        value={typeof resolvedValue === 'number' || typeof resolvedValue === 'string' ? resolvedValue : ''}
        onCommit={(next) => onChange(fieldKey, next)}
        onDraftStateChange={(state) => onNumberDraftChange?.(fieldId, state)}
      />
    );
  }
  return (
    <TextInput
      key={fieldKey}
      label={property.label ?? fieldKey}
      value={String(resolvedValue)}
      onChange={(event) => onChange(fieldKey, event.currentTarget.value)}
    />
  );
}

function updateGraphSection(
  modelGraph: MethodBuilderProps['modelGraph'],
  section: GraphSection,
  updater: (layers: NonNullable<MethodBuilderProps['modelGraph']['encoder']>) => NonNullable<MethodBuilderProps['modelGraph']['encoder']>,
) {
  const currentLayers = ((modelGraph[section] as NonNullable<MethodBuilderProps['modelGraph']['encoder']> | undefined) ?? []);
  return { ...modelGraph, [section]: updater(currentLayers) };
}

export function SequentialMethodBuilder({
  method,
  modelConfig,
  modelGraph,
  layers,
  validation,
  onConfigChange,
  onGraphChange,
  onNumberDraftChange,
}: MethodBuilderProps) {
  const [expandedBlocks, setExpandedBlocks] = useState<Record<string, boolean>>({});
  const [expandedLayers, setExpandedLayers] = useState<Record<string, boolean>>({});
  const [encoderAddCategory, setEncoderAddCategory] = useState<string | null>('Convolution');
  const [decoderAddCategory, setDecoderAddCategory] = useState<string | null>('Convolution');
  const [encoderAddType, setEncoderAddType] = useState<string | null>('Conv2d');
  const [decoderAddType, setDecoderAddType] = useState<string | null>('ConvTranspose2d');

  const layerByType = useMemo(() => new Map(layers.map((layer) => [layer.type, layer])), [layers]);
  const layerOptions = useMemo(
    () => layers.map((layer) => ({ value: layer.type, label: `${layer.label} (${layer.category})` })),
    [layers],
  );
  const layerCategories = useMemo(
    () =>
      Array.from(new Set(layers.map((layer) => layer.category)))
        .sort()
        .map((category) => ({ value: category, label: category })),
    [layers],
  );
  const layerOptionsByCategory = useMemo(() => {
    const byCategory = new Map<string, { value: string; label: string }[]>();
    for (const layer of layers) {
      const entries = byCategory.get(layer.category) ?? [];
      entries.push({ value: layer.type, label: layer.label });
      byCategory.set(layer.category, entries);
    }
    for (const [category, entries] of byCategory.entries()) {
      byCategory.set(category, entries.sort((left, right) => left.label.localeCompare(right.label)));
    }
    return byCategory;
  }, [layers]);

  function toggleBlock(blockId: string) {
    setExpandedBlocks((current) => ({ ...current, [blockId]: current[blockId] !== true }));
  }

  function toggleLayer(layerId: string) {
    setExpandedLayers((current) => ({ ...current, [layerId]: current[layerId] !== true }));
  }

  function updateLayer(section: GraphSection, layerId: string, partial: Record<string, unknown>) {
    onGraphChange((current) =>
      updateGraphSection(current, section, (sectionLayers) =>
        sectionLayers.map((layer) => (layer.id === layerId ? { ...layer, ...partial } : layer)),
      ),
    );
  }

  function updateLayerConfig(section: GraphSection, layerId: string, key: string, value: unknown) {
    onGraphChange((current) =>
      updateGraphSection(current, section, (sectionLayers) =>
        sectionLayers.map((layer) => (layer.id === layerId ? { ...layer, config: { ...layer.config, [key]: value } } : layer)),
      ),
    );
  }

  function addLayer(section: GraphSection, layerType: string | null) {
    if (!layerType) return;
    onGraphChange((current) =>
      updateGraphSection(current, section, (sectionLayers) => [...sectionLayers, makeLayer(layerType, layerByType)]),
    );
  }

  function removeLayer(section: GraphSection, layerId: string) {
    onGraphChange((current) => updateGraphSection(current, section, (sectionLayers) => sectionLayers.filter((layer) => layer.id !== layerId)));
  }

  function moveLayer(section: GraphSection, layerId: string, direction: -1 | 1) {
    onGraphChange((current) =>
      updateGraphSection(current, section, (sectionLayers) => {
        const nextLayers = [...sectionLayers];
        const index = nextLayers.findIndex((layer) => layer.id === layerId);
        const nextIndex = index + direction;
        if (index < 0 || nextIndex < 0 || nextIndex >= nextLayers.length) return sectionLayers;
        [nextLayers[index], nextLayers[nextIndex]] = [nextLayers[nextIndex], nextLayers[index]];
        return nextLayers;
      }),
    );
  }

  function renderLayerList(section: GraphSection) {
    const sectionLayers = ((modelGraph[section] as NonNullable<typeof modelGraph.encoder> | undefined) ?? []);
    const addCategory = section === 'encoder' ? encoderAddCategory : decoderAddCategory;
    const setAddCategory = section === 'encoder' ? setEncoderAddCategory : setDecoderAddCategory;
    const addType = section === 'encoder' ? encoderAddType : decoderAddType;
    const setAddType = section === 'encoder' ? setEncoderAddType : setDecoderAddType;
    const exactLayerOptions = addCategory ? (layerOptionsByCategory.get(addCategory) ?? []) : [];

    return (
      <CollapsibleMethodSection
        blockId={section}
        title={section === 'encoder' ? 'Encoder' : 'Decoder'}
        subtitle="Ordered layer stack"
        rightSection={<Badge variant="light">{sectionLayers.length} layer(s)</Badge>}
        expandedBlocks={expandedBlocks}
        toggleBlock={toggleBlock}
      >
        <Stack gap="sm">
          <Group align="flex-end">
            <Select
              label="Layer category"
              data={layerCategories}
              value={addCategory}
              onChange={(nextCategory) => {
                setAddCategory(nextCategory);
                setAddType(nextCategory ? (layerOptionsByCategory.get(nextCategory)?.[0]?.value ?? null) : null);
              }}
              flex={1}
            />
            <Select label="Layer" data={exactLayerOptions} value={addType} onChange={setAddType} searchable flex={1} />
            <Button leftSection={<Plus size={16} />} variant="light" onClick={() => addLayer(section, addType)} disabled={!addType}>
              Add layer
            </Button>
          </Group>

          {sectionLayers.map((layer, index) => {
            const definition = layerByType.get(layer.type);
            const layerSpec = validation?.layer_specs.find((spec) => spec.section === section && spec.layer_id === layer.id);
            const layerExpanded = expandedLayers[layer.id] === true;
            return (
              <Paper key={layer.id} withBorder p="sm" radius="sm" className={`model-layer-card model-section-${section}`}>
                <Stack gap="sm">
                  <Group justify="space-between" align="flex-start">
                    <Group gap="xs">
                      <ActionIcon
                        variant="subtle"
                        onClick={() => toggleLayer(layer.id)}
                        aria-label={layerExpanded ? `Collapse ${layer.type}` : `Expand ${layer.type}`}
                      >
                        {layerExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                      </ActionIcon>
                      <Badge>{index + 1}</Badge>
                      <Select
                        data={layerOptions}
                        value={layer.type}
                        onChange={(nextType) => {
                          if (!nextType) return;
                          updateLayer(section, layer.id, {
                            type: nextType,
                            config: { ...(layerByType.get(nextType)?.default_config ?? {}) },
                          });
                        }}
                        searchable
                        w={260}
                      />
                      {layerSpec && (
                        <Badge variant="light" color="gray">
                          {layerSpec.input_label} {'->'} {layerSpec.output_label}
                        </Badge>
                      )}
                    </Group>
                    <Group gap={4}>
                      <ActionIcon variant="subtle" disabled={index === 0} onClick={() => moveLayer(section, layer.id, -1)}>
                        <ArrowUp size={16} />
                      </ActionIcon>
                      <ActionIcon
                        variant="subtle"
                        disabled={index === sectionLayers.length - 1}
                        onClick={() => moveLayer(section, layer.id, 1)}
                      >
                        <ArrowDown size={16} />
                      </ActionIcon>
                      <ActionIcon color="red" variant="subtle" onClick={() => removeLayer(section, layer.id)}>
                        <Trash2 size={16} />
                      </ActionIcon>
                    </Group>
                  </Group>
                  <Collapse in={layerExpanded}>
                    <Stack gap="sm">
                      {definition?.shape_notes && (
                        <Text size="xs" c="dimmed">
                          {definition.shape_notes}
                        </Text>
                      )}
                      <SimpleGrid cols={{ base: 1, sm: 2, xl: 3 }}>
                        {Object.entries(definition?.config_schema.properties ?? {}).map(([key, property]) =>
                          layerConfigField({
                            value: layer.config[key],
                            fieldKey: key,
                            property,
                            fieldId: `layer.${section}.${layer.id}.${key}`,
                            onChange: (configKey, value) => updateLayerConfig(section, layer.id, configKey, value),
                            onNumberDraftChange,
                          }),
                        )}
                      </SimpleGrid>
                    </Stack>
                  </Collapse>
                </Stack>
              </Paper>
            );
          })}
        </Stack>
      </CollapsibleMethodSection>
    );
  }

  return (
    <Stack gap="md">
      <CollapsibleMethodSection
        blockId="input"
        title="Input"
        subtitle="Source tensor shape"
        expandedBlocks={expandedBlocks}
        toggleBlock={toggleBlock}
      >
        <SchemaForm
          schema={method.method_schema}
          config={modelConfig as ModelConfig}
          keys={['input_channels', 'input_width', 'input_height']}
          fieldPrefix="method.input"
          onChange={onConfigChange}
          onNumberDraftChange={onNumberDraftChange}
        />
      </CollapsibleMethodSection>
      {renderLayerList('encoder')}
      <CollapsibleMethodSection
        blockId="latent"
        title="Latent"
        subtitle="Implicit flatten and projection bridge"
        expandedBlocks={expandedBlocks}
        toggleBlock={toggleBlock}
      >
        <SchemaForm
          schema={method.method_schema}
          config={modelConfig as ModelConfig}
          keys={['latent_dim', 'kl_weight']}
          fieldPrefix="method.latent"
          onChange={onConfigChange}
          onNumberDraftChange={onNumberDraftChange}
        />
      </CollapsibleMethodSection>
      {method.builder_kind === 'sequential_variational_autoencoder' && (
        <Alert color="violet" title="VAE latent block">
          The latent bridge stores mu/logvar metadata. Static validation always runs; the optional Torch check runs only when Torch is installed.
        </Alert>
      )}
      {renderLayerList('decoder')}
      <CollapsibleMethodSection
        blockId="output"
        title="Output"
        subtitle="Reconstruction settings"
        expandedBlocks={expandedBlocks}
        toggleBlock={toggleBlock}
      >
        <SchemaForm
          schema={method.method_schema}
          config={modelConfig as ModelConfig}
          keys={['output_activation']}
          fieldPrefix="method.output"
          onChange={onConfigChange}
          onNumberDraftChange={onNumberDraftChange}
        />
      </CollapsibleMethodSection>
    </Stack>
  );
}
