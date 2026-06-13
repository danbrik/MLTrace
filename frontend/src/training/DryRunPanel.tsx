import {
  Alert,
  Badge,
  Button,
  Collapse,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { ChevronDown, ChevronUp, FlaskConical } from 'lucide-react';
import { useState } from 'react';

import { dryRunTrainingPipeline } from '../api';
import type { PreprocessingPreviewImage, TrainingPipelineDryRun, TrainingPipelinePayload } from '../types';

function imageCaption(preview: PreprocessingPreviewImage): string {
  return `${preview.width}x${preview.height} · ${preview.channels} ch · ${preview.dtype} · [${preview.value_min.toFixed(3)}, ${preview.value_max.toFixed(3)}]`;
}

function PreviewCard({ title, src, caption }: { title: string; src: string; caption: string }) {
  return (
    <Stack gap={4}>
      <Text fw={700} size="sm">
        {title}
      </Text>
      <img src={src} alt={title} className="preview-image" />
      <Text size="xs" c="dimmed">
        {caption}
      </Text>
    </Stack>
  );
}

/**
 * Runs the dummy test: first training image -> preprocessing -> model forward
 * pass with random weights, and renders before/after/model-output side by side.
 */
export function DryRunPanel({
  payload,
  disabled,
}: {
  payload: TrainingPipelinePayload | null;
  disabled: boolean;
}) {
  const [result, setResult] = useState<TrainingPipelineDryRun | null>(null);
  const [loading, setLoading] = useState(false);
  const [showAllSteps, setShowAllSteps] = useState(false);

  async function handleRun() {
    if (!payload) return;
    setLoading(true);
    setResult(null);
    try {
      const next = await dryRunTrainingPipeline(payload);
      setResult(next);
      notifications.show({
        color: next.valid ? 'green' : 'red',
        title: next.valid ? 'Dummy test passed' : 'Dummy test failed',
        message: next.valid ? `Mode: ${next.mode}` : next.errors[0] ?? 'Unknown error',
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setResult({
        valid: false,
        mode: 'failed',
        errors: [message],
        warnings: [],
        logs: [],
        training_dataset_name: null,
        source_image_path: null,
        source_timestamp: null,
        preprocessing_previews: [],
        model_output: null,
        note: null,
      });
      notifications.show({ color: 'red', title: 'Dummy test failed', message });
    } finally {
      setLoading(false);
    }
  }

  const previews = result?.preprocessing_previews ?? [];
  const before = previews[0];
  const after = previews.length > 1 ? previews[previews.length - 1] : undefined;

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Group justify="space-between" align="center">
          <Title order={3}>Dummy Test</Title>
          <Button
            leftSection={<FlaskConical size={18} />}
            loading={loading}
            disabled={disabled}
            onClick={handleRun}
          >
            Run dummy test
          </Button>
        </Group>
        <Text size="sm" c="dimmed">
          Takes the first image of the first training set, runs it through the preprocessing pipeline and a
          randomly initialized model to validate the composition end to end.
        </Text>

        {result && (
          <Stack gap="md">
            {result.errors.length > 0 && (
              <Alert color="red" title="Dummy test failed">
                <Stack gap={4}>
                  {result.errors.map((error, index) => (
                    <Text key={index} size="sm">
                      {error}
                    </Text>
                  ))}
                </Stack>
              </Alert>
            )}
            {result.warnings.length > 0 && (
              <Alert color="yellow" title="Warnings">
                <Stack gap={4}>
                  {result.warnings.map((warning, index) => (
                    <Text key={index} size="sm">
                      {warning}
                    </Text>
                  ))}
                </Stack>
              </Alert>
            )}
            {result.note && (
              <Alert color="blue" title="First contribution to the artifact">
                {result.note}
              </Alert>
            )}

            {result.source_image_path && (
              <Text size="xs" c="dimmed" className="mono path-text">
                Source: {result.source_image_path}
                {result.training_dataset_name ? ` (training set: ${result.training_dataset_name})` : ''}
              </Text>
            )}

            {before && (
              <SimpleGrid cols={{ base: 1, md: result.model_output ? 3 : 2 }}>
                <PreviewCard
                  title="Before preprocessing"
                  src={before.image_data_url}
                  caption={imageCaption(before)}
                />
                {after && (
                  <PreviewCard
                    title="After preprocessing"
                    src={after.image_data_url}
                    caption={imageCaption(after)}
                  />
                )}
                {result.model_output && (
                  <PreviewCard
                    title="Model output (random weights)"
                    src={result.model_output.image_data_url}
                    caption={`in ${result.model_output.input_shape.join('x')} -> out ${result.model_output.output_shape.join('x')} · ${result.model_output.elapsed_ms} ms`}
                  />
                )}
              </SimpleGrid>
            )}

            {previews.length > 2 && (
              <Stack gap="xs">
                <Button
                  variant="subtle"
                  size="compact-sm"
                  leftSection={showAllSteps ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  onClick={() => setShowAllSteps((current) => !current)}
                >
                  {showAllSteps ? 'Hide' : 'Show'} all {previews.length} preprocessing steps
                </Button>
                <Collapse in={showAllSteps}>
                  <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
                    {previews.map((preview, index) => (
                      <Stack key={preview.node_id} gap={4}>
                        <Group gap="xs">
                          <Badge size="xs" variant="filled" color="yellow">
                            {index + 1}
                          </Badge>
                          <Text size="sm" fw={600}>
                            {preview.label}
                          </Text>
                        </Group>
                        <img src={preview.image_data_url} alt={preview.label} className="preview-image" />
                        <Text size="xs" c="dimmed">
                          {imageCaption(preview)}
                        </Text>
                      </Stack>
                    ))}
                  </SimpleGrid>
                </Collapse>
              </Stack>
            )}

            {result.logs.length > 0 && (
              <Stack gap={2}>
                <Text fw={700} size="sm">
                  Logs
                </Text>
                {result.logs.map((log, index) => (
                  <Text key={index} size="xs" className="mono" c="dimmed">
                    {log}
                  </Text>
                ))}
              </Stack>
            )}
          </Stack>
        )}
      </Stack>
    </Paper>
  );
}
