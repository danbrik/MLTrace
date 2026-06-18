import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Group,
  Modal,
  NumberInput,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
  Textarea,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';

import { StepCard } from '../components/StepCard';
import { Copy, Edit3, Info, Plus, Save, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import {
  cleanupTrainingDatasetInvalidRules,
  createTrainingDataset,
  deleteTrainingDataset,
  getTrainingDataset,
  listDatasets,
  listTrainingDatasets,
  previewTrainingDataset,
  updateTrainingDataset,
} from '../api';
import type { Dataset, DatasetFolder, TrainingDataset, TrainingDatasetPreview, TrainingDatasetRuleInput } from '../types';

type RuleRow = TrainingDatasetRuleInput & {
  localId: string;
};

type FolderChoice = {
  dataset: Dataset;
  folder: DatasetFolder;
  value: string;
  label: string;
  signature: string | null;
  metadataLabel: string;
  min: string;
  max: string;
};

const USAGE_OPTIONS = [
  { value: 'train', label: 'Train' },
  { value: 'test', label: 'Test' },
  { value: 'validation', label: 'Validation' },
  { value: 'mixed', label: 'Mixed' },
];

function toInputDateTime(value: string | null): string {
  if (!value) return '';
  return value.slice(0, 19);
}

function formatTimestamp(value: string | null): string {
  if (!value) return 'n/a';
  return value.replace('T', ' ');
}

function formatNameTimestamp(value: string): string {
  return value.replace('T', ' ');
}

function usageLabelText(value: string): string {
  return USAGE_OPTIONS.find((option) => option.value === value)?.label ?? value;
}

function usageLabelColor(value: string): string {
  if (value === 'test') return 'orange';
  if (value === 'validation') return 'violet';
  if (value === 'mixed') return 'gray';
  return 'teal';
}

function summaryKeys(summary: Record<string, number> | null): string[] {
  return summary ? Object.keys(summary).sort() : [];
}

function folderImageSignature(folder: DatasetFolder): string | null {
  if (!folder.resolution_summary || !folder.extension_summary || !folder.image_metadata) return null;
  const resolution = summaryKeys(folder.resolution_summary).join(',');
  const extension = summaryKeys(folder.extension_summary).join(',');
  const dtype = String(folder.image_metadata.dtype ?? 'unknown-dtype');
  const channels = folder.image_metadata.channels == null ? 'unknown-ch' : `${folder.image_metadata.channels}ch`;
  const mode = String(folder.image_metadata.mode ?? 'unknown-mode');
  return `${resolution} | ${extension} | ${dtype} | ${channels} | ${mode}`;
}

function folderMetadataLabel(folder: DatasetFolder): string {
  const resolution = summaryKeys(folder.resolution_summary).join(',') || 'size n/a';
  const extension = summaryKeys(folder.extension_summary).join(',') || 'filetype n/a';
  const metadata = folder.image_metadata ?? {};
  const dtype = metadata.dtype ? String(metadata.dtype) : 'dtype n/a';
  const channels = metadata.channels == null ? 'channels n/a' : `${metadata.channels}ch`;
  return `${resolution}, ${extension}, ${dtype}, ${channels}`;
}

function generatedTrainingDatasetName(rules: RuleRow[]): string {
  const starts = rules.map((rule) => rule.start_timestamp).filter(Boolean).sort();
  const ends = rules.map((rule) => rule.end_timestamp).filter(Boolean).sort();
  if (starts.length === 0 || ends.length === 0) return '';
  return `${formatNameTimestamp(starts[0])} to ${formatNameTimestamp(ends[ends.length - 1])}`;
}

function newRule(folderChoices: FolderChoice[]): RuleRow | null {
  const choice = folderChoices[0];
  if (!choice) return null;
  return {
    localId: crypto.randomUUID(),
    folder_id: choice.folder.id,
    start_timestamp: choice.min,
    end_timestamp: choice.max,
    stride: 1,
  };
}

function selectedText(trainingDataset: TrainingDataset): string {
  return `${trainingDataset.total_selected_images} images`;
}

export function TrainingDatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [inspectedDataset, setInspectedDataset] = useState<TrainingDataset | null>(null);
  const [loadedDataset, setLoadedDataset] = useState<TrainingDataset | null>(null);
  const [isEditingLoaded, setIsEditingLoaded] = useState(false);
  const [name, setName] = useState('');
  const [nameEdited, setNameEdited] = useState(false);
  const [usageLabel, setUsageLabel] = useState('train');
  const [usageFilter, setUsageFilter] = useState<string | null>(null);
  const [notes, setNotes] = useState('');
  const [rules, setRules] = useState<RuleRow[]>([]);
  const [preview, setPreview] = useState<TrainingDatasetPreview | null>(null);
  const [loading, setLoading] = useState(false);
  const isReadOnly = Boolean(loadedDataset && !isEditingLoaded);

  async function refresh() {
    const [nextDatasets, nextTrainingDatasets] = await Promise.all([
      listDatasets(),
      listTrainingDatasets(),
    ]);
    setDatasets(nextDatasets.filter((dataset) => dataset.status === 'ready'));
    setTrainingDatasets(nextTrainingDatasets);
  }

  useEffect(() => {
    refresh().catch((error) => {
      notifications.show({ color: 'red', title: 'Could not load train/test datasets', message: error.message });
    });
  }, []);

  const folderChoices = useMemo<FolderChoice[]>(
    () =>
      datasets.flatMap((dataset) =>
        dataset.folders
          .filter((folder) => folder.first_timestamp && folder.last_timestamp)
          .map((folder) => ({
            dataset,
            folder,
            value: String(folder.id),
            signature: folderImageSignature(folder),
            metadataLabel: folderMetadataLabel(folder),
            label: `${dataset.name} / ${folder.relative_path} (${folderMetadataLabel(folder)})`,
            min: toInputDateTime(folder.first_timestamp),
            max: toInputDateTime(folder.last_timestamp),
          })),
      ),
    [datasets],
  );

  const folderChoiceById = useMemo(
    () => new Map(folderChoices.map((choice) => [choice.folder.id, choice])),
    [folderChoices],
  );

  const folderOptions = useMemo(
    () => folderChoices.map((choice) => ({ value: choice.value, label: choice.label })),
    [folderChoices],
  );

  useEffect(() => {
    if (rules.length === 0 && folderChoices.length > 0) {
      const rule = newRule(folderChoices);
      if (rule) setRules([rule]);
    }
  }, [folderChoices, rules.length]);

  const invalidRules = useMemo(
    () =>
      rules.filter((rule) => {
        const choice = folderChoiceById.get(rule.folder_id);
        if (!choice) return true;
        return (
          !rule.start_timestamp ||
          !rule.end_timestamp ||
          rule.start_timestamp < choice.min ||
          rule.end_timestamp > choice.max ||
          rule.end_timestamp < rule.start_timestamp
        );
      }),
    [folderChoiceById, rules],
  );

  const generatedName = useMemo(() => generatedTrainingDatasetName(rules), [rules]);

  const selectedSignatures = useMemo(
    () =>
      rules
        .map((rule) => folderChoiceById.get(rule.folder_id)?.signature ?? null)
        .filter((signature): signature is string => Boolean(signature)),
    [folderChoiceById, rules],
  );

  const signatureError = useMemo(() => {
    if (rules.length === 0) return null;
    const missing = rules.some((rule) => !folderChoiceById.get(rule.folder_id)?.signature);
    if (missing) return 'Every selected folder must have indexed image metadata. Rescan the source dataset if metadata is missing.';
    const unique = [...new Set(selectedSignatures)];
    if (unique.length > 1) {
      return `All ranges in one train/test dataset must use the same image data. Found: ${unique.join(' / ')}.`;
    }
    return null;
  }, [folderChoiceById, rules, selectedSignatures]);

  const filteredTrainingDatasets = useMemo(
    () => trainingDatasets.filter((dataset) => !usageFilter || (dataset.usage_label ?? 'train') === usageFilter),
    [trainingDatasets, usageFilter],
  );

  useEffect(() => {
    if (!nameEdited && generatedName) {
      setName(generatedName);
    }
  }, [generatedName, nameEdited]);

  function updateRule(localId: string, patch: Partial<RuleRow>) {
    if (isReadOnly) return;
    setPreview(null);
    setRules((current) =>
      current.map((rule) => (rule.localId === localId ? { ...rule, ...patch } : rule)),
    );
  }

  function handleFolderChange(localId: string, folderId: string | null) {
    if (!folderId) return;
    const choice = folderChoiceById.get(Number(folderId));
    if (!choice) return;
    updateRule(localId, {
      folder_id: choice.folder.id,
      start_timestamp: choice.min,
      end_timestamp: choice.max,
    });
  }

  function handleAddRule() {
    if (isReadOnly) return;
    const rule = newRule(folderChoices);
    if (!rule) return;
    setRules((current) => [...current, rule]);
    setPreview(null);
  }

  function handleRemoveRule(localId: string) {
    if (isReadOnly) return;
    setRules((current) => current.filter((rule) => rule.localId !== localId));
    setPreview(null);
  }

  function rulesPayload(): TrainingDatasetRuleInput[] {
    return rules.map(({ localId: _localId, ...rule }) => rule);
  }

  async function handlePreview() {
    setLoading(true);
    try {
      const nextPreview = await previewTrainingDataset({ rules: rulesPayload() });
      setPreview(nextPreview);
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Preview failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleSave() {
    setLoading(true);
    try {
      await createTrainingDataset({
        name,
        usage_label: usageLabel,
        notes,
        rules: rulesPayload(),
      });
      setName('');
      setNameEdited(false);
      setUsageLabel('train');
      setNotes('');
      setPreview(null);
      await refresh();
      notifications.show({ color: 'green', title: 'Train/test dataset saved', message: name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Save failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  }

  function resetBuilder() {
    setLoadedDataset(null);
    setIsEditingLoaded(false);
    setName('');
    setNameEdited(false);
    setUsageLabel('train');
    setNotes('');
    setPreview(null);
    const rule = newRule(folderChoices);
    setRules(rule ? [rule] : []);
  }

  function applyLoadedDataset(details: TrainingDataset) {
    setLoadedDataset(details);
    setIsEditingLoaded(false);
    setName(details.name);
    setNameEdited(true);
    setUsageLabel(details.usage_label ?? 'train');
    setNotes(details.notes ?? '');
    setRules(
      details.rules.map((rule) => ({
        localId: crypto.randomUUID(),
        folder_id: rule.folder_id,
        start_timestamp: toInputDateTime(rule.start_timestamp),
        end_timestamp: toInputDateTime(rule.end_timestamp),
        stride: rule.stride,
      })),
    );
    setPreview(null);
  }

  async function handleLoad(trainingDataset: TrainingDataset) {
    try {
      const details = await getTrainingDataset(trainingDataset.id);
      applyLoadedDataset(details);
      notifications.show({ color: 'blue', title: 'Train/test dataset loaded', message: details.name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Load failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  async function handleUpdate() {
    if (!loadedDataset) return;
    setLoading(true);
    try {
      const updated = await updateTrainingDataset(loadedDataset.id, {
        name,
        usage_label: usageLabel,
        notes,
        rules: rulesPayload(),
      });
      applyLoadedDataset(updated);
      await refresh();
      notifications.show({ color: 'green', title: 'Train/test dataset updated', message: updated.name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Update failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleSaveAsNew() {
    setLoading(true);
    try {
      const saved = await createTrainingDataset({
        name,
        usage_label: usageLabel,
        notes,
        rules: rulesPayload(),
      });
      applyLoadedDataset(saved);
      await refresh();
      notifications.show({ color: 'green', title: 'Train/test dataset saved as new', message: saved.name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Save as new failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleCleanupInvalidRules(trainingDataset: TrainingDataset) {
    try {
      const updated = await cleanupTrainingDatasetInvalidRules(trainingDataset.id);
      if (loadedDataset?.id === updated.id) applyLoadedDataset(updated);
      if (inspectedDataset?.id === updated.id) setInspectedDataset(updated);
      await refresh();
      notifications.show({ color: 'green', title: 'Invalid rules cleaned up', message: updated.name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Cleanup failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  async function handleDelete(trainingDataset: TrainingDataset) {
    const confirmed = window.confirm(`Delete train/test dataset "${trainingDataset.name}"?`);
    if (!confirmed) return;

    try {
      await deleteTrainingDataset(trainingDataset.id);
      if (inspectedDataset?.id === trainingDataset.id) {
        setInspectedDataset(null);
      }
      if (loadedDataset?.id === trainingDataset.id) {
        resetBuilder();
      }
      await refresh();
      notifications.show({ color: 'green', title: 'Train/test dataset deleted', message: trainingDataset.name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Delete failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  async function handleInspect(trainingDataset: TrainingDataset) {
    try {
      const details = await getTrainingDataset(trainingDataset.id);
      setInspectedDataset(details);
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Inspect failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  const canSubmit = Boolean(name.trim() && rules.length > 0 && invalidRules.length === 0 && !signatureError);

  return (
    <Stack gap="lg">
      <div>
        <Title order={2}>Train/Test Datasets</Title>
        <Text c="dimmed" size="sm">
          Save reusable dataset rules from indexed folders, time ranges, and sampling stride. Labels are filters only;
          train sets can still be used for testing and test sets can still be used for training.
        </Text>
      </div>

      <StepCard title="Create train/test dataset" color="blue">
          {loadedDataset && (
            <Alert
              color={loadedDataset.is_update_locked ? 'yellow' : 'blue'}
              title={isReadOnly ? 'Loaded read-only' : 'Editing loaded dataset'}
            >
              <Stack gap="xs">
                <Text size="sm">
                  {loadedDataset.name} is loaded. {isReadOnly ? 'Press Edit to change it or save a copy.' : 'Changes can be updated in place or saved as a new dataset.'}
                </Text>
                {loadedDataset.update_lock_reasons.map((reason) => (
                  <Text key={reason} size="sm">
                    {reason}
                  </Text>
                ))}
                {loadedDataset.integrity_warnings.map((warning) => (
                  <Text key={warning} size="sm">
                    {warning}
                  </Text>
                ))}
              </Stack>
            </Alert>
          )}
          <Group align="flex-end" grow>
            <TextInput
              label="Name"
              placeholder="AE normal training set v1"
              value={name}
              description="Generated from the earliest selected start and latest selected end."
              disabled={isReadOnly}
              onChange={(event) => {
                setNameEdited(true);
                setName(event.currentTarget.value);
              }}
            />
            <Select
              label="Label"
              data={USAGE_OPTIONS}
              value={usageLabel}
              disabled={isReadOnly}
              onChange={(value) => setUsageLabel(value ?? 'train')}
            />
          </Group>
          <Textarea
            label="Notes"
            placeholder="Normal state, selected February ranges"
            value={notes}
            disabled={isReadOnly}
            onChange={(event) => setNotes(event.currentTarget.value)}
          />

          {folderChoices.length === 0 && (
            <Alert color="blue" title="No scanned folders">
              Add and scan at least one dataset before creating a train/test dataset.
            </Alert>
          )}

          {folderChoices.length > 0 && (
            <>
              <Group justify="space-between">
                <Title order={4}>Selection ranges</Title>
                <Button leftSection={<Plus size={18} />} variant="light" onClick={handleAddRule} disabled={isReadOnly}>
                  Add range
                </Button>
              </Group>

              <ScrollArea>
                <Table verticalSpacing="sm">
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Dataset / folder</Table.Th>
                      <Table.Th>Image data</Table.Th>
                      <Table.Th>Start</Table.Th>
                      <Table.Th>End</Table.Th>
                      <Table.Th>Allowed range</Table.Th>
                      <Table.Th>Stride</Table.Th>
                      <Table.Th>Preview</Table.Th>
                      <Table.Th />
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {rules.map((rule, index) => {
                      const choice = folderChoiceById.get(rule.folder_id);
                      const invalid = invalidRules.some((invalidRule) => invalidRule.localId === rule.localId);
                      return (
                        <Table.Tr key={rule.localId}>
                          <Table.Td>
                            <Select
                              data={folderOptions}
                              value={String(rule.folder_id)}
                              onChange={(folderId) => handleFolderChange(rule.localId, folderId)}
                              searchable
                              disabled={isReadOnly}
                            />
                          </Table.Td>
                          <Table.Td>
                            <Stack gap={2}>
                              <Text size="xs">{choice?.metadataLabel ?? 'n/a'}</Text>
                              <Text size="xs" c="dimmed">
                                {choice?.signature ?? 'Missing metadata'}
                              </Text>
                            </Stack>
                          </Table.Td>
                          <Table.Td>
                            <TextInput
                              type="datetime-local"
                              step={1}
                              min={choice?.min}
                              max={choice?.max}
                              error={invalid ? 'Out of range' : undefined}
                              value={rule.start_timestamp}
                              disabled={isReadOnly}
                              onChange={(event) =>
                                updateRule(rule.localId, { start_timestamp: event.currentTarget.value })
                              }
                            />
                          </Table.Td>
                          <Table.Td>
                            <TextInput
                              type="datetime-local"
                              step={1}
                              min={choice?.min}
                              max={choice?.max}
                              error={invalid ? 'Out of range' : undefined}
                              value={rule.end_timestamp}
                              disabled={isReadOnly}
                              onChange={(event) =>
                                updateRule(rule.localId, { end_timestamp: event.currentTarget.value })
                              }
                            />
                          </Table.Td>
                          <Table.Td>
                            <Text size="xs" c="dimmed">
                              {choice ? `${formatTimestamp(choice.min)} to ${formatTimestamp(choice.max)}` : 'n/a'}
                            </Text>
                          </Table.Td>
                          <Table.Td>
                            <NumberInput
                              min={1}
                              value={rule.stride}
                              disabled={isReadOnly}
                              onChange={(value) =>
                                updateRule(rule.localId, { stride: typeof value === 'number' ? value : 1 })
                              }
                            />
                          </Table.Td>
                          <Table.Td>
                            {preview?.rules[index]
                              ? `${preview.rules[index].selected_images} / ${preview.rules[index].matching_images}`
                              : 'n/a'}
                          </Table.Td>
                          <Table.Td>
                            <ActionIcon
                              color="red"
                              variant="subtle"
                              aria-label="Remove rule"
                              onClick={() => handleRemoveRule(rule.localId)}
                              disabled={isReadOnly}
                            >
                              <Trash2 size={18} />
                            </ActionIcon>
                          </Table.Td>
                        </Table.Tr>
                      );
                    })}
                  </Table.Tbody>
                </Table>
              </ScrollArea>

              {invalidRules.length > 0 && (
                <Alert color="red" title="Invalid time range">
                  Start and end must stay inside the selected folder bounds.
                </Alert>
              )}

              {signatureError && (
                <Alert color="red" title="Image data mismatch">
                  {signatureError}
                </Alert>
              )}

              {preview && (
                <Alert color="green" title="Preview">
                  {preview.total_selected_images} selected images from {preview.total_matching_images} matching images.
                </Alert>
              )}

              <Group justify="space-between">
                <Button variant="subtle" onClick={resetBuilder}>
                  Reset
                </Button>
                <Group>
                <Button
                  variant="light"
                  onClick={handlePreview}
                  loading={loading}
                  disabled={rules.length === 0 || invalidRules.length > 0 || Boolean(signatureError)}
                >
                  Preview counts
                </Button>
                {loadedDataset ? (
                  <>
                    <Button
                      leftSection={<Edit3 size={18} />}
                      variant="light"
                      onClick={() => setIsEditingLoaded(true)}
                      disabled={!isReadOnly || loadedDataset.is_update_locked}
                    >
                      Edit
                    </Button>
                    {isEditingLoaded && (
                      <Button leftSection={<Save size={18} />} onClick={handleUpdate} loading={loading} disabled={!canSubmit}>
                        Update
                      </Button>
                    )}
                    <Button leftSection={<Copy size={18} />} onClick={handleSaveAsNew} loading={loading} disabled={!canSubmit}>
                      Save as new
                    </Button>
                  </>
                ) : (
                  <Button
                    leftSection={<Save size={18} />}
                    onClick={handleSave}
                    loading={loading}
                    disabled={!canSubmit}
                  >
                    Save train/test dataset
                  </Button>
                )}
                </Group>
              </Group>
            </>
          )}
      </StepCard>

      <StepCard
        title="Saved train/test datasets"
        color="cyan"
        action={
          <Select
            label="Label filter"
            data={USAGE_OPTIONS}
            value={usageFilter}
            onChange={setUsageFilter}
            clearable
          />
        }
      >
          <ScrollArea>
            <Table verticalSpacing="sm" striped>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>Label</Table.Th>
                  <Table.Th>Source paths</Table.Th>
                  <Table.Th>Image data</Table.Th>
                  <Table.Th>Images</Table.Th>
                  <Table.Th>Created</Table.Th>
                  <Table.Th>Notes</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {filteredTrainingDatasets.map((trainingDataset) => (
                  <Table.Tr key={trainingDataset.id}>
                    <Table.Td>
                      <Stack gap={4}>
                        <Text>{trainingDataset.name}</Text>
                        {trainingDataset.invalid_rule_count > 0 && (
                          <Badge color="yellow" variant="light">
                            {trainingDataset.invalid_rule_count} invalid rule(s)
                          </Badge>
                        )}
                      </Stack>
                    </Table.Td>
                    <Table.Td>
                      <Badge color={usageLabelColor(trainingDataset.usage_label ?? 'train')} variant="light">
                        {usageLabelText(trainingDataset.usage_label ?? 'train')}
                      </Badge>
                    </Table.Td>
                    <Table.Td>
                      <Group gap="xs">
                        {trainingDataset.dataset_names.map((datasetName) => (
                          <Badge key={datasetName} variant="light">
                            {datasetName}
                          </Badge>
                        ))}
                      </Group>
                    </Table.Td>
                    <Table.Td>
                      <Group gap={4}>
                        {(trainingDataset.image_signatures ?? []).length > 0 ? (
                          (trainingDataset.image_signatures ?? []).map((signature) => (
                            <Badge key={signature} size="xs" variant="outline" color="gray">
                              {signature}
                            </Badge>
                          ))
                        ) : (
                          <Badge size="xs" variant="outline" color="gray">
                            n/a
                          </Badge>
                        )}
                      </Group>
                    </Table.Td>
                    <Table.Td>{selectedText(trainingDataset)}</Table.Td>
                    <Table.Td>{formatTimestamp(trainingDataset.created_at)}</Table.Td>
                    <Table.Td>{trainingDataset.notes ?? ''}</Table.Td>
                    <Table.Td>
                      <Group gap="xs" justify="flex-end">
                        <Button size="xs" variant="light" onClick={() => handleLoad(trainingDataset)}>
                          Load
                        </Button>
                        <Tooltip label="Inspect ranges">
                          <ActionIcon
                            variant="subtle"
                            aria-label="Inspect train/test dataset"
                            onClick={() => handleInspect(trainingDataset)}
                          >
                            <Info size={18} />
                          </ActionIcon>
                        </Tooltip>
                        {trainingDataset.invalid_rule_count > 0 && (
                          <Button
                            size="xs"
                            variant="light"
                            color="yellow"
                            onClick={() => handleCleanupInvalidRules(trainingDataset)}
                            disabled={trainingDataset.is_update_locked}
                          >
                            Cleanup
                          </Button>
                        )}
                        <Tooltip label="Delete">
                          <ActionIcon
                            color="red"
                            variant="subtle"
                            aria-label="Delete train/test dataset"
                            onClick={() => handleDelete(trainingDataset)}
                          >
                            <Trash2 size={18} />
                          </ActionIcon>
                        </Tooltip>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
      </StepCard>

      <Modal
        opened={inspectedDataset !== null}
        onClose={() => setInspectedDataset(null)}
        title={inspectedDataset ? `Train/test dataset: ${inspectedDataset.name}` : 'Train/test dataset'}
        size="xl"
      >
        {inspectedDataset && (
          <Stack gap="md">
            <Alert color="blue" title="Image count">
              {usageLabelText(inspectedDataset.usage_label ?? 'train')} set with {inspectedDataset.total_selected_images} selected
              images from {inspectedDataset.total_matching_images} matching images.
            </Alert>
            <ScrollArea h={420}>
              <Table verticalSpacing="sm">
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>#</Table.Th>
                    <Table.Th>Dataset</Table.Th>
                    <Table.Th>Folder</Table.Th>
                    <Table.Th>Image data</Table.Th>
                    <Table.Th>Start</Table.Th>
                    <Table.Th>End</Table.Th>
                    <Table.Th>Stride</Table.Th>
                    <Table.Th>Images</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {inspectedDataset.rules.map((rule, index) => (
                    <Table.Tr key={rule.id}>
                      <Table.Td>{index + 1}</Table.Td>
                      <Table.Td>
                        <Stack gap={2}>
                          <Text size="sm">{rule.dataset_name}</Text>
                          <Text size="xs" c="dimmed" className="mono">
                            {rule.dataset_root_path}
                          </Text>
                        </Stack>
                      </Table.Td>
                      <Table.Td className="mono">{rule.folder_relative_path}</Table.Td>
                      <Table.Td>
                        <Stack gap={2}>
                          <Text size="xs">{rule.folder_image_signature ?? 'n/a'}</Text>
                          <Text size="xs" c="dimmed">
                            {rule.folder_image_metadata
                              ? `${String(rule.folder_image_metadata.dtype ?? 'dtype n/a')}, ${String(
                                  rule.folder_image_metadata.channels ?? 'channels n/a',
                                )}ch`
                              : 'metadata n/a'}
                          </Text>
                        </Stack>
                      </Table.Td>
                      <Table.Td>{formatTimestamp(rule.start_timestamp)}</Table.Td>
                      <Table.Td>{formatTimestamp(rule.end_timestamp)}</Table.Td>
                      <Table.Td>{rule.stride}</Table.Td>
                      <Table.Td>
                        {rule.selected_images} / {rule.matching_images}
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </ScrollArea>
          </Stack>
        )}
      </Modal>
    </Stack>
  );
}
